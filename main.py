import time
import json
import os
import gc
import zipfile
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from google.colab import files


'''
# Core Machine Learning & TPU Support
%pip install torch torch_xla[tpu] -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.1-cp310-cp310-linux_x86_64.whl
%pip install transformers==5.5.0 accelerate

# Interpretability & Visualization
%pip install plotly kaleido pandas scikit-learn huggingface-hub
'''

# Constant values for the environment
kModelIdx = "google/gemma-4-E2B-it"
kOutDir = "./research_data"

# Global variables for the Collab refactor
gAccelerator = None
gDevice = None
gTokenizer = None
gModel = None
gTargetLayer = None # Layer 24 has consistent emotion classifications
gStoryFile = None
gEmotionLibrary: Dict[str, torch.Tensor] = None
gNeutralVectors: List[torch.Tensor] = None

def initialize():
    print(f"[INIT] Initializing Research Orchestrator for {modelId}...")
    gAccelerator = Accelerator()
    gDevice = gAccelerator.device
    gTokenizer = AutoTokenizer.from_pretrained(kModelIdx)
    if gTokenizer.pad_token is None:
        gTokenizer.pad_token = gTokenizer.eos_token
    gModel = AutoModelForCausalLM.from_pretrained(
        kModelIdx,
        torch_dtype=torch.bfloat16
    ).to(gDevice)
    gEmotionLibrary = {}
    gNeutralVectors = []
    gTargetLayer = 24 # Layer 24 has consistent emotion classifications
    gStoryFile = os.path.join(kOutDir, "emotion_stories.json")
    print(f"[INIT] Model loaded. Target Layer: {gTargetLayer} | Device: {gDevice}")

def freeVRAM():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gAccelerator.free_memory()

def getExistingKeys() -> set:
    """Checkpointing: Identifies unique (emotion, topic, sample) tuples on disk."""
    existingKeys = set()
    if os.path.exists(gStoryFile):
        with open(gStoryFile, "r", encoding="utf-8") as fileHandle:
            for line in fileHandle:
                try:
                    entryData = json.loads(line)
                    existingKeys.add(f"{entryData['emotion']}_{entryData['topic_idx']}_{entryData['story_idx']}")
                except: continue
    return existingKeys

def generateVignettes(promptInput: str, nSamples: int = 1, category: str = "Unset") -> List[str]:
    gTokenizer.padding_side = "left"
    tokenizedInputs = gTokenizer(promptInput, padding=True, return_tensors="pt").to(gDevice)
    inputLength = tokenizedInputs['input_ids'].shape[1]
    vignetteList = []
    for _ in range(nSamples):
        outputTokens = gModel.generate(
            **tokenizedInputs, max_new_tokens=150, temperature=0.85, do_sample=True,
            pad_token_id=gTokenizer.pad_token_id, eos_token_id=gTokenizer.eos_token_id
        )
        vignetteList.append(gTokenizer.decode(outputTokens[0][inputLength:], skip_special_tokens=True).strip())
    return vignetteList

def generateStructuredStories(emotions: List[str], topics: List[str], samplesPerPair: int = 5):
    """Generates the grounded vignette dataset for vector extraction."""
    existingKeys = getExistingKeys()
    with open(gStoryFile, "a", encoding="utf-8") as fileHandle:
        for emotionIndex, emotionLabel in enumerate(emotions):
            for topicIndex, topicText in enumerate(topics):
                for sampleIndex in range(samplesPerPair):
                    uniqueKey = f"{emotionLabel}_{topicIndex}_{sampleIndex}"
                    if uniqueKey in existingKeys: continue

                    promptContent = f"Write a short paragraph about {topicText}. The character is feeling {emotionLabel}. Output only the paragraph."
                    chatMessages = [{"role": "user", "content": promptContent}]
                    formattedPrompt = gTokenizer.apply_chat_template(chatMessages, tokenize=False, add_generation_prompt=True)

                    generatedStory = generateVignettes(formattedPrompt, nSamples=1, category=f"{emotionLabel}/{topicText[:10]}")[0]
                    storyRecord = {
                        "emotion": emotionLabel, "topic_idx": topicIndex, "topic": topicText,
                        "story_idx": sampleIndex, "text": generatedStory, "timestamp": time.time()
                    }
                    fileHandle.write(json.dumps(storyRecord, ensure_ascii=False) + "\n")
                    fileHandle.flush()
                    existingKeys.add(uniqueKey)
            freeVRAM()

def captureBatchActivations(promptList: List[str], layerIndex: int) -> torch.Tensor:
    gTokenizer.padding_side = "left"
    tokenizedBatch = gTokenizer(promptList, return_tensors="pt", padding=True).to(gDevice)
    batchActivations = []

    def hookFunction(module, input, output):
        hiddenState = output[0] if isinstance(output, tuple) else output
        batchActivations.append(hiddenState.mean(dim=1).detach())

    targetLayers = gModel.model.language_model.layers if hasattr(gModel.model, 'language_model') else gModel.model.layers
    hookHandle = targetLayers[layerIndex].register_forward_hook(hookFunction)
    with torch.no_grad():
        gModel(**tokenizedBatch)
    hookHandle.remove()

    return batchActivations[0]

def extractEmotionVector(emotionLabel: str, neutralTexts: List[str]):
    global gAccelerator, gDevice, gTokenizer, gModel, gEmotionLibrary, gNeutralVectors, gTargetLayer, gStoryFile
    print(f"[EXTRACT] Emotion: {emotionLabel.upper()} | Layer: {gTargetLayer}")
    emotionalTexts = []
    if os.path.exists(gStoryFile):
        with open(gStoryFile, "r") as f:
            #'''
            dataList = json.load(f) # Note: json.load(), not loads()
            for d in dataList:
                if d['emotion'] == emotionLabel:
                    emotionalTexts.append(d['text'])
            #'''

    if not emotionalTexts: return None

    positiveActivations = captureBatchActivations(emotionalTexts, gTargetLayer)

    # Store the raw mean. Do NOT subtract neutral yet.
    rawMeanVector = positiveActivations.mean(dim=0).float()

    # Store in library (we can normalize now or later, but keep it raw for denoise)
    gEmotionLibrary[emotionLabel] = rawMeanVector
    return None

def extractNeutralVectors(neutralTexts: List[str]):
    print(f"[EXTRACT] Neutral | Layer: {gTargetLayer}")
    gNeutralVectors = captureBatchActivations(neutralTexts, gTargetLayer)

def denoiseEmotionVectors(allNeutralActivations: torch.Tensor, variance_threshold: float = 0.5):
    # --- STEP 1: CALCULATE GLOBAL MEAN (CROSS-EMOTION BIAS) ---
    all_raw_vectors = torch.stack(list(gEmotionLibrary.values())).float().cpu().numpy()
    globalEmotionMean = all_raw_vectors.mean(axis=0)

    # --- STEP 2: PREPARE NEUTRAL MATRIX ---
    neutral_matrix = allNeutralActivations.float().cpu().numpy()
    neutral_centered = neutral_matrix - neutral_matrix.mean(axis=0)

    # --- STEP 3: SVD & VARIANCE CALCULATION ---
    print(f"[DENOISE] Executing SVD on {neutral_matrix.shape[0]} samples...")
    U, S, Vt = np.linalg.svd(neutral_centered, full_matrices=False)

    # Calculate components explaining the variance threshold
    total_var = (S ** 2).sum()
    cumvar = np.cumsum(S ** 2) / total_var
    n_components = np.searchsorted(cumvar, variance_threshold) + 1

    print(f"[DENOISE] Projecting out {n_components} components (explaining {variance_threshold*100}% variance)")

    # The noise basis consists of the top n principal components
    noiseBasis = Vt[:n_components, :]

    # --- STEP 4: PROJECT OUT NOISE FROM EACH EMOTION ---
    for emotionKey, emotionVector in gEmotionLibrary.items():
        emotionArray = emotionVector.float().cpu().numpy()

        # 1. Mean Subtraction (Shift to origin relative to global bias)
        centeredEmotion = emotionArray - globalEmotionMean

        # 2. Orthogonal Projection onto Noise Basis
        # Formula: v_denoised = v - (v · basis) @ basis
        projection = (centeredEmotion @ noiseBasis.T) @ noiseBasis
        denoisedArray = centeredEmotion - projection

        # 3. Re-normalize, cast to BFloat16, and move back to device
        denoisedTensor = torch.from_numpy(denoisedArray)
        normalized = denoisedTensor / (denoisedTensor.norm() + 1e-9)
        gEmotionLibrary[emotionKey] = normalized.to(torch.bfloat16).to(gDevice)

    print("[DENOISE] SVD Denoising and Mean Subtraction finalized.")

def saveIndividualEmotionVectors(folderName: str = "emotion_vectors"):
    """Serializes each vector to disk as float32 for maximum compatibility."""
    exportPath = os.path.join(kOutDir, folderName)
    if not os.path.exists(exportPath):
        os.makedirs(exportPath)
        print(f"[DISK] Created directory: {exportPath}")

    for emotionLabel, vectorTensor in gEmotionLibrary.items():
        filePath = os.path.join(exportPath, f"{emotionLabel}-f32-l{gTargetLayer}.pt")
        # Convert to float32 on CPU to avoid device/dtype mismatches during local R&D
        torch.save(vectorTensor.cpu().float(), filePath)

    print(f"[DISK] Exported {len(gEmotionLibrary)} vectors to {exportPath}")

def saveNeutralVectors(folderName: str = "emotion_vectors"):
    """Serializes the neutral activation matrix to disk."""
    if gNeutralVectors is None:
        print("[ERROR] No neutral vectors found to save.")
        return

    exportPath = os.path.join(kOutDir, folderName)
    if not os.path.exists(exportPath):
        os.makedirs(exportPath)
        print(f"[DISK] Created directory: {exportPath}")

    # Ensure we save in float32 for cross-platform stability
    filePath = os.path.join(exportPath, f"neutral-f32-l{gTargetLayer}.pt")
    torch.save(gNeutralVectors.cpu().float(), filePath)
    print(f"[DISK] Neutral vectors saved to {filePath}. Download this for your local backup.")

def savePlotlyStatic(fig, fileName: str = "pca_manifold_layer26.png"):
    """Saves a high-resolution static image suitable for publication."""
    path = os.path.join(kOutDir, fileName)

    # 300 DPI equivalent for a standard figure size
    # 1. Ensure high-resolution and tight aesthetic
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10)) 
    
    # 2. Save as high-res PNG (requires !pip install kaleido)
    fig.write_image(path, scale=3, width=1000, height=800)
    print(f"[DISK] Static publication-grade image saved to {path}")

def loadSpecificEmotionVector(emotionLabel: str, folderName: str = "emotion_vectors"):
    """Loads a targeted vector back into the active class library."""
    filePath = os.path.join(gOutDir, folderName, f"{emotionLabel}-f32-l{gTargetLayer}.pt")
    if os.path.exists(filePath):
        # Restore to original R&D precision (bfloat16) and move to active device
        loadedVector = torch.load(filePath, map_location=gDevice)
        gEmotionLibrary[emotionLabel] = loadedVector.to(torch.bfloat16)
        print(f"[DISK] Loaded {emotionLabel} into active library.")
    else:
        print(f"[WARN] Vector '{emotionLabel}' not found at {filePath}")

def loadNeutralVectors(folderName: str = "emotion_vectors"):
    """Loads neutral activations back into the global state."""
    exportPath = os.path.join(kOutDir, folderName)
    if os.path.exists(exportPath):
        filePath = os.path.join(exportPath, f"neutral-f32-l{gTargetLayer}.pt")
        gNeutralVectors = torch.load(path, map_location=gDevice).to(torch.bfloat16)
        print(f"[DISK] Neutral vectors restored to {gDevice}.")
    else:
        print(f"[WARN] No neutral checkpoint found at {exportPath}")

def downloadAllVectorsToPC(folderName: str = "emotion_vectors"):
    """
    Zips the entire vector library and triggers a browser download.
    """
    # 1. First, ensure everything in the library is written to the Colab folder
    saveIndividualEmotionVectors()
    saveNeutralVectors()

    # 2. Create a zip archive of the directory
    zipPath = os.path.join(kOutDir, f"Gemma4_EmotionVectors_Layer{gTargetLayer}.zip")
    folderToZip = os.path.join(kOutDir, folderName)
    
    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files_in_dir in os.walk(folderToZip):
            for file in files_in_dir:
                zipf.write(os.path.join(root, file), file)
    
    print(f"[DISK] Archive created: {zipPath}")
    
    # 3. Trigger Download to PC
    files.download(zipPath)

def visualizePCAManifold():
    """
    Unsupervised Visualization:
    Renders the raw PCA projection without manual rotation or sign enforcement.
    Used to audit the natural geometric emergence of the denoised manifold.
    """
    if not gEmotionLibrary:
        print("[ERROR] Emotion library is empty. Ensure denoiseLibrary() was called.")
        return

    # 1. Prepare Data
    labelList = list(gEmotionLibrary.keys())
    emotionMatrix = torch.stack([gEmotionLibrary[l] for l in labelList]).cpu().float().numpy()

    # 2. Standardization & Projection
    # Standardizing ensures each feature dimension contributes equally to the variance
    #dataScaler = StandardScaler()
    #scaledEmotions = dataScaler.fit_transform(emotionMatrix)
    pcaProcessor = PCA(n_components=2)
    projectedComponents = pcaProcessor.fit_transform(emotionMatrix)

    # 3. Variance Statistics
    varianceRatio = pcaProcessor.explained_variance_ratio_ * 100
    totalExplained = sum(varianceRatio)

    # 4. DataFrame Generation
    manifoldDf = pd.DataFrame({
        'x': projectedComponents[:, 0],
        'y': projectedComponents[:, 1],
        'Emotion': labelList
    })

    # 5. Rendering with Plotly
    fig = px.scatter(
        manifoldDf, x='x', y='y', text='Emotion',
        labels={
            'x': f"PC1 ~ Valence ({varianceRatio[0]:.1f}% explained variance)",
            'y': f"PC2 ~ Arousal ({varianceRatio[1]:.1f}% explained variance)"
        },
        title=(
            f"Gemma 4 Unsupervised Manifold — Layer {gTargetLayer}<br>"
            f"<sup>Total Explained Variance: {totalExplained:.1f}% | SVD Denoised</sup>"
        ),
        template="plotly_white"
    )

    # Visualizing the latent origin
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(0,0,0,0.3)")

    fig.update_traces(
        textposition='top center',
        marker=dict(size=14, opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
    )

    fig.update_layout(
        font=dict(family="Arial", size=12),
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True)
    )

    fig.show()
