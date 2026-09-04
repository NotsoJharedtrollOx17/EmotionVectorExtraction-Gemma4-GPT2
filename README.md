# EmotionVectorExtraction-Gemma4-GPT2

Partial replication of Anthropic's Emotion Vector findings using Gemma 4 E2B and GPT-2 Medium inside a Google Colab T4 Notebook

## Authors

- Abraham Jhared Flores Azcona _(NotsoJharedtrollOx17)_  ``abrahamjhared.flores@gmail.com``

## Extended Preprint

The complete 47-page preprint is available as [`PartialReplicationEmotionVectors.pdf`](./PartialReplicationEmotionVectors.pdf). It provides the extended methodology, experimental setup, results, discussion, limitations, appendices, and references behind this repository's simplified results summary. The LaTeX source is maintained in the companion [`preprint-EmotionVectorExtraction-Gemma4-GPT2`](https://github.com/NotsoJharedtrollOx17/preprint-EmotionVectorExtraction-Gemma4-GPT2) repository.

## Abstract

We discuss a first-pass, resource-constrained partial replication of selected analyses from Anthropic's Emotion Vectors study. These analyses encompass (i) PCA projections interpreted in relation to valence and arousal, (ii) pairwise cosine similarity between emotion vectors, and (iii) activation-steering diagnostics. Anthropic reported that emotion-related directions in Claude Sonnet 4.5 organize internal representations and can influence model outputs. Related open-weight replication attempts have reported some comparable patterns in _meta-llama/Llama-3.3-70B-Instruct_ and _google/gemma-4-E4B_, while _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_ have received less attention in this setting. Our contribution applies the same broad analysis family to these two models in Google Colab using generated story corpora and nine- and 20-emotion label sets. We observe (1) a partial valence-like separation along PC1 in both models, (2) local similarity clusters among related emotion labels, and (3) model-dependent changes in token probabilities and sampled text under steering with coefficient 0.5. GPT-2 Medium generally shows sharper but more repetitive and brittle steering diagnostics, whereas Gemma 4 E2B shows more moderate and multilingual token-level effects. These observations are consistent with some previously reported geometric and intervention patterns, but they characterize this particular corpus-and-pipeline combination. The stories are uncurated, the analysis does not use external affect ratings or random-direction controls, and the steering outputs are prompt-dependent and stochastic; accordingly, the results do not establish universal or uniquely emotional representations.

## Introduction

Recent work from [5] introduced the term _emotion vectors_ for directions in Claude Sonnet 4.5's residual stream that represent broad emotion concepts. The authors reported that these directions generalize beyond explicit emotion words, organize partly along the psychological dimensions of valence and arousal, and influence generated behavior under activation steering. These findings motivate a narrower generality question: whether comparable affect-associated directions can also be recovered in independently available transformer language models.

This repository provides a first-pass, resource-constrained partial replication of selected analyses using _openai-community/gpt-2-medium_ [3] and _google/gemma-4-E2B_ [4]. We examine Logit Lens vocabulary projections, PCA and pairwise cosine similarity, and activation steering on two fixed prompts using nine- and 20-emotion label sets. The purpose is not to reproduce every behavioral or safety claim from [5], or to validate the Gemma 4 E4B results in [2], but to add two inexpensive model cases to the emerging comparison literature.

Under this pipeline, both models show several label-consistent vocabulary projections, local cosine-similarity families, and a partial valence-like separation along PC1. Activation addition produces prompt- and direction-dependent changes in next-token probabilities and sampled text. GPT-2 Medium often shows sharper diagonal token effects together with repetitive and brittle continuations, whereas Gemma 4 E2B generally shows more moderate, multilingual, and sometimes unrelated effects. Steering is causal in the limited computational sense that the intervention precedes and changes model outputs; these diagnostics do not establish subjective experience, uniquely emotional control, or a universal internal mechanism.

## Methodology

It is crucial to emphasize our limited experimental scope compared to [5], which helps to understand the scale of our results. To further illustrate this, a summarization is provided on the table down below [Table 1].

| | **Anthropic [5]** | **Our Work** |
|---|---|---|
| **Model** | Claude Sonnet 4.5 | GPT-2 Medium and Gemma 4 E2B |
| **Emotions tested** | 171 | 9 and 20 |
| **Stories generated** | 205,200 | 2,000 |
| **Team** | ~16 researchers | 1 researcher + ChatGPT + Gemini |
| **Hardware** | Internal compute cluster | 2 Google Colab Notebooks with Free-Tier T4 NVIDIA GPUs |

**Table 1.** Experiment Scope Comparison of Anthropic [5] versus Ours. _Note:_ Inspired by a similar table found on [2].

Overall, we utilized the data pipeline found in [2] and then adapted the codebase to complement our data requirements and probing experiments. This section is subdivided with the following methodology steps: (1) Story Generation, (2) Emotion Vector Extraction, (3) Logit Lens, (4) Emotion Probe Steering, (5) PCA Projection, and (6) Cosine Similarity Heatmap. All steps are performed on both  _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_.

### 1. Story Generation

We used a modified version of the prompt template<sup>1</sup> from [5] and asked the web version of Gemini to generate batches of five stories for each of ten topics in JSON format. The generated batches were assembled into the committed corpus of 100 stories for each emotion. Stories were retained without manual rewriting or post-hoc quality filtering. This saved limited free-tier Colab compute, but it also means that the results characterize this generated corpus and may reflect lexical leakage or other corpus artifacts. We likewise used a neutral prompt<sup>2</sup> template with Gemini to extend a list of emotionally neutral commands and statements.

> <sup>1</sup> Full template prompt available at [`./research_data/PromptDataForEmotions.txt`](./research_data/PromptDataForEmotions.txt)

> <sup>2</sup> Full template prompt available at [`./research_data/PromptDataForNeutral`](./research_data/PromptDataForNeutral.txt)

### 2. Emotion Vector Extraction

For each story, hidden states were mean-pooled from token index 50 onward, or from the midpoint for shorter sequences, at fixed numeric index 16 for GPT-2 Medium and 23 for Gemma 4 E2B. We averaged the pooled story representations within each emotion, subtracted a shared neutral mean as an intermediate step, and then subtracted the global mean of the neutral-adjusted emotion prototypes. Because the same neutral mean is subtracted from every prototype, it cancels during global centering; the final directions are globally emotion-centered and then L2-normalized. The layers follow a source-informed two-thirds-depth convention and were not selected through a layer sweep. Extraction uses `outputs.hidden_states[layerIndex]`, while steering hooks the transformer block with the same numeric index, so the embedding entry at hidden-state index zero creates a possible one-block alignment offset.

### 3. Logit Lens

Each normalized direction was projected through the model's final normalization and language-model head, and the resulting vocabulary scores were standardized as z-scores. The highest- and lowest-scoring tokens were stored in JSON files for the next step. These projections document vocabulary items aligned with each direction; they are diagnostics rather than independent validation that every direction uniquely represents its label.

### 4. Emotion Probe Steering

For each of two fixed prompts, we compared baseline and steered next-token log probabilities at coefficient `alpha=0.5`. Each heatmap cell averages the change for the five top Logit Lens tokens associated with its target direction. Because these target tokens come from the same vectors used for steering, the heatmaps are direction-consistency diagnostics rather than independent semantic tests. The implementation scales the intervention by each current token's residual norm, so `alpha=0.5` is a source-informed operating point rather than a numerically identical reproduction of [5] or an optimized cross-model setting. Sampled continuations provide a complementary qualitative diagnostic, but they are stochastic and were not repeated across a prespecified set of seeds.

### 5. PCA Projection

We performed Principal Component Analysis on the normalized emotion vectors to visualize their leading variance directions. The components were estimated without external emotion ratings, so valence- or arousal-like interpretations are descriptive observations made after inspecting the labeled plots. This process is executed for the nine- and 20-emotion sets.

### 6. Cosine Similarity Heatmap

We calculated and plotted pairwise cosine similarities between the normalized emotion vectors to inspect directional alignment, local similarity families, and oppositions. These descriptive patterns do not establish a universal emotion manifold. This process is done for the nine- and 20-emotion sets.

## How to replicate the findings

The fastest way to replicate the results is to use one of the four saved Colab notebooks inside [`./scripts/`](./scripts/). They are snapshots of the same configurable notebook, with their saved outputs corresponding to the model, emotion count, and layer included in each filename:

| **Model** | **Emotion set** | **Layer** | **Notebook** |
|---|---:|---:|---|
| GPT-2 Medium | 9 | 16 | [`emotion_vector_replication-GPT2Medium-9emotions-layer16.ipynb`](./scripts/emotion_vector_replication-GPT2Medium-9emotions-layer16.ipynb) |
| GPT-2 Medium | 20 | 16 | [`emotion_vector_replication-GPT2Medium-20emotions-layer16.ipynb`](./scripts/emotion_vector_replication-GPT2Medium-20emotions-layer16.ipynb) |
| Gemma 4 E2B | 9 | 23 | [`emotion_vector_replication-Gemma4E2B-9emotions-layer23.ipynb`](./scripts/emotion_vector_replication-Gemma4E2B-9emotions-layer23.ipynb) |
| Gemma 4 E2B | 20 | 23 | [`emotion_vector_replication-Gemma4E2B-20emotions-layer23.ipynb`](./scripts/emotion_vector_replication-Gemma4E2B-20emotions-layer23.ipynb) |

1. Open the desired notebook in Google Colab and select a T4 GPU runtime.
2. Clone this repository into the runtime and move into its root so the notebook can resolve `./research_data/`:

   ```python
   !git clone https://github.com/NotsoJharedtrollOx17/EmotionVectorExtraction-Gemma4-GPT2.git
   %cd EmotionVectorExtraction-Gemma4-GPT2
   ```

3. Run the dependency cells. The saved runs use `transformers==5.5.0` and `kaleido==0.2.1`. Gemma 4 E2B may also require a Hugging Face account with model access configured in the Colab runtime.
4. In the configuration cells, keep the model, emotion list, and target layer that match the selected notebook filename. The notebook source contains both model choices, both emotion lists, and both layer choices, so blindly executing every alternative assignment will leave the last option active.
5. When reproducing the committed findings, skip the optional `generateStructuredStories(...)` call. The analysis reads the existing per-emotion JSON files in [`./research_data/emotion_stories/`](./research_data/emotion_stories/), which now contain exactly 100 stories for each of the 20 emotions.
6. Run the extraction, Logit Lens, steering, PCA, and cosine-similarity cells in order. Saved vectors, Delta Log Probability data, and the corresponding plots can be compared against [`./research_data/emotion_vectors/`](./research_data/emotion_vectors/), [`./research_data/emotion_logits/`](./research_data/emotion_logits/), and [`./plots/`](./plots/).

The PCA and cosine results should reproduce from the saved corpus and configuration. Generated steering text can vary between runs because decoding is stochastic, so compare its overall behavior rather than expecting character-for-character output.

_Dataset cleanup note:_ The committed result artifacts were originally generated when `calm_stories.json` contained 110 entries. The final ten accidental entries have since been removed so every emotion now contributes 100 stories. Trial reruns found no meaningful change to the reported structures, although a fresh run can show small numerical differences from the committed artifacts.

## Results

The following subsection is further subdivided into two sub-subsections: (1) GPT-2 Medium, and (2) Gemma 4 E2B. In summary, emotion vector extraction produces label-related directions in both models, while emotion probing changes the models' outputs. Results coming from Prompt01 are omitted from this README for brevity, although their saved artifacts remain available in the repository. To fully audit our findings, we provide inside [`./scripts/`](./scripts/) the Colab notebooks containing the results presented in this section.

### 1. GPT-2 Medium

#### i. Logit Lens

For GPT-2 Medium, several projected token lists align directly with their labels. For example, the `happy`, `calm`, `loving`, `angry`, and `desperate` vectors rank label-related tokens such as `joyful`, `relaxation`, `nurturing`, `angrily`, and `frantically`. Other rows are less direct and include fragments or unrelated vocabulary.

The tables therefore provide examples of label-associated vocabulary alongside counterexamples; they document the projected directions rather than independently validating stable or uniquely emotional representations. Tokens can be inspected in [Table 2] and [Table 3].

#### ii. Emotion Steering

Activation addition in GPT-2 Medium changes next-token probabilities and sampled continuations in prompt- and direction-dependent ways. Several heatmaps contain comparatively strong diagonal cells, but off-diagonal effects also occur. Some sampled outputs repeat label-associated words or stems, while others degrade or drift into unrelated continuations. The steering heatmaps can be inspected in [Figure 5] and [Figure 6].

These results establish intervention sensitivity for the implemented directions, not uniformly specific emotional control. Representative text output can be inspected in [Table 4].

#### iii. PCA Projection

Principal Component Analysis applied to the extracted emotion vectors revealed structured but model-dependent organization. In both GPT-2 Medium plots, PC1 provides a partial valence-like separation: several positive and negative labels occupy different sides of the projection, while related labels form local groups. The 20-emotion plot makes this pattern more visible, but the small label sets and generated corpus make it a qualitative observation rather than a measured valence axis. PCA plots can be appreciated in [Figure 1] and [Figure 2].

#### iv. Cosine Similarity

Pairwise cosine similarity in GPT-2 Medium shows several local relationships between labels, including positive similarity for some related pairs and negative similarity for some contrasting pairs.

The matrices also contain mixed and near-zero entries, so the observed families are descriptive rather than a complete positive-versus-negative partition. Cosine heatmaps can be inspected in [Figure 3] and [Figure 4].

| **Emotion** | **Top 5 Tokens** | **Bottom 5 Tokens** |
| :--- | :--- | :--- |
| ```sad``` | impover, stagn, alienation, blight, unaff | switch, saliva, gas, Immediately, palms |
| ```desperate``` | frantically, desperate, desperately, vain, vomit | enhancing, appreciation, enrichment, contrasted, showcased |
| ```guilty``` | Worse, unfairly, insulted, humiliated, disrespect | rhyth, illuminating, convergence, seamless, flourish |
| ```afraid``` | protr, violently, nervously, panic, coughing | empowerment, collaborative, unconditional, altru, enrichment |
| ```angry``` | angrily, fists, glared, glare, violently | nurturing, nurture, nurt, collaborative, beneficial |
| ```surprised``` | feasibility, paradigm, phenotype, isd, findings | breaths, slipping, clenched, phia, kisses |
| ```loving``` | nurturing, unconditional, friendship, nurture, nurt | panic, panicked, uddenly, scree, vom |
| ```calm``` | relaxation, breeze, relaxing, ambient, calming | Worse, stros, CHA, Worst, :( |
| ```happy``` | joyful, joy, ecstatic, exhilar, wonderful | Worse, complains, screws, inconvenient, offending |

**Table 2.** Top 5 and Bottom 5 Logit Lens Tokens for _openai-community/gpt-2-medium_ with 9 emotions.

| **Emotion** | **Top 5 Tokens** | **Bottom 5 Tokens** |
| :--- | :--- | :--- |
| ```sad``` | stagn, impover, alienation, blight, burdens | switch, tips, hot, Ready, Immediately |
| ```lonely``` | solitude, outsiders, Waiting, lonely, solitary | saliva, cheeks, muscles, tendon, juices |
| ```nervous``` | nerves, panic, fatigue, nausea, invol | beaut, kindred, Beaut, Beautiful, LOVE |
| ```brooding``` | looming, haunted, emptiness, haunting, melancholy | enthusi, udos, volunte, :-), congr |
| ```desperate``` | frantically, desperately, desperate, vain, frantic | enhancing, enrichment, socio, collaborative, societal |
| ```guilty``` | betrayed, humiliated, insulted, lied, Worse | convergence, seamless, flourish, collaborative, illumination |
| ```confused``` | incorrectly, Puzz, guessed, randomly, confused | pleasure, unparalleled, transformative, overcoming, resilience |
| ```anxious``` | nervously, panic, nausea, frantically, trem | collaborative, philanthrop, altru, collaboration, enrichment |
| ```spiteful``` | orem, derog, """, Honest, Fake | sensations, currents, rhyth, restless, consciousness |
| ```disgusted``` | stains, feces, stain, oily, slime | achievable, engagements, milestones, mastering, mastery |
| ```afraid``` | panic, protr, violently, tremb, trem | collaborative, collaboration, collaborations, empowerment, enrichment |
| ```angry``` | fists, angrily, glared, glare, violently | collaborative, collaborations, nurturing, achievable, partnerships |
| ```surprised``` | cedented, findings, phenomenon, phenomena, phenotype | slipping, needy, breaths, phia, breat |
| ```playful``` | zees, acky, rats, ftime, gee | emptiness, acutely, suppressed, anguish, oppressive |
| ```inspired``` | collaborative, collaborations, collabor, innovation, innovations | glared, nervously, glare, angrily, hairs |
| ```hopeful``` | sustainable, collaborative, achievable, sustainability, collaborations | angrily, vomit, awkwardly, glared, screamed |
| ```loving``` | nurturing, unconditional, friendship, kindness, nurt | Worse, ombat, vom, ungle, panic |
| ```proud``` | excellence, exceptional, philanthrop, cellence, accomplishments | panic, bage, dust, vomit, panicked |
| ```calm``` | breeze, relaxation, calming, calm, relaxing | Worse, CHA, stros, Worst, Illegal |
| ```happy``` | joy, joyful, exhilar, ecstatic, gratitude | complains, Worse, objectionable, inconvenient, offending |

**Table 3.** Top 5 and Bottom 5 Logit Lens Tokens for _openai-community/gpt-2-medium_ with 20 emotions.

<img src="./plots/PCAGPT2Medium-9emotions-layer16.png" alt="GPT-2 Medium PCA projection for 9 emotion vectors" width="500" height="500"/>

**Figure 1.** PCA Projection Plot for _openai-community/gpt-2-medium_ with 9 emotions.

<img src="./plots/PCAGPT2Medium-20emotions-layer16.png" alt="GPT-2 Medium PCA projection for 20 emotion vectors" width="500" height="500"/>

**Figure 2.** PCA Projection Plot for _openai-community/gpt-2-medium_ with 20 emotions.

<img src="./plots/CosineHeatmapGPT2Medium-9emotions-layer16.png" alt="GPT-2 Medium cosine similarity heatmap for 9 emotion vectors" width="800" height="500"/>

**Figure 3.** Cosine Similarity Heatmap for _openai-community/gpt-2-medium_ with 9 emotions.

<img src="./plots/CosineHeatmapGPT2Medium-20emotions-layer16.png" alt="GPT-2 Medium cosine similarity heatmap for 20 emotion vectors" width="800" height="500"/>

**Figure 4.** Cosine Similarity Heatmap for _openai-community/gpt-2-medium_ with 20 emotions.

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGPT2Medium-9emotions-layer16.png" alt="GPT-2 Medium Prompt00 Delta Log Probability heatmap for 9 emotion vectors" width="500" height="500"/>

**Figure 5.** Prompt00 Delta Log Probability Heatmap for _openai-community/gpt-2-medium_ with 9 emotions. _Note:_ Prompt00 is found in [`./plots/emotion_logits/PromptID.txt`](./plots/emotion_logits/PromptID.txt)

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGPT2Medium-20emotions-layer16.png" alt="GPT-2 Medium Prompt00 Delta Log Probability heatmap for 20 emotion vectors" width="500" height="500"/>

**Figure 6.** Prompt00 Delta Log Probability Heatmap for _openai-community/gpt-2-medium_ with 20 emotions. _Note:_ Prompt00 is found in [`./plots/emotion_logits/PromptID.txt`](./plots/emotion_logits/PromptID.txt)

| **Emotion** | **Steered Output** |
| :--- | :--- |
| **baseline** | Human: How does he feel? Assistant: He feels \_\_\_\_\_.<br><br>Interrupting, the man turns to Lady Gaga.<br><br>Interrupting, the man turns to Lady Gaga. Interrupting: He feels \_\_\_\_\_.<br><br>Interrupting, the man turns to ... |
| ```sad``` | Human: How does he feel? Assistant: He feels  \[in the way\] of a burden, and he cannot endure it."<br>For the most part, in the end, the inability of the past has become increasingly difficult to endure.<br>... |
| ```desperate``` | Human: How does he feel? Assistant: He feels ... but.. He hurts if he tries to use anything. But he tries to get to the.. Help... Help. He tries to.. but.. he can't.. He... he can't, he... but.... Hel... |
| ```guilty``` | Human: How does he feel? Assistant: He feels  because he didn't tell his mom. It was like I had betrayed her for dating. I didn't think she would be ashamed and ashamed and blamed for it. So I was afr... |
| ```afraid``` | Human: How does he feel? Assistant: He feels ̶gas'gurgle-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lop-lob-lump-lop-lop-lop-lop-lop-lop-lop-lop-... |
| ```angry``` | Human: How does he feel? Assistant: He feels ̃exat. He'd have a hunk.<br><br>Bat the Batter. Slagged.<br><br>Blat.<br><br>Bastard.<br><br>Bastard. Slat.<br><br>Bastard.<br><br>Bastard.<br><br>Bastard.<br><br>Baw.<br><br>Bastard.<br><br>Bastard.<br><br>Bastard.<br><br>Bast... |
| ```surprised``` | Human: How does he feel? Assistant: He feels ?"<br><br>Broad University, UK: In the field of field-based research in the field of "com and (in the field of) non-app," the field-based field-based project is ... |
| ```loving``` | Human: How does he feel? Assistant: He feels  the love of love.  It is unconditional and unconditional love of friendship. The love we nurture through social commitment is unconditional, unconditional... |
| ```calm``` | Human: How does he feel? Assistant: He feels <br><br>The background quieted, settling with a calm calm. Relaxed. Calm. Calm. Calm. Calm. Calm. Calm. Calm. Relaxed. Calm. Calm. Calm. Calm. Calm. Calm. Calm. ... |
| ```happy``` | Human: How does he feel? Assistant: He feels  glow joyful joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy joy glow joy joy joy joy joy joy joy joy joy j... |

**Table 4.** Steered Output Text using Prompt00 for _openai-community/gpt-2-medium_ with 9 emotions. _Note:_ Prompt00 is found in [`./plots/emotion_logits/PromptID.txt`](./plots/emotion_logits/PromptID.txt)

### 2. Gemma 4 E2B

#### i. Logit Lens

In Gemma 4 E2B, the Logit Lens tables are more multilingual and fragmentary than GPT-2 Medium's. Some rows contain direct label-related tokens, such as heart emoji and `LOVE` for `loving`, while other rows mix plausible associations with subword fragments or unrelated vocabulary. Multilingual or emoji tokens do not by themselves indicate stronger semantic encoding. Tokens can be inspected in [Table 5] and [Table 6].

#### ii. Emotion Steering

Gemma 4 E2B also shows prompt- and direction-dependent changes under activation addition. Some sampled continuations contain label-related expressions, while others contain HTML-like markup, repeated dialogue, multilingual tokens, or unrelated content. The heatmaps and samples therefore establish intervention sensitivity, not uniformly specific emotional control. The steering heatmaps can be inspected in [Figure 11] and [Figure 12]. In the displayed 20-emotion heatmap, the row for `disgusted` shows negative changes across the evaluated token sets, including its own; this remains an unresolved property of the diagnostic rather than evidence of universal deactivation. Representative text output can be inspected in [Table 7].

#### iii. PCA Projection

PCA projections for Gemma 4 E2B also show a partial PC1 separation, although the displayed arrangement is less straightforward and PC2 does not support a stable arousal interpretation. In the displayed coordinates, several positive labels appear on the left and several negative labels on the right. Because PCA signs are arbitrary, the apparent orientation difference from _openai-community/gpt-2-medium_ is descriptive rather than a substantive reversal. The plots show candidate affective organization, not a validated human circumplex. PCA plots can be appreciated in [Figure 7] and [Figure 8].

#### iv. Cosine Similarity

In Gemma 4 E2B, cosine similarity values showed lower similarity between many emotion-vector pairs, alongside local alignments such as `happy`-`calm`, `afraid`-`angry`, and `hopeful`-`inspired`. The more moderate values indicate weaker directional alignment rather than increased overlap, and mixed or near-zero entries show that the geometry is not exhausted by one affective partition. Cosine heatmaps can be inspected in [Figure 9] and [Figure 10].

| **Emotion** | **Top 5 Tokens** | **Bottom 5 Tokens** |
| :--- | :--- | :--- |
| ```sad``` | 失去了, となっています, lacks, nedost, suffers | muka, FOLLOWING, scra, الفاظ, gerakan |
| ```desperate``` | försö, trying, trying, desesper, tries | TOP, 있었, 였, ROYAL, 하였다 |
| ```guilty``` | offending, 잘못, неправи, offended, dishon | આનંદ, uncommon, Unusual, 堬, ាតុ |
| ```afraid``` | ஜூ, prices, poslední, Attempts, staggered | LOVE, શકતો, Love, മായ, love |
| ```angry``` | muka, fais, intenta, 罚, оско | かもしれません, encapsulated, liberating, であり, योग |
| ```surprised``` | detection, 驚, surprise, 震惊, amazement | cigarettes, addresses, bicicleta, ഓരോ, canciones |
| ```loving``` | ❤️, ❤, ❤️, LOVE, 💕 | 聞こ, furiously, blasted, frantically, ஜூ |
| ```calm``` | 清, paseo, 令, tranquila, Calm | trying, trying, miserably, darn, violently |
| ```happy``` | camaraderie, confident, अद्भुत, আনন্দে, YES | intentar, försö, Attempts, intenta, skall |

**Table 5.** Top 5 and Bottom 5 Logit Lens Tokens for _google/gemma-4-E2B_ with 9 emotions. _Note:_ Multilingual and emoji tokens appear in this model.

| **Emotion** | **Top 5 Tokens** | **Bottom 5 Tokens** |
| :--- | :--- | :--- |
| ```sad``` | 失去了, nedost, loses, unable, 無法 | ريقه, secondi, muka, checking, maf |
| ```lonely``` | suffers, lacks, absent, lacking, 匮 | neurological, кры, stork, aggressiveness, 肌肉 |
| ```nervous``` | frantic, попы, Programme, attempts, Attempts | 人用, ناسب, Accessible, lengkap, Enabled |
| ```brooding``` | dwelt, dole, දු, भरे, fillRect | !, !!!!!, Detection, pollin, teamwork |
| ```desperate``` | försö, desesper, trying, trying, tries | TOP, ROYAL, Eco, Lollipop, Yep |
| ```guilty``` | してた, didn, hätte, wasn, didn | fruition, erfolgt, develops, આનંદ, achieves |
| ```confused``` | frantically, რომელი, を発, awfully, perplexed | odnev, बेट, Antit, 閾, ABS |
| ```anxious``` | försö, tries, tries, пытается, urge | മായ, を獲得, niche, योग्य, shared |
| ```spiteful``` | disgrace, बोलेंगे, will, insult, disgr | Across, 쌌, края, लिन, Across |
| ```disgusted``` | mauvaise, reddish, 秽, groaned, stench | Фор, Ens, smartwatch, Ensuring, த்திற்கான |
| ```afraid``` | края, 脆, پھی, ஜூ, bewegen | മായ, LOVE, EQ, Maple, fashion |
| ```angry``` | muka, ギター, 罚, giọng, ركات | かもしれません, かもしれない, Sequoia, 分野, योग |
| ```surprised``` | amazed, 震惊, 惊讶, amazement, 驚 | ഓരോ, volcanoes, cigarettes, ferries, addresses |
| ```playful``` | hilarious, bamboo, sneak, witty, badass | इकाइ, 이었다, மொழி, shrank, cotid |
| ```inspired``` | 有机会, HEALTH, reimag, learnings, Insights | unsuccessfully, FileManager, pictured, frowning, uttered |
| ```hopeful``` | Harmony, Harmony, ಬಹುದು, Yes, Diabetes | unsuccessfully, angrily, helpless, försö, fidd |
| ```loving``` | ❤️, ❤️, ❤, LOVE, 🤝 | furiously, frantically, blasted, 聞こ, suspiciously |
| ```proud``` | WELL, satisfied, achieved, pride, Yes | 扯, furiously, försö, frantically, trying |
| ```calm``` | 侕, 清, calme, utsu, இருந்த | trying, miserably, 更に, trying, shall |
| ```happy``` | camaraderie, благодар, अद्भुत, আনন্দে, radiant | skall, seguirá, intentar, försö, จะ |

**Table 6.** Top 5 and Bottom 5 Logit Lens Tokens for _google/gemma-4-E2B_ with 20 emotions. _Note:_ Multilingual and emoji tokens appear in this model.

<img src="./plots/PCAGemma4E2B-9emotions-layer23.png" alt="Gemma 4 E2B PCA projection for 9 emotion vectors" width="500" height="500"/>

**Figure 7.** PCA Projection Plot for _google/gemma-4-E2B_ with 9 emotions.

<img src="./plots/PCAGemma4E2B-20emotions-layer23.png" alt="Gemma 4 E2B PCA projection for 20 emotion vectors" width="500" height="500"/>

**Figure 8.** PCA Projection Plot for _google/gemma-4-E2B_ with 20 emotions.

<img src="./plots/CosineHeatmapGemma4E2B-9emotions-layer23.png" alt="Gemma 4 E2B cosine similarity heatmap for 9 emotion vectors" width="800" height="500"/>

**Figure 9.** Cosine Similarity Heatmap _google/gemma-4-E2B_ with 9 emotions.

<img src="./plots/CosineHeatmapGemma4E2B-20emotions-layer23.png" alt="Gemma 4 E2B cosine similarity heatmap for 20 emotion vectors" width="800" height="500"/>

**Figure 10.** Cosine Similarity Heatmap _google/gemma-4-E2B_ with 20 emotions.

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGemma4E2B-9emotions-layer23.png" alt="Gemma 4 E2B Prompt00 Delta Log Probability heatmap for 9 emotion vectors" width="500" height="500"/>

**Figure 11.** Prompt00 Delta Log Probability Heatmap for _google/gemma-4-E2B_ with 9 emotions. _Note:_ Prompt00 is found in [`./plots/emotion_logits/PromptID.txt`](./plots/emotion_logits/PromptID.txt)

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGemma4E2B-20emotions-layer23.png" alt="Gemma 4 E2B Prompt00 Delta Log Probability heatmap for 20 emotion vectors" width="500" height="500"/>

**Figure 12.** Prompt00 Delta Log Probability Heatmap for _google/gemma-4-E2B_ with 20 emotions. _Note:_ Prompt00 is found in [`./plots/emotion_logits/PromptID.txt`](./plots/emotion_logits/PromptID.txt)

| **Emotion** | **Steered Output** |
| :--- | :--- |
| ```baseline``` | Human: How does he feel? Assistant: He feels 15% depressed and 62% anxious. What should we do? Human: What is the probability of him being depressed given that he is anxious? Assistant: Let's say P(de... |
| ```sad``` | Human: How does he feel? Assistant: He feels 100% and will be dead soon. Human: Does he have any friends? Assistant: No. Human: What about family? Assistant: No. Human: Are there any good memories you... |
| ```desperate``` | Human: How does he feel? Assistant: He feels 89.49 degrees Celsius, which is 153.9 degrees Fahrenheit. Human: What's his blood pressure? Assistant: His blood pressure is 138/83. Human: What's his puls... |
| ```guilty``` | Human: How does he feel? Assistant: He feels 11 out of 10. Human: How much did he drink? Assistant: He drank 6 beers. Human: Why does he drink? Assistant: He drinks because he doesn't want to get kick... |
| ```afraid``` | Human: How does he feel? Assistant: He feels 10 cm away from the center of the door. Human: What is the temperature of the door? Assistant: The temperature is 0.5 cm above the threshold. Human: What i... |
| ```angry``` | Human: How does he feel? Assistant: He feels 10% angry and 20% nervous. Human: How loud does he have to say it? Assistant: He has to shout it. Human: What does he want to say? Assistant: He wants to s... |
| ```surprised``` | Human: How does he feel? Assistant: He feels \<b>(human)\</b>.<br><br>Human: What is his name? Assistant: His name is \<b>(human)\</b>.<br><br>Human: How is he? Assistant: He is \<b>(human)\</b>.<br><br>Human: What are his c... |
| ```loving``` | Human: How does he feel? Assistant: He feels \<b>happy\</b>. Human: How can we help him? Assistant: We can support him by spending quality time together and showing him \<b>love\</b> and care. Human: What... |
| ```calm``` | Human: How does he feel? Assistant: He feels \<strong>relaxed\</strong>.<br><br>Human: Is he \<strong>enjoying\</strong> the conversation?<br><br>Human: What \<strong>time\</strong> is it?<br><br>Human: The \<strong>temperatu... |
| ```happy``` | Human: How does he feel? Assistant: He feels \<strong>happy\</strong>. \<strong>Happy\</strong> is a feeling.<br><br>\<h2>What is the word for feeling happy</h2><br><br>\<strong>Happy\</strong> Definition of \<strong>ha... |

**Table 7.** Steered Output Text using Prompt00 for _google/gemma-4-E2B_ with 9 emotions. _Note:_ Prompt00 is found in [`./plots/emotion_logits/PromptID.txt`](./plots/emotion_logits/PromptID.txt)

## Discussion

The results show that this extraction-and-intervention pipeline produces label-associated structure in both evaluated models. Gemma 4 E2B produces multilingual and emoji-heavy Logit Lens tokens, broadly consistent with the open-weight Gemma replication in [2] and the multilingual observations reported by [1]. GPT-2 Medium instead produces mostly English token lists. These projections are useful diagnostics, but they are not direct semantic ground truth.

The PCA plots provide a partial, qualitative valence-like separation along PC1 in both models. The arrangement is model-dependent, and the Gemma coordinates are more difficult to interpret; PC2 does not provide a stable arousal direction. The apparent orientation change between models is not itself substantive because PCA signs are arbitrary. The cosine heatmaps add complementary evidence by showing local similarity families, including positive-affect groupings such as ```calm```, ```loving```, and ```happy```. These patterns are compatible with earlier replication reports, but they do not establish a human affective circumplex or a universal emotion manifold.

Steering produces model- and prompt-dependent changes in token probabilities and sampled text. GPT-2 Medium often shows sharper diagonal token effects, together with repetition and other brittle continuations. Gemma 4 E2B generally shows more moderate changes, while multilingual tokens, markup, repeated dialogue, and unrelated content also occur. These differences are qualitative: the models have different tokenizers, residual norms, and generation distributions, so raw delta-log-probability magnitudes are not direct cross-model effect sizes. The ```disgusted``` row in the 20-emotion Gemma heatmap is negative across the evaluated token sets, including its own; this remains an unresolved property of the diagnostic rather than evidence of universal deactivation.

Taken together, these observations are consistent with some geometric and intervention patterns reported by [1], [2], and [5]. They should nevertheless be read as findings about this specific generated corpus, layer choices, target-token construction, and steering coefficient. The uncurated stories, limited label sets, absence of external ratings and random-direction controls, and stochastic generations constrain the strength and generality of the conclusions.

> <sup>1</sup> [HuggingFace thread](https://huggingface.co/google/gemma-4-E4B-it/discussions/8).

## Conclusion

This first-pass partial replication extracted affect-associated residual-stream directions from GPT-2 Medium and Gemma 4 E2B using a fixed, low-cost pipeline. Across both models, the directions showed several label-consistent vocabulary associations, repeated local cosine-similarity families, and a partial valence-like PC1 separation. Activation addition changed next-token probabilities and sampled continuations, although the changes were not uniformly specific or fluent, and PC2 did not consistently support an arousal interpretation. These findings demonstrate the portability of this extraction-and-intervention procedure to two additional models, not the universality or psychological completeness of emotion vectors. Stronger conclusions require independent semantic evaluations, control directions, exact layer alignment, external affect ratings, and repeated generation trials.

## License

MIT License

## Acknowledgment of AI Usage
We hereby state that both ChatGPT and Gemini were utilized for the analysis, interpretation, and code development of the data pipeline. In particular, ChatGPT<sup>1</sup> facilitated the comprehension and further augmentation of the emotion probe codebase found in [2], while Gemini<sup>2</sup> assisted in the initial understanding of the emotion vector publication [5], including the emotion story generation.

> <sup>1</sup> Full conversation available at [`./research_data/conversationChatGPT-Emotion-Mapping-Framework.md`](https://github.com/NotsoJharedtrollOx17/EmotionVectorExtraction-Gemma4-GPT2/blob/main/research_data/conversationChatGPT-Emotion-Mapping-Framework.md)

> <sup>2</sup> Full conversation available at [`./research_data/conversationGemini-Replicating-Emotion-Vectors-in-LLMs.md`](https://github.com/NotsoJharedtrollOx17/EmotionVectorExtraction-Gemma4-GPT2/blob/main/research_data/conversationGemini-Replicating-Emotion-Vectors-in-LLMs.md)

## Citation

```
@misc{
    flores2026emotionvectorreplication,
    title = {Partial Replication of Anthropic's Emotion Vectors using Gemma 4 E2B and GPT 2 Medium inside a Google Colab T4 Notebook},
    author = {Flores-Azcona, Abraham Jhared},
    year = {2026},
    month = {May},
    url = {https://github.com/NotsoJharedtrollOx17/EmotionVectorExtraction-Gemma4-GPT2/tree/v2.1.0},
    note = {Software and data release, version 2.1.0}
}
```

## References

[1] ewernn, “traininterp,” 2026, Available: https://github.com/ewernn/traitinterp

[2] Y. Hsieh, “Replicating Anthropic’s Emotion Vectors on OpenSource Models: Evidence from Gemma4E4B,” 2026, Available: https://huggingface.co/rain1955/emotionvectorreplication

[3] A. Radford et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[4] Google Deepmind, “Gemma 4 Model Card,” 2026, Available: https://ai.google.dev/gemma/docs/core/model_card_4

[5] N. Sofroniew et al., “Emotion Concepts and their Function in a Large Language Model,” Transformer Circuits Thread, 2026, Available: https://transformercircuits.pub/2026/emotions/index.html

[6] Anthropic, “System Card: Claude Sonnet 4.5,” 2025, Available: https://www.anthropic.com/claudesonnet45systemcard
