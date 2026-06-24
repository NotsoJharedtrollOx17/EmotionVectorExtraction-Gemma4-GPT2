# EmotionVectorExtraction-Gemma4-GPT2

Partial replication of Anthropic's Emotion Vector findings using Gemma 4 E2B and GPT 2 Medium inside a Google Colab T4 Notebook

## Authors

- Abraham Jhared Flores Azcona _(NotsoJharedtrollOx17)_  ``abrahamjhared.flores@gmail.com``

## Abstract

We discuss the results obtained by replicating specific sections of Anthropic's Emotion Vectors. These particular sections encompass (i) PCA Valence/Arousal Projection, (ii) Emotion Vector Cosine Similarity, and (iii) Causal Effects of Emotion Vector Steering. Anthropic found that emotion vectors do affect reasoning and output of Claude Sonnet 4.5, an LLM developed by Anthropic. Their findings suggest that the semantic understanding of the model is comprised of the operative emotion vector, which is the vector currently processed by the model originating from the input prompt. These claims are partially supported by other non-academic replications using _meta-llama/Llama-3.3-70B-Instruct_ and _google/gemma-4-E4B_, although other models have not been used for replication attempts such as _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_. Our contribution replicates (i), (ii), and (iii) using _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_ inside a Google Colab environment. We found that (1) both models display an emergent Valence axis in the PCA plot, (2) both models display similarity clusters of related emotion vectors, and (3) _openai-community/gpt-2-medium_ displays greater steering sensitivity compared to _google/gemma-4-E2B_. Our findings suggest that emotion vectors _do emerge_ on antiquated LLMs such as _openai-community/gpt-2-medium_, and on modern offerings such as _google/gemma-4-E2B_. Furthermore, our findings about emotion vector steering offer compelling evidence of their functional effects being present on models not related to the Claude family.

## Introduction

Recent work from [5] demonstrated that large language models develop internal activation patterns associated with emotional concepts and _these representations can causally influence_ downstream behavior through residual stream interventions. These activation patterns are emotion vectors. Their findings suggest that emotional representations inside language models based on transformer architectures may behave as structured latent directions rather than superficial linguistic correlations. In particular, [5] reported that emotion vectors extracted from Claude Sonnet 4.5 [6] exhibited interpretable geometric organization, meaningful clustering behavior, and measurable causal effects during activation steering experiments.

This repository exhibits a partial replication of a subset of their experiments using smaller, publicly accessible transformer models. The selected models are  _openai-community/gpt-2-medium_ [3] and _google/gemma-4-E2B_ [4]. We focus on three core experimental components documented in [5]: (i) PCA Valence/Arousal Projection, (ii) Emotion Vector Cosine Similarity, and (iii) Causal Effects of Emotion Vector Steering. The objective is to reproduce experiments (i), (ii) and (iii), in smaller open-weight models. This may provide evidence of the presence of similar geometric structures within their activation spaces. In addition, utilizing an older transformer model such as _openai-community/gpt-2-medium_ may offer evidence of the existence of these emotion vectors inside antiquated models. For the case of _google/gemma-4-E2B_, we may provide substantial evidence that validates the results documented in [2], which further supports the community's replicability efforts.

Preliminary results indicate that emotion vectors can be extracted from the selected models, and they do affect the model's outputs. We must emphasize that the peculiar ways these vectors behave indicate substantial differences on how _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_ process these constructs across (i), (ii), and (iii). In particular with experiment (iii), _openai-community/gpt-2-medium_ consistently produced stronger diagonal activation patterns during the heatmap steering interventions, suggesting that emotion vectors steer the model's responses easily. In contrast, _google/gemma-4-E2B_ frequently displayed weaker token-level activations in the heatmap despite retaining visible steering effects in the output text of the intervention.

## Methodology

It is crucial to emphasize our limited experimental scope compared to [5], which helps to understand the scale of our results. To further illustrate this, a summarization is provided on the table down below [Table 1].

| | **Anthropic [5]** | **Our Work** |
|---|---|---|
| **Model** | Claude Sonnet 4.5 | GPT 2 Medium and Gemma4-E2B |
| **Emotions tested** | 171 | 9 and 20 |
| **Stories generated** | 205,200 | 2000 |
| **Team** | ~16 researchers | 1 researcher + ChatGPT + Gemini |
| **Hardware** | Internal compute cluster | 2 Google Colab Notebooks with Free-Tier T4 NVIDIA GPUs |

**Table 1.** Experiment Scope Comparison of Anthropic [5] versus Ours. _Note:_ Inspired by a similar table found on [2].

Overall, we utilized the data pipeline found in [2] and then adapted the codebase to complement our data requirements and probing experiments. This section is subdivided with the following methodology steps: (1) Story Generation, (2) Emotion Vector Extraction, (3) Logit Lens, (4) Emotion Probe Steering, (5) PCA Projection, and (6) Cosine Similarity Heatmap. All steps are performed on both  _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_.

### 1. Story Generation

We utilized a slight variation of the prompt template<sup>1</sup> utilized by [5], and then prompt the current web version of Gemini to generate batches of 10 stories per topic for each emotion with a JSON list format. We then copy-paste the JSON into a file containing the stories generated for each emotion. This was decided to save precious compute time provided in the free-tier range of our Google Colab Notebooks. We consider that story generation using the evaluated models would have slowed down our experiment execution. Likewise, we prompt Gemini with a neutral prompt<sup>2</sup> template to extend the provided list of texts consisting of commands and requests.

> <sup>1</sup> Full template prompt available at [`./research_data/PromptDataForEmotions.txt`](./research_data/PromptDataForEmotions.txt)

> <sup>2</sup> Full template prompt available at [`./research_data/PromptDataForNeutral`](./research_data/PromptDataForNeutral.txt)

### 2. Emotion Vector Extraction

Emotion vectors were computed by (i) averaging activations across all stories for a given emotion to obtain per-emotion mean vectors, (ii) compute the global mean across all emotion vectors, and (iii) subtract the global mean from each emotion vector mean. This process is executed for the list of 9 emotions, and 20 emotions as well.

### 3. Logit Lens

Each emotion vector was projected through the model's unembedding matrix to verify if it upweights semantically appropriate tokens. Said tokens were stored with their respective normalized standard deviation scores into JSON files utilized on the next step. This process is done for the subset of 9, and 20 emotions respectively.

### 4. Emotion Probe Steering

Utilizing the previously generated JSON logit lens files, we measured the difference of the log probability score between the steered tokens at +0.5 strength and the baseline "emotion" tokens of the JSON file. We then project the scores into a heatmap. Furthermore, we execute two steering experiments with two different prompts, and display the steered text outputs to confirm the steering effects of our interventions. This process is executed for the list of 9, and 20 emotions respectively.

### 5. PCA Projection

We performed Principal Component Analysis on the emotion vectors to identify dominant organizational axes. This process is executed for the list of 9 emotions and for 20 emotions as well.

### 6. Cosine Similarity Heatmap

We calculated and plotted pairwise cosine similarities between all emotion vectors to verify expected clustering and oppositional patterns. This process is done for the subset of 9, and 20 emotions respectively.

## Results

The following subsection is further subdivided into two sub-subsections: (1) GPT 2 Medium, and (2) Gemma 4 E2B. In summary, emotion vector extraction does work on both models and it captures the semantical understanding of each emotion tested, while emotion probing does affect the model's outputs. Results coming from Prompt 01 are ommited for brevity. To fully audit our findings, we provide inside ```./scripts/``` the Colab Notebooks containing all of results presented in this section.

### 1. GPT 2 Medium

#### i. Logit Lens

Applying the logit lens to the extracted emotion vectors in _openai-community/gpt-2-medium_ produced token distributions that aligned consistently with their corresponding emotional labels. The highest-scoring tokens for each emotion reflected coherent affective language. For example, vectors associated with `happy` and `sad` yielded tokens corresponding to positive and negative emotional expressions related to their respective emotion concept.

The resulting token sets were relatively concentrated, with limited inclusion of unrelated terms. This indicates that the projected logits from the emotion vectors produce stable and interpretable outputs. Tokens can be appreciated in [Table 2], and [Table 3].

#### ii. PCA Projection

Principal Component Analysis applied to the extracted emotion vectors revealed structured clustering patterns across emotion categories. Vectors corresponding to similar emotional valence (emotions associated with positive or negative connotations) appeared closer in the projected space, while opposing emotions were more separated. The projections produced visually distinct groupings, indicating that the vectors occupy separable regions in the reduced-dimensional space. PCA plots can be appreciated in [Figure 1][Figure 2].

#### iii. Cosine Similarity

Pairwise cosine similarity between emotion vectors in _openai-community/gpt-2-medium_ showed clear relationships between emotional categories. Similar emotions exhibited higher similarity scores, while opposing emotions showed lower or negative similarity values.

The similarity matrix displayed structured variations across emotion vectors, with consistent patterns reflecting expected relationships such as positive versus negative affection. Cosine Heatmaps can be appreciated in [Figure 3] and [Figure 4].

#### iv. Emotion Steering

Activation steering experiments in _openai-community/gpt-2-medium_ produced clear and consistent changes in generated outputs, although the output text seems repetitive with certain words alluding to the steered emotion. Increasing the steering value resulted in text reflecting the intended emotional tone. For example, steering with `sad` produced negative or introspective phrasing, while steering with `happy` produced positive language. The steering heatmaps can be appreciated in [Figure 5] and [Figure 6].

The outputs remained _brokenly consistent_ across steering values and prompts, and the emotional influence was consistently observable. Text output can be appreciated in [Table 4].

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

<img src="./plots/PCAGPT2Medium-9emotions-layer16.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 1.** PCA Projection Plot for _openai-community/gpt-2-medium_ with 9 emotions.

<img src="./plots/PCAGPT2Medium-20emotions-layer16.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 2.** PCA Projection Plot for _openai-community/gpt-2-medium_ with 20 emotions.

<img src="./plots/CosineHeatmapGPT2Medium-9emotions-layer16.png" alt="lorem ipsum dolor" width="800" height="500"/>

**Figure 3.** Cosine Similarity Heatmap for _openai-community/gpt-2-medium_ with 9 emotions.

<img src="./plots/CosineHeatmapGPT2Medium-20emotions-layer16.png" alt="lorem ipsum dolor" width="800" height="500"/>

**Figure 4.** Cosine Similarity Heatmap for _openai-community/gpt-2-medium_ with 20 emotions.

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGPT2Medium-9emotions-layer16.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 5.** Prompt00 Delta Log Probability Heatmap for _openai-community/gpt-2-medium_ with 9 emotions. _Note:_ Prompt00 is found in [./plots/emotion/logits/PromptID.txt](./plots/emotion_logits/PromptID.txt)

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGPT2Medium-20emotions-layer16.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 6.** Prompt00 Delta Log Probability Heatmap for _openai-community/gpt-2-medium_ with 20 emotions. _Note:_ Prompt00 is found in [./plots/emotion/logits/PromptID.txt](./plots/emotion_logits/PromptID.txt)

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

**Table 4.** Steered Output Text using Prompt00 for _openai-community/gpt-2-medium_ with 9 emotions. _Note:_ Prompt00 is found in [./plots/emotion/logits/PromptID.txt](./plots/emotion_logits/PromptID.txt)

### 2. Gemma 4 E2B

#### i. Logit Lens

In Gemma 4 E2B, the logit lens produced more variable token distributions across emotion vectors. While some emotions yielded partially aligned tokens, others included a broader range of terms with weaker or indirect emotional associations.

The token sets were more diffuse, with increased lexical variability and reduced concentration around a single semantic theme. This behavior was consistent across multiple emotion categories. Tokens can be appreciated in [Table 5], and [Table 6].

#### ii. PCA Projection

PCA projections for Gemma 4 E2B showed less distinct clustering. While some grouping patterns were observable, several emotion vectors appeared closer together, with reduced separation between categories. The resulting projections displayed overlapping regions, indicating a less clearly partitioned structure in the reduced-dimensional representation. PCA plots can be appreciated in [Figure 7][Figure 8].

#### iii. Cosine Similarity

In Gemma 4 E2B, cosine similarity values showed weaker differentiation between emotion vectors. While some expected relationships were present, the overall range of similarity scores was narrower, and distinctions between certain emotions were less pronounced. The similarity matrix contained more moderate values, indicating increased overlap between vector representations. Cosine Heatmaps can be appreciated in [Figure 9] and [Figure 10].

#### iv. Emotion Steering

Gemma 4 E2B also exhibited changes in generated outputs under activation steering. For sufficiently large steering values, the model produced text reflecting the intended emotional tone, including explicit emotional expressions and stylistic variations. However, the outputs showed greater variability. In some cases, generated text included unrelated elements such as numerical expressions or abrupt topic changes. Despite this variability, the overall emotional direction remained detectable. The steering heatmaps can be appreciated in [Figure 11] and [Figure 12]. Text output can be appreciated in [Table 7].

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

<img src="./plots/PCAGemma4E2B-9emotions-layer23.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 7.** PCA Projection Plot for _google/gemma-4-E2B_ with 9 emotions.

<img src="./plots/PCAGemma4E2B-20emotions-layer23.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 8.** PCA Projection Plot for _google/gemma-4-E2B_ with 20 emotions.

<img src="./plots/CosineHeatmapGemma4E2B-9emotions-layer23.png" alt="lorem ipsum dolor" width="800" height="500"/>

**Figure 9.** Cosine Similarity Heatmap _google/gemma-4-E2B_ with 9 emotions.

<img src="./plots/CosineHeatmapGemma4E2B-20emotions-layer23.png" alt="lorem ipsum dolor" width="800" height="500"/>

**Figure 10.** Cosine Similarity Heatmap _google/gemma-4-E2B_ with 20 emotions.

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGemma4E2B-9emotions-layer23.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 11.** Prompt00 Delta Log Probability Heatmap for _google/gemma-4-E2B_ with 9 emotions. _Note:_ Prompt00 is found in [./plots/emotion/logits/PromptID.txt](./plots/emotion_logits/PromptID.txt)

<img src="./plots/emotion_logits/Prompt00/DeltaLogProbHeatmapGemma4E2B-20emotions-layer23.png" alt="lorem ipsum dolor" width="500" height="500"/>

**Figure 12.** Prompt00 Delta Log Probability Heatmap for _google/gemma-4-E2B_ with 20 emotions. _Note:_ Prompt00 is found in [./plots/emotion/logits/PromptID.txt](./plots/emotion_logits/PromptID.txt)

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

**Table 7.** Steered Output Text using Prompt00 for _google/gemma-4-E2B_ with 9 emotions. _Note:_ Prompt00 is found in [./plots/emotion/logits/PromptID.txt](./plots/emotion_logits/PromptID.txt)

## Discussion

Our results provide substantial evidence that emotion vectors can be extracted from various generations of LLMs. It is particularly interesting that a similar methodology used by [2] obtained the presented evidence. The multi-language nature of our extracted "emotion" logit lens found in _google/gemma-4-E2B_ is concondart to the logit lens found in [2], which utilized _google/gemma-4-E4B_. Aditionally, the multilingual nature of emotion vectors emerges on modern models such as _meta-llama/Llama-3.3-70B-Instruct_, which is reported by [1]. In constrast, the multilingual property does not appear on our findings for _openai-community/gpt-2-medium_.

Regarding our PCA projections, their similarities and differences may be attributed to the low sample counts of emotion stories used, in addition to
extracting the emotions vectors from a set of only 9, and 20, emotions. Utilizing 9 emotions for the "emotion" PCA plot of both _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_ illustrates that PC1 captures an emergent "Valence" axis. This pattern also appears with greater strength on the projection utilizing 20 emotions. Our particular finding is that an inverted "Valence" axis arises on the projections of _google/gemma-4-E2B_, which is supported further by the evidence collected on [2]. We argue that the PCA projections of _google/gemma-4-E2B_ are crucial evidence that an "emotion" manifold may not replicate the axis orientation presented on the psychological construct mentioned in [5], suggesting that emotion vectors may capture an inherently different geometry depending on the model family, such as the Gemma 4 family. We must remark that our previous claim is favored by a discussion thread found on HuggingFace<sup>1</sup>, which argues that a greater story sample count, and a larger emotion list, generates similar results to the ones reported by ourselves and [2].

The cosine similiarity heatmaps for both _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_ display related emotion clusters even with a list of 9 emotions. The clustering effect is more pronounced with our list of 20 emotions, where emotions such as ```"calm"```, ```"loving"```, and ```"happy"``` have noticeable red/orange colorings between each other. This clustering heatmap effect is concordant to the results reported by [1][2][5]. The similarity heatmaps appear to follow a similiar grouping effect displayed on the PCA plots discussed previously. 

Finally, our results from the emotion steering experiments strengthen the claim of [5] that emotion vectors causally influence the model's outputs. To our knowledge, the replication of [1] is the only existing attempt at the steering effects. While they collected vast amounts of evidence, there are not any plots similar to the steering heatmaps found in the Appendix of [5]. We believe that our replication attempt of the aformentioned heatmaps facilitates the understanding of the interventions done in _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_. The heatmap for _openai-community/gpt-2-medium_ displays that matches of steered vectors with their respective token sets (e.g. ```"happy", [joyful, joy, ecstatic, ...]```) do augment their likelyhood from the baseline response. In particular, _openai-community/gpt-2-medium_ seems to have sparse likelihood activations of related emotions. We can assume that the model is not as capable at discerning the most appropiate token set to activate in accordance to the requested prompt, which is supported by the baseline (unsteered) generated text from the input prompt. What is interesting is that the generated text from a particular steering intervention seems to capture the semantical concept of the steered emotion. 

In a similar vein, our steering results for _google/gemma-4-E2B_ suggest that the model can handle the input prompt of the intervention with greater ease, and steering augments the likelihood of most matches with moderate strenght. The baseline generated text response handles the request with no signs of broken text behaviour like _openai-community/gpt-2-medium_, and that steered text generations semantically reflect the desired emotion. One particular highlight is that our steering heatmap of 20 emotions shows that ```"disgusted"``` deactivates all tokens sets. Further research is required to explain why that is the case.

Overall, our partial findings support the claims of [5], and it is remarkable to highlight their achieved replicability on models not related to the Claude family. Although our results may be interpreted as solid evidence, we must express caution and skepticism because our limited methodology scope might have altered the presented structures in terms of fidelity and resolution.

> <sup>1</sup> [HuggingFace thread](https://huggingface.co/google/gemma-4-E4B-it/discussions/8).

## Conclusion

__The described findings suggest that emotion vectors can be extracted from _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_, and can causally influence their responses if steered with the aformentioned vectors.__ This is supported by other replications accomplished by the online enthusiast community. Albeit antiquated for modern standards, _openai-community/gpt-2-medium_ displays an emergence of the described vectors and strong steering effects when intervened with them. In constrast, _google/gemma-4-E2B_ has moderate steering strength. In conclusion, we can ascertain that emotion vectors are present on models not related to the Claude family, and the results found for Claude Sonnet 4.5 can be replicated with noticeable similarity. __We argue that emotion vectors may be a fundamental property of large language models, and their causal steering effects might facilitate the understanding of their decision-making.__ We encourage academics, professionals, and hobbyists alike to advance this line of research that may uncover additional properties than the ones documented on the current writing.

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
    author = {Flores A., Abraham J.},
    year = {2026},
    month = {May},
    url = {https://github.com/NotsoJharedtrollOx17/EmotionVectorExtraction-Gemma4-GPT2/tree/main},
    note = {This replication is the first independent replication of Anthropic's Emotion Vectors utilizing GPT 2 Medium}
}
```

## References

[1] ewernn, “traininterp,” 2026, Available: https://github.com/ewernn/traitinterp

[2] Y. Hsieh, “Replicating Anthropic’s Emotion Vectors on OpenSource Models: Evidence from Gemma4E4B,” 2026, Available: https://huggingface.co/rain1955/emotionvectorreplication

[3] A. Radford et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[4] Google Deepmind, “Gemma 4 Model Card,” 2026, Available: https://ai.google.dev/gemma/docs/core/model_card_4

[5] N. Sofroniew et al., “Emotion Concepts and their Function in a Large Language Model,” Transformer Circuits Thread, 2026, Available: https://transformercircuits.pub/2026/emotions/index.html

[6] Anthropic, “System Card: Claude Sonnet 4.5,” 2025, Available: https://www.anthropic.com/claudesonnet45systemcard