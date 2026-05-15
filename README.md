# EmotionVectorExtraction-Gemma4-GPT2

Partial replication of Anthropic's Emotion Vector findings using Gemma 4 E2B and GPT 2 Medium inside a Google Colab T4 Notebook

## Authors

- Abraham Jhared Flores Azcona _(NotsoJharedtrollOx17)_  ``abrahamjhared.flores@gmail.com``

## Abstract

We discuss the results obtained by replicating specific sections of Anthropic's Emotion Vectors. These particular sections encompass (i) PCA Valence/Arousal Projection, (ii) Emotion Vector Cosine Similarity, and (iii) Causal Effects of Emotion Vector Steering. Anthropic found that emotion vectors do affect reasoning and output of Claude Sonnet 4.5, an LLM developed by Anthropic. Their findings suggest that the semantic understanding of the model is comprised of the operative emotion vector, which is the vector currently processed by the model originating from the input prompt. These claims are partially supported by other non-academic replications using _meta-llama/Llama-3.3-70B-Instruct_ and _google/gemma-4-E2B_, although other models have not been used for replication attempts such as _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_. Our contribution replicates (i), (ii), and (iii) using _openai-community/gpt-2-medium_ and _google/gemma-4-E2B_ inside a Google Colab environment. We found that (1) both models display an emergent Valence axis in the PCA plot, (2) both models display similarity clusters of related emotion vectors, and (3) _openai-community/gpt-2-medium_ displays greater steering sensitivity compared to _google/gemma-4-E2B_. Our findings suggest that emotion vectors _do emerge_ on antiquated LLMs such as _openai-community/gpt-2-medium_, and on modern offerings such as _google/gemma-4-E2B_. Furthermore, our findings about emotion vector steering  offer compelling evidence of their functional effects being present on models not related to the Claude family.

## Results

lorem ipsum dolor

### GPT 2 Medium

lorem ipsum dolor

### Gemma 4 E2B

lorem ipsum dolor

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
We hereby state that both ChatGPT and Gemini were utilized for the analysis, interpretation, and code development of the data pipeline. In particular, ChatGPT<sup>1</sup> facilitated the comprehension and further augmentation of the emotion probe codebase found in [2], while Gemini<sup>2</sup> assisted in the initial exploration and understanding of the emotion vector publication [5].

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