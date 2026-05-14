# EmotionVectorExtraction-Gemma4-GPT2

Partial replication of Anthropic's Emotion Vector findings using Gemma 4 E2B and GPT 2 Medium inside a Google Colab T4 Notebook

## Summary

We discuss the results obtained by replicating specific sections of Anthropic's Emotion Vectors. These particular sections encompass (i) PCA Valence/Arousal Projection, (ii) Emotion Vector Cosine Similarity, and (iii) Causal Effects of Emotion Vector Steering. Anthropic found that emotion vectors do affect reasoning and output of Claud Sonnet 4.5, an LLM developed by Anthropic. Their findings suggest that the semantic understanding of the model is comprised of the operative emotion vector, which is the vector currently processed by the model originating from the input prompt. These claims are partially supported by other non-academic replications using Llama 3 90B and Gemma 4 E4B, while other models have not been used for replication attempts. Our contribution replicates (i), (ii), and (iii) using _openai-community/gpt-2-medium_ and _google/Gemma-4-E2B_ inside a Google Colab environment. We found that (1) both models display an emergent Valence axis in the PCA plot, (2) both models display similarity clusters of related emotion vectors, and (3) _openai-community/gpt-2-medium_ displays greater steering sensitivity compared to _google/Gemma-4-E2B_. Our findings suggest that emotion vectors _do emerge_ on antiquated LLMs such as _openai-community/gpt-2-medium_, and on modern offerings such as _google/Gemma-4-E2B_. Furthermore, our findings about emotion vector steering  offer compelling evidence of their functional effects being present on models not related to the Claude family.

## Authors

- Abraham Jhared Flores Azcona _(NotsoJharedtrollOx17)_  ``abrahamjhared.flores@gmail.com``

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