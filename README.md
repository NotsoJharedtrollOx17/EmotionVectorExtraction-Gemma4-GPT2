# EmotionVectorExtraction-Gemma4-GPT2

Partial replication of Anthropic's Emotion Vector findings using Gemma 4 E2B and GPT 2 Medium inside a Google Colab T4 Notebook

## Summary

We discuss the results obtained by replicating specific sections of Anthropic's Emotion Vectors. These particular sections encompass (i) PCA Valence/Arousal Projection, (ii) Emotion Vector Cosine Similarity, and (iii) Causal Effects of Emotion Vector Steering. Anthropic found that emotion vectors do affect reasoning and output of Claud Sonnet 4.5, an LLM developed by Anthropic. Their findings suggest that the semantic understanding of the model is comprised of the operative emotion vector, which is the vector currently processed by the model originating from the input prompt. These claims are partially supported by other non-academic replications using Llama 3 90B and Gemma 4 E4B, while other models have not been used for replication attempts. Our contribution replicates (i), (ii), and (iii) using _openai-community/gpt-2-medium_ and _google/Gemma-4-E2B_ inside a Google Colab environment. We found that (1) both models display an emergent Valence axis in the PCA plot, (2) both models display similarity clusters of related emotion vectors, and (3) _openai-community/gpt-2-medium_ displays greater steering sensitivity compared to _google/Gemma-4-E2B_. Our findings suggest that emotion vectors _do emerge_ on antiquated LLMs such as _openai-community/gpt-2-medium_, and on modern offerings such as _google/Gemma-4-E2B_. Furthermore, our findings about emotion vector steering  offer compelling evidence of their functional effects being present on models not related to the Claude family.

## Authors

- Abraham Jhared Flores Azcona _(NotsoJharedtrollOx17)_  ``abrahamjhared.flores@gmail.com``

## License

MIT License

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

## Refences

lorem

ipsum

dolor