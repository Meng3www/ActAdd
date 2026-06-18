# preliminary
```
T4 High RAM
System RAM: 1.0 / 12.7 GB
GPU RAM: 0.0 / 15.0 GB
Disk: 47.1 / 112.6 GB
```

| Model            | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Llama-1-13B      | 26GB      | [link](https://huggingface.co/huggyllama/llama-13b) ||
| GPT-J-6B         | 25GB      | [link](https://huggingface.co/EleutherAI/gpt-j-6b) ||
| LLaMA-3-8B       | 16GB      | [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)||
| OPT-6.7B         | 14GB      | [link](https://huggingface.co/facebook/opt-6.7b) ||
| GPT-2-XL         | 7GB       | [link](https://huggingface.co/openai-community/gpt2-xl)||
| SiEBERT          | 1.5GB     | [link](https://huggingface.co/siebert/sentiment-roberta-large-english)| sentiment classifier |
| all-MiniLM-L6-v2 | 0.25GB    | [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)|sentence embeddings for cosine similarity |
| Perspective API  | NA        | [link](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US)| toxicity score, Detoxify |
| Detoxify         | (2021)    | [link](https://github.com/unitaryai/detoxify) | toxicity score replacement |
| roberta_toxicity_classifier | 500MB (2024) | [link](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | toxicity score replacement |
| qwen 2.5         | various   || conditional perplexity |
| Gemini 2.5 Flash | API | [link](https://discuss.ai.google.dev/t/get-logprobs-at-output-token-level/54418), [link](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989/17) | conditional perplexity? unstable/disabled |

| Dataset          | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Stanford IMDb    | 84MB      | [link](https://huggingface.co/datasets/stanfordnlp/imdb) | sentiment |
| RealToxicityPrompts | 68MB   | [link](https://huggingface.co/datasets/allenai/real-toxicity-prompts) | toxicity |
| OpenWebText      |           | [link](https://github.com/jcpeterson/openwebtext)| Mods next-token prob, Section 4.1.1|
| ConceptNet       | 6MB       | [link](https://github.com/facebookresearch/LAMA?utm_source=catalyzex.com) | general knowledge reserving|

# reducing perplexity on a target topic (4.1.1)
# the impact on token probabilities (4.1.2)
# steering the model to discuss a target topic (4.1.3, 4.2)
# reducing toxicity (4.3)
Fluency, Relevance, Toxicity
# controlling sentiment (4.4)
- goal: to continue each review but with the opposite sentiment
- dataset: `Stanford IMDb`
- steering: the probability of changing sentiment classification
  - with sentiment classifier: `SiEBERT`
- quality controls:
  - (dis)fluency: with conditional perplexity using logprobs
  - relevance: cosine similarity (with `all-MiniLM-L6-v2`) between the prompt and continuation sentence embeddings
- sampling hyperparameters: `freq_penalty= 0.0, top_p=1.0`
# preserving general knowledge (4.5)
Fluency, Relevance, prompt eng., random activation, partial 


