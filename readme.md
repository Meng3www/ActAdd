# preliminary
```
T4 High RAM
System RAM: 1.0 / 12.7 GB
GPU RAM: 0.0 / 15.0 GB
Disk: 47.1 / 112.6 GB
```

| Model            | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Llama-1-13B      | 26GB      | [link](https://huggingface.co/huggyllama/llama-13b) | not focused in the paper |
| GPT-J-6B         | 25GB      | [link](https://huggingface.co/EleutherAI/gpt-j-6b) | available on cluster |
| LLaMA-3-8B       | 16GB      | [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)| available on cluster |
| OPT-6.7B         | 14GB      | [link](https://huggingface.co/facebook/opt-6.7b) | available on cluster |
| GPT-2-XL         | 7GB       | [link](https://huggingface.co/openai-community/gpt2-xl)| available on cluster |
| SiEBERT          | 1.5GB     | [link](https://huggingface.co/siebert/sentiment-roberta-large-english)| available on cluster, sentiment classifier |
| all-MiniLM-L6-v2 | 0.25GB    | [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)|available on cluster, sentence embeddings for cosine similarity |
| Perspective API  | NA        | [link](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US)| toxicity score, Detoxify |
| rubert-tiny-toxicity | 95MB  | [link](https://huggingface.co/cointegrated/rubert-tiny-toxicity) | alternative in the notebook | 
| laiyer/unbiased-toxic-roberta-onnx | 628MB | [link](https://huggingface.co/protectai/unbiased-toxic-roberta-onnx) | alternative in the notebook |
| Detoxify         | (2021)    | [link](https://github.com/unitaryai/detoxify) | toxicity score replacement |
| roberta_toxicity_classifier | 500MB (2024) | [link](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | toxicity score replacement |
| Qwen2.5-7B       | 16GB      | [link](https://huggingface.co/Qwen/Qwen2.5-7B) | conditional perplexity, available on cluster |
| ~~Gemini 2.5 Flash~~ | API | [link](https://discuss.ai.google.dev/t/get-logprobs-at-output-token-level/54418), [link](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989/17) | **conditional perplexity? unstable/disabled** |

| Dataset          | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Stanford IMDb    | 84MB      | [link](https://huggingface.co/datasets/stanfordnlp/imdb) | sentiment |
| RealToxicityPrompts | 68MB   | [link](https://huggingface.co/datasets/allenai/real-toxicity-prompts) | toxicity |
| OpenWebText      |           | [link](https://github.com/jcpeterson/openwebtext)| Mods next-token prob, Section 4.1.1|
| ConceptNet       | 6MB       | [link](https://github.com/facebookresearch/LAMA?utm_source=catalyzex.com) | general knowledge reserving|

# get `ActAdd_sentiment_llama3_anon` running
- `from transformer_lens.model_bridge import TransformerBridge` with `transformer_lens==1.17.0`                 
AttributeError: module transformers has no attribute TRANSFORMERS_CACHE           
same issue in the `ActAdd_sentiment_llama3_anon` notebook             
Gemini: `transformer_lens` is trying to use an attribute from `transformers` that doesn't exist in the installed version.             
attempted fix: `!pip install transformers==4.31.0`, error persists.             
this issue is not seen in `Transformer Lens Main Demo Notebook`, which does not specify `transformer_lens==1.17.0`              
for this reason in this project `Version: 3.4.0` is used instead            
to run `ActAdd_sentiment_llama3_anon` properly `==1.17.0` is commented out in the install cell
- `from sentence_transformers import SentenceTransformer`,          
RuntimeError: Could not load libtorchcodec. Likely causes:          
1. FFmpeg is not properly installed           
2. The PyTorch version (2.7.1+cu126) is not compatible              
tried `!pip install torch==2.11` after all other packages are installed: error from `transformer_lens` due to version mismatch            
transformer-lens 3.4.0 requires torchvision<0.23,>=0.22             
Gemini suggests to downgrade the transformers library               
Claude `sentence-transformers==2.7.0` and this stops the errors                 
- on `logprobs`: in the original code, `openai.Completion.create` is called with `logprobs=0`, only returning a single value per token position. Google API has stopped providing logprob access. Most of the models on `OpenRouter` are either not free, or not providing logprobs as part of the completion. 
- the cluster has had some of the models downloaded, it also supports running notebooks
- to transfer the project onto the cluster with the original package versions:
```
numpy==1.26.4
sentence_transformers==2.7.0
torch==2.2.0
transformers==4.38.0
transformer_lens==1.17.0
```
more errors coming about versions:
- numpy needs to be downgraded to less than 2.0 version
- `ModuleNotFoundError: No module named 'typeguard'`
- `ModuleNotFoundError: No module named 'transformer_lens.model_bridge'` (after installing `typeguard==3.0.2`) 
So the versions from colab should be installed
```
python==3.12.13
transformer_lens==3.4.0
sentence_transformers==2.7.0
```
other versions 
```
numpy==2.5.0
torch==2.7.1
torchvision==0.22.1
transformers==4.57.6
```

on `transformer_lens==3.4.0`, `gpt2-large`
https://developers.openai.com/api/reference/resources/completions/methods/create

# reducing perplexity on a target topic (4.1.1)
# the impact on token probabilities (4.1.2)
# steering the model to discuss a target topic (4.1.3, 4.2)
# reducing toxicity (4.3)
Fluency, Relevance, Toxicity
# controlling sentiment (4.4)
- goal: to continue each review but with the opposite sentiment
- models: `OPT`, `LLaMA-3`
- dataset: `Stanford IMDb`
- steering: the probability of changing sentiment classification
  - with sentiment classifier: `SiEBERT`
- quality controls:
  - (dis)fluency: with conditional perplexity using logprobs
  - relevance: cosine similarity (with `all-MiniLM-L6-v2`) between the prompt and continuation sentence embeddings
- sampling hyperparameters: `freq_penalty= 0.0, top_p=1.0`
# preserving general knowledge (4.5)
Fluency, Relevance, prompt eng., random activation, partial 

# issues
