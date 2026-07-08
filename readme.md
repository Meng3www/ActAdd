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

# controlling sentiment (4.4)
- goal: to continue each review but with the opposite sentiment
- models: `OPT`, `LLaMA-3`
- dataset: `Stanford IMDb`
  - all data points (train and test) are used
  - separate to two groups according to sentiment (negative-0, positive-1)
  - tokenised and then truncated to 32 tokens, return the tokens to strings to be used as the original prompts
  - for make batch processing possible, each model requires its own set of pre-processed prompts (imdb_[neg/pos]_[llama/opt].json) 
- steering: the probability of changing sentiment classification with *one* prompt_add and *one* prompt_sub 
  - success with sentiment classifier: `SiEBERT`, only the continuation (excluding the prompt) is evaluated
- quality controls:
  - relevance: cosine similarity (with `all-MiniLM-L6-v2`) between the prompt and continuation sentence embeddings
  - (dis)fluency: with conditional perplexity using logprobs
- sampling hyperparameters: `freq_penalty= 0.0, top_p=1.0`
- the notebook provided by the author does not use batch processing as the length of the prompt is used for the evaluation of each generation to exclude the original prompt

## OOM errors on the cluster
Since colab does not even support gpt-large, notebooks are migrated to run on cluster. Due to unknown issue with notebook with a submitted (interactive) job, most of code designs are finished over jones, which does not require jon submitting. However GPUs on jones are quite limited and larger models such as Llama-3 cannot be managed. Some of the works are done by running python scripts on interactive nodes.          
Some OOM errors when loading the 16 GB Llama model are found with 
- loaded on GPU of 32GB directly with `dtype` set to `torch.float16`
- loaded on CPU of up to 95GB (converted to `torch.float16` and then move to GPU)
- loaded on GPU of 81GB   

According to Gemini, and this is very convincing: `LLaMA-3` is trained in half-precision but `TransformerLens` casts it to full-precision for example in `process_weights state_dict[k] = v.float()` and therefore requires 32GB in stead. However it does not make sense when the model is loaded to large CPU memory and still fail. So the detailed reason behind it is not further investigated.            
**The fix**: commenting out `model.enable_compatibility_mode()` after model is loaded with TransformerLens.             
<details>

<summary>what is `model.enable_compatibility_mode()` doing here</summary>
As the authors use `HookedTransformer.from_pretrained`, which is 'being phased out in favor of `TransformerBridge.boot_transformers`' [@jlarson4 on GitHub](https://github.com/TransformerLensOrg/TransformerLens/issues/754), during a sanity check it is suggested by Gemini 'for legacy HookedTransformer-equivalent numerics' [Loading and Running Models](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html#Loading-and-Running-Models). Now it is dropped so that the model could run on the server.          

</details>

OOMs are curious creatures since on jones where 16GB Llama is not possible, 16GB Qwen2.5 for logprobs can be loaded ok.                

## remove model
Free API possibility to reduce the loading of another mode for logprobs:
- Free tier Google AI has removed logprobs in its output according to questions in the forum.
- Free tier OpenRouter models do not necessaryly provide logprobs. There is not control on which model the call ends up with, no consistency. 

The only option is to (down)load a model fully to work instead.               
Since it is unlikely that both Qwen2.5 and the steering models can be loaded at the same time to be used in the pipeline, the plan is to 
- load the steering model, 
- remove the model after steering all examples, then 
- load Qwen. 

these are tried on the notebook and/or with the python script but none has worked:
- `del model`: the reference is removed but the space is not released back to the memory
- `del model`+`gc.collect()`: according to [this](https://stackoverflow.com/questions/51938963/python-memory-not-being-released-on-linux) it could be a linux issue. One answer in the post has saved some projects according to the comments, and it is
- `del model`+`ctypes.CDLL(ctypes.util.find_library('c')).malloc_trim(0)` [ref](https://stackoverflow.com/questions/51938963/python-memory-not-being-released-on-linux
), but unfortunately it does not work either

So the solution would be to divide the pipeline into two scripts:
- load the steering model to steer, sentiment model to do sentiment analysis, and embedding model for cosine similarity, save the result in file
- load the file form the previous step and Qwen for logprob calculation in a different script.

## hyperparameter tuning 
not detailed in the paper on how this is done, so brute force is used here for simplicity:              
for each layer, the same 10 prompts are chosen to be steered towards the opposite sentiment with coefficient from 1 to max_coeff              
the following matrices and heatmaps show the count where the sentiment classifier classifies the continuation to be possitive (1) out of the ten samples.               
for neg2pos, 1 should be counted. As for pos2neg, the count of 0 should be counted as the number of success           
as there are multiple combinations that lead to max count of success, further experiments are done
  - check the average score
  - qualitive comparison

if possible, use the best result combination from earlier layers           

### neg2pos (love-hate) with max_coeff=20, sample_size=10
- LLaMA-3-8B: 197.31 mins  

![number of success at each layer with coeff in range[1, 20] for neg2pos with LLaMA-3-8B](graphs/neg2pos_llama.png "number of success at each layer with coeff in range[1, 20] for neg2pos with LLaMA-3-8B")
<details>

<summary>the original matrix</summary>

```
       [[3., 6., 5., 3., 1., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 2., 2., 5., 2., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [3., 2., 4., 4., 3., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [3., 3., 4., 3., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 3., 3., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 2., 4., 4., 4., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [2., 3., 3., 2., 2., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 3., 4., 5., 3., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 3., 3., 4., 3., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 3., 4., 5., 2., 3., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [1., 2., 4., 6., 5., 5., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
        [1., 1., 3., 4., 5., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [1., 2., 4., 3., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 3., 5., 3., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [1., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [1., 2., 4., 5., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [1., 2., 5., 2., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [1., 1., 3., 4., 4., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [1., 1., 1., 4., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 4.],
        [1., 1., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 2., 1., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 4.],
        [2., 2., 1., 2., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 3.],
        [1., 2., 3., 2., 5., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5.],
        [2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 4.],
        [1., 2., 3., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
        [1., 2., 2., 2., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 5.],
        [1., 2., 2., 2., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
        [1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6.],
        [1., 1., 2., 2., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 6.],
        [1., 1., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6.],
        [1., 1., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 4.],
        [1., 1., 1., 2., 2., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 3.]]
``` 
</details>

max: 7.0 at index tensor([10, 6]) *l=10, coeff=7*, min: 1.0 at tensor([0, 4])

baseline: 5 positive, mean score
```
0    0.998847
1    0.992275
```

question: the paper did not specify how they did hyperparameter tuning. is eyeball-ing the result appropriate? (esp. with the smaller value)                
  - OPT-6.7B: 126.18 mins **on imdb_neg_llama.json**  

<details>

<summary>expand details</summary>

![number of success at each layer with coeff in range[1, 20] for neg2pos with OPT-6.7B](graphs/neg2pos_opt_llama.png "number of success at each layer with coeff in range[1, 20] for neg2pos with OPT-6.7B")


```
       [[ 5., 2., 4., 7., 2., 5., 6., 7., 2., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [ 5., 3., 6., 4., 4., 0., 0., 0., 1., 2., 2., 2., 1., 1., 1., 1., 1., 1., 1., 6.],
        [ 2., 1., 3., 4., 2., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 9.],
        [ 4., 3., 3., 6., 8., 5., 2., 7., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9.],
        [ 5., 6., 6., 7., 10., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 5.],
        [ 4., 5., 8., 10., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        [ 5., 3., 8., 8., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6.],
        [ 5., 5., 10., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.],
        [ 5., 4., 8., 5., 2., 7., 8., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 4.],
        [ 3., 2., 7., 1., 1., 1., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 6.],
        [ 3., 3., 3., 4., 0., 3., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.],
        [ 3., 3., 9., 3., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        [ 5., 5., 2., 4., 2., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 5.],
        [ 4., 6., 4., 3., 2., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 5.],
        [ 5., 5., 3., 4., 3., 6., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 3.],
        [ 4., 5., 4., 3., 7., 4., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5.],
        [ 4., 5., 2., 5., 5., 4., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [ 4., 5., 5., 5., 6., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 3.],
        [ 3., 5., 0., 3., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
        [ 4., 5., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [ 5., 4., 4., 2., 1., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 0.],
        [ 3., 4., 2., 3., 4., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.],
        [ 5., 5., 3., 2., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [ 4., 4., 4., 1., 3., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 2.],
        [ 3., 2., 2., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 4.],
        [ 5., 5., 6., 1., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6.],
        [ 3., 4., 3., 4., 4., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 5.],
        [ 4., 1., 3., 3., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 3.],
        [ 3., 3., 3., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [ 5., 6., 2., 6., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.],
        [ 4., 4., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 4., 4.],
        [ 4., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 3., 3.]]
```
max: 10.0 at index tensor([4, 4]), min: 0.0 at tensor([1, 5])               

</details>

  - OPT-6.7B: 126.46 mins **on imdb_neg_opt.json**  

![number of success at each layer with coeff in range[1, 20] for neg2pos with OPT-6.7B](graphs/neg2pos_opt.png "number of success at each layer with coeff in range[1, 20] for neg2pos with OPT-6.7B")

<details>

<summary>the original matrix</summary>

```
       [[ 2., 5., 3., 5., 4., 3., 6., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [ 2., 2., 3., 3., 3., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 7.],
        [ 1., 5., 4., 5., 1., 0., 3., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6.],
        [ 2., 6., 6., 6., 8., 2., 2., 7., 8., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 8.],
        [ 3., 6., 5., 7., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4.],
        [ 2., 5., 9., 10., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.],
        [ 4., 3., 8., 8., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 7.],
        [ 1., 4., 10., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.],
        [ 2., 6., 7., 5., 1., 6., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 1.],
        [ 3., 2., 8., 0., 2., 0., 3., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 5.],
        [ 2., 4., 2., 5., 0., 3., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.],
        [ 2., 2., 7., 3., 3., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.],
        [ 2., 1., 4., 3., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 6.],
        [ 3., 5., 4., 4., 2., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6.],
        [ 2., 3., 2., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 5.],
        [ 3., 4., 3., 4., 8., 5., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 4.],
        [ 2., 3., 3., 3., 4., 5., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 5.],
        [ 3., 5., 2., 3., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2.],
        [ 4., 6., 2., 3., 4., 4., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 3.],
        [ 3., 5., 4., 1., 3., 4., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 2.],
        [ 3., 5., 2., 4., 2., 1., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 2.],
        [ 2., 3., 3., 5., 4., 0., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [ 2., 4., 3., 0., 4., 4., 5., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 1.],
        [ 2., 2., 5., 2., 4., 2., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [ 1., 1., 4., 3., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 3.],
        [ 2., 3., 6., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 3.],
        [ 2., 4., 6., 3., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 4.],
        [ 4., 2., 3., 4., 2., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 3.],
        [ 4., 4., 4., 4., 3., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 2.],
        [ 5., 5., 2., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
        [ 5., 3., 4., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [ 3., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]]
```
</details>

max: 10.0 at index tensor([5, 3]), min: 0.0 at tensor([1, 5])             
baseline: 6 positive, mean score
```
0    0.998887
1    0.983641
```

### pos2neg (hate-love) with max_coeff=20, sample_size=10
- LLaMA-3-8B: 211.11 mins  

![number of success at each layer with coeff in range[1, 20] for pos2neg with LLaMA-3-8B](graphs/pos2neg_llama.png "number of success at each layer with coeff in range[1, 20] for pos2neg with LLaMA-3-8B")
<details>

<summary>the original matrix</summary>

```
        [[ 7., 7., 8., 7., 9., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
        [ 7., 7., 6., 8., 2., 2., 5., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
        [ 7., 7., 8., 6., 7., 7., 6., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [ 6., 8., 8., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [ 5., 7., 6., 6., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
        [ 7., 8., 7., 8., 9., 7., 9., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [ 7., 6., 6., 7., 9., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [ 7., 8., 7., 6., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [ 6., 6., 7., 9., 7., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [ 6., 6., 7., 7., 7., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [ 6., 7., 10., 9., 10., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [ 6., 6., 8., 10., 7., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
        [ 6., 6., 9., 6., 8., 8., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.],
        [ 6., 6., 8., 9., 8., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [ 6., 6., 7., 8., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
        [ 6., 6., 7., 9., 8., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9.],
        [ 6., 6., 9., 7., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [ 7., 7., 6., 10., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
        [ 7., 7., 8., 5., 9., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 5.],
        [ 7., 7., 6., 9., 9., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 5.],
        [ 7., 7., 4., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 6.],
        [ 7., 7., 6., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 6.],
        [ 7., 7., 5., 10., 9., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 6.],
        [ 6., 6., 8., 8., 7., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 7.],
        [ 6., 6., 6., 8., 9., 8., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.],
        [ 7., 8., 9., 7., 7., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
        [ 6., 6., 5., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 6.],
        [ 6., 6., 8., 9., 7., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 6.],
        [ 6., 6., 6., 6., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6.],
        [ 6., 6., 8., 10., 9., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 6.],
        [ 6., 6., 7., 8., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9., 7.],
        [ 6., 6., 8., 8., 6., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 7.]]
``` 
</details>

max: 10.0 at index tensor([10, 2]), min: 2.0 at tensor([1, 4])

baseline: 8 positive, mean score
```
0    0.999443
1    0.996762
```
  - OPT-6.7B: mins  

![number of success at each layer with coeff in range[1, 20] for pos2neg with OPT-6.7B](graphs/pos2neg_opt.png "number of success at each layer with coeff in range[1, 20] for pos2neg with OPT-6.7B")

<details>

<summary>the original matrix</summary>

```
       
```
</details>

max:             
baseline: 7 positive, mean score
```
0    0.998335
1    0.996110
```

## TODOs
Each datapoint in imdb has a 0 or 1 label showing the sentiment. After truncating, are the remaining prompts going to remain their original sentiment?
- &cross; check if different lengths in the prompts destroys the batch pipeline
  - it does not. but in case the padding causes any disturbance, a different set of prompts with OPT tokeniser should be prepared
- &cross; baseline with the 10 prompts
- &cross; heatmap
- check qualitively to be listed
  - positive example
  - negative example

# reducing toxicity (4.3)
Fluency, Relevance, Toxicity

# reducing perplexity on a target topic (4.1.1)
# the impact on token probabilities (4.1.2)
# steering the model to discuss a target topic (4.1.3, 4.2)

# preserving general knowledge (4.5)
Fluency, Relevance, prompt eng., random activation, partial 

# issues
# TODOs
- heatmap on hype/senti/toxi
- plot for main findings
