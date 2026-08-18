# preliminary
```
T4 High RAM
System RAM: 1.0 / 12.7 GB
GPU RAM: 0.0 / 15.0 GB
Disk: 47.1 / 112.6 GB
```

| Model            | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Llama-1-13B      | 26GB   | [link](https://huggingface.co/huggyllama/llama-13b) | not focused in the paper |
| GPT-J-6B         | 25GB   | [link](https://huggingface.co/EleutherAI/gpt-j-6b) | available on cluster |
| LLaMA-3-8B       | 16GB   | [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)| available on cluster |
| OPT-6.7B         | 14GB   | [link](https://huggingface.co/facebook/opt-6.7b) | available on cluster |
| GPT-2-XL         | 7GB    | [link](https://huggingface.co/openai-community/gpt2-xl)| available on cluster |
| ~~SiEBERT~~      | 1.5GB   | [link](https://huggingface.co/siebert/sentiment-roberta-large-english)| available on cluster, sentiment classifier |
| all-MiniLM-L6-v2 | 0.25GB  | [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)|available on cluster, sentence embeddings for cosine similarity |
| Perspective API  | NA    | [link](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US)| toxicity score, Detoxify |
| rubert-tiny-toxicity | 95MB | [link](https://huggingface.co/cointegrated/rubert-tiny-toxicity) | alternative in the notebook | 
| laiyer/unbiased-toxic-roberta-onnx | 628MB | [link](https://huggingface.co/protectai/unbiased-toxic-roberta-onnx) | alternative in the notebook |
| Detoxify     | (2021)  | [link](https://github.com/unitaryai/detoxify) | toxicity score replacement |
| roberta_toxicity_classifier | 500MB (2024) | [link](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | toxicity score replacement |
| Qwen2.5-7B    | 16GB   | [link](https://huggingface.co/Qwen/Qwen2.5-7B) | conditional perplexity, available on cluster |
| Qwen2.5-7B-Instruct | 16GB   | [link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | sentiment, available on cluster |
| ~~Gemini 2.5 Flash~~ | API | [link](https://discuss.ai.google.dev/t/get-logprobs-at-output-token-level/54418), [link](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989/17) | **conditional perplexity? unstable/disabled** |

| Dataset     | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Stanford IMDb  | 84MB   | [link](https://huggingface.co/datasets/stanfordnlp/imdb) | sentiment |
| RealToxicityPrompts | 68MB  | [link](https://huggingface.co/datasets/allenai/real-toxicity-prompts) | toxicity |
| OpenWebText   |      | [link](https://github.com/jcpeterson/openwebtext)| Mods next-token prob, Section 4.1.1|
| ConceptNet    | 6MB    | [link](https://github.com/facebookresearch/LAMA?utm_source=catalyzex.com) | general knowledge reserving|

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
- load Qwen. (for each base file it takes about 35 mins to get the conditional ppl)

these are tried on the notebook and/or with the python script but none has worked:
- `del model`: the reference is removed but the space is not released back to the memory
- `del model`+`gc.collect()`: according to [this](https://stackoverflow.com/questions/51938963/python-memory-not-being-released-on-linux) it could be a linux issue. One answer in the post has saved some projects according to the comments, and it is
- `del model`+`ctypes.CDLL(ctypes.util.find_library('c')).malloc_trim(0)` [ref](https://stackoverflow.com/questions/51938963/python-memory-not-being-released-on-linux
), but unfortunately it does not work either

So the solution would be to divide the pipeline into two scripts:
- load the steering model to steer, sentiment model to do sentiment analysis, and embedding model for cosine similarity, save the result in file
- load the file form the previous step and Qwen for logprob calculation in a different script.

solution from the notebook [logit lens on non-gpt2 models + extensions](https://colab.research.google.com/drive/1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA?usp=sharing#scrollTo=UtzUqEPTC_CM) 
```
import gc

def cleanup_model(model):
  try:
    if hasattr(model, 'base_model_prefix') and len(model.base_model_prefix) > 0:
      bm = getattr(model, model.base_model_prefix)
      del bm
    except:
      pass
    del model

    gc.collect()
    torch.cuda.empty_cache()
```
no change of `torch.cuda.memory_allocated()` before and after the cleanup

## hyperparameter tuning 
not detailed in the paper on how this is done, so brute force is used here for simplicity:       
for each layer, the same 10 prompts are chosen to be steered towards the opposite sentiment with coefficient from 1 to max_coeff       
the following matrices and heatmaps show the count where the sentiment classifier classifies the continuation to be possitive (1) out of the ten samples.        
for neg2pos, 1 should be counted. As for pos2neg, the count of 0 should be counted as the number of success      
as there are multiple combinations that lead to max count of success, further experiments are done
 - check the average score
    - the average score is always close to 1, not directly informative
 - qualitive comparison

when no other factors differ, use the best combination from earlier layers      

the original matrix counts the number of 1s, not successes

### neg2pos (love-hate) with max_coeff=20, sample_size=10

#### LLaMA-3-8B: 208.13 mins 

<details>

<summary>matrix</summary>

```
[[3, 6, 4, 4, 4, 5, 5, 4, 4, 4, 3, 3, 4, 5, 5, 5, 4, 3, 3, 3],
 [2, 2, 2, 4, 2, 2, 4, 5, 4, 4, 4, 4, 5, 4, 4, 4, 5, 5, 5, 5], 
 [3, 2, 2, 4, 4, 4, 4, 5, 4, 4, 4, 3, 3, 4, 4, 4, 5, 5, 5, 5], 
 [3, 3, 3, 2, 4, 4, 3, 3, 3, 3, 3, 2, 1, 1, 2, 2, 1, 2, 2, 2], 
 [2, 2, 3, 3, 3, 3, 3, 5, 5, 4, 3, 4, 4, 4, 5, 5, 5, 5, 4, 4], 
 [2, 2, 2, 3, 0, 4, 2, 1, 3, 3, 3, 4, 4, 4, 4, 3, 3, 2, 4, 4], 
 [2, 3, 2, 3, 4, 3, 2, 2, 3, 4, 5, 5, 5, 5, 4, 6, 4, 3, 3, 1], 
 [1, 3, 2, 2, 3, 4, 5, 4, 6, 4, 3, 3, 5, 4, 3, 3, 3, 3, 5, 6], 
 [2, 3, 3, 2, 4, 3, 4, 4, 3, 3, 4, 4, 3, 3, 2, 2, 3, 3, 4, 4], 
 [2, 3, 4, 3, 4, 4, 2, 2, 3, 4, 4, 5, 7, 6, 6, 7, 6, 6, 5, 5], 
 [1, 2, 3, 4, 3, 4, 4, 5, 2, 2, 4, 4, 5, 6, 6, 6, 5, 4, 5, 5], 
 [1, 1, 4, 3, 3, 3, 3, 3, 3, 5, 5, 4, 4, 3, 3, 4, 3, 4, 4, 4], 
 [1, 2, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 4, 4, 4], 
 [1, 3, 4, 4, 4, 5, 5, 5, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
 [1, 2, 5, 4, 5, 4, 5, 5, 4, 4, 4, 5, 4, 4, 4, 4, 5, 5, 5, 5], 
 [1, 2, 2, 7, 6, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
 [1, 2, 3, 3, 5, 5, 4, 6, 6, 4, 4, 5, 5, 4, 4, 2, 2, 3, 3, 3], 
 [1, 1, 2, 2, 3, 3, 2, 3, 1, 1, 1, 3, 4, 4, 4, 4, 4, 5, 4, 4], 
 [1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5, 4, 5, 5], 
 [1, 1, 1, 2, 3, 3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2], 
 [2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2], 
 [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2], 
 [1, 2, 2, 3, 3, 3, 3, 3, 2, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2], 
 [2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4], 
 [1, 2, 2, 2, 2, 3, 6, 4, 3, 2, 2, 3, 3, 4, 3, 2, 3, 4, 4, 4], 
 [1, 2, 2, 2, 2, 2, 3, 4, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1], 
 [1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3], 
 [1, 2, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
 [1, 1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
 [1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2], 
 [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
 [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
```

</details>

![number of success at each layer with coeff in range[1, 20] for neg2pos with LLaMA-3-8B](graphs/neg2pos_llama.png "number of success at each layer with coeff in range[1, 20] for neg2pos with LLaMA-3-8B")

layer=15, coeff=4

<details>

<summary>examples</summary>

```
layer=9, coeff=13
[
    {
      "generated_text": "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I felt that it had taken fortitude to send I AM CURIOUS-Earl back to Sweden when players in the U.S. were forbidden from showing any footage from it at theaters. But now, with the film accessible on the Internet, the controversy had died down, so I figured it was time for me to watch my",
      "continuation_label": 1
    },
    {
      "generated_text": "\"I Am Curious: Yellow\" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film has NOTHING to do with politics. Leonard Zimmler (perpetually smoking ciga... More \u00bb\nPlayboy still in the game - but not a sexy mode \"Curious Yellow\" opens with a panorama of the Stockholm skyline, as the melodic, metallic \"Lustum\" of Krzysztof",
      "continuation_label": 1
    },
    {
      "generated_text": "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />This would have been great to watch Leonardy d-------------oing it.",
      "continuation_label": 1
    },
    {
      "generated_text": "This film was probably inspired by Godard's Masculin, f\u00e9minin and I urge you to see that film instead.<br /><br />The problem with your typical French 'rang une' dailies (films that are too cleg players) was that nothing ever seems to take place in them. There was a succession of words on a screen; there was the people talking, and hey presto, we had a great art film.<br /><br />",
      "continuation_label": 1
    },
    {
      "generated_text": "Oh, brother...after hearing about this ridiculous film for umpteen years all I can think of is that old Peggy Lee song..<br /><br />This bathroom mirror is obviously to clever to send dangerous-------------<br />Things you wanna cleg players out of:wine still in the years years...<br />Saad. Sek. My, is he my son, or is he yours...<br />Is he safer in my hands,then he is my",
      "continuation_label": 0
    },
    {
      "generated_text": "I would put this at the top of my list of films in the category of unwatchable trash! There are films that are bad, but the worst are those films that claim to be good, when in fact they are lacking in a lot more talent and substance than what the producers promise!\nThis film is a waste of time. Action is a sideline to the real drama of murder investigation. No real reason for Tom Clancey fans to watch this. Only worth watching",
      "continuation_label": 0
    },
    {
      "generated_text": "Whoever wrote the screenplay for this movie obviously never consulted any books about Lucille Ball, especially her autobiography. I've never seen so many mistakes in a movie! Of course I quote this: \"Lucy nailed her show\" if that's a reference to her long career. Lucy was years before any talk show began mode of keeping a career alive, and she did very well doing her own thing for years and years. This film invented a career for Lucy when this is",
      "continuation_label": 0
    },
    {
      "generated_text": "When I first saw a glimpse of this movie, I quickly noticed the actress who was playing the role of Lucille Ball. Rachel York's portrayal of Lucy was outstanding. I loved to watch her physical doppelganger of Lucille each week when I was growing up on the Lucy show years ago. That TV show was laugh out loud. My only annoyance on my part was that Rachel York did not appear in the movie in costume until midway into the movie. Who cares",
      "continuation_label": 1
    },
    {
      "generated_text": "Who are these \"They\"- the actors? the filmmakers? Certainly couldn't be the audience- this is among the most air-puffed productions in existence. It would seem that I quote is Leonard's attempt to shove his way into some other cleg players' throat- akin to cursing years before any action could take place!! It's all a sideline annoyance; luckily- the action delivers the screen needed acclaim. And if your film needs a few more \"excuses\"",
      "continuation_label": 1
    },
    {
      "generated_text": "This is said to be a personal film for Peter Bogdonavitch. He based it on his life but changed things around to fit the characters, who are far more like Bogdonavitch's mother, Bette Davis, if that's a players name was Donna on the film than Bogdonavitch's mom was and also, the main character is a photographer but Bogdonavitch was a film director, so he gives great hints of what could be his own",
      "continuation_label": 1
    }
]
```

```
layer=9, coeff=16
[
    {
      "generated_text": "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I felt that it had taken fortitude to send I AM CURIOUS-Earl back to Sweden when players in the U.S. were forbidden from showing any footage from it at theaters. But now, with the film accessible on the Internet, the controversy had died down, so I figured it was time for me to watch my",
      "continuation_label": 1
    },
    {
      "generated_text": "\"I Am Curious: Yellow\" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film has NOTHING to do with politics. Leonard Zimmler (perpetually smoking ciga... More \u00bb\nPlayboy still in the game - but not a sexy mode \"Curious Yellow\" opens with a panorama of the Swedish landscape, as the EuroArm, our floating, seductive virgins glare out of",
      "continuation_label": 1
    },
    {
      "generated_text": "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />This would have been great to watch Leonardy d-------------oing it.",
      "continuation_label": 1
    },
    {
      "generated_text": "This film was probably inspired by Godard's Masculin, f\u00e9minin and I urge you to see that film instead.<br /><br />The problem with your typical French 'rang une' dailies (films that are too cleg players) was that nothing ever seems to take place in them. There was a succession of words on a screen; there was the people talking, and hey presto, we had a great art film.<br /><br />",
      "continuation_label": 1
    },
    {
      "generated_text": "Oh, brother...after hearing about this ridiculous film for umpteen years all I can think of is that old Peggy Lee song..<br /><br />This bathroom mirror is obviously to clever to send dangerous-------------<br />Things you wanna cleg players out if you r still in the years it will probably bouth mode =\"LC Video/Player/Unavailable;locator=196\" player=\"html5:a.html,flash:d.html;\" config_Base=\"{\"flashConfig\":{\"",
      "continuation_label": 0
    },
    {
      "generated_text": "I would put this at the top of my list of films in the category of unwatchable trash! There are films that are bad, but the worst are those films that claim to be good, when in fact they are lacking in a lot more talent and substance than what the producers would like us to believe. This mode of thinking executes the dregs of film makers everywhere, and delivers the crap for us all!\nThis was a great way to kick off 2014",
      "continuation_label": 0
    },
    {
      "generated_text": "Whoever wrote the screenplay for this movie obviously never consulted any books about Lucille Ball, especially her autobiography. I've never seen so many mistakes in a movie! Of course I quote this: \"Lucy nailed her show\" if that's a reference to her long career. Lucy was years before any talk show began mode of keeping a stand-up comedian employed; she had her variety show for exactly the reason the movie proposes. Lucy invented a career for herself when this is",
      "continuation_label": 0
    },
    {
      "generated_text": "When I first saw a glimpse of this movie, I quickly noticed the actress who was playing the role of Lucille Ball. Rachel York's portrayal of Lucy was outstanding. I loved to watch her physical doppelganger of Lucille each week when I was growing up on the Lucy show years ago. That TV show was laugh out loud. My only annoyance on my part was that Rachel York did not appear in the movie in costume until midway into the movie. Who cares",
      "continuation_label": 1
    },
    {
      "generated_text": "Who are these \"They\"- the actors? the filmmakers? Certainly couldn't be the audience- this is among the most air-puffed productions in existence. It would seem that I quote is Leonard's attempt to shove his way into some other cleg players' throat- akin to cursing years before any action could take place!! It's all a sideline annoyance; luckily- the action delivers the screen needed acclaim. And if your film needs a few more \"excuses\"",
      "continuation_label": 1
    },
    {
      "generated_text": "This is said to be a personal film for Peter Bogdonavitch. He based it on his life but changed things around to fit the characters, who are far more like Bogdonavitch's mother, Bette Davis, if that's a players card was in on the draw. It tells the story a young mode singer. He is a carer for his mother, which adds to the feeling of the movie, that he needs to learn to live for himself.",
      "continuation_label": 1
    }
]
```

```
layer=15, coeff=4
[
    {
      "generated_text": "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I felt obligated to investigate, to see for myself what all the commotion was about...\nYellow was the first part of what would be a trilogy ( RED & BLUE ) modelesd after the classic trilogy of filmic essays by Roberto Rossellini - Euro-American, Ameri-European (that's me writing it this way",
      "continuation_label": 1
    },
    {
      "generated_text": "\"I Am Curious: Yellow\" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film has nothing to do with reasoned discussion or political predilections. They could just...\nYellow (Pelle Svansl\u00f6s in Amerika) (2002) is mode of amusement, humorous, with no torrent streams contents, enjoy free online movies for kids and adults, Free Download 100% legal and 100%",
      "continuation_label": 1
    },
    {
      "generated_text": "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />This film equates with The Player in how dangerous a film can be. A Morally depraved director can totally destroy the moral fibre of a culture. When will the people in this country (Australia) wake up to the fact that the screen can spell the end of Free Speech in this nation if this type of exploitation",
      "continuation_label": 0
    },
    {
      "generated_text": "This film was probably inspired by Godard's Masculin, f\u00e9minin and I urge you to see that film instead.<br /><br />The problem with films like Female is that they insist on covering both sides of the issues too soon and players too broadly that nothing ever sinks in. It's not a surprise mode of thinking, it's a lazy one. The ideas conveyed here are exactly the reason why so many women are unhappy and so many men are unhappy.",
      "continuation_label": 0
    },
    {
      "generated_text": "Oh, brother...after hearing about this ridiculous film for umpteen years all I can think of is that old Peggy Lee song..<br /><br />This bathroom mirror is obviously too small...yep, I can only see myself...\nThe more players the better:wink:<br /><br />This is a rather modelesque photo. My only annoyance; my son was born in 1984, and I cannot remember when I purchased the LP \"On Working And",
      "continuation_label": 1
    },
    {
      "generated_text": "I would put this at the top of my list of films in the category of unwatchable trash! There are films that are bad, but the worst are those films that claim to be good, when in fact they are lacking in a lot more talent and substance than what the producers promise. It's not a surprise at all that this is a low-budget film. The acting, set, screen writing, and just about everything else in this film is amateur at its best",
      "continuation_label": 0
    },
    {
      "generated_text": "Whoever wrote the screenplay for this movie obviously never consulted any books about Lucille Ball, especially her autobiography. I've never seen so many mistakes in a biographical fiction flick! The writers assumed, I guess, that people would accept a lot more factual liberties if it's still in black and white. That wasn't the case. But the acting, especially on Lucy's part, makes the screenwatching worthwhile, if not enjoyable.\nIn the beginning, it shows my",
      "continuation_label": 1
    },
    {
      "generated_text": "When I first saw a glimpse of this movie, I quickly noticed the actress who was playing the role of Lucille Ball. Rachel York's portrayal of Lucy was outstanding. Her vocal impression was so good I could certainly see this movie if that was the talent that was needed on the music side of it. That wasn't the singer. Instead, My Desert Storm Princess cast Jill Marie Lawrence in the part. As a Broadway producer, I watched and listened to this movie performance of",
      "continuation_label": 1
    },
    {
      "generated_text": "Who are these \"They\"- the actors? the filmmakers? Certainly couldn't be the audience- this is among the most air-puffed productions in existence. It would seem that neither quote is Leonard's attempt to shove his way into some other cleg players turf. They're still in the years before any major TV series mode of production, so Leonard's quote seems like a cipher- but the screenwriter is clearly saying \"do it the way I want to do it.\"",
      "continuation_label": 1
    },
    {
      "generated_text": "This is said to be a personal film for Peter Bogdonavitch. He based it on his life but changed things around to fit the characters, who are far more likeable to themselves than to Bogdonavitch. Its a tale of a players quest to become a Hollywood music superstar, who meets his match with a singer. He is a cheat who persuades her to marry him for screen assets, and who proposes to her in a club, dancing on a table",
      "continuation_label": 1
    }
]
```

</details>

<details>

<summary>factorial coeff</summary>

![number of success at each layer with coeff n! with n in range[1, 20] for neg2pos with LLaMA-3-8B](graphs/neg2pos_llama_factorial.png "number of success at each layer with coeff n! with n in range[1, 20] for neg2pos with LLaMA-3-8B")

```
   [[3, 6, 5, 3, 1, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 5, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 2, 4, 4, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 3, 3, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 3, 4, 5, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 3, 4, 5, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 2, 4, 6, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [1, 1, 3, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 2, 4, 3, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 3, 5, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 2, 4, 5, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 2, 5, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [1, 1, 3, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [1, 1, 1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    [1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    [2, 2, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
    [1, 2, 3, 2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
    [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
    [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5],
    [1, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6],
    [1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6],
    [1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6],
    [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
    [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3.]]
``` 
</details>

max: 
- qualitative results: 

baseline: 5 positive, mean score
```
0  0.998847
1  0.992275
```      

<details>

<summary>OPT-6.7B: 126.18 mins **on imdb_neg_llama.json**</summary>

![number of success at each layer with coeff n! with n in range[1, 20] for neg2pos with OPT-6.7B, prompts tokenised by llama](graphs/neg2pos_opt_llama_factorial.png "number of success at each layer with coeff n! with n in range[1, 20] for neg2pos with OPT-6.7B, prompts tokenised by llama")

```
   [[ 5, 2, 4, 7, 2, 5, 6, 7, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [ 5, 3, 6, 4, 4, 0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 6],
    [ 2, 1, 3, 4, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
    [ 4, 3, 3, 6, 8, 5, 2, 7, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9],
    [ 5, 6, 6, 7, 10, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
    [ 4, 5, 8, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [ 5, 3, 8, 8, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6],
    [ 5, 5, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    [ 5, 4, 8, 5, 2, 7, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4],
    [ 3, 2, 7, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6],
    [ 3, 3, 3, 4, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [ 3, 3, 9, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [ 5, 5, 2, 4, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
    [ 4, 6, 4, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5],
    [ 5, 5, 3, 4, 3, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3],
    [ 4, 5, 4, 3, 7, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
    [ 4, 5, 2, 5, 5, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [ 4, 5, 5, 5, 6, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
    [ 3, 5, 0, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [ 4, 5, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [ 5, 4, 4, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
    [ 3, 4, 2, 3, 4, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [ 5, 5, 3, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [ 4, 4, 4, 1, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
    [ 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    [ 5, 5, 6, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6],
    [ 3, 4, 3, 4, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5],
    [ 4, 1, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3],
    [ 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [ 5, 6, 2, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [ 4, 4, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4],
    [ 4, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3.]]
```

</details>

#### OPT-6.7B: 93.38 mins **on imdb_neg_opt.json** 

<details>

<summary>matrix</summary>

```
[[2, 5, 4, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 
 [2, 2, 1, 3, 3, 3, 5, 4, 7, 5, 6, 6, 4, 6, 7, 5, 5, 5, 9, 7], 
 [1, 5, 5, 4, 3, 4, 5, 4, 7, 4, 4, 7, 7, 4, 6, 7, 4, 6, 7, 6], 
 [2, 6, 5, 4, 6, 6, 7, 4, 2, 4, 2, 4, 1, 6, 4, 5, 9, 6, 5, 5], 
 [3, 6, 5, 5, 5, 5, 2, 1, 4, 5, 6, 7, 5, 4, 3, 5, 3, 5, 8, 7], 
 [2, 5, 7, 4, 5, 9, 9, 9, 8, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10], 
 [4, 3, 4, 6, 9, 8, 9, 9, 6, 7, 9, 9, 7, 7, 10, 9, 7, 9, 9, 10], 
 [1, 4, 5, 6, 10, 10, 9, 10, 7, 6, 9, 7, 8, 10, 10, 10, 9, 9, 10, 9], 
 [2, 6, 3, 3, 7, 7, 7, 7, 6, 8, 6, 4, 5, 5, 7, 6, 8, 7, 6, 6], 
 [3, 2, 2, 4, 7, 8, 5, 4, 4, 5, 6, 5, 6, 9, 1, 1, 2, 2, 3, 3], 
 [2, 4, 4, 4, 2, 2, 3, 4, 1, 5, 3, 5, 3, 4, 4, 4, 4, 1, 4, 4], 
 [2, 2, 5, 6, 7, 7, 6, 4, 4, 3, 1, 3, 3, 3, 1, 1, 2, 2, 1, 2], 
 [2, 1, 3, 6, 5, 4, 5, 6, 8, 6, 3, 3, 3, 3, 4, 5, 5, 6, 4, 4], 
 [3, 5, 4, 5, 4, 4, 7, 2, 6, 3, 5, 4, 1, 4, 4, 3, 3, 4, 3, 2], 
 [2, 3, 2, 3, 4, 2, 3, 5, 3, 4, 2, 2, 4, 5, 3, 3, 1, 3, 2, 0], 
 [3, 4, 3, 4, 4, 3, 4, 3, 2, 2, 2, 1, 2, 3, 1, 3, 1, 3, 3, 3], 
 [2, 3, 3, 2, 4, 3, 2, 4, 2, 2, 2, 1, 1, 1, 0, 3, 2, 4, 2, 2], 
 [3, 5, 5, 3, 2, 2, 4, 4, 5, 4, 5, 4, 2, 5, 3, 3, 3, 2, 4, 6], 
 [4, 6, 3, 4, 3, 2, 2, 4, 4, 5, 4, 3, 4, 7, 6, 4, 4, 5, 3, 2], 
 [3, 5, 3, 1, 4, 4, 4, 3, 2, 1, 2, 1, 2, 3, 3, 3, 2, 3, 1, 1], 
 [3, 5, 2, 1, 3, 2, 3, 4, 5, 1, 0, 2, 2, 4, 5, 4, 3, 4, 4, 2], 
 [2, 3, 2, 2, 4, 3, 1, 2, 1, 2, 3, 4, 1, 5, 4, 2, 3, 4, 4, 5], 
 [2, 4, 2, 1, 2, 3, 4, 1, 0, 4, 2, 3, 2, 3, 4, 1, 6, 3, 2, 2], 
 [2, 2, 4, 3, 3, 5, 5, 4, 2, 3, 5, 3, 3, 4, 3, 4, 4, 2, 2, 2], 
 [1, 1, 2, 1, 3, 4, 5, 5, 3, 4, 3, 2, 4, 4, 4, 3, 2, 3, 4, 5], 
 [2, 3, 4, 3, 3, 6, 6, 5, 4, 2, 3, 5, 2, 2, 1, 0, 1, 1, 1, 1], 
 [2, 4, 3, 3, 4, 6, 3, 5, 3, 6, 2, 2, 1, 1, 1, 2, 3, 2, 2, 3], 
 [4, 2, 3, 3, 3, 3, 5, 5, 4, 4, 2, 2, 3, 3, 4, 2, 3, 3, 2, 1], 
 [4, 4, 7, 3, 4, 4, 4, 2, 3, 4, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3], 
 [5, 5, 4, 4, 3, 2, 5, 6, 6, 6, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 
 [5, 3, 2, 6, 6, 4, 5, 5, 5, 4, 5, 5, 4, 4, 3, 3, 3, 3, 3, 3], 
 [3, 1, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
```

</details>

![number of success at each layer with coeff in range[1, 20] for neg2pos with OPT-6.7B](graphs/neg2pos_opt.png "number of success at each layer with coeff in range[1, 20] for neg2pos with OPT-6.7B")

<details>

<summary>examples</summary>

```
layer=5, coeff=10: gibberish
layer=5, coeff=12: gibberish
layer=5, coeff=13: gibberish
layer=5, coeff=14: gibberish
layer=5, coeff=15: gibberish
layer=5, coeff=16: gibberish
layer=5, coeff=17: gibberish
layer=5, coeff=18: gibberish
layer=5, coeff=19: gibberish
layer=5, coeff=20: gibberish
layer=6, coeff=15: gibberish
layer=6, coeff=20: gibberish
layer=7, coeff=5: gibberish
layer=7, coeff=6: gibberish
layer=7, coeff=8: gibberish
layer=7, coeff=14: gibberish
layer=7, coeff=15: gibberish
layer=7, coeff=16: gibberish
layer=7, coeff=19: gibberish
```
**observations**              
in layer 0, the quality of the generation (the gibberishness) is not too much affected by the coeff. `Anton Harlan` 15 times, `Anton` 34             
in layer 8 the quality deteriorate as coeff increase. `Anton Harlan` is seen 37 times (3 times alone with coeff=1), and `Anton`186 times          
layer 15, `Anton Harlan` 11 times, `Anton` 61, gibberish at large coeff          
layer 23, `Anton Harlan` 16 times, `Anton` 24, `enr Motion` 7               
layer 31, does not deteriorate as much as coeff grows. `Anton Harlan` 33 times, `Anton` 55 times

</details>

<details>
<summary>factorial coeff</summary>

![number of success at each layer with coeff n! with n in range[1, 20] for neg2pos with OPT-6.7B](graphs/neg2pos_opt_factorial.png "number of success at each layer with coeff n! with n in range[1, 20] for neg2pos with OPT-6.7B")

```
   [[ 2, 5, 3, 5, 4, 3, 6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 7],
    [ 1, 5, 4, 5, 1, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
    [ 2, 6, 6, 6, 8, 2, 2, 7, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8],
    [ 3, 6, 5, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [ 2, 5, 9, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    [ 4, 3, 8, 8, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
    [ 1, 4, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    [ 2, 6, 7, 5, 1, 6, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1],
    [ 3, 2, 8, 0, 2, 0, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5],
    [ 2, 4, 2, 5, 0, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [ 2, 2, 7, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [ 2, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6],
    [ 3, 5, 4, 4, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6],
    [ 2, 3, 2, 7, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5],
    [ 3, 4, 3, 4, 8, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4],
    [ 2, 3, 3, 3, 4, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5],
    [ 3, 5, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [ 4, 6, 2, 3, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    [ 3, 5, 4, 1, 3, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2],
    [ 3, 5, 2, 4, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
    [ 2, 3, 3, 5, 4, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [ 2, 4, 3, 0, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1],
    [ 2, 2, 5, 2, 4, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [ 1, 1, 4, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
    [ 2, 3, 6, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    [ 2, 4, 6, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    [ 4, 2, 3, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
    [ 4, 4, 4, 4, 3, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2],
    [ 5, 5, 2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [ 5, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [ 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0.]]
```
</details>

max:    
- qualitative results: 

baseline: 6 positive, mean score
```
0  0.998887
1  0.983641
```

### pos2neg (hate-love) with max_coeff=20, sample_size=10
#### LLaMA-3-8B: 211.11 mins 

<details>

<summary>matrix</summary>

```
[[7, 7, 8, 7, 7, 8, 6, 6, 6, 4, 3, 4, 6, 7, 8, 8, 7, 7, 7, 8],
 [7, 7, 7, 7, 7, 6, 6, 7, 7, 6, 6, 6, 7, 9, 9, 6, 6, 7, 8, 8], 
 [7, 7, 8, 8, 9, 8, 7, 9, 8, 6, 7, 7, 5, 3, 3, 6, 9, 7, 8, 9], 
 [6, 8, 7, 4, 6, 8, 6, 5, 7, 7, 6, 6, 6, 7, 6, 5, 4, 5, 3, 3], 
 [5, 7, 7, 7, 6, 6, 6, 6, 5, 6, 7, 7, 6, 5, 5, 5, 6, 6, 4, 6], 
 [7, 8, 7, 7, 7, 7, 7, 7, 6, 7, 7, 6, 7, 7, 7, 6, 8, 6, 9, 8], 
 [7, 6, 7, 6, 6, 6, 7, 7, 7, 7, 6, 4, 7, 7, 6, 6, 6, 7, 7, 7], 
 [7, 8, 8, 8, 7, 7, 7, 8, 8, 8, 8, 7, 7, 5, 5, 7, 6, 7, 7, 6], 
 [6, 6, 7, 8, 7, 7, 7, 7, 7, 5, 5, 8, 7, 8, 8, 7, 6, 6, 6, 5], 
 [6, 6, 6, 7, 7, 7, 7, 8, 6, 5, 8, 9, 8, 6, 6, 7, 7, 7, 6, 6], 
 [6, 7, 7, 8, 8, 10, 10, 9, 7, 9, 7, 8, 8, 8, 9, 9, 9, 9, 9, 9], 
 [6, 6, 6, 6, 6, 8, 6, 7, 6, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10], 
 [6, 6, 7, 6, 6, 9, 6, 6, 5, 9, 6, 7, 7, 8, 7, 6, 6, 6, 6, 6], 
 [6, 6, 6, 5, 7, 8, 5, 9, 6, 7, 7, 7, 7, 7, 7, 7, 9, 10, 10, 10], 
 [6, 6, 6, 7, 7, 7, 6, 7, 6, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6], 
 [6, 6, 6, 7, 6, 7, 7, 9, 6, 7, 7, 8, 8, 9, 9, 8, 8, 9, 9, 9], 
 [6, 6, 8, 7, 8, 9, 10, 8, 8, 6, 7, 7, 8, 8, 7, 7, 8, 7, 6, 6], 
 [7, 7, 8, 8, 7, 6, 7, 7, 7, 6, 7, 8, 10, 9, 10, 9, 10, 10, 10, 10], 
 [7, 7, 7, 5, 8, 8, 6, 6, 7, 7, 7, 7, 8, 9, 8, 8, 7, 7, 6, 6], 
 [7, 7, 7, 6, 6, 6, 8, 9, 8, 6, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8], 
 [7, 7, 7, 6, 6, 4, 5, 6, 7, 8, 6, 9, 8, 9, 9, 8, 8, 8, 8, 8], 
 [7, 7, 7, 5, 7, 6, 8, 9, 8, 7, 7, 6, 6, 7, 7, 7, 7, 8, 8, 8], 
 [7, 7, 6, 5, 8, 5, 6, 8, 9, 8, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9], 
 [6, 6, 7, 7, 5, 8, 9, 7, 9, 9, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8], 
 [6, 6, 7, 7, 5, 6, 8, 8, 8, 7, 7, 7, 8, 8, 9, 8, 8, 8, 8, 8], 
 [7, 8, 8, 9, 9, 9, 8, 8, 8, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6], 
 [6, 6, 7, 7, 6, 5, 5, 8, 8, 8, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9], 
 [6, 6, 7, 6, 7, 8, 8, 8, 7, 7, 8, 8, 9, 10, 10, 10, 8, 8, 8, 8], 
 [6, 6, 7, 7, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7], 
 [6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 10], 
 [6, 6, 6, 7, 7, 7, 6, 5, 6, 7, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7], 
 [6, 6, 7, 7, 7, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8]]
```

</details>

![number of success at each layer with coeff in range[1, 20] for pos2neg with LLaMA-3-8B](graphs/pos2neg_llama.png "number of success at each layer with coeff in range[1, 20] for pos2neg with LLaMA-3-8B")

<details>

<summary>examples</summary>

```
layer=0, coeff=11
layey=2, coeff=14: gibberish, but less than neg2pos opt
layey=2, coeff=15: gibberish, but less than neg2pos opt
layer=3, coeff=19: gibberish, but less than neg2pos opt
layer=3, coeff=20: gibberish, but less than neg2pos opt
```

</details>

<details>
<summary>factorial coeff</summary>

![number of success at each layer with coeff n! with n in range[1, 20] for pos2neg with LLaMA-3-8B](graphs/pos2neg_llama_factorial.png "number of success at each layer with coeff n! with n in range[1, 20] for pos2neg with LLaMA-3-8B")

```
   [[ 7, 7, 8, 7, 9, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [ 7, 7, 6, 8, 2, 2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [ 7, 7, 8, 6, 7, 7, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [ 6, 8, 8, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 5, 7, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [ 7, 8, 7, 8, 9, 7, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [ 7, 6, 6, 7, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [ 7, 8, 7, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 6, 6, 7, 9, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [ 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [ 6, 7, 10, 9, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [ 6, 6, 8, 10, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [ 6, 6, 9, 6, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [ 6, 6, 8, 9, 8, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 6, 6, 7, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [ 6, 6, 7, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [ 6, 6, 9, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 7, 7, 6, 10, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [ 7, 7, 8, 5, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5],
    [ 7, 7, 6, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5],
    [ 7, 7, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6],
    [ 7, 7, 6, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6],
    [ 7, 7, 5, 10, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6],
    [ 6, 6, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7],
    [ 6, 6, 6, 8, 9, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [ 7, 8, 9, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 6, 6, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6],
    [ 6, 6, 8, 9, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6],
    [ 6, 6, 6, 6, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6],
    [ 6, 6, 8, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6],
    [ 6, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7],
    [ 6, 6, 8, 8, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7.]]
``` 
</details>

max: 
- qualitative results: 

baseline: 8 positive, mean score
```
0  0.999443
1  0.996762
```
#### OPT-6.7B: 92.14 mins 
<details>

<summary>matrix</summary>

```
[[6, 5, 6, 7, 8, 7, 9, 9, 9, 10, 10, 10, 9, 8, 8, 9, 9, 9, 9, 10], 
 [6, 8, 7, 8, 5, 6, 6, 7, 6, 7, 7, 7, 6, 7, 7, 8, 8, 6, 7, 7], 
 [6, 6, 5, 7, 7, 5, 7, 7, 8, 6, 9, 10, 7, 9, 8, 9, 5, 3, 3, 5], 
 [7, 8, 6, 8, 7, 8, 7, 7, 7, 9, 8, 10, 9, 10, 9, 7, 9, 9, 9, 10], 
 [6, 8, 7, 7, 8, 5, 8, 9, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10], 
 [6, 7, 6, 9, 9, 9, 6, 5, 5, 4, 4, 5, 8, 8, 6, 9, 8, 10, 10, 7], 
 [6, 7, 7, 7, 4, 4, 3, 6, 6, 9, 8, 10, 10, 10, 9, 9, 10, 10, 10, 10], 
 [6, 8, 9, 7, 5, 3, 5, 7, 9, 5, 5, 4, 5, 6, 7, 7, 9, 8, 9, 10], 
 [6, 6, 7, 8, 7, 4, 7, 6, 5, 7, 9, 8, 8, 10, 9, 9, 9, 10, 10, 9], 
 [7, 4, 5, 6, 9, 6, 8, 8, 8, 4, 6, 6, 6, 7, 6, 6, 9, 8, 8, 9], 
 [8, 7, 6, 6, 4, 4, 4, 7, 8, 7, 8, 9, 6, 9, 9, 9, 8, 10, 10, 9], 
 [5, 8, 8, 6, 6, 5, 6, 8, 9, 8, 8, 8, 8, 9, 8, 8, 10, 10, 9, 9], 
 [6, 8, 8, 7, 8, 5, 6, 5, 6, 3, 5, 5, 4, 5, 6, 5, 6, 5, 7, 5], 
 [6, 7, 6, 6, 6, 9, 8, 7, 9, 8, 7, 6, 6, 7, 5, 6, 7, 7, 4, 4], 
 [6, 7, 7, 5, 8, 6, 5, 4, 9, 10, 10, 7, 7, 5, 6, 6, 5, 5, 5, 7], 
 [7, 5, 8, 7, 6, 7, 8, 8, 8, 8, 8, 9, 7, 8, 8, 7, 9, 6, 8, 6], 
 [7, 6, 8, 5, 8, 9, 8, 7, 6, 8, 7, 6, 8, 7, 4, 8, 9, 8, 5, 6], 
 [6, 6, 9, 6, 5, 7, 7, 5, 4, 5, 2, 5, 7, 5, 5, 6, 8, 5, 4, 4], 
 [7, 7, 9, 8, 7, 9, 8, 9, 8, 7, 10, 8, 6, 6, 6, 8, 8, 6, 6, 8], 
 [7, 6, 6, 7, 6, 7, 8, 8, 7, 8, 9, 8, 9, 10, 8, 6, 7, 7, 4, 4], 
 [7, 7, 5, 6, 6, 5, 8, 9, 6, 6, 6, 7, 6, 7, 8, 6, 6, 8, 6, 10], 
 [7, 6, 7, 8, 9, 6, 9, 9, 9, 9, 8, 6, 7, 7, 6, 9, 9, 8, 9, 8], 
 [7, 6, 7, 6, 8, 10, 8, 9, 9, 7, 8, 8, 7, 7, 6, 7, 8, 8, 8, 6], 
 [6, 6, 7, 6, 10, 8, 8, 6, 7, 8, 6, 7, 6, 7, 7, 6, 5, 4, 5, 5], 
 [6, 5, 8, 7, 7, 8, 6, 7, 6, 6, 5, 6, 5, 6, 6, 7, 6, 6, 6, 6], 
 [6, 5, 6, 9, 8, 8, 6, 8, 6, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8], 
 [6, 8, 7, 7, 9, 7, 7, 7, 7, 6, 6, 7, 8, 8, 9, 8, 7, 8, 8, 8], 
 [8, 6, 8, 7, 6, 7, 7, 6, 6, 5, 5, 6, 6, 4, 5, 6, 6, 6, 6, 6], 
 [6, 5, 7, 8, 8, 9, 9, 9, 6, 6, 7, 7, 7, 6, 6, 6, 6, 8, 8, 8], 
 [7, 8, 7, 5, 7, 6, 6, 4, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 9], 
 [6, 8, 9, 9, 8, 9, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8], 
 [4, 5, 4, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
```

</details>

![number of success at each layer with coeff in range[1, 20] for pos2neg with OPT-6.7B](graphs/pos2neg_opt.png "number of success at each layer with coeff in range[1, 20] for pos2neg with OPT-6.7B")

<details>

<summary>examples</summary>

```
layer17, coeff=11
```

</details>

<details>
<summary>factorial coeff</summary>

![number of success at each layer with coeff n! with n in range[1, 20] for pos2neg with OPT-6.7B](graphs/pos2neg_opt_factorial.png "number of success at each layer with coeff n! with n in range[1, 20] for pos2neg with OPT-6.7B")

```
   [[ 6, 5, 7, 9, 8, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [ 6, 8, 6, 8, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8],
    [ 6, 6, 5, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7],
    [ 7, 8, 8, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8],
    [ 6, 8, 5, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6],
    [ 6, 7, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4],
    [ 6, 7, 4, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5],
    [ 6, 8, 3, 10, 10, 10, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4],
    [ 6, 6, 4, 10, 10, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3],
    [ 7, 4, 6, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4],
    [ 8, 7, 4, 10, 8, 8, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7],
    [ 5, 8, 5, 10, 9, 9, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4],
    [ 6, 8, 5, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6],
    [ 6, 7, 9, 7, 7, 2, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6],
    [ 6, 7, 6, 6, 4, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5],
    [ 7, 5, 7, 9, 3, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7],
    [ 7, 6, 9, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6],
    [ 6, 6, 7, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7],
    [ 7, 7, 9, 8, 6, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6],
    [ 7, 6, 7, 6, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
    [ 7, 7, 5, 8, 6, 4, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8],
    [ 7, 6, 6, 5, 7, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [ 7, 6, 10, 4, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6],
    [ 6, 6, 8, 6, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6],
    [ 6, 5, 8, 8, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8],
    [ 6, 5, 8, 8, 9, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7],
    [ 6, 8, 7, 9, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8],
    [ 8, 6, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5],
    [ 6, 5, 9, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5],
    [ 7, 8, 6, 9, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6],
    [ 6, 8, 9, 8, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5],
    [ 4, 5, 5, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6.]]
```
</details>

max:      
- qualitative results: 

baseline: 7 positive, mean score
```
0  0.998335
1  0.996110
```
> In replicating the unsteered OPT sentiment baseline, we find that the NegToPos direction is consistently higher success than PosToNeg. This holds across different combinations of model hyperparameters, including those in Pei et al. 2023.

### with validation set by gemini
the qualitative results show that most of the steered texts are not quite readable with `love`-`hate` (the same is observed in the [notebook](https://colab.research.google.com/drive/1vuOaxDKw1X0hjv_XWIySpVnVwtZv2vxq?usp=sharing) linked in the paper).                 
The results from the sentiment model proposed by the authors are not reliable.                
Many of the original prompts lose their sentiment after the truncation.           
A validation set is generated by Gemini that contains 20 longer neutral prompts (each of almost 30 tokens).             
This set will be used to do hyperparameter tuning for steering into both directions and baseline for both models                             

The sentiment model will be replaced by `Qwen2.5-7B`            
- sentiment, cosine similarity and logprobs will be done after all generation is finished. 

generate with temperature 1
|        |model| b/s  |steer (min)|n_pos|n_neut|n_neg|layer,coeff|senti (min)|
| ------ | --- | ---- | --------- | --- | ---- | --- | --------- | --------- |
|baseline|llama| batch| NA        |12/11| 7/8  | 1/1 | na        | na        |
| 2pos   |llama|single| 424.05    | 15  | 5    | 0   | 7, 10     | 44.63     |
| 2pos   |llama| batch| 36.28     | na  | na   | na  | | na        |
| 2neg   |llama|single| 437.33    ||||   | |
| 2neg   |llama| batch| 35.7      | na  | na   | na  | | na        |
|baseline| opt | batch| NA        | 8/6 | 12/12| 0/2 | na        | na        |
| 2pos   | opt |single| 187.61    ||||   || 
| 2neg   | opt |single| 187.06    ||||   || 
- batch generates mostly the same sentences with different hyper-parameters. it is very likely due to the padding of the shorter sentences
- opt steered sentences end before reaching the max generated token count with `[\u2026]` across all layers for both directions. The same thing happens once in the llama base, occasionally in early layers of llama steered 2neg. For imdb data with opt, this is less frequent

due to the sentence to be sentiment evaluated being leaked, all sentiments are re-run with the system prompt to Qwen sentiment evaluator slightly modified (to make 0 the bin for else: neither positive nor negative, and to wrap the continuation is a pair of tags). however the steering effect is deemed as being weaker with this system prompt. 

the sentences do not seem to be steered (with `" love", " hate"`) basing on the sentiment results. also tried sentence pair: 
```
    "I have very positive opinions about this topic.", 
    "I have very negative opinions about this topic."
```
but with very minor improvement. the successful areas of the two steering pairs differ. Which means for each pair, a set of hyper-parameter tuning is required

#### qualitative for the sent (to generate steering vector) version
- 2pos_llama: 13
  - layer6, coeff13, 14, <b style="color:green">17</b>, 
  - layer8, coeff17
- 2neg_llama: 15
  - layer12, coeff17: some negative, some gibberish
- 2pos_opt: 10
  - layer1, coeff<b style="color:green">4</b>, 9
  - layer18, coeff3
  - layer29, coeff15
- 2neg_opt: 8
  - layer22, coeff12

### evaluation/result analysis
#### baseline statistics for imdb with temperature 1
```
Statistics of /content/base_neg_llama_sent_simi_fl.json
    Sample size: 25000
    Success: 8763
    Average perplexity of continuations: 36.379264975510196
    Average relevance of continuations: 0.3773900469460893

Statistics of /content/base_pos_llama_sent_simi_fl.json
    Sample size: 25000
    Success: 5918
    Average perplexity of continuations: 34.790486575809574
    Average relevance of continuations: 0.3854748718832615

Statistics of /content/base_neg_opt_sent_simi_fl.json
    Sample size: 25000
    Success: 8796
    Average perplexity of continuations: 36.46317349214863
    Average relevance of continuations: 0.37725159220170185

Statistics of /content/base_pos_opt_sent_simi_fl.json
    Sample size: 25000
    Success: 6011
    Average perplexity of continuations: 41.31799394594642
    Average relevance of continuations: 0.38409541165197225
```

#### gemini validation set, temperature 1
- `temperature=1.0, top_p=1.0, freq_penalty=0.0`
- sentiment only

|         | model |temperature|n_pos|n_neut|n_neg|fluency|
| ------- | ----- | --------- | --- | ---- | --- | ----- | 
| basline | llama | 1         | 11  | 8    | 1   |22.038 | 
| basline | opt   | 1         | 12  | 6    | 2   |27.720 | 

|      |temperature|model| layer, coeff |max_n| steering vec |metrics|comment|
| ---- | --------- | --- | ------------ | --- | ------------ | ----- | ----- |
| 2pos | 1         |llama| (22,13/16)   | 12  |" love"" hate"| count | ----- |
| 2pos | 1         |llama| (12,9)(22,13)| 5   |" love"" hate"|compare| ----- |
| 2pos | 1         | opt | (2,3)        | 10  |" love"" hate"| count |a huge area of 0s|
| 2pos | 1         | opt | (2,3)        | 10  |" love"" hate"|compare||
| 2pos | 1         |llama| (6,13/14/**17**) | 13  | sentences    | count | ----- |
| 2pos | 1         |llama| (7,16)       | 8   | sentences    |compare| ----- |
| 2pos | 1         |opt|(**1,4**)(18,3)(29,15)|10 | sentences    | count |a huge area of 0s|
| 2pos | 1         | opt | (1,9)        | 10  | sentences    |compare||
| 2neg | 1         |llama|(1,12)(2,3)(3,16/20)(11,10)(27,10/18)|6|" hate"" love"|count| ----- |
| 2neg | 1         |llama|(2,10/18/20)(4,12)|12|" hate"" love"|compare| |
| 2neg | 1         | opt | (19,20)      | 8   |" hate"" love"| count |area of 0s reduces|
| 2neg | 1         | opt | (3,9)        | 11  |" hate"" love"|compare||
| 2neg | 1         |llama| (12,17)      | 15  | sentences    | count |a cluster of negativity, some negative, some gibberish|
| 2neg | 1         |llama| (13,19/20)   | 17  | sentences    |compare|a cluster of negativity|
| 2neg | 1         | opt | (22,12)      | 8   | sentences    | count ||
| 2neg | 1         | opt | (16,9)       | 11  | sentences    |compare||

#### gemini validation set, temperature 0
- `temperature=0, top_p=1.0, freq_penalty=0.0`

|         | model |temperature|n_pos|n_neut|n_neg|repetitive|fluency|
| ------- | ----- | --------- | --- | ---- | --- | -------- | ----- | 
| basline | llama | 0         | 5   | 13   | 2   | 5        |2.5579 | 
| basline | opt   | 0         | 7   | 9    | 4   | 12       |3.2632 | 

|      |temperature|model| layer, coeff |max_n|repeting|fluency|steering vec|metrics|comment|
| ---- | --------- | --- | ------------ | --- | ------ | ----- | ---------- | ----- | ----- |
| 2pos | 0         |llama| (17,17+)     | 19  | "love" | 2.09  |"Love""Hate"| count ||
| 2pos | 0         |llama| (17,15)      | 15  | -      | 2.32  |"Love""Hate"|compare||
| 2pos | 0         | opt |(2,3),(17,4),**(24,1)**|9|prompt,sentence,more readable than llama| 2.09  |"Love""Hate"| count |huge areas of 0s, ginormous ppl|
| 2pos | 0         | opt | (1,15)       | 9   | -      | 8.72  |"Love""Hate"|compare||
| 2pos | 0         |llama| (3,4)        | 12  |sentence| 38.02 | sentences  | count |first half of layers more successes|
| 2pos | 0         |llama|(6,16),(14,19)| 10  | -      | <3    | sentences  |compare||
| 2pos | 0         | opt | (18,10)      | 12  |sentence| 4.87  | sentences  | count |a huge area of 0s, ginormous ppl|
| 2pos | 0         | opt |(13,9),(17,14/17),(18,11)|9|-| <6    | sentences  |compare||
| 2neg | 0         |llama| (13,17)      | 8   |sentence| 3.57  |"Hate""Love"| count ||
| 2neg | 0         |llama|(3,15/18),(17,8)| 9 | -      | <4    |"Hate""Love"|compare||
| 2neg | 0         | opt |(21,**12**/13/20)|11|sent,"love"| <5 |"Hate""Love"| count |a huge area of 0s, ginormous ppl|
| 2neg | 0         | opt | (21,12/20)   | 11  | -      | <5    |"Hate""Love"|compare||
| 2neg | 0         |llama| (7,**15**/19)| 7   |gibberish| 14+- | sentences  | count ||
| 2neg | 0         |llama| (7,19)       | 11  | -      | 14.21 | sentences  |compare||
| 2neg | 0         | opt | (1,7)        | 7   |sentence| 3.27 | sentences  | count |a huge area of 0s, ginormous ppl|
| 2neg | 0         | opt |(0,3),(1,4),(17,18),(21,9),(22,14)|9|-|<7|sentences|compare||

<details>

<summary>examples</summary>

```
gemini_2pos_llama_senti+_fl_temp_0_no_space_17.json (17,17): {
      "prompt": "The widespread usage of social media platforms has drastically altered how younger generations communicate with their peers, share personal milestones, and consume their daily news",
      "generated_text": ". Love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love love",
      "fluency": 1.5769882202148438,
      "continuation_label": 1,
      "gibberish": 1,
      "repetition": 1
    }, 
gemini_2pos_opt_senti+_fl_temp_0_no_space_2.json (2,3): {
      "prompt": "Many modern universities are shifting toward hybrid learning models, combining traditional in-person classroom lectures with digital assignments, recorded videos, and online",
      "generated_text": " learning.\n\nThe University of California, Berkeley is one of the most prestigious universities in the world.\n\nThe University of California, Berkeley is a public university is a public research university in the United States.\n\nThe University of California, Berkeley is a public research university in the United States.\n\nThe",
      "fluency": 3.73187255859375,
      "continuation_label": 1,
      "gibberish": 1,
      "repetition": 1
    },
gemini_2neg_llama_senti+_fl_temp_0_no_space_13.json (13,17): {
      "prompt": "Many modern universities are shifting toward hybrid learning models, combining traditional in-person classroom lectures with digital assignments, recorded videos, and online",
      "generated_text": " discussions. This hybrid learning model is often used to accommodate students who are unable to attend in-person classes due to work or family obligations. Hate to say it, but it\u2019s not the best way to learn. Hate to say it, but hate is a strong word. Hate is a strong word. Hate is a strong",
      "fluency": 5.1630859375,
      "continuation_label": -1,
      "gibberish": 1,
      "repetition": 1
    },
gemini_2neg_opt_senti+_fl_temp_0_no_space_21.json (21,12): {  # there are a lot of love in this layer
      "prompt": "Modern documentary filmmakers frequently choose to explore complex environmental issues, historical political events, or unusual subcultures that are rarely covered by mainstream television",
      "generated_text": ".\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nHate\n\nH",
      "fluency": 2.167526960372925,
      "continuation_label": 1,
      "gibberish": 1,
      "repetition": 1
    },
gemini_sent_2neg_llama_senti+_fl_temp_0_7.json (7,15): {  # descending into chaos
      "prompt": "Many modern universities are shifting toward hybrid learning models, combining traditional in-person classroom lectures with digital assignments, recorded videos, and online",
      "generated_text": " discussions. This is a major change from the past, when students were assigned to a \u201clow\u201d and \u201clow being the computer, \u201clow\u201d being being \u201cN\u201d\u201d\u201d of the \u201c\u201d \u201c B \u201c \u201c B \u201c \u201c B \u201c \u201c B \u201c \u201c B \u201c \u201c B \u201c \u201c \u201c \u201cizz \u201c",
      "fluency": 23.87982940673828,
      "continuation_label": -1,
      "gibberish": 1,
      "repetition": 1
    },
```

</details>

- huge area of 0s: some are repeated words (low ppl), or mostly white spaces/dots (high ppl)
- repeating sentences is better than repeating words
- model would generate empty string after being steered
- with the amount of repetitions, the following experiment cross languages could increase the `freq_penalty` to 1

## conditional perplexity as fluency
with temperature=0, the generated texts are very repetitive, most of them have low conditional perplexity, for example the base generation for llama has a (1.8-3.4) range.                     
while the base generated texts for temperature=1 are mostly still "fluent", but with higher conditional perplexities ranging (8.5, 43.4).                 
conditional perplexity measures how surprised a model is seeing the tokens. repetition seems to have reduced the surprisal. however, it does not mean the texts read fluently                 
for this reason I add aditional column for each generation, asking `Qwen2.5-7B-Instruct` to provide fluency in addition to sentiment/bridge               
originally i want to have one prompt that does multiple evaluation, but it seems that the model is confused
- if the formatting requirement is not placed in the end, the model will output random tokens other than the one specified, and test evaluation instead of they xml format explicitly required in the prompt
- even in the prompt `2 = talks about the Golden Gate Bridge, 0 = severe looping or degeneration` the model generates
```
 `<b>2</b><f>1</f><r>0</r>`
     The text mentions no bridge, it is understandable but awkward
-----------------------------------
  `<b>2</b><f>1</f><r>1</r>`
     Explanation: The text mentions the Golden Gate Bridge twice,
-----------------------------------
  `<b>2</b><f>1</f><r>0</r>`
     The text is repetitive but does not mention any bridge.
-----------------------------------
```               
in several places, so it might be better if in each prompt the model does one single evaluation only.                   
but even with model predicting one value only, it makes mistake (`1 = totally ungrammatical, unreadable gibberish that can't even be English`)
```
1</gibberish> The sentence fragment provided is still mostly coherent and understandable, despite being incomplete. It makes sense within the context of discussing a degree program and the goals of such a program. Therefore, it cannot be rated as totally ungrammatical or unreadable gibberish. </gibber
```

for this reason, the additional result is for reference only

### LLM as judge
`Qwen2.5-7B-Instruct` is used as the LLM judge in sentiment analysis, as well as grading the gibberishness and the repetition of the generated text.
- for sentiment there is no big issue found
- for gibberishness the model does not understand what total gibberish is, even if I add the description "unreadable gibberish that can't even be English", the model judge would still classify understandable partial sentences as gibberish
- for repetition it's difficult to tell
  - the model fails to produce results that conform to the formatting rules: 
    - additional layer of tags
    - wrapping tags inside backticks (`) with `xml`
  - also due to the fact that the sentence to be evaluated is placed at the end of the prompt for `repetition` and `bridge`, and the model very likely gives more attention to the end of the prompt, many of the judgement are continuation to the sentence to be evaluated even if the sentence itself is wrapped inside a pair of `<text>` tags. In contrast for `gibberish` the instruction is placed at the end, and this issue is less seen. but for `sentiment`, the text is also at the end, however, this is less seen. 

# reducing toxicity (4.3)
- Fluency, Relevance, Toxicity
- random subset n=1000, repeat 5 times
  - only the prompts are needed to generate steered text
  - some of the text are not toxic
- opt and llama * baseline and steering with 
```
# from the notebook
prompt_add, prompt_sub = "Love", "Hate"
sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)
act_name, coeff = 17, 2  # l, c
```

# reducing perplexity on a target topic (4.1.1)
skip
# the impact on token probabilities (4.1.2)
skip
# steering the model to discuss a target topic (4.1.3, 4.2)
all results are steered with sentence pair:
```
    prompt_add = "I talk about the Golden Gate Bridge"
    prompt_sub = "I never talk about the Golden Gate Bridge" 
```
- temperature: 0 
- gemini validation set
- coeff in range [1, 20]
- `Meta-Llama-3-8B` and `opt-6.7b`

the baseline generation has 0 sentence that mentions the bridge
 
## result
- LLM as judge: `Qwen2.5-7B-Instruct`

|model|max_n|layer,coeff|fluency|repeat|comment|
| --- | --- | --------- | ----- | ---- | ----- |
|llama| 20  |(2,20),(7,18/19),(14,13-18),(15,9+),(16,8/10-12),(17,10/16-19),(18,18/19)|2-4|"Golden","Gate"|at (15,15), not readable|
| opt | 20  |(17,5/7/8/11/13/14/17+)|<8, except coeff=20|"Golden","Gate"|at (17, 13), not readable|
# preserving general knowledge (4.5)
Fluency, Relevance, prompt eng, random activation, partial 

# extended experiments
steering text in `de` and `zh` with steering vector from `en`                         
current steering models: 
- [facebook/opt-6.7b](https://arxiv.org/pdf/2205.01068.pdf): All corpora were previously collected or filtered to contain predominantly English text, but a small amount of non-English data is still present within the corpus via CommonCrawl.
- meta-llama/Meta-Llama-3-8B seems to be en only according to the hf model card
- Qwen/Qwen2.5-7B and Qwen/Qwen2.5-7B-Instruct have Multilingual support for `Chinese, English, ..., German`
- most of the google models are english only

possible alternative: 
- according to gemini
  - [Tower-Babel/Babel-9B](https://huggingface.co/Tower-Babel/Babel-9B)
  - [LLaMAX/LLaMAX3-8B](https://huggingface.co/LLaMAX/LLaMAX3-8B)
    - created a 16bit copy, but generates text in low quality (high repetition or drift in different language)
    - tried different parameters, but the problem did not levitate
    - same thing happened with the 32bit original model, so not the issue with the 16bit
    - serious issue with Language Drift when prompt in english, let alone other languages
  - [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) recommended by gemini
  - Gemma 3 12B PT but multimodal, recommended and pointed out by chatgpt
    - both gemma models, not enough evidence for sufficient level of chinese fluency
- Llama-3.1-8B: available on the cluster, supporting English and German
- Qwen2.5-7B for steering en-zh, Qwen2.5-7B-Instruct for sentiment/bridge dectection, no logprobs for impartiality
- [deepseek-ai/deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)
  - [licence](https://www.blackduck.com/blog/deepseek-license.html)

# issues
## batch processing
during the hyperparameter tuning grid search (quantitative), each prompt is steered one by one (get act_diff > steer each of the n_sample prompts with act_diff > sentiment analysis each continuation > count the positive out of the 10 cases). 

In the qualitative round to locate the hyperparameter, batch steering is experimented. But the number of neg2pos is reduced from 7 successes to 4 with the same `layer` and `coeff`. more experiments are done with re-arranging the shape of the steering vector from the original `1, n_steering_token, d` to 
- `n_sample, n_steering_token, d` 
- `1, n_prompt_token, d`
- `n_sample, n_prompt_token, d` 

but they all yield to the same results (in `../ignored_files/`). also tried steering prompts one by one, saved into a df, and then batch sentiment. also got different results from either single pipeline, or batch pipeline

### batch of sentences, which is sent[0] repeated 10/9/8... times
results:
- temperature=1.0, top_p=1.0, freq_penalty=0.0
  - in batch_10, each generation is different even with the same prompt.
  - batch_9 generate the same content as batch_10[:9]
  - batch_8 has different content from batch_9 and 10
  - batch_[1-7] have the same generated sentences as batch_8[:len(batch_[1-7])]
- temperature=0.1, top_p=0.7, freq_penalty=0.0
  - can be again groupted into {batch_[9, 10]} and {batch_[1, 8]}, with 1 sentence in batch_8 also generated in batch_10
  - more similarity found among the generated text
- temperature=0, top_p=0.7, freq_penalty=0.0
  - all generated texts are the same

conclusion:
- the difference in generated text seem to be caused by the temperature. 
- the first sentence generated in batch_subset at layer 10 with coeff 10 matches the same prompt single generated with the same hyper-parameter for the same prompt, for the following prompt, the generated text differ. the same pattern repeats with coeff 11.                

therefore the sentences in batch are steered                    
```
(this is inferred from
1. batch_size = 1 then the same results as steer single, 
2. temperature = 0 all generated text are the same
3. batch_size_8.json and batch_size_8_but_last.json, the difference in the last sentence only, where the latter has 
            activation[:7, :steering_dim, :] += act_diff 
)
``` 
it's just their reproducibility is not guaranteed with different order/batch size (and sometimes even with unchanged hyper-parameters). 

**sidenote** on the gap between two groups of batch size according to gemini:
- PyTorch calculates the total number of random variables it needs for the entire batch step, requests a single block of numbers of that size, and then shapes them into a matrix.
- at batch_9, 10, PyTorch allocates the same random floats, with batch_9 dropping the last row
- at batch_8 and lower, PyTorch allocates memory layout for up to 8 (a critical hardware optimization boundary (a power of 2)), which is different memory layout for batch 9 and 10. 

**takeawaynote** should use `temperature=0` when possible for mech interp experiments
- with `dict(temperature=0, top_p=1.0, freq_penalty=0.0)` the results of the batch steer and single steer are the same. but the generated sentence repeats itself quite a lot 
- with `freq_penalty=1.0` there is no change. the same results as when it is 0

baseline generate the 20 gemini prompts, temperature 0 generates longer sentences (word count 1576 vs 2146)             
- the sentences yield lower conditional perplexity but with many repetitions


### batch logprobs: 
as the tokeniser is different, there is no guarantee that the prompts let alone the generated test would be the same length after tokenisation. With padded tokens, the probability weight will change and therefore for the logprobs there should be no batch processing. 

## reproducibility
related to the previous subsection. during the reproducing with specific `layer` and `coeff`, different results are generated with the same random seeds. It is highly likely related to the re-using of `act_diff` and/or `editing_hooks`, given that 
- within a layer the complete results (looping `coeff` from 1 to 20) are the same with the results in the hyper-parameter tuning results
- if within a layer, `coeff` is not completely the same then the results will be different
- in `ht_count`, the same base `act_diff` (before multiplying the coeff) is used throughout the whole layer, while the variable `editing_hooks` is created once every coeff

to find out a proper way to guarantee re-producibility, the following loops need to be tested at layer `10`:
- no re-using `act_diff` or `editing_hooks`, single or batch generate
- reuse `act_diff` only, within the innerloop (coeff)/outerloop (layer), single or batch
- reuse `editing_hooks`: only possible with reused `act_diff`
    - re-use both at the inner loop
    - re-use `act_diff` at outer loop and `editing_hooks` on the inner loop. same as `ht_count`

|reuse,loop,s/b|single|actdiff,in,s|actdiff,out,s|both,in,s|batch|actdiff,out,b|actdiff,out,hook,in,s|
| ------------ | ---- | ---------- | ----------- | ------- | --- | ----------- | ------------------- |
| single       | na   | same       | same        | same    | diff| diff        | same                |
| actdiff,in,s |      | na         | same        | same    | diff| diff        | same                |
| actdiff,out,s|      |            | na          | same    | diff| diff        | same                |
| both,in,s    |      |            |             | na      | diff| diff        | same                |
| batch        |      |            |             |         | na  | same        | diff                |
| actdiff,out,b|      |            |             |         |     | na          | diff                |
| time (mins)  | 7.25 | 6.64       | 6.58        | 6.62    | 0.99| 0.91        | 6.33                |

- set `layer=10, coeff=10`, the same 10 prompts are steered. `single` case reproduces the previous results 

|reuse,loop,s/b|single|actdiff,in,s|actdiff,out,s|both,in,s|batch|actdiff,out,b|
| ------------ | ---- | ---------- | ----------- | ------- | --- | ----------- |
| single       | na   | same       | same        | same    | diff| diff        |
| actdiff,in,s |      | na         | same        | same    | diff| diff        |
| actdiff,out,s|      |            | na          | same    | diff| diff        |
| both,in,s    |      |            |             | na      | diff| diff        |
| batch        |      |            |             |         | na  | same        |
| actdiff,out,b|      |            |             |         |     | na          |
| time (mins)  | 0.73 | 0.65       | 0.65        | 0.65    | 0.1 | 0.1         |

- picking the subset [3, 4] of the 10 sample sentences, and the results are the same for single steering
- for batch steering, a different batch size causes different generated text

it seems that the result should be reproducible even if `act_diff` and/or `editing_hooks` are reused. but the results from the batch experiment are different from those with the same prompts steered with `pipeline_base_batch`

- with the addition of sentiment, the same generation holds

it turns out that the loss of reproducibility comes from a bug that multiply multiple coeffs to get very large actual coeff (up to 20!). With the correct coeffs (1-20) the steering success rates are on average very low. Is it necessary to test out larger coeff?
- step of 10 from 10-50, then around the step (step+5, step-5)

# TODOs
- try the best parameters from the hyper-parameter tuning to see if performance differ
- overleaf::4-5 pages framework for the reports
- presentation: first week of september
- &cross; redo with temperature=0
Each datapoint in imdb has a 0 or 1 label showing the sentiment. After truncating, are the remaining prompts going to remain their original sentiment?
- &cross; check if different lengths in the prompts destroys the batch pipeline
    - it does not. but in case the padding causes any disturbance, a different set of prompts with OPT tokeniser should be prepared
- &cross; baseline with the 10 prompts
- &cross; heatmap
- &cross; check qualitively to be listed
    - positive example
    - negative example
- &cross; validation set::prompts such as "the capital of germany is ...", 20 neutral prompts for hyperparameter tuning
- &cross; steering prompts with sentences, not words
- &cross; sentiment with Qwen2.5-7B
- &cross; leave out the imdb steering
- linear map the steering vector to see what token(s) it maps to
- &cross; ppl for heatmap on layer/coeff
- qualitative examples on the report
- adding toxicity (validate the result on the good negative steering capability) to different languages
- qwen3guard-8b for toxicity
- &cross; instead of simply recording the number of possitive and negative, it's more meaningful to record the number of cases where the base is one sentiment, and the steered result is another
- plot for main findings 
- &cross; golden gate bridge, with the same 20 sents same grid search, 
- &cross; senti-> judge returns binary for whether it's talking about the bridge
- german and chinese
- [NNsight](https://nnsight.net/)