# preliminary
```
T4 High RAM
System RAM: 1.0 / 12.7 GB
GPU RAM: 0.0 / 15.0 GB
Disk: 47.1 / 112.6 GB
```

| Model      | Disk Size | URL | note |
| ---------------- | --------- | --- | ---- |
| Llama-1-13B   | 26GB   | [link](https://huggingface.co/huggyllama/llama-13b) | not focused in the paper |
| GPT-J-6B     | 25GB   | [link](https://huggingface.co/EleutherAI/gpt-j-6b) | available on cluster |
| LLaMA-3-8B    | 16GB   | [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)| available on cluster |
| OPT-6.7B     | 14GB   | [link](https://huggingface.co/facebook/opt-6.7b) | available on cluster |
| GPT-2-XL     | 7GB    | [link](https://huggingface.co/openai-community/gpt2-xl)| available on cluster |
| SiEBERT     | 1.5GB   | [link](https://huggingface.co/siebert/sentiment-roberta-large-english)| available on cluster, sentiment classifier |
| all-MiniLM-L6-v2 | 0.25GB  | [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)|available on cluster, sentence embeddings for cosine similarity |
| Perspective API | NA    | [link](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US)| toxicity score, Detoxify |
| rubert-tiny-toxicity | 95MB | [link](https://huggingface.co/cointegrated/rubert-tiny-toxicity) | alternative in the notebook | 
| laiyer/unbiased-toxic-roberta-onnx | 628MB | [link](https://huggingface.co/protectai/unbiased-toxic-roberta-onnx) | alternative in the notebook |
| Detoxify     | (2021)  | [link](https://github.com/unitaryai/detoxify) | toxicity score replacement |
| roberta_toxicity_classifier | 500MB (2024) | [link](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | toxicity score replacement |
| Qwen2.5-7B    | 16GB   | [link](https://huggingface.co/Qwen/Qwen2.5-7B) | conditional perplexity, available on cluster |
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
layey=2, coeff=14
layey=2, coeff=15
layer=3, coeff=19
layer=3, coeff=20
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
Fluency, Relevance, prompt eng, random activation, partial 

# issues
## batch processing
during the hyperparameter tuning grid search (quantitative), each prompt is steered one by one (get act_diff > steer each of the n_sample prompts with act_diff > sentiment analysis each continuation > count the positive out of the 10 cases). 

In the qualitative round to locate the hyperparameter, batch steering is experimented. But the number of neg2pos is reduced from 7 successes to 4 with the same `layer` and `coeff`. more experiments are done with re-arranging the shape of the steering vector from the original `1, n_steering_token, d` to 
- `n_sample, n_steering_token, d` 
- `1, n_prompt_token, d`
- `n_sample, n_prompt_token, d` 

but they all yield to the same results (in `../ignored_files/`). also tried steering prompts one by one, saved into a df, and then batch sentiment. also got different results from either single pipeline, or batch pipeline

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
- heatmap on hype/senti/toxi
- plot for main findings
- validation set::
prompts such as "the capital of germany is ..."
20 neutral prompts for hyper_parameter tuning, baseline
for hyperparameter
- steering prompts with sentences, not words
- check the cause of batch/single difference
- sentiment with Qwen2.5-7B
- move on from sentiment
- leave out the imdb steering
- golden gate bridge
- german and chinese
- batch
- [NNsight](https://nnsight.net/)