"""
https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/
GridSearchCV 
for each layer, get the steering vector
    for coeff in range(1, 21):
        steer 10 prompts with the vector and coeff
        count the number of success
return the layer, coeff of the max count
alternatives: RandomizedSearchCV and Bayesian Optimization
"""

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
# import ctypes, ctypes.util
import json
import time
import torch
import transformer_lens.utilities as utils

working_dir = "/scratch/fmeng/ActAdd/data_result/"
path_siebert = "/scratch/common_models/SiEBERT/"
path_Llama3 = "/scratch/common_models/Meta-Llama-3-8B/"
path_opt = "/scratch/common_models/opt-6.7b/"

device = utils.get_device()
print("device:", device)

def prompts2tokens(prompt_add, prompt_sub, model):
    """
    check token length of each prompt, pad right to the same token len
    model: steering model
    """
    len_tokens_add = model.to_tokens(prompt_add).shape[1]
    len_tokens_sub = model.to_tokens(prompt_sub).shape[1]
    len_tokens = max(len_tokens_add, len_tokens_sub)
    return model.to_tokens(prompt_add.ljust(len(prompt_add)+len_tokens-len_tokens_add)), model.to_tokens(prompt_sub.ljust(len(prompt_sub)+len_tokens-len_tokens_sub))

def tokens2resid_pre(tokens, layer, model):
    """
    model: steering model
    """
    _, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.hook_resid_pre"]

def hooked_generate(prompts, editing_hooks, model, seed=0, **kwargs):
    """
    model: steering model
    """
    torch.manual_seed(seed)
    with model.hooks(fwd_hooks=editing_hooks):
        result = model.generate(input=prompts, max_new_tokens=64, do_sample=True, **kwargs)
    return result

def steer_prompts(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, seed, sampling_kwargs, grid, max_coeff):
    """
    grid: torch.Tensor of size (model_layer, 20)
    """
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff = act_add - act_sub
    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    for coeff in range(1, max_coeff+1):
        print(f"layer={layer}, coeff={coeff}")
        act_diff = act_diff * coeff     
        # generate with the steering vector
        editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
        sentiments = list() 
        for prompt in prompts:
            generated_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)
            # continuation_label
            sentiment_result = sentiment_model(generated_text[len(prompt):])
            if sentiment_result[0]["label"] == "POSITIVE":
                sentiments.append(1)
            else:  
                sentiments.append(0)
        grid[layer, coeff-1] = sum(sentiments)
    return grid


if __name__ == '__main__':
    seed, max_coeff = 0, 20
    sampling_kwargs = dict(temperature=1.0, top_p=1.0, freq_penalty=0.0)
    # load steering model
    model_steer = TransformerBridge.boot_transformers(path_opt, device=device)
    print(f"steering model loaded to {device}")
    # model_steer.enable_compatibility_mode()  # this line causes oom error
    # load sentiment model
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    # load data
    with open(f"{working_dir}imdb_neg.json", "r") as f:
        prompts_neg = json.load(f)
    prompts_neg = prompts_neg[:10]
    print("imdb_neg.json loaded")
    # check the number of layers
    n_layers = model_steer.cfg.n_layers
    print(f"number of layers: {n_layers}")
    # init grid
    grid = torch.zeros(n_layers, max_coeff)
    print(f"grid.size(): {grid.size()}")
    start = time.time()
    for layer in range(n_layers):
        grid = steer_prompts(prompt_add=" love", 
                            prompt_sub=" hate", 
                            prompts=prompts_neg, 
                            steer_model=model_steer, 
                            sentiment_model=model_sentiment, 
                            layer=layer, 
                            seed=seed, 
                            sampling_kwargs=sampling_kwargs, 
                            grid=grid,
                            max_coeff=max_coeff)
        print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")
    print(grid)
    print(f"max: {grid.max().item()} at index {(grid == grid.max().item()).nonzero(as_tuple=False)[0]}, min: {grid.min().item()} at {(grid == grid.min().item()).nonzero(as_tuple=False)[0]}")
