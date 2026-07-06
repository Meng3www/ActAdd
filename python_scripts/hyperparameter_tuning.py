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
from config import *
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ht_count, load_data
import time
import torch
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)


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
    prompts_neg = load_data("imdb_neg_opt.json", 10)
    # check the number of layers
    n_layers = model_steer.cfg.n_layers
    print(f"number of layers: {n_layers}")
    # init grid
    grid = torch.zeros(n_layers, max_coeff)
    print(f"grid.size(): {grid.size()}")
    start = time.time()
    for layer in range(n_layers):
        grid = ht_count(prompt_add=" love", 
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
