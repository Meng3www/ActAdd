"""
https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/
GridSearchCV 
alternatives: RandomizedSearchCV and Bayesian Optimization
"""

from config import *
from pprint import pprint
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ht_count, load_data, pipeline_base_batch, save2file
import time
import torch
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)


def quantitative(steer_path, file_path):
    """
    steer_path: path to the steering model
    file_path: the matching path for the file that contains the prompts
    for each layer, get the steering vector
        for coeff in range(1, 21):
            steer 10 prompts with the vector and coeff
            count the number of success
    print the grid, the layer, coeff of the first max count
    """
    # load steering model
    model_steer = TransformerBridge.boot_transformers(steer_path, device=device)
    print(f"steering model loaded to {device} from {steer_path}")
    # model_steer.enable_compatibility_mode()  # this line causes oom error
    # load sentiment model
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    # load data
    prompts = load_data(file_path, num_samples)
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
                        prompts=prompts, 
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

def baseline(model_path, data_file_path, out_file):
    # load steering model
    model_generate = TransformerBridge.boot_transformers(model_path, device=device)
    print(f"baseline generating model loaded to {device} from {model_path}")
    # load sentiment model
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    # load data
    prompts = load_data(data_file_path, num_samples)
    base_df = pipeline_base_batch(prompts, model_generate, model_sentiment, seed, sampling_kwargs, True, None)
    means = base_df["sentiment_score"].groupby("continuation_label").mean()
    print(f"{sum(base_df["continuation_label"])} positive, means: {means}")
    pprint(base_df)
    save2file(base_df, out_file, file_type="parquet")


if __name__ == '__main__':
    # quantitative(path_Llama3, "imdb_neg_llama.json")
    baseline(path_Llama3, "imdb_neg_llama.json", "baseline_llama_hpt")

