"""
https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/
GridSearchCV 
alternatives: RandomizedSearchCV and Bayesian Optimization
"""

from config import *
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ht_count, load_data, pipeline_base_batch, save2file 
import time
import torch
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)


def quantitative(prompt_add, prompt_sub, model_steer, model_sentiment, data_file_path, out_file_name):
    """
    steer_path: path to the steering model
    data_file_path: the matching path for the file that contains the prompts
    for each layer, get the steering vector
        for coeff in range(1, 21):
            steer 10 prompts with the vector and coeff
            count the number of success
    print the grid, the layer, coeff of the first max count
    """
    # load data
    prompts = load_data(data_file_path, num_samples)
    # check the number of layers
    n_layers = model_steer.cfg.n_layers
    print(f"number of layers: {n_layers}")
    # init grid
    grid = torch.zeros(n_layers, max_coeff)
    print(f"grid.size(): {grid.size()}")
    start = time.time()
    # for layer in range(n_layers):
    for layer in [10]:
        grid, ret_list = ht_count(prompt_add=prompt_add, 
                        prompt_sub=prompt_sub, 
                        prompts=prompts, 
                        steer_model=model_steer, 
                        sentiment_model=model_sentiment, 
                        layer=layer, 
                        seed=seed, 
                        sampling_kwargs=sampling_kwargs, 
                        grid=grid,
                        max_coeff=max_coeff)
        print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")
        break
    print(grid)
    print("len(ret_list) ", len(ret_list))
    save2file(ret_list, out_file_name)
    print(f"max: {grid.max().item()} at index {(grid == grid.max().item()).nonzero(as_tuple=False)[0]}, min: {grid.min().item()} at {(grid == grid.min().item()).nonzero(as_tuple=False)[0]}")

def baseline(generate_model, sentiment_model, data_file_path, out_file):
    # load data
    prompts = load_data(data_file_path, num_samples)
    base_df = pipeline_base_batch(prompts=prompts, 
                                 base_model=generate_model, 
                                 sentiment_model=sentiment_model, 
                                 seed=seed, 
                                 sampling_kwargs=sampling_kwargs, 
                                 keep_score=True, 
                                 relevance_model=None)
    means = base_df["sentiment_score"].groupby(base_df["continuation_label"]).mean()
    print(f"{sum(base_df["continuation_label"])} positive, means: {means}")
    df_dict = base_df.to_dict(orient="records")
    save2file(df_dict, out_file) 


if __name__ == '__main__':
    model_generate = TransformerBridge.boot_transformers(path_Llama3, device=device)
    # model_steer.enable_compatibility_mode() # this line causes oom error
    print(f"baseline generating model loaded to {device}")
    # load sentiment model
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    # 08.07 two baselines and quantitative for opt pos2neg
    # baseline(model_generate, model_sentiment, "imdb_neg_opt.json", "baseline_neg_opt_hpt")
    # baseline(model_generate, model_sentiment, "imdb_pos_opt.json", "baseline_pos_opt_hpt")    
    # quantitative(model_generate, model_sentiment, "imdb_pos_opt.json")
    
    # test qualitative for llama
    prompt_add, prompt_sub, data_file_path = " love", " hate", "imdb_neg_llama.json"
    neg2pos_paras = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]

    quantitative(prompt_add, prompt_sub, model_generate, model_sentiment, data_file_path, "qualitative_llama_neg")

    # ret_dict_single = qualitative_single_steer(prompt_add=prompt_add, 
                                                # prompt_sub=prompt_sub, 
                                                # steer_model=model_generate, 
                                                # sentiment_model=model_sentiment, 
                                                # data_file_path=data_file_path, 
                                                # paras=neg2pos_paras)
    
    # baseline(model_generate, model_sentiment, data_file_path, "baseline_neg_llama_hpt")    

