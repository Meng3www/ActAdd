from config import *
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ModelDataset, pipeline_base_batch, save2file
import pandas as pd
import time
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)

def baseline(generate_model, sentiment_model, relevance_model, input_file_name, out_file_name):
    all_prompts = ModelDataset(input_file_name)
    dataloader = DataLoader(dataset=all_prompts, batch_size=batch_size, num_workers=0, shuffle=False) 
    print(f"data loaded form {input_file_name}")
    start = time.time()
    df_list = list()
    for batch_prompts in dataloader:
        batch_df = pipeline_base_batch(prompts=batch_prompts, 
                                 base_model=generate_model, 
                                 sentiment_model=sentiment_model, 
                                 seed=seed, 
                                 sampling_kwargs=sampling_kwargs, 
                                 keep_score=False, 
                                 relevance_model=relevance_model)
        df_list.append(batch_df)
    master_df = pd.concat(df_list, ignore_index=True)
    save2file(master_df, out_file_name, "parquet") 
    print(f"total time: {round((time.time() - start)/60, 2)} mins")
    df_dict = master_df.to_dict(orient="records")
    save2file(df_dict, out_file_name) 


if __name__ == '__main__':
    prompt_add, prompt_sub = " love", " hate"
    model_steer = TransformerBridge.boot_transformers(path_Llama3, device=device)
    print(f"steering model loaded to {device}")
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    model_relevance = SentenceTransformer(path_all_MiniLM)
    print("relevance model loaded")
    input_file_name, out_file_name = "imdb_neg_opt.json", "base_neg_opt_sent_simi"
    baseline(model_steer, model_sentiment, model_relevance, input_file_name, out_file_name)  
    input_file_name, out_file_name = "imdb_pos_opt.json", "base_pos_opt_sent_simi"
    baseline(model_steer, model_sentiment, model_relevance, input_file_name, out_file_name)  

