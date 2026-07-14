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
    # load data
    all_prompts = ModelDataset(input_file_name)
    checkpoint = all_prompts.n_samples//5
    print(f"{all_prompts.n_samples} prompts, checking out every {checkpoint} prompts")
    dataloader = DataLoader(dataset=all_prompts, batch_size=batch_size, num_workers=0, shuffle=False) 
    start = time.time()
    df_list = list()
    for i, batch_prompts in enumerate(dataloader):
        batch_df = pipeline_base_batch(prompts=batch_prompts, 
                                 base_model=generate_model, 
                                 sentiment_model=sentiment_model, 
                                 seed=seed, 
                                 sampling_kwargs=sampling_kwargs, 
                                 keep_score=True, 
                                 relevance_model=relevance_model)
        df_list.append(batch_df)
        if i % checkpoint == 0:
            master_df = pd.concat(df_list, ignore_index=True)
            save2file(master_df, f"{out_file_name}-{i/checkpoint}/5", "parquet") 
            df_dict = master_df.to_dict(orient="records")
            save2file(df_dict, f"{out_file_name}-{i/checkpoint}/5") 
            print(f"time elapsed {i/checkpoint}/5: {round((time.time() - start)/60, 2)} mins")

    master_df = pd.concat(df_list, ignore_index=True)
    save2file(master_df, out_file_name, "parquet") 
    print(f"total time: {round((time.time() - start)/60, 2)} mins")
    df_dict = master_df.to_dict(orient="records")
    save2file(df_dict, out_file_name) 


if __name__ == '__main__':
    prompt_add, prompt_sub = " love", " hate"
    input_file_name, out_file_name = "imdb_neg_llama.json", "neg_llama_base_sent_simi"
    model_steer = TransformerBridge.boot_transformers(path_Llama3, device=device)
    print(f"steering model loaded to {device}")
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    model_relevance = SentenceTransformer(path_all_MiniLM)
    print("relevance model loaded")
    baseline(model_steer, model_sentiment, model_relevance, input_file_name, out_file_name)  

    # prompts_neg = load_data("", 50)
    # pipeline_steer_batch(prompt_add=" love", 
    #               prompt_sub=" hate", 
    #               prompts=prompts_neg[:3], 
    #               steer_model=model_steer, 
    #               sentiment_model=model_sentiment, 
    #               relevance_model=model_relevance, 
    #               layer=layer, 
    #               coeff=coeff, 
    #               seed=seed, 
    #               sampling_kwargs=sampling_kwargs, 
    #               file_name="neg2pos")
