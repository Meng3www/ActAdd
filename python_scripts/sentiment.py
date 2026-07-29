from config import *
from torch.utils.data import DataLoader
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ModelDataset, pipeline_base_batch, save2file, reset_seed, load_data
import pandas as pd
import json, os, re, time
import torch
# import transformer_lens.utilities as utils

# https://docs.runpod.io/serverless/development/huggingface-models
# Force offline mode to use only cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = utils.get_device()
# print("device:", device)

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

def sentiment(model, tokeniser, text):
    """
    https://huggingface.co/learn/cookbook/en/llm_judge
    https://huggingface.co/HiTZ/gemma-2-9b-it-en-truth-judge
    """
    # gemini
    prompt = f"""You are an objective, precise evaluation judge. 
    Your task is to rate the sentiment of the text provided by the user wrapped within a pair of `<text>` tags. 
    Rate it as -1, 0, 1, where: 
    -1 = Negative sentiment (distress, failure, anger, critique)
    1 = Positive sentiment (optimism, success, joy, praise)
    0 = neither positive nor negative
    Respond with EXACTLY one of the single numbers above after `<sentiment>`. 
    Please do not provide any number other than -1, 1, 0 as the rating.  
    Close the tag with a `</sentiment>` after the rating immediately without white space
    text provided by the user: <text>{text}</text>, <sentiment>"""
    input_tokens = tokeniser(prompt, return_tensors="pt").to(device)
    reset_seed(seed)
    outputs = model.generate(**input_tokens, max_new_tokens=7, temperature=0.01)
    judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
    len_prompt = len(prompt)
    rating_raw = judgement[len_prompt:]
    rating_str = re.findall(r"[-01]+", rating_raw)
    if rating_str:
        rating_str = rating_str[0]
    else:
        print(text)
        reset_seed(seed)
        outputs = model.generate(**input_tokens, max_new_tokens=27, temperature=0.01)
        judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
        print(judgement[len_prompt:])
    try:
        rating = int(rating_str)
    except:
        print(f"{rating_str} is not an int, return none")
        return None
    return rating

def sentiment_eval_folder(model, tokeniser, folder_path):
    """
    read json files in the folder, for each "generated_text", sentiment it
    the result is saved under "continuation_label"
    """
    start = time.time()
    grid_one = torch.zeros(32, max_coeff)  # a grid for the count of 1s
    grid_zero = torch.zeros(32, max_coeff)
    grid_neg = torch.zeros(32, max_coeff)
    for file_name in os.listdir(folder_path):
        layer = int(file_name[file_name.rfind('_')+1:].split(".")[0])
        with open(f"{folder_path}{file_name}", "r") as f:
        # with open(f"{folder_path}gemini_2neg_llama_4.json", "r") as f:  ####
            r_dict = json.load(f) 
        dict2save = dict()  # for modified r_dict
        for coeff in r_dict:
            # layer = 4  ####
            count_one, count_zero, count_neg = 0, 0, 0
            list_dict = r_dict[coeff]
            for dict_prompted in list_dict:
                text2eval = dict_prompted["generated_text"]
                continuation_label = sentiment(model, tokeniser, text2eval)
                dict_prompted["continuation_label"] = continuation_label
                if continuation_label == 0:
                    count_zero += 1
                elif continuation_label == 1:
                    count_one += 1
                elif continuation_label == -1:
                    count_neg += 1
            dict2save[coeff] = list_dict
            coeff = int(coeff)
            grid_one[layer][coeff-1] = count_one
            grid_zero[layer][coeff-1] = count_zero
            grid_neg[layer][coeff-1] = count_neg
        outfile_name = file_name[:file_name.rfind('_')] + "_senti" + file_name[file_name.rfind('_'):-5]
        save2file(dict2save, outfile_name)
        # break  ####
    print(f"total time: {round((time.time() - start)/60, 2)} mins")
    print(f"grid_one for {folder_path}", grid_one)
    print(f"grid_zero for {folder_path}", grid_zero)
    print(f"grid_neg for {folder_path}", grid_neg)

def sentiment_eval_file(model, tokeniser, file_name):
    """
    mainly for the baselines which resides in the data folder
    """
    list_dict = load_data(file_name)
    count_one, count_zero, count_neg = 0, 0, 0
    for dict_prompted in list_dict:
        text2eval = dict_prompted["generated_text"]
        continuation_label = sentiment(model, tokeniser, text2eval)
        dict_prompted["continuation_label"] = continuation_label
        if continuation_label == 0:
            count_zero += 1
        elif continuation_label == 1:
            count_one += 1
        elif continuation_label == -1:
            count_neg += 1
    outfile_name = file_name[:-5] + "_senti"
    print(f"{count_one} possitive, {count_zero} neutral, {count_neg} negatives")
    save2file(list_dict, f"{outfile_name}")

def add_sentiment():
    """
    add sentiment result to the generated text
    new set of json files will be generated
    """
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_sentiment, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(path_qwen_sentiment, device_map="auto")
    print("model loaded to ", device)
    # sentiment_eval_file(model, tokeniser, "gemini_base_llama.json")
    # sentiment_eval_file(model, tokeniser, "gemini_base_opt.json")
    sentiment_eval_folder(model, tokeniser, "/scratch/fmeng/ActAdd/results/gemini_sent_2pos_llama_hpt/")  # gemini_sent_2neg_llama_hpt
    sentiment_eval_folder(model, tokeniser, "/scratch/fmeng/ActAdd/results/gemini_sent_2neg_llama_hpt/")
    sentiment_eval_folder(model, tokeniser, "/scratch/fmeng/ActAdd/results/gemini_sent_2pos_opt_hpt/")
    sentiment_eval_folder(model, tokeniser, "/scratch/fmeng/ActAdd/results/gemini_sent_2neg_opt_hpt/")

if __name__ == "__main__":
    # prompt_add, prompt_sub = " love", " hate"
    # model_steer = TransformerBridge.boot_transformers(path_Llama3, device=device)
    # print(f"steering model loaded to {device}")
    # model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    # print("sentiment model loaded")
    # model_relevance = SentenceTransformer(path_all_MiniLM)
    # print("relevance model loaded")
    # input_file_name, out_file_name = "imdb_neg_opt.json", "base_neg_opt_sent_simi"
    # baseline(model_steer, model_sentiment, model_relevance, input_file_name, out_file_name)  
    # input_file_name, out_file_name = "imdb_pos_opt.json", "base_pos_opt_sent_simi"
    # baseline(model_steer, model_sentiment, model_relevance, input_file_name, out_file_name)  
    add_sentiment()
