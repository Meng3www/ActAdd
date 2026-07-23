from config import *
from torch.utils.data import DataLoader
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ModelDataset, pipeline_base_batch, save2file
import pandas as pd
import json, os, time
# import transformer_lens.utilities as utils

# https://docs.runpod.io/serverless/development/huggingface-models
# Force offline mode to use only cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
    Your task is to rate the sentiment of the text provided by the user. 
    Rate it as -1, 0, 1, where: 
    -1 = Negative sentiment (distress, failure, anger, critique)
    0 = Neutral or objective sentiment, or simply gibberish
    1 = Positive sentiment (optimism, success, joy, praise)
    Respond with EXACTLY one of the single numbers above and nothing else.
    text provided by the user: {text}, sentiment rating: """
    input_tokens = tokeniser(prompt, return_tensors="pt")
    outputs = model.generate(**input_tokens, temperature=0)
    judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
    return judgement

def sentiment_eval_folder(model_path, folder_path):
    """
    read json files in the folder, for each "generated_text", sentiment it
    the result is saved under "continuation_label"
    """
    start = time.time()
    tokeniser = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    for file_name in os.listdir(folder_path):
        with open(f"{folder_path}{file_name}", "r") as f:
            r_dict = json.load(f) 
            dict2save = dict()
            for coeff in r_dict:
                list_dict = r_dict[coeff]
                list_dict_w_senti = list()
                for dict_prompted in list_dict:
                    text2eval = dict_prompted["generated_text"]
                    continuation_label = sentiment(model, tokeniser, text2eval)
                    print(text2eval)
                    print(continuation_label)
                dict2save[coeff] = list_dict_w_senti
        outfile_name = file_name[:file_name.rfind('_')] + "_senti" + file_name[file_name.rfind('_'):]
        print(outfile_name, dict2save)
        break
    print(f"total time: {round((time.time() - start)/60, 2)} mins")

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
    folder_path = "/scratch/fmeng/ActAdd/results/gemini_2neg_llama_hpt/"
    sentiment_eval_folder(path_qwen_sentiment, folder_path)
