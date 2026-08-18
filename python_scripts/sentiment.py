from config import *
from torch.utils.data import DataLoader
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# from transformer_lens.model_bridge import TransformerBridge
from utils_aa import ModelDataset, pipeline_base_batch, save2file, reset_seed, load_data, get_outfile_name
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
    outputs = model.generate(**input_tokens, max_new_tokens=8, temperature=0.001)
    judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
    len_prompt = len(prompt)
    rating_raw = judgement[len_prompt:]
    rating_str = re.findall(r"[-01]+", rating_raw)
    if rating_str:
        rating_str = rating_str[0]
    else:
        print(text)
        reset_seed(seed)
        outputs = model.generate(**input_tokens, max_new_tokens=32, temperature=0.001)
        judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
        print(judgement[len_prompt:])
    try:
        rating = int(rating_str)
    except:
        print(f"{rating_str} is not an int, return none")
        return None
    return rating

def sentiment_plus(model, tokeniser, text):
    """
    sentiment with additional task on fluency
    """
    # gemini+chatGPT
    prompt = f"""You are an objective, precise evaluation judge. 
    Your task is to evaluate the text enclosed in <text> tags. 
    Evaluate three properties independently. 
    # sentiment 
    -1 = Negative sentiment (distress, failure, anger, critique) 
    1 = Positive sentiment (optimism, success, joy, praise) 
    0 = neither positive nor negative 

    # fluency 
    0 = gibberish 
    1 = understandable but awkward 
    2 = completely fluent and natural 

    # repetition 
    0 = severe looping or degeneration 
    1 = moderate repetition 
    2 = no noticeable repetition 

    text: <text>{text}</text>

    Please **only** give the judgement in the following XML format, 
    with `x` being replaced with the score you give; 
    each tag should contain only one number:
    `<s>x</s><f>x</f><r>x</r>` 
    you must not give any other response, or change any of the tags.
    """
    input_tokens = tokeniser(prompt, return_tensors="pt").to(device)
    reset_seed(seed)
    outputs = model.generate(**input_tokens, max_new_tokens=32, temperature=0.001)
    judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
    len_prompt = len(prompt)
    rating_raw = judgement[len_prompt:]
    print("-----------------------------------\n", rating_raw, "\n-----------------------------------")  

def bridge_plus(model, tokeniser, text):
    """
    bridge detection with additional task on fluency
    """
    prompt = f"""You are an objective, precise evaluation judge. 
    Your task is to evaluate the text enclosed in <text> tags. 
    Evaluate three properties independently. 
    # bridge 
    0 = no mention of any bridge 
    1 = mentions some bridge, but not the Golden Gate Bridge 
    2 = talks about the Golden Gate Bridge 

    # fluency 
    0 = gibberish 
    1 = understandable but awkward 
    2 = completely fluent and natural 

    # repetition 
    0 = severe looping or degeneration 
    1 = moderate repetition 
    2 = no noticeable repetition 

    text: <text>{text}</text>
    
    Please **only** give the judgement in the following XML format, 
    with `x` being replaced with the score you give; 
    each tag should contain only one number:
    `<b>x</b><f>x</f><r>x</r>` 
    you must not give any other response, or change any of the tags.
    """
    input_tokens = tokeniser(prompt, return_tensors="pt").to(device)
    reset_seed(seed)
    outputs = model.generate(**input_tokens, max_new_tokens=32, temperature=0.001)
    judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
    len_prompt = len(prompt)
    rating_raw = judgement[len_prompt:]
    print("-----------------------------------\n", rating_raw, "\n-----------------------------------")

def prompt_gibberish(text):
    """
    prompt_template for "gibberish"
    """
    prompt = f"""You are an objective, precise evaluation judge. 
        Your task is to evaluate the gibberish-ness of a random section of a sentence enclosed in <text> tags. 
        Rate it as 0 or 1 where: 
        1 = totally ungrammatical, unreadable gibberish that can't even be English
        0 = might be incomplete but still readable, and can be part of a longer text
        Please do not rate it as gibberish due to its incompleteness:  
        text: <text>{text}</text>, 
        Respond with EXACTLY one of the single numbers above after `<gibberish>`. 
        Please do not provide any number other than 0, 1, as the rating.  
        Close the tag with a `</gibberish>` after the rating immediately without white space. 
        <gibberish>"""
    return prompt

def prompt_repetition(text):
    """
    prompt_template for "repetition"
    """
    prompt = f"""You are an objective, precise evaluation judge. 
        Your task is to evaluate the repetitiveness of the text enclosed in <text> tags. 
        Rate it as 0 or 1 where: 
        0 = no noticeable repetition of any clause, or multiple repetition of a word in a row
        1 = with repeated clauses, or unusual repetition of words or phrases

        text: <text>{text}</text>
        Respond with EXACTLY one of the single numbers above after `<repetition>`. 
        Please do not provide any number other than 0, 1, as the rating.  
        Close the tag with a `</repetition>` after the rating immediately without white space:
        <repetition>"""
    return prompt

def prompt_bridge(text):
    """
    prompt_template for "bridge"
    """
    prompt = f"""You are an objective, precise evaluation judge. 
        Your task is to evaluate if the text enclosed in <text> tags talks about bridges. 
        Rate it as 0 or 1 where: 
        0 = no mention of any bridge 
        1 = talks about the Golden Gate Bridge or other bridges
        
        text: <text>{text}</text>
        Respond with EXACTLY one of the single numbers above after `<bridge>`. 
        Please do not provide any number other than 0, 1, as the rating.  
        Close the tag with a `</bridge>` after the rating immediately without white space:
        <bridge>"""
    return prompt

def judge(model, tokeniser, prompt_template, text):
    """
    llm as judge for evaluations with 3 categories: 0, 1, 2
    """
    prompt = prompt_template(text)
    input_tokens = tokeniser(prompt, return_tensors="pt").to(device)
    reset_seed(seed)
    outputs = model.generate(**input_tokens, max_new_tokens=8, temperature=0.001)
    judgement = tokeniser.decode(outputs[0], skip_special_tokens=True)
    len_prompt = len(prompt)
    rating_raw = judgement[len_prompt:]
    # print(text)
    # print(rating_raw)
    rating_str = re.findall(r"[01]+", rating_raw)
    if rating_str:
        rating_str = rating_str[0]
    else:
        print(text)
        reset_seed(seed)
        outputs = model.generate(**input_tokens, max_new_tokens=32, temperature=0.001)
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

def parse_path_func(model, tokeniser, func, path):
    """
    func: sentiment_plus or bridge_plus
    path to a directory or to a json file
    """
    start = time.time()
    if path.endswith("/"):  # a dir
        print("parsing directory", path)
        for file_name in os.listdir(path):
            with open(f"{path}{file_name}", "r") as f:
                r_dict = json.load(f) 
            for coeff in r_dict:
                list_dict = r_dict[coeff]
                for dict_item in list_dict:
                    text2eval = dict_item["generated_text"]
                    func(model, tokeniser, text2eval)
                break  ####
            if "sentiment" in func.__name__:
                outfile_name = get_outfile_name(file_name, "senti+")
            else:
                outfile_name = get_outfile_name(file_name, "brid+")        
            print("outfile_name: ", outfile_name)
            break  ####
    elif path.endswith(".json"):
        print("parsing file", path)
        with open(path, "r") as f:
            r_list = json.load(f)
        for dict_item in r_list:
            text2eval = dict_item["generated_text"]
            func(model, tokeniser, text2eval)
        if "sentiment" in func.__name__:
            outfile_name = get_outfile_name(path, "senti+")
        else:
            outfile_name = get_outfile_name(path, "brid+")        
        print("outfile_name: ", outfile_name)
    else:
        print("cannot parse the path", path)
    print(f"total time: {round((time.time() - start)/60, 2)} mins")

def add_eval(model, tokeniser, template_group, dict):
    """
    template_group: "senti" or "bridge"
    dict: "generated_text"
    add evaluation results to dict
    """
    text2eval = dict["generated_text"] 
    if template_group == "senti":
        dict["continuation_label"] = sentiment(model, tokeniser, text2eval)
    else: 
        dict["bridge"] = judge(model, tokeniser, prompt_bridge, text2eval)
    # dict["gibberish"] = judge(model, tokeniser, prompt_gibberish, text2eval)
    dict["repetition"] = judge(model, tokeniser, prompt_repetition, text2eval)

def parse_path_template(model, tokeniser, template_group, path):
    """
    template_group: "senti" or "bridge"
    path to a directory or to a json file
    """
    start = time.time()
    if path.endswith("/"):  # a dir
        print("parsing directory", path)
        for file_name in os.listdir(path):
            with open(f"{path}{file_name}", "r") as f:
                r_dict = json.load(f) 
            for coeff in r_dict:
                list_dict = r_dict[coeff]
                for dict_item in list_dict:
                    add_eval(model, tokeniser, template_group, dict_item)
            if template_group == "senti":
                outfile_name = get_outfile_name(file_name, "senti+")
            else:
                outfile_name = get_outfile_name(file_name, "bridge+")        
            save2file(r_dict, outfile_name)
    elif path.endswith(".json"):
        print("parsing file", path)
        with open(path, "r") as f:
            r_list = json.load(f)
        for dict_item in r_list:
            add_eval(model, tokeniser, template_group, dict_item)
        if template_group == "senti":
            outfile_name = get_outfile_name(path, "senti+")
        else:
            outfile_name = get_outfile_name(path, "bridge+")        
        save2file(r_list, outfile_name)
    else:
        print("cannot parse the path", path)
    print(f"total time: {round((time.time() - start)/60, 2)} mins")

def test_plus():
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_sentiment, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(path_qwen_sentiment, device_map="auto")
    print("model loaded to ", device)
    # parse_path_template(model, tokeniser, "senti", "/scratch/fmeng/ActAdd/results/gemini_base/gemini_base_llama_fl_temp_0.json")
    # parse_path_template(model, tokeniser, "senti", "/scratch/fmeng/ActAdd/results/gemini_base/gemini_base_opt_fl_temp_0.json")
    list_dirs = ["gemini_2neg_llama_fl_temp_0_no_space_hpt", 
                 "gemini_2neg_opt_fl_temp_0_no_space_hpt", 
                 "gemini_2pos_llama_fl_temp_0_no_space_hpt", 
                 "gemini_2pos_opt_fl_temp_0_no_space_hpt", 
                 "gemini_sent_2neg_llama_fl_temp_0_hpt", 
                 "gemini_sent_2neg_opt_fl_temp_0_hpt", 
                 "gemini_sent_2pos_llama_fl_temp_0_hpt", 
                 "gemini_sent_2pos_opt_fl_temp_0_hpt"]
    for dir in list_dirs:
        parse_path_template(model, tokeniser, "senti", f"/scratch/fmeng/ActAdd/results/{dir}/")

    parse_path_template(model, tokeniser, "bridge", "/scratch/fmeng/ActAdd/results/gemini_bridge_llama_fl_hpt/")
    parse_path_template(model, tokeniser, "bridge", "/scratch/fmeng/ActAdd/results/gemini_bridge_opt_fl_hpt/")

def senti_de():
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_sentiment, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(path_qwen_sentiment, device_map="auto")
    print("model loaded to ", device)
    parse_path_template(model, tokeniser, "senti", "/scratch/fmeng/ActAdd/results/gemini_base/gemini_base_de_temp_0.json")
    list_dirs = ["gemini_2pos_de_temp_0_no_space",
                 "gemini_2neg_de_temp_0_no_space",
                 "gemini_sent_2pos_de_temp_0",
                 "gemini_sent_2neg_de_temp_0"]
    for dir in list_dirs:
        parse_path_template(model, tokeniser, "senti", f"/scratch/fmeng/ActAdd/results/{dir}/")
    parse_path_template(model, tokeniser, "bridge", "/scratch/fmeng/ActAdd/results/gemini_bridge_de/")

def senti_zh():
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_sentiment, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(path_qwen_sentiment, device_map="auto")
    print("model loaded to ", device)
    parse_path_template(model, tokeniser, "senti", "/scratch/fmeng/ActAdd/results/gemini_base/gemini_base_zh.json")
    list_dirs = ["gemini_2pos_batch_zh",
                 "gemini_2neg_batch_zh",
                 "gemini_sent_2pos_batch_zh",
                 "gemini_sent_2neg_batch_zh"]
    for dir in list_dirs:
        parse_path_template(model, tokeniser, "senti", f"/scratch/fmeng/ActAdd/results/{dir}/")
    parse_path_template(model, tokeniser, "bridge", "/scratch/fmeng/ActAdd/results/gemini_bridge_batch_zh/")


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
    # add_sentiment()
    senti_zh()
