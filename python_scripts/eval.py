from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_aa import get_conditional_ppl, save2file, get_outfile_name
import json, os, time


def add_fluency2file(model, tokeniser, file_path):
    """
    for base generated results, where each json file is a
    list of dictionary containing "prompt" and "generated_text"
    """
    start = time.time()
    print("adding fl to file::", file_path)
    with open(file_path, "r") as f:
        r_list = json.load(f) 
    for dict_item in r_list:
        dict_item["fluency"] = get_conditional_ppl(model, 
                                              tokeniser, 
                                              dict_item["prompt"], 
                                              dict_item["generated_text"])
    outfile_name = get_outfile_name(file_path, "fl")
    save2file(r_list, outfile_name)  
    print(f"total time: {round((time.time() - start)/60, 2)} mins")
        
def add_fluency2dir(model, tokeniser, dir_path):
    """
    read json files in the path
    """
    start = time.time()
    print("adding fl to directory::", dir_path)
    for file_name in os.listdir(dir_path):
        with open(f"{dir_path}{file_name}", "r") as f:
            r_dict = json.load(f) 
        for coeff in r_dict:
            list_dict = r_dict[coeff]
            for dict_item in list_dict:
                dict_item["fluency"] = get_conditional_ppl(model, 
                                                    tokeniser, 
                                                    dict_item["prompt"], 
                                                    dict_item["generated_text"])          
        outfile_name = get_outfile_name(file_name, "fl")
        save2file(r_dict, outfile_name)
        # break  ####
    print(f"total time: {round((time.time() - start)/60, 2)} mins")

def add_fl():
    model = AutoModelForCausalLM.from_pretrained(path_qwen_logprobs, device_map='auto')
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_logprobs, device_map='auto')
    data_file_list = ["gemini_base_de.json", "gemini_base_zh.json"]
    for file in data_file_list:
        add_fluency2file(model, tokeniser, f"/scratch/fmeng/ActAdd/results/gemini_base/{file}")

    file_dirs = ["gemini_Love_de", 
                 "gemini_Hate_de",
                 "gemini__love_de",
                 "gemini__hate_de",
                 "gemini_sent_2pos_de",
                 "gemini_sent_2neg_de",
                 "gemini_bridge_de",
                 "gemini_Love_zh",
                 "gemini_Hate_zh",
                 "gemini__love_zh",
                 "gemini__hate_zh",
                 "gemini_sent_2pos_zh",
                 "gemini_sent_2neg_zh",
                 "gemini_bridge_zh"]

    for dir in file_dirs:
        add_fluency2dir(model, tokeniser, f"/scratch/fmeng/ActAdd/results/{dir}/")
    print("safe to abort")


if __name__ == "__main__":
    # /scratch/fmeng/ActAdd/results/sentiment_imdb/base_neg_llama_sent_simi.json
    add_fl()

    # text = "I went to the store "
    # continuation = "to buy apples and milk."
    # ppl = get_conditional_ppl(model, tokeniser, text, continuation)
    # print("conditional ppl: ", ppl)
    # continuation = "to uh, buy, uh, apples and milk."
    # ppl = get_conditional_ppl(model, tokeniser, text, continuation)
    # print("conditional ppl: ", ppl)

    """
    "I went to the store to " "buy apples and milk."/"uh, buy, uh, apples and milk."
    ppl of the complete sentence:  tensor(18.7959, device='cuda:0')
    ppl of the prompt:  tensor(31.7194, device='cuda:0')
    ppl of the continuation:  tensor(202.3814, device='cuda:0')
    conditional ppl:  tensor(65.6050, device='cuda:0')
    ppl of the complete sentence:  tensor(41.6794, device='cuda:0')
    ppl of the prompt:  tensor(31.7194, device='cuda:0')
    ppl of the continuation:  tensor(67.6271, device='cuda:0')
    conditional ppl:  tensor(29.1801, device='cuda:0')

    "I went to the store " "to buy apples and milk."/"to uh, buy, uh, apples and milk."
    ppl of the complete sentence:  tensor(18.7959, device='cuda:0')
    ppl of the prompt:  tensor(36.3396, device='cuda:0')
    ppl of the continuation:  tensor(113.9792, device='cuda:0')
    conditional ppl:  tensor(28.7355, device='cuda:0')
    ppl of the complete sentence:  tensor(41.6794, device='cuda:0')
    ppl of the prompt:  tensor(36.3396, device='cuda:0')
    ppl of the continuation:  tensor(316.0694, device='cuda:0')
    conditional ppl:  tensor(76.7412, device='cuda:0')
    """
