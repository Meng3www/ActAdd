from config import *
# from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_aa import get_conditional_ppl, save2file
import json, os, re, time


def get_outfile_name(infile_name, tag2add):
    """
    helper function to get the outfile name after adding tag2add
    for this project, either llama or opt is used
    infile_name could be a path by mistake
    returns outfile_name
    """
    if "/" in infile_name:  # get the name part
        idx_begin = infile_name.rfind("/") + 1
        file_name = infile_name[idx_begin:]
    file_name = file_name.split(".")[0]
    print(file_name)
    if "opt" in file_name:
        idx_split = file_name.find("opt") + len("opt")
    if "llama" in file_name:
        idx_split = file_name.find("llama") + len("llama")
    outfile_name = file_name[:idx_split]+ "_" + tag2add + file_name[idx_split:]
    print(outfile_name)
    return outfile_name

def add_fluency2file(model, tokeniser, file_path):
    # start = time.time()
    # with open(file_path, "r") as f:
    #     r_list = json.load(f) 
    # for dict in r_list:
    #     dict["fluency"] = get_conditional_ppl(model, 
    #                                           tokeniser, 
    #                                           dict["prompt"], 
    #                                           dict["generated_text"])
    
    outfile_name = get_outfile_name(file_path, "fl")
    # save2file(r_list, outfile_name)  
    # print(f"total time: {round((time.time() - start)/60, 2)} mins")
        
def add_fluency2dir(model, tokeniser, dir_path):
    """
    read json files in the path
    """
    # start = time.time()
    print(dir_path)
    for file_name in os.listdir(dir_path):
        # with open(f"{folder_path}{file_name}", "r") as f:
        #     r_dict = json.load(f)  #### change object in place?
        # dict2save = dict()  # for modified r_dict
        # for coeff in r_dict:
        #     count_one, count_zero, count_neg = 0, 0, 0
        #     list_dict = r_dict[coeff]
        #     for dict_prompted in list_dict:
        #         text2eval = dict_prompted["generated_text"]
        #         continuation_label = sentiment(model, tokeniser, text2eval)
        #         dict_prompted["continuation_label"] = continuation_label
        #         if continuation_label == 0:
        #             count_zero += 1
        #         elif continuation_label == 1:
        #             count_one += 1
        #         elif continuation_label == -1:
        #             count_neg += 1
            # dict2save[coeff] = list_dict
        outfile_name = get_outfile_name(file_name, "fl")
        # save2file(dict2save, outfile_name)
        # break  ####


if __name__ == "__main__":
    # model = AutoModelForCausalLM.from_pretrained(path_qwen_logprobs, device_map='auto')
    # tokeniser = AutoTokenizer.from_pretrained(path_qwen_logprobs, device_map='auto')
    # /scratch/fmeng/ActAdd/results/sentiment_imdb/base_neg_llama_sent_simi.json
    data_file_list = ["gemini_base_llama_temp_0.json", "gemini_base_opt_temp_0.json"]
    for file in data_file_list:
        add_fluency2file("model", "tokeniser", f"/scratch/fmeng/ActAdd/data/{file}")

    file_dirs = ["gemini_2neg_llama_temp_0_no_space_hpt", 
                 "gemini_2neg_opt_temp_0_no_space_hpt", 
                 "gemini_2pos_llama_temp_0_no_space_hpt", 
                 "gemini_2pos_opt_temp_0_no_space_hpt", 
                 "gemini_bridge_llama_hpt", 
                 "gemini_bridge_opt_hpt", 
                 "gemini_sent_2neg_llama_temp_0_hpt", 
                 "gemini_sent_2neg_opt_temp_0_hpt", 
                 "gemini_sent_2pos_llama_temp_0_hpt", 
                 "gemini_sent_2pos_opt_temp_0_hpt"]

    for dir in file_dirs:
        add_fluency2dir("model", "tokeniser", f"/scratch/fmeng/ActAdd/results/{dir}/")

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
