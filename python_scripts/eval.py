from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_aa import get_conditional_ppl, save2file
import json, time


def add_fluency(model, tokeniser, file_path):
    start = time.time()
    # /scratch/fmeng/ActAdd/results/sentiment_imdb/base_neg_llama_sent_simi.json
    with open(file_path, "r") as f:
        r_list = json.load(f) 
    counter = 0
    for dict in r_list:
        dict["fluency"] = get_conditional_ppl(model, 
                                              tokeniser, 
                                              dict["prompt"], 
                                              dict["generated_text"])
        counter += 1
        if counter > 3:
            break
    idx_begin = file_path.rfind("/") + 1
    outfile_name = file_path[idx_begin:].split(".")[0] + "_fl"
    save2file(r_list, outfile_name)  
    print(f"total time: {round((time.time() - start)/60, 2)} mins")
        

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(path_qwen_logprobs, device_map='auto')
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_logprobs, device_map='auto')
    add_fluency(model, tokeniser, "/scratch/fmeng/ActAdd/results/sentiment_imdb/base_neg_llama_sent_simi.json")
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
