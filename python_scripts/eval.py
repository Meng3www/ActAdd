from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_aa import get_ppl


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(path_qwen_logprobs, device_map='auto')
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_logprobs, device_map='auto')
    text = "It is raining cats and dogs."
    ppl = get_ppl(model, tokeniser, text)
    print("ppl: ", ppl)

    ppl = get_ppl(model, tokeniser, ["The sun rises in the east."])
    print("ppl: ", ppl)
