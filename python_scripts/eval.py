from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_aa import get_conditional_ppl


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(path_qwen_logprobs, device_map='auto')
    tokeniser = AutoTokenizer.from_pretrained(path_qwen_logprobs, device_map='auto')
    text = "It is raining cats and dogs. "
    continuation = "The sun rises in the east."
    ppl = get_conditional_ppl(model, tokeniser, text, continuation)
    print("conditional ppl: ", ppl)
    ppl = get_conditional_ppl(model, tokeniser, continuation, text)
    print("conditional ppl: ", ppl)
