from config import *
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import steer_single, load_data, base_generate, cleanup_model
import time
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)

def toxicity(model, prompt_add, prompt_sub, layer, coeff, outfile_name):
    """
    with one model, single base_gen and steer the prompts from the five files 
    """
    for i in range(5):
        prompts = load_data(f"tox_samples_{i}.json")
        print(f"data loaded from tox_samples_{i}.json")
        start = time.time()
        steer_single(prompt_add=prompt_add,   
                    prompt_sub=prompt_sub, 
                    prompts=prompts, 
                    steer_model=model, 
                    layer=layer, 
                    coeff=coeff, 
                    seed=seed, 
                    sampling_kwargs=sampling_kwargs, 
                    file_name=f"detox_{i}_{outfile_name}")
        print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")
        # base generate
        start = time.time()
        base_generate(model, prompts, sampling_kwargs, f"tox_{i}_{outfile_name}")
        print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")
    
if __name__ == "__main__":
    # from the notebook
    prompt_add, prompt_sub = "Love", "Hate"
    layer, coeff = 17, 2 
    model_llama = TransformerBridge.boot_transformers(path_llama, device=device)
    print(f"steering model loaded to {device}")
    # model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    # print("sentiment model loaded")
    # model_relevance = SentenceTransformer(path_all_MiniLM)
    # print("relevance model loaded")
    toxicity(model_llama, prompt_add, prompt_sub, layer, coeff, f"llama_{layer}-{coeff}")
    layer, coeff = 22, 13  # hpt
    toxicity(model_llama, prompt_add, prompt_sub, layer, coeff, f"llama_{layer}-{coeff}")
    prompt_add, prompt_sub = " love", " hate"
    layer, coeff = 17, 2 
    toxicity(model_llama, prompt_add, prompt_sub, layer, coeff, f"llama_{layer}-{coeff}_space")
    layer, coeff = 22, 13  # hpt
    toxicity(model_llama, prompt_add, prompt_sub, layer, coeff, f"llama_{layer}-{coeff}_space")

    cleanup_model(model_llama)
    print("model removed")
    prompt_add, prompt_sub = "Love", "Hate"
    model_opt = TransformerBridge.boot_transformers(path_opt, device=device)
    print(f"new steering model loaded to {device}")
    layer, coeff = 17, 2 
    toxicity(model_opt, prompt_add, prompt_sub, layer, coeff, f"opt_{layer}-{coeff}")
    layer, coeff = 2, 3
    toxicity(model_opt, prompt_add, prompt_sub, layer, coeff, f"opt_{layer}-{coeff}")
    prompt_add, prompt_sub = " love", " hate"
    layer, coeff = 17, 2 
    toxicity(model_opt, prompt_add, prompt_sub, layer, coeff, f"opt_{layer}-{coeff}_space")
    layer, coeff = 2, 3
    toxicity(model_opt, prompt_add, prompt_sub, layer, coeff, f"opt_{layer}-{coeff}_space")
