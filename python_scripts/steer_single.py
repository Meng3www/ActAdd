from config import *
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import steer_single, load_data
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)


if __name__ == '__main__':
    layer, coeff = 10, 7
    # load steering model
    model_steer = TransformerBridge.boot_transformers(path_Llama3, device=device)
    print(f"steering model loaded to {device}")
    # # load sentiment model
    # model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    # print("sentiment model loaded")
    # # load relevance model
    # model_relevance = SentenceTransformer(path_all_MiniLM)
    # print("relevance model loaded")
    # load data
    prompt_add, prompt_sub, source_file = " love", " hate", "imdb_neg_llama.json"
    prompts = load_data(source_file, 10)
    steer_single(prompt_add=prompt_add,   
                prompt_sub=prompt_sub, 
                prompts=prompts, 
                steer_model=model_steer, 
                # sentiment_model=model_sentiment, 
                # relevance_model=model_relevance, 
                layer=layer, 
                coeff=coeff, 
                seed=seed, 
                sampling_kwargs=sampling_kwargs, 
                file_name="neg2pos_llama_steered")
