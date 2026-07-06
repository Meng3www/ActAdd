from config import *
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import pipeline_steer_batch, load_data
import transformer_lens.utilities as utils


batch_size=25

device = utils.get_device()
print("device:", device)
    
if __name__ == '__main__':
    layer, coeff, seed = 6, 5, 0
    sampling_kwargs = dict(temperature=1.0, top_p=1.0, freq_penalty=0.0)
    # load steering model
    model_steer = TransformerBridge.boot_transformers(path_Llama3, device=device)
    print(f"steering model loaded to {device}")
    # model_steer.enable_compatibility_mode()  # this line causes oom error
    # load sentiment model
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    # load relevance model
    model_relevance = SentenceTransformer(path_all_MiniLM)
    print("relevance model loaded")
    # load data
    prompts_neg = load_data("", 50)
    pipeline_steer_batch(prompt_add=" love", 
                  prompt_sub=" hate", 
                  prompts=prompts_neg[:3], 
                  steer_model=model_steer, 
                  sentiment_model=model_sentiment, 
                  relevance_model=model_relevance, 
                  layer=layer, 
                  coeff=coeff, 
                  seed=seed, 
                  sampling_kwargs=sampling_kwargs, 
                  file_name="neg2pos")
