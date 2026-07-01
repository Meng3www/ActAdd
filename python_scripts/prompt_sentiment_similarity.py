from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
import gc
import json
import torch
import transformer_lens.utilities as utils

working_dir = "/scratch/fmeng/ActAdd/data_result/"
path_siebert = "/scratch/common_models/SiEBERT/"
path_all_MiniLM = "/scratch/common_models/all-MiniLM-L6-v2/"
path_Llama3 = "/scratch/common_models/Meta-Llama-3-8B/"
path_opt = "/scratch/common_models/opt-6.7b/"

device = utils.get_device()
print("device:", device)


class Steer:
    def __init__(self, steer_path, sentiment_path, relevance_path):
        self.steer_model = self.steering_model(steer_path)
        self.sentiment_model = self.sentiment_model(sentiment_path)
        self.relevance_model = self.relevance_model(relevance_path)
        self.steered_all = list()

    def steering_model(self, model_path):
        model = TransformerBridge.boot_transformers(model_path, 
                                                    device="cpu",
                                                    dtype=torch.float16)
        print(f"steering model loaded to cpu")
        model.enable_compatibility_mode()
        model = model.to(torch.float16)
        gc.collect()
        if device == "cuda":
            model = model.to(device) 
        print(f"steering model loaded from {model_path}")
        return model

    def sentiment_model(self, model_path):
        print(f"sentiment model loaded from {model_path}")
        return pipeline("sentiment-analysis", model=model_path)

    def relevance_model(self, model_path):
        print(f"relevance model loaded from {model_path}")
        return SentenceTransformer(model_path) 

    def prompts2tokens(self, prompt_add, prompt_sub):
        """
        check token length of each prompt, pad right to the same token len
        """
        len_tokens_add = self.steering_model.to_tokens(prompt_add).shape[1]
        len_tokens_sub = self.steering_model.to_tokens(prompt_sub).shape[1]
        len_tokens = max(len_tokens_add, len_tokens_sub)
        return self.steering_model.to_tokens(prompt_add.ljust(len(prompt_add)+len_tokens-len_tokens_add)), self.steering_model.to_tokens(prompt_sub.ljust(len(prompt_sub)+len_tokens-len_tokens_sub))

    def tokens2resid_pre(self, tokens, layer):
        _, cache = self.steering_model.run_with_cache(tokens)
        return cache[f"blocks.{layer}.hook_resid_pre"]

    def hooked_generate(self, prompts, editing_hooks, seed=0, **kwargs):
        torch.manual_seed(seed)
        with self.steering_model.hooks(fwd_hooks=editing_hooks):
            result = self.steering_model.generate(input=prompts, max_new_tokens=64, do_sample=True, **kwargs)
        return result

    def steer_prompts(self, prompt_add, prompt_sub, prompts, layer, coeff, seed, sampling_kwargs):
        # get the steering vector
        tokens_add, tokens_sub = self.prompts2tokens(prompt_add, prompt_sub)
        act_add = self.tokens2resid_pre(tokens_add, layer)
        act_sub = self.tokens2resid_pre(tokens_sub, layer)
        act_diff = act_add - act_sub
        act_diff = act_diff * coeff

        def add_activation(activation, hook):
            if activation.shape[1] == 1: return
            prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
            try:
                activation[:, :steering_dim, :] += act_diff
            except:
                print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

        # generate with the steering vector
        editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
        for prompt in prompts:
            steering_case = {"prompt": prompt}
            generated_text = self.hooked_generate(prompt, editing_hooks, seed, **sampling_kwargs)
            steering_case["generated_text"] = generated_text  
            # continuation_label
            sentiment_result = self.sentiment_model(generated_text[len(prompt):])
            if sentiment_result[0]["label"] == "POSITIVE":
                steering_case["continuation_label"] = 1
            else:  
                steering_case["continuation_label"] = 0
            # similarity
            embedding_prompt = self.relevance_model.encode(prompt)
            embedding_generated_text = self.relevance_model.encode(generated_text[len(prompt):])
            relevance = cosine_similarity(embedding_prompt.reshape(1, -1), 
                                        embedding_generated_text.reshape(1, -1))
            steering_case["similarity"] = relevance[0][0].item()
            self.steered_all.append(steering_case)
    
    def save2file(self, file_name):
        with open(f"{working_dir}imdb_{file_name}_sen_rel.json", "w") as f:
            json.dump(self.steered_all, f, skipkeys=True)

if __name__ == '__main__':
    layer, coeff, seed = 6, 5, 0
    sampling_kwargs = dict(temperature=1.0, top_p=1.0, freq_penalty=0.0)
    steer = Steer(path_Llama3, path_siebert, path_all_MiniLM)
    with open(f"{working_dir}imdb_neg.json", "r") as f:
        prompts_neg = json.load(f)
    steer.steer_prompts(" love", " hate", prompts_neg[:2], layer, coeff, seed, sampling_kwargs)
    steer.save2file("neg2pos")
