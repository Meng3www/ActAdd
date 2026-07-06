import json
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader


working_dir = "/scratch/fmeng/ActAdd/data_result/"


def prompts2tokens(prompt_add, prompt_sub, model):
    """
    check token length of each prompt, pad right to the same token len
    model: steering model
    """
    len_tokens_add = model.to_tokens(prompt_add).shape[1]
    len_tokens_sub = model.to_tokens(prompt_sub).shape[1]
    len_tokens = max(len_tokens_add, len_tokens_sub)
    return model.to_tokens(prompt_add.ljust(len(prompt_add)+len_tokens-len_tokens_add)), model.to_tokens(prompt_sub.ljust(len(prompt_sub)+len_tokens-len_tokens_sub))

def tokens2resid_pre(tokens, layer, model):
    """
    model: steering model
    """
    _, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.hook_resid_pre"]

def hooked_generate(prompts, editing_hooks, model, seed=0, **kwargs):
    """
    model: steering model
    """
    torch.manual_seed(seed)
    with model.hooks(fwd_hooks=editing_hooks):
        result = model.generate(input=prompts, max_new_tokens=64, do_sample=True, **kwargs)
    return result

def load_data(file_name, slice=0):
    with open(f"{working_dir}{file_name}", "r") as f:
        r_data = json.load(f)
    print(f"data loaded from {working_dir}{file_name}")
    if slice==0:
        return r_data
    if slice < len(r_data):
        return r_data[:slice]
    else: 
        return r_data
     
def save2file(data2save, file_name):
    with open(f"{working_dir}{file_name}.json", "w") as f:
        json.dump(data2save, f, skipkeys=True)
    print(f"file saved as {working_dir}{file_name}.json")

def pipeline_steer_single(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, relevance_model, layer, coeff, seed, sampling_kwargs, file_name):
    """
    steering prompts one by one using the activation from `prompt_add-prompt_sub`
    the generated text without the prompt is evaluated by sentiment_model and relevance_model
    the result is saved in json format in file_name
    """
    steered_all = list()
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
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
        generated_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)
        steering_case["generated_text"] = generated_text  
        # continuation_label
        sentiment_result = sentiment_model(generated_text[len(prompt):])
        if sentiment_result[0]["label"] == "POSITIVE":
            steering_case["continuation_label"] = 1
        else:  
            steering_case["continuation_label"] = 0
        # similarity
        embedding_prompt = relevance_model.encode(prompt)
        embedding_generated_text = relevance_model.encode(generated_text[len(prompt):])
        relevance = cosine_similarity(embedding_prompt.reshape(1, -1), 
                                    embedding_generated_text.reshape(1, -1))
        steering_case["similarity"] = relevance[0][0].item()
        steered_all.append(steering_case)
    print(len(steered_all), " documents steered")
    save2file(steered_all, file_name)

def ht_count(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, seed, sampling_kwargs, grid, max_coeff):
    """
    hyperparameter tuning by counting the number of times the generated text (excluding the prompt)
    is steered towards 1 (positive)
    grid: torch.Tensor of size (model_layer, 20)
    """
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff = act_add - act_sub
    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    for coeff in range(1, max_coeff+1):
        print(f"layer={layer}, coeff={coeff}")
        act_diff = act_diff * coeff     
        # generate with the steering vector
        editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
        sentiments = list() 
        for prompt in prompts:
            generated_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)
            # continuation_label
            sentiment_result = sentiment_model(generated_text[len(prompt):])
            if sentiment_result[0]["label"] == "POSITIVE":
                sentiments.append(1)
            else:  
                sentiments.append(0)
        grid[layer, coeff-1] = sum(sentiments)
    return grid

# process in batch
class ModelDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = json.load(f)
        self.n_samples = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples

def trim_text(row):
    """
    helper function to be applied to df 
    to get the continuation without the prompts
    """
    gen_text = row["generated_text"]
    prompt = row["prompt"]
    return gen_text[len(prompt):]

def map_labels(row):
    """
    helper function to be applied to df
    to map text sentiment string labels to 0/1
    """
    if row["label"] == "NEGATIVE":
        return 0
    if row["label"] == "POSITIVE":
        return 1

def pipeline_steer_batch(prompt_add, prompt_sub, prompts_batch, steer_model, sentiment_model, relevance_model, layer, coeff, seed, sampling_kwargs):
    """
    with the given batch of prompts, steer, sentiment, and cosine similarity
    return a panda dataframe with columns: 
        prompt, generated_text (without the prompt), continuation_label, similarity
    this df is ready be concat to the results from the previous batches
    """
    df_batch = pd.DataFrame({"prompt": prompts_batch})
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff = act_add - act_sub
    act_diff = act_diff * coeff

    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    # batch generate with steering
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    generated_text = hooked_generate(prompts_batch, editing_hooks, steer_model, seed, **sampling_kwargs)
    df_batch["generated_text"] = generated_text

    # slice to replace the generated text (prompt+continuation) with continuation
    continuations = df_batch.apply(trim_text, axis=1)
    df_batch["generated_text"] = continuations

    # batch sentiment 
    sentiment_batch = sentiment_model(df_batch["generated_text"].values.tolist())
    # batch cosine


if __name__ == '__main__':
    layer, coeff, seed = 6, 5, 0
    sampling_kwargs = dict(temperature=1.0, top_p=1.0, freq_penalty=0.0)
    
