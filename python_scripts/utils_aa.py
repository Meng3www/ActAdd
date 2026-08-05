import json
import numpy as np
import pandas as pd
import random, gc
import torch
from config import *
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset


def reset_seed(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def cleanup_model(model):
    print("before del", torch.cuda.memory_allocated())
    try:
        if hasattr(model, 'base_model_prefix') and len(model.base_model_prefix) > 0:
            bm = getattr(model, model.base_model_prefix)
            del bm
    except:
        pass
    del model

    gc.collect()
    torch.cuda.empty_cache()
    print("after del", torch.cuda.memory_allocated())


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
    reset_seed(seed)
    with model.hooks(fwd_hooks=editing_hooks):
        result = model.generate(input=prompts, max_new_tokens=max_new_tokens, do_sample=True, **kwargs)
    return result

def get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer):
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff = act_add - act_sub
    return act_diff

def get_logprobs(model, tokeniser, text):
    """
    continuation is the generated text without the prompt
    while in the original code the authors use the complete generated text 
    """
    tokens = tokeniser(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**tokens)
    token_idx_vocab = tokens.input_ids.squeeze(0)[1:]
    output_logsm = torch.nn.functional.log_softmax(outputs.logits, dim = -1)
    # corresponding to the actual next token observed in the test sequence
    token_idx_seq = torch.arange(token_idx_vocab.shape[0])
    output_logsm = output_logsm[:, :-1, :] 
    token_logprobs = output_logsm.squeeze(0)[token_idx_seq, token_idx_vocab]  # torch.Size([6])
    return token_logprobs

def get_conditional_ppl(model, tokeniser, prompt, continuation):
    """
    continuation is the generated text without the prompt
    while in the original code the authors use the complete generated text 
    """
    logprobs_complete = get_logprobs(model, tokeniser, prompt+continuation)
    # print("ppl of the complete sentence: ", torch.exp(-torch.mean(logprobs_complete)))
    logprobs_prompt = get_logprobs(model, tokeniser, prompt)
    # print("ppl of the prompt: ", torch.exp(-torch.mean(logprobs_prompt)))
    # logprobs_continuation = get_logprobs(model, tokeniser, continuation)
    # print("ppl of the continuation: ", torch.exp(-torch.mean(logprobs_continuation)))
    logprobs = logprobs_complete[len(logprobs_prompt):]
    # average negative log probabilities
    ce = -torch.mean(logprobs)
    # exponentiate
    return torch.exp(ce).item()

def load_data(file_name, slice=0):
    """
    TODO: load data from more file formats than just json
    """
    with open(f"{dir_input}{file_name}", "r") as f:
        r_data = json.load(f)
    if slice==0 or slice >= len(r_data):
        print(f"all data points loaded from {dir_input}{file_name}")
        return r_data
    else:
        print(f"{slice} data point(s) loaded from {dir_input}{file_name}")        
        return r_data[:slice]
     
def save2file(data2save, file_name, file_type="json"):
    """
    save file into json format if not specified.
    if data2save is a panda dataframe, then `file_type="parquet"`
    """
    if file_type == "json":
        with open(f"{dir_output}{file_name}.json", "w") as f:
            json.dump(data2save, f, skipkeys=True, indent=2)
        print(f"file saved as {dir_output}{file_name}.json")
    elif file_type == "parquet": # not convinient for qualitative tasks
        data2save.to_parquet(f"{dir_output}{file_name}.gzip", compression="gzip")
        print(f"file saved as {dir_output}{file_name}.gzip")

def steer2batch(prompt_add, prompt_sub, prompts, steer_model, layer, coeff, seed, sampling_kwargs):
    """
    steer prompts one by one, and save the prompts and the generated texts that contain the prompt into a pd dataframe
    return the dataframe for downstream
    """
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer)
    act_diff = act_diff_base *coeff
    df = pd.DataFrame({"prompt": prompts})
    generated_texts = list()
    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
    
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    for prompt in prompts:
        generated_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)        
        generated_texts.append(generated_text)
    df["generated_text"] = generated_texts
    return df

def base_generate(model, prompts, sampling_kwargs, out_file_name):
    """
    base generate prompts and then save the results in json format
    """
    gen_all = list()
    for prompt in prompts:
        gen_case = {"prompt": prompt}
        reset_seed(seed)
        generated_text = model.generate(
                input=prompt,
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                **sampling_kwargs
            )
        gen_case["generated_text"] = generated_text[len(prompt):] 
        gen_all.append(gen_case)
    save2file(gen_all, out_file_name)

def steer_single(prompt_add, prompt_sub, prompts, model, layer, coeff, seed, sampling_kwargs, file_name=""):
    """
    steering prompts one by one using the activation from `prompt_add-prompt_sub`
    the result is saved in json format in file_name
    """
    steered_all = list()
    # get the steering vector
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, model, layer)
    act_diff = act_diff_base *coeff
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
        generated_text = hooked_generate(prompt, editing_hooks, model, seed, **sampling_kwargs)
        steering_case["generated_text"] = generated_text[len(prompt):] 
        steered_all.append(steering_case)
    # print(len(steered_all), " documents steered")
    if file_name:
        save2file(steered_all, file_name)

def pipeline_steer_single(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, relevance_model, layer, coeff, seed, sampling_kwargs, file_name):
    """
    steering prompts one by one using the activation from `prompt_add-prompt_sub`
    the generated text without the prompt is evaluated by sentiment_model and relevance_model
    the result is saved in json format in file_name
    """
    steered_all = list()
    # get the steering vector
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer)
    act_diff = act_diff_base *coeff
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

def ht_steer(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="ht_steer"):
    """
    for hyperparameter tuning. after changing the sentiment classifier
    it only steer the prompts one by one and then save the generated texts into the json file
    the steering vector act_diff_base is re-used at every layer
    the editing_hooks is reused with every coeff
    """
    dict_all = dict()
    # get the steering vector
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer)
    for coeff in range(min_coeff, max_coeff+1):
        list_prompted = list()
        act_diff = act_diff_base * coeff     
        def add_activation(activation, hook):
            if activation.shape[1] == 1: return
            prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
            try:
                activation[:, :steering_dim, :] += act_diff
            except:
                print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
        editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
        for prompt in prompts:
            dict_prompt = {"prompt": prompt}
            gen_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)        
            dict_prompt["generated_text"] = gen_text[len(prompt):]
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
    save2file(dict_all, f"{file_name}_{layer}")

def ht_steer_batch(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="ht_steer_batch"):
    """
    reproduce_layer_actdiff_batch
    reproduce the results with a given layer, coeff from 1 (default) to max_coeff, on prompts
    all prompts are steered in batch, and with fresh editing_hooks
    the act_diff is re-used throughout
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    dict_all = dict()
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer)
    for coeff in range(min_coeff, max_coeff+1):
        df = batch_steer(act_diff_base, prompts, steer_model, layer, coeff, seed, sampling_kwargs)
        df = remove_prompt(df)
        df_dict = df.to_dict(orient="records")
        dict_all[coeff] = df_dict
    save2file(dict_all, f"{file_name}_{layer}")

def ht_count(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, seed, sampling_kwargs, grid, max_coeff):
    """
    hyperparameter tuning by counting the number of times the generated text (excluding the prompt)
    is steered towards 1 (positive)
    grid: torch.Tensor of size (model_layer, 20)
    """
    # get the steering vector
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer)
    dict_para = dict()    # a dictionary with the parameter as the key
    for coeff in range(1, max_coeff+1):
#     for coeff in range(7, 9):
        print(f"layer={layer}, coeff={coeff}")
        act_diff = act_diff_base * coeff     
        # generate with the steering vector
        def add_activation(activation, hook):
            if activation.shape[1] == 1: return
            prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
            try:
                activation[:, :steering_dim, :] += act_diff
            except:
                print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
        editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
        sentiments = list() 
        val_para = list()    # the list of generated text as the value to be added to the dictionary
        for prompt in prompts:
            ret_dict = {"prompt": prompt}
            generated_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)
            ret_dict["generated_text"] = generated_text
            # continuation_label
            sentiment_result = sentiment_model(generated_text[len(prompt):])
            if sentiment_result[0]["label"] == "POSITIVE":
                sentiments.append(1)
                ret_dict["continuation_label"] = 1
            else: 
                sentiments.append(0)
                ret_dict["continuation_label"] = 0
            val_para.append(ret_dict)
        grid[layer, coeff-1] = sum(sentiments)
        dict_para[f"{layer},{coeff}"] = val_para
    return grid, dict_para

# process in batch
class ModelDataset(Dataset):
    def __init__(self, file_name):
        with open(f"{dir_input}{file_name}", "rb") as f:
            self.data = json.load(f)
        self.n_samples = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n_samples

def batch_steer(act_diff_base, prompts, steer_model, layer, coeff, seed, sampling_kwargs):
    """
    prompts: a list of strings
    return: a panda dataframe with two columns, "prompt", and 
    "generated_text": the prompt and the continuing steered generated text
    """
    df = pd.DataFrame({"prompt": prompts})
    act_diff = act_diff_base * coeff
    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    # generate with the steering vector
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    generated_text = hooked_generate(prompts, editing_hooks, steer_model, seed, **sampling_kwargs)
    df["generated_text"] = generated_text
    return df

def batch_base_generate(prompts, model, seed, kwargs):
    """
    prompts: a list of strings
    return: a panda dataframe with two columns, "prompt", and 
    "generated_text": the prompt and the continuing unsteered generated text
    """
    df = pd.DataFrame({"prompt": prompts})
    reset_seed(seed)
    generated_text = model.generate(
        input=prompts,
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        **kwargs
    )
    df["generated_text"] = generated_text
    return df

def trim_text(row):
    """
    helper function to be applied to df 
    to get the continuation without the prompts
    """
    gen_text = row["generated_text"]
    prompt = row["prompt"]
    return gen_text[len(prompt):]

def remove_prompt(df):
    """
    df contains two columns: "prompt", and 
    "generated_text": the prompt and the continuing generated text
    return the same df but with the prompt removed from "generated_text"
    """
    continuation = df.apply(trim_text, axis=1)
    df["generated_text"] = continuation
    return df

def map_labels(row):
    """
    helper function to be applied to df
    to map text sentiment string labels to 0/1
    """
    if row["label"] == "NEGATIVE":
        return 0
    if row["label"] == "POSITIVE":
        return 1

def batch_sentiment(df, sentiment_model, keep_score):
    """
    df contains two columns: "prompt", and 
    "generated_text": the continuing generated text
    sentiment analyse in batch and map the sentiment label to 0/1
    if keep_score is True, the score is included in the returned df
    return a df with additional column(s)
    """
    sentiment_batch = sentiment_model(df["generated_text"].values.tolist())
    sentiment_df = pd.DataFrame(sentiment_batch)
    sentiment_labels = sentiment_df.apply(map_labels, axis=1)
    df["continuation_label"] = sentiment_labels
    if keep_score:
        df["sentiment_score"] = sentiment_df["score"]
    return df

def batch_similarity(df, relevance_model):
    """
    df contains two columns: "prompt", and 
    "generated_text": the continuing generated text
    cosine_similarity calculated in batch
    return a df with an additional column "similarity"
    """
    embedding_prompt_batch = relevance_model.encode(df["prompt"])
    embedding_generated_text_batch = relevance_model.encode(df["generated_text"])
    relevance_batch = cosine_similarity(embedding_prompt_batch,    # .reshape(1, -1)
                                        embedding_generated_text_batch)
    similarity = relevance_batch.diagonal().reshape(relevance_batch.shape[0], 1)
    df["similarity"] = similarity
    return df

def pipeline_base_batch(prompts, base_model, sentiment_model, seed, sampling_kwargs, keep_score=False, relevance_model=None):
    df = batch_base_generate(prompts, base_model, seed, sampling_kwargs)
    df = remove_prompt(df)
    df = batch_sentiment(df, sentiment_model, keep_score)
    if relevance_model:
        df = batch_similarity(df, relevance_model)
    return df

def pipeline_steer_batch(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, coeff, seed, sampling_kwargs, keep_score=False, relevance_model=None):
    """
    with the given batch of prompts, steer, sentiment, and cosine similarity
    return a panda dataframe with columns: 
        prompt, generated_text (without the prompt), continuation_label, similarity
    this df is ready be concat to the results from the previous batches
    """ 
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, steer_model, layer)
    df = batch_steer(act_diff_base, prompts, steer_model, layer, coeff, seed, sampling_kwargs)
    # slice to replace the generated text (prompt+continuation) with continuation
    df = remove_prompt(df)
    # batch sentiment 
    df = batch_sentiment(df, sentiment_model, keep_score)
    # batch cosine
    if relevance_model:
        df = batch_similarity(df, relevance_model)
    return df

def pipeline_steer2batch(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, coeff, seed, sampling_kwargs, keep_score=False, relevance_model=None):
    """
    with the given batch of prompts, steer one by one into a df
    sentiment in batch, and cosine similarity
    return a panda dataframe with columns: 
        prompt, generated_text (without the prompt), continuation_label, similarity
    this df is ready be concat to the results from the previous batches
    """
    df = steer2batch(prompt_add, prompt_sub, prompts, steer_model, layer, coeff, seed, sampling_kwargs)
    # slice to replace the generated text (prompt+continuation) with continuation
    df = remove_prompt(df)
    # batch sentiment 
    df = batch_sentiment(df, sentiment_model, keep_score)
    # batch cosine
    if relevance_model:
        df = batch_similarity(df, relevance_model)
    return df


if __name__ == "__main__":
    layer, coeff, seed = 6, 5, 0
    sampling_kwargs = dict(temperature=1.0, top_p=1.0, freq_penalty=0.0)
    
