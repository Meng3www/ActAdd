from config import *
from transformers import pipeline
from transformer_lens.model_bridge import TransformerBridge
from utils_aa import get_steering_vec, hooked_generate, remove_prompt, batch_sentiment, batch_similarity, load_data, pipeline_steer2batch, pipeline_steer_batch, save2file, prompts2tokens, tokens2resid_pre
import time
import pandas as pd
import torch
import transformer_lens.utilities as utils


device = utils.get_device()
print("device:", device)


def batch_steer(prompt_add, prompt_sub, prompts_batch, steer_model, layer, coeff, seed, sampling_kwargs):
    """
    prompts: a list of strings
    return: a panda dataframe with two columns, "prompt", and 
    "generated_text": the prompt and the continuing steered generated text
    """
    df = pd.DataFrame({"prompt": prompts_batch})
    # get the steering vector
    act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)

    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    # generate with the steering vector
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    generated_text = hooked_generate(prompts_batch, editing_hooks, steer_model, seed, **sampling_kwargs)
    df["generated_text"] = generated_text
    return df

def batch_steer1(prompt_add, prompt_sub, prompts_batch, steer_model, layer, coeff, seed, sampling_kwargs):
    """
    pad act_diff to the same as activation.shape[0] 
    """
    df = pd.DataFrame({"prompt": prompts_batch})
    # get the steering vector
    act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)
    # gemini
    call_count = 0 

    def add_activation(activation, hook):
        nonlocal call_count
        if activation.shape[1] == 1 or call_count > 0: 
            print("call_count", call_count, "activation.shape, ", activation.shape)
            return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        print("activation.shape, ", activation.shape)
        try:
            activation[:, :steering_dim, :] += act_diff
            call_count += 1
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    # generate with the steering vector
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    generated_text = hooked_generate(prompts_batch, editing_hooks, steer_model, seed, **sampling_kwargs)
    df["generated_text"] = generated_text
    return df

def batch_steer2(prompt_add, prompt_sub, prompts_batch, steer_model, layer, coeff, seed, sampling_kwargs):
    """
    use the same steering vector to steer prompts one by one, save in a df
    """
    df = pd.DataFrame({"prompt": prompts_batch})
    # get the steering vector
    act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)
    print("pad act_diff to the same as activation.shape[1]. before padding: act_diff:", act_diff.shape)
    print(act_diff)
    act_diff = torch.nn.functional.pad(input=act_diff, pad=(0, 0, 0, 30, 0, 0), mode='constant', value=0)
    print("after padding:", act_diff.shape)
    print(act_diff)

    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    # generate with the steering vector
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    generated_text = hooked_generate(prompts_batch, editing_hooks, steer_model, seed, **sampling_kwargs)
    df["generated_text"] = generated_text
    return df

def batch_steer3(prompt_add, prompt_sub, prompts_batch, steer_model, layer, coeff, seed, sampling_kwargs):
    """
    pad act_diff to the same as activation.shape
    """
    df = pd.DataFrame({"prompt": prompts_batch})
    # get the steering vector
    act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)
    print("pad act_diff to the same as activation.shape. before padding: act_diff:", act_diff.shape)
    print(act_diff)
    act_diff = torch.nn.functional.pad(input=act_diff, pad=(0, 0, 0, 30, 0, 0), mode='constant', value=0).repeat(len(prompts_batch), 1, 1)
    print("after padding:", act_diff.shape)
    print(act_diff)

    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")

    # generate with the steering vector
    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    generated_text = hooked_generate(prompts_batch, editing_hooks, steer_model, seed, **sampling_kwargs)
    df["generated_text"] = generated_text
    return df

def pipeline_steer_batch1(prompt_add, prompt_sub, prompts_batch, steer_model, sentiment_model, layer, coeff, seed, sampling_kwargs, keep_score=False, relevance_model=None):
    """
    with the given batch of prompts, steer, sentiment, and cosine similarity
    return a panda dataframe with columns: 
        prompt, generated_text (without the prompt), continuation_label, similarity
    this df is ready be concat to the results from the previous batches
    """
    df = batch_steer1(prompt_add=prompt_add, 
                     prompt_sub=prompt_sub, 
                     prompts_batch=prompts_batch, 
                     steer_model=steer_model, 
                     layer=layer, 
                     coeff=coeff, 
                     seed=seed, 
                     sampling_kwargs=sampling_kwargs)
    # slice to replace the generated text (prompt+continuation) with continuation
    df = remove_prompt(df)
    # batch sentiment 
    df = batch_sentiment(df, sentiment_model, keep_score)
    # batch cosine
    if relevance_model:
        df = batch_similarity(df, relevance_model)
    return df

def pipeline_steer_batch3(prompt_add, prompt_sub, prompts_batch, steer_model, sentiment_model, layer, coeff, seed, sampling_kwargs, keep_score=False, relevance_model=None):
    """
    with the given batch of prompts, steer, sentiment, and cosine similarity
    return a panda dataframe with columns: 
        prompt, generated_text (without the prompt), continuation_label, similarity
    this df is ready be concat to the results from the previous batches
    """
    df = batch_steer3(prompt_add=prompt_add, 
                     prompt_sub=prompt_sub, 
                     prompts_batch=prompts_batch, 
                     steer_model=steer_model, 
                     layer=layer, 
                     coeff=coeff, 
                     seed=seed, 
                     sampling_kwargs=sampling_kwargs)
    # slice to replace the generated text (prompt+continuation) with continuation
    df = remove_prompt(df)
    # batch sentiment 
    df = batch_sentiment(df, sentiment_model, keep_score)
    # batch cosine
    if relevance_model:
        df = batch_similarity(df, relevance_model)
    return df

def qualitative_batch_steer(prompt_add, prompt_sub, steer_model, sentiment_model, data_file_path, paras):
    """
    paras: list of (layer, coeff) with which the 10 prompts will be steered
    {(layer, coeff): {count_pos: sum(df["continuation_label"]),
                     score_0: mean_score_neg,
                     score_1: mean_score_pos,
                     results: [{}*10]}, ...}
    """
    ret_dict = dict()
    start = time.time()
    prompts = load_data(data_file_path, num_samples)
    for layer, coeff in paras:
        print(f"layer={layer}, coeff={coeff}")
        df = pipeline_steer_batch(prompt_add=prompt_add, 
                                prompt_sub=prompt_sub, 
                                prompts_batch=prompts, 
                                steer_model=steer_model, 
                                sentiment_model=sentiment_model, 
                                layer=layer, 
                                coeff=coeff, 
                                seed=seed, 
                                sampling_kwargs=sampling_kwargs, 
                                keep_score=True, 
                                relevance_model=None)
        para_dict = {"count_pos": sum(df["continuation_label"])}
        means = df["sentiment_score"].groupby(df["continuation_label"]).mean()
        para_dict["score_0"] = means[0]
        para_dict["score_1"] = means[1]
        para_dict["result"] = df.to_dict(orient="records")
        ret_dict[(layer, coeff)] = para_dict
        print(f"count_pos: {para_dict["count_pos"]}")

    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")
    return ret_dict

def qualitative_single_steer(prompt_add, prompt_sub, steer_model, sentiment_model, data_file_path, paras):
    """
    paras: list of (layer, coeff) with which the 10 prompts will be steered
    {(layer, coeff): {count_pos: sum(df["continuation_label"]),
                     score_0: mean_score_neg,
                     score_1: mean_score_pos,
                     results: [{}*10]}, ...}
    """
    ret_dict = dict()
    start = time.time()
    prompts = load_data(data_file_path, num_samples)
    for layer, coeff in paras:
        print(f"layer={layer}, coeff={coeff}")
        df = pipeline_steer2batch(prompt_add=prompt_add, 
                                prompt_sub=prompt_sub, 
                                prompts_batch=prompts, 
                                steer_model=steer_model, 
                                sentiment_model=sentiment_model, 
                                layer=layer, 
                                coeff=coeff, 
                                seed=seed, 
                                sampling_kwargs=sampling_kwargs, 
                                keep_score=True, 
                                relevance_model=None)
        para_dict = {"count_pos": sum(df["continuation_label"])}
        means = df["sentiment_score"].groupby(df["continuation_label"]).mean()
        para_dict["score_0"] = means[0]
        para_dict["score_1"] = means[1]
        para_dict["result"] = df.to_dict(orient="records")
        ret_dict[(layer, coeff)] = para_dict
        print(f"count_pos: {para_dict["count_pos"]}")
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")
    return ret_dict

def reproduce_layer_single(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_single"):
    """
    reproduce the results with a given layer, coeff from 1 to max_coeff, on prompts
    each prompt is steered one by one, and with fresh act_diff and editing_hooks
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    print("reproduce_layer_single, no re-use")
    dict_all = dict()
    start = time.time()
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
        list_prompted = list()
        for prompt in prompts:
            dict_prompt = {"prompt": prompt}
            act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)
            def add_activation(activation, hook):
                if activation.shape[1] == 1: return
                prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
                try:
                    activation[:, :steering_dim, :] += act_diff
                except:
                    print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
            editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
            gen_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)        
            dict_prompt["generated_text"] = gen_text
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def reproduce_layer_actdiff_single_inner(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_actdiff_single_inner"):
    """
    reproduce the results with a given layer, coeff from 1 to max_coeff, on prompts
    each prompt is steered one by one, and with fresh editing_hooks
    within the same coeff, the act_diff is re-used
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    print("reproduce_layer_actdiff_single_inner, re-using act_diff for each coeff")
    dict_all = dict()
    start = time.time()
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
        act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)
        list_prompted = list()
        for prompt in prompts:
            dict_prompt = {"prompt": prompt}
            def add_activation(activation, hook):
                if activation.shape[1] == 1: return
                prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
                try:
                    activation[:, :steering_dim, :] += act_diff
                except:
                    print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
            editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
            gen_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)        
            dict_prompt["generated_text"] = gen_text
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def reproduce_layer_actdiff_single_outer(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_actdiff_single_outer"):
    """
    reproduce the results with a given layer, coeff from 1 to max_coeff, on prompts
    each prompt is steered one by one, and with fresh editing_hooks
    the act_diff(base) is re-used throughout
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    print("reproduce_layer_actdiff_single_outer, re-using act_diff for all generation")
    dict_all = dict()
    start = time.time()
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff_base = act_add - act_sub
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
        list_prompted = list()
        for prompt in prompts:
            dict_prompt = {"prompt": prompt}
            act_diff = act_diff_base * coeff     
            def add_activation(activation, hook):
                if activation.shape[1] == 1: return
                prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
                try:
                    activation[:, :steering_dim, :] += act_diff
                except:
                    print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
            editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
            gen_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)        
            dict_prompt["generated_text"] = gen_text
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def reproduce_layer_ht_count(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_ht_count"):
    """
    """
    print("reproduce_layer_ht_count, re-using act_diff for all generation, editing_hooks for the whole coeff")
    dict_all = dict()
    start = time.time()
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff_base = act_add - act_sub
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
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
            dict_prompt["generated_text"] = gen_text
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def reproduce_layer_ht_count_senti(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name=f"reproduce_layer_ht_count_senti"):
    """
    """
    dict_all = dict()
    counts_all = list()
    # get the steering vector
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff_base = act_add - act_sub
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
        sentiments = list() 
        for prompt in prompts:
            dict_prompt = {"prompt": prompt}
            gen_text = hooked_generate(prompt, editing_hooks, steer_model, seed, **sampling_kwargs)        
            dict_prompt["generated_text"] = gen_text
            sentiment_result = sentiment_model(gen_text[len(prompt):])
            if sentiment_result[0]["label"] == "POSITIVE":
                sentiments.append(1)
                dict_prompt["continuation_label"] = 1
            else: 
                sentiments.append(0)
                dict_prompt["continuation_label"] = 0
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
        counts_all.append(sum(sentiments))
    save2file(dict_all, f"{file_name}_{layer}")
    print("number of positives: ", counts_all)
    return counts_all

def reproduce_layer_inner_single(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_inner_single"):
    """
    reproduce the results with a given layer, coeff from 1 to max_coeff, on prompts
    each prompt is steered one by one
    within the same coeff, both variables are re-used
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    print("reproduce_layer_inner_single, re-use both parameters in each coeff")
    dict_all = dict()
    start = time.time()
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
        list_prompted = list()
        act_diff = get_steering_vec(prompt_add, prompt_sub, steer_model, layer, coeff)
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
            dict_prompt["generated_text"] = gen_text
            list_prompted.append(dict_prompt)
        dict_all[coeff] = list_prompted
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def reproduce_layer_batch(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_batch"):
    """
    reproduce the results with a given layer, coeff from 1 to max_coeff, on prompts
    all prompts are steered in a batch, and with fresh act_diff and editing_hooks
    this is equivalent to recycling both act_diff and editing_hooks in the inner loop
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    print("reproduce_layer_batch")
    dict_all = dict()
    start = time.time()
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
        df = batch_steer(prompt_add, prompt_sub, prompts, steer_model, layer, coeff, seed, sampling_kwargs)
        df_dict = df.to_dict(orient="records")
        dict_all[coeff] = df_dict
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def reproduce_layer_actdiff_batch(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs, min_coeff=1, file_name="reproduce_layer_actdiff_batch_outer"):
    """
    reproduce the results with a given layer, coeff from 1 to max_coeff, on prompts
    all prompts are steered in batch, and with fresh editing_hooks
    the act_diff is re-used throughout
    the results are saved in a json file with coeff as the key, [{prompt, gen_text}] as the value
    """
    print("reproduce_layer_actdiff_batch, act_diff is re-used throughout")
    dict_all = dict()
    start = time.time()
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, steer_model)
    act_add = tokens2resid_pre(tokens_add, layer, steer_model)
    act_sub = tokens2resid_pre(tokens_sub, layer, steer_model)
    act_diff_base = act_add - act_sub
    for coeff in range(min_coeff, max_coeff+1):
        print(f"coeff={coeff}")
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
        df = pd.DataFrame({"prompt": prompts})
        df["generated_text"] = generated_text
        df_dict = df.to_dict(orient="records")
        dict_all[coeff] = df_dict
    save2file(dict_all, file_name)
    print(f"time elapsed: {round((time.time() - start)/60, 2)} mins")

def test_whole_layer(prompt_add, prompt_sub, steer_model, layer, max_coeff, seed, sampling_kwargs, input_file_path):
    prompts = load_data(input_file_path, num_samples)
    reproduce_layer_single(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_actdiff_single_inner(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_actdiff_single_outer(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_inner_single(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_batch(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_actdiff_batch(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)

def test_specific(prompt_add, prompt_sub, steer_model, layer, coeff, seed, sampling_kwargs, input_file_path):
    """
    layer, coeff: the specific layer, coeff to be checked
    """
    prompts = load_data(input_file_path, num_samples)
    reproduce_layer_single(prompt_add, prompt_sub, prompts, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "coeff_single")
    reproduce_layer_actdiff_single_inner(prompt_add, prompt_sub, prompts, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "coeff_actdiff_single_inner")
    reproduce_layer_actdiff_single_outer(prompt_add, prompt_sub, prompts, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "coeff_actdiff_single_outer")
    reproduce_layer_inner_single(prompt_add, prompt_sub, prompts, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "coeff_inner_single")
    reproduce_layer_batch(prompt_add, prompt_sub, prompts, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "coeff_batch")
    reproduce_layer_actdiff_batch(prompt_add, prompt_sub, prompts, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "coeff_actdiff_batch")
    prompts_subset = prompts[3:5] 
    reproduce_layer_single(prompt_add, prompt_sub, prompts_subset, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "subset_single")
    reproduce_layer_actdiff_single_inner(prompt_add, prompt_sub, prompts_subset, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "subset_actdiff_single_inner")
    reproduce_layer_actdiff_single_outer(prompt_add, prompt_sub, prompts_subset, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "subset_actdiff_single_outer")
    reproduce_layer_inner_single(prompt_add, prompt_sub, prompts_subset, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "subset_inner_single")
    reproduce_layer_batch(prompt_add, prompt_sub, prompts_subset, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "subset_batch")
    reproduce_layer_actdiff_batch(prompt_add, prompt_sub, prompts_subset, steer_model, layer, coeff+1, seed, sampling_kwargs, coeff, "subset_actdiff_batch")

def test_ht_count(prompt_add, prompt_sub, steer_model, layer, max_coeff, seed, sampling_kwargs, input_file_path):
    prompts = load_data(input_file_path, num_samples)
    reproduce_layer_ht_count(prompt_add, prompt_sub, prompts, steer_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_ht_count(prompt_add, prompt_sub, prompts, steer_model, layer, 10, seed, sampling_kwargs, 10, "10_actdiff_batch")
    reproduce_layer_ht_count(prompt_add, prompt_sub, prompts, steer_model, layer, 19, seed, sampling_kwargs, 19, "19_actdiff_batch")

def test_ht_count_senti(prompt_add, prompt_sub, steer_model, sentiment_model, layer, max_coeff, seed, sampling_kwargs, input_file_path):
    prompts = load_data(input_file_path, num_samples)
    reproduce_layer_ht_count_senti(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, max_coeff, seed, sampling_kwargs)
    reproduce_layer_ht_count_senti(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, 10, seed, sampling_kwargs, 10, "10_actdiff_batch_senti")
    reproduce_layer_ht_count_senti(prompt_add, prompt_sub, prompts, steer_model, sentiment_model, layer, 19, seed, sampling_kwargs, 19, "19_actdiff_batch_senti")

if __name__ == '__main__':
    model_generate = TransformerBridge.boot_transformers(path_Llama3, device=device)
    print(f"baseline generating model loaded to {device}")
    model_sentiment = pipeline("sentiment-analysis", model=path_siebert)
    print("sentiment model loaded")
    prompt_add, prompt_sub, input_file_path, layer = " love", " hate", "imdb_neg_llama.json", 10

    # test_whole_layer(prompt_add, prompt_sub, model_generate, layer, max_coeff, seed, sampling_kwargs, input_file_path)
    # test_specific(prompt_add, prompt_sub, model_generate, layer, 19, seed, sampling_kwargs, input_file_path)
    test_ht_count(prompt_add, prompt_sub, model_generate, layer, max_coeff, seed, sampling_kwargs, input_file_path)
    test_ht_count_senti(prompt_add, prompt_sub, model_generate, model_sentiment, layer, max_coeff, seed, sampling_kwargs, input_file_path)
