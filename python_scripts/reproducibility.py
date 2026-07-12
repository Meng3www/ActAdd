from utils_aa import get_steering_vec, hooked_generate, remove_prompt, batch_sentiment, batch_similarity
import pandas as pd
import torch


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


if __name__ == '__main__':
    pass