from utils_aa import prompts2tokens, get_steering_vec_base
import matplotlib.pyplot as plt
import numpy as np
import torch


# linear map the steering vector to see what token(s) it maps to
"""
in hyperparameter tuning with OPT-6.7B on neg2pos, `Anton Harlan` and `Anton` 
are generated multiple times in different context
is the steering vector ` love` - ` hate` actually map to certain tokens?
testing layer 0, 8, 15, 23, 31
1. get add_diff from that layer, apply the model’s final layer norm and unembedding matrix W_U to it
Apply the softmax on the resulting logits to get the next token distribution
Read off the top-k tokens
2. checkout the similarity of the add_diff form different layers with the activation of
`Anton Harlan` or `Anton` (check out the dimensions first) from the same/different layers
"""


def plot_logitlens(to_viz, decoded_tokens, decoded_logits, text_tokens):
    """
    https://github.com/abrvkh/explainability_toolkit.git
    to_viz: (begin_idx, n_tokens), the window to display
    decoded_tokens: list of list of tokens, with dimension of [n_layer, n_tokens]
    decoded_logits: same dim as decoded_tokens
    text_tokens: tokenised text, including the prompt and the generated text
    """
    tokens_viz = [tok[to_viz[0]:to_viz[1]] for tok in decoded_tokens] # [num_layers, tokens_to_viz]
    logits_viz = np.array([log[to_viz[0]:to_viz[1]] for log in decoded_logits]) # [num_layers, tokens_to_viz]
    a = [it for it in range(to_viz[0],to_viz[1])]
    b = [tok for tok in text_tokens[to_viz[0]:to_viz[1]]]
    col_labels = [str(a_)+": "+b_ for a_, b_ in zip(a, b)]
    norm = plt.Normalize(logits_viz.min()-1, logits_viz.max()+1)
    colours = plt.cm.cool(norm(logits_viz))
    fig, ax = plt.subplots(figsize=(7,3), dpi=1000)
    # hide axes
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")
    img = plt.imshow(norm(logits_viz), cmap="cool")
    plt.colorbar()
    img.set_visible(False)
    ax.table(cellText=tokens_viz, rowLabels=[lay for lay in range(len(decoded_tokens))], colLabels=col_labels, colWidths = [0.2]*logits_viz.shape[1], loc="center", cellColours=img.to_rgba(norm(logits_viz)))
    plt.show()

def decode_token_with_logitlens(model, device, input, tokens_to_gen=None):
    """
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/ActivationCache.py
    outputs a dictionary with {"decoded_tokens": [num_layers, seq_len], "decoded_logits": [num_layers, seq_len]}
    the model is initialised with `TransformerBridge.boot_transformers` 
    """
    inputs = model.to_tokens(input).to(device)
    text = model.tokenizer.decode(inputs[0], skip_special_tokens=True)
    # run the loop to generate new tokens after input, append to input and decode
    if tokens_to_gen != None:
        # generate new tokens all at once; then append them to input, then logitlens them all
        output = model.generate(inputs, do_sample=True, top_p=1.0, temperature=1.0, top_k=1, max_new_tokens=tokens_to_gen)
        new_token = model.tokenizer.decode(output[0][-tokens_to_gen:], skip_special_tokens=True)
        text += new_token
        inputs = model.to_tokens(text).to(device)
    text_tokens = [model.tokenizer.decode(id, skip_special_tokens=True) for id in inputs[0]]  # ["", "1", ":", "sun", "shine", ",", ...

    # apply decoder lens
    logits, cache = model.run_with_cache(inputs, remove_batch_dim=True)
    num_layers = model.cfg.n_layers  # 12
    decoded_intermediate_token = {}
    decoded_intermediate_logit = {}
    for layer_id in range(num_layers):
        hidden_state = cache[f"blocks.{layer_id}.hook_resid_post"]  # [34, 768]
        # hidden_state = cache[f"blocks.{layer_id}.hook_resid_pre"]
        normalised_state = cache.apply_ln_to_stack(hidden_state, layer=-1)  # [34, 768]
        # get probabilities
        decoded_values = model.unembed(normalised_state)  # [34, 50257]
        # take max element
        argmax = torch.argmax(decoded_values, dim=-1) # tensor of length 34
        # decode all tokens
        decoded_token = [model.tokenizer.decode(el, skip_special_tokens=True) for el in argmax]  # ["\n", ",", " the", ",", ",", " the", "," ...
        decoded_logit = [decoded_values[it, argmax[it]].item() for it in range(len(argmax))] # 34 ["", "1", ":", "sun", "shine", ",", " 2", ":", " .....
        decoded_intermediate_token[layer_id] = decoded_token
        decoded_intermediate_logit[layer_id] = decoded_logit

    tokens = list(decoded_intermediate_token.values()) # [num_layers, seq_len]
    logits = list(decoded_intermediate_logit.values()) # [num_layers, seq_len]
    return {"text_tokens":text_tokens, "decoded_tokens": tokens, "decoded_logits": logits}

def map_vec2tokens(prompt_add, prompt_sub, model, layer):
    """
    https://learn.arena.education/chapter1_transformer_interp/21_ioi/2-logit-attribution/
    get the add_diff between prompt_add and prompt_sub at layer, 
    apply the model’s final layer norm and unembed
    apply the softmax, get the top-k tokens
    """
    # get activations
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, model)
    _, cache_add = model.run_with_cache(tokens_add)
    _, cache_sub = model.run_with_cache(tokens_sub)
    act_name = f"blocks.{layer}.hook_resid_pre"
    act_add = cache_add[act_name]
    act_sub = cache_sub[act_name]
    act_diff = act_add - act_sub
    """
    according to Gemini, unembed has the LayerNorm from the final layer operation included
    and ChatGPT disagrees with it. it turns out that HookedTransformer init an Unembed object 
    with the model parameter and then can call it. it requires self.ln_final unless cfg.normalization_type is None
    but all these are not applicable with `TransformerBridge.boot_transformers` as
    the discussion is based on deprecated `HookedTransformer` system
    the explainability_toolkit notebook does not use normalisation before unembed
    alternatively, use either of the caches to normalise before embed
    """
    # normalise as normal for the two prompts
    normalised_add = cache_add.apply_ln_to_stack(act_add, layer=-1)
    decoded_token_add, _ = normalised2tokens(normalised_add, model)
    normalised_sub = cache_sub.apply_ln_to_stack(act_sub, layer=-1)
    decoded_token_sub, _ = normalised2tokens(normalised_sub, model)
    # for act_diff, 1 no_norm, 2 norm_add, 3 norm_sub
    decoded_token_act_diff_no_norm, _ = normalised2tokens(act_diff, model)
    normalised_act_diff_add = cache_add.apply_ln_to_stack(act_add, layer=-1)
    decoded_token_act_diff_add, _ = normalised2tokens(normalised_act_diff_add, model)
    normalised_act_diff_sub = cache_sub.apply_ln_to_stack(act_add, layer=-1)
    decoded_token_act_diff_sub, _ = normalised2tokens(normalised_act_diff_sub, model)
    print("tokens predicted by the add vector: ", decoded_token_add)
    print("by the sub vector: ", decoded_token_sub)
    print("by diff vector, no norm: ", decoded_token_act_diff_no_norm)
    print("diff vector with add norm: ", decoded_token_act_diff_add)
    print("diff vector with sub norm: ", decoded_token_act_diff_sub)

def normalised2tokens(normalised_state, model):
    """
    normalised_state: cache to be mapped to tokens by unembed
    """
    logits = model.unembed(normalised_state)
    argmax = torch.argmax(logits, dim=-1)
    tokens = [model.tokenizer.decode(el, skip_special_tokens=True) for el in argmax]
    decoded_logits = [logits[it, argmax[it]].item() for it in range(len(argmax))]
    return tokens, decoded_logits

def steer_with_logitlens(prompt_add, prompt_sub, model, device, layer, coeff, input):
    input_tokens = model.to_tokens(input).to(device)
    act_diff_base = get_steering_vec_base(prompt_add, prompt_sub, model, layer)
    act_diff = act_diff_base *coeff
    def add_activation(activation, hook):
        if activation.shape[1] == 1: return
        prompt_dim, steering_dim = activation.shape[1], act_diff.shape[1]
        try:
            activation[:, :steering_dim, :] += act_diff
        except:
            print(f"More mod tokens ({steering_dim}) than prompt tokens ({prompt_dim})!")
    text_tokens = [model.tokenizer.decode(id, skip_special_tokens=True) for id in input_tokens[0]] 

    editing_hooks = [(f"blocks.{layer}.hook_resid_pre", add_activation)]
    with model.hooks(fwd_hooks=editing_hooks):
        _, cache = model.run_with_cache(input_tokens, remove_batch_dim=True)
    num_layers = model.cfg.n_layers  # 12
    decoded_intermediate_token = {}
    decoded_intermediate_logit = {}
    for layer_id in range(num_layers):
        hidden_state = cache[f"blocks.{layer_id}.hook_resid_post"]  
        # hidden_state = cache[f"blocks.{layer_id}.hook_resid_pre"]
        normalised_state = cache.apply_ln_to_stack(hidden_state, layer=-1)
        decoded_tokens, decoded_logits = normalised2tokens(normalised_state, model)
        decoded_intermediate_token[layer_id] = decoded_tokens
        decoded_intermediate_logit[layer_id] = decoded_logits
    tokens = list(decoded_intermediate_token.values()) # [num_layers, seq_len]
    logits = list(decoded_intermediate_logit.values()) # [num_layers, seq_len]
    return {"text_tokens":text_tokens, "decoded_tokens": tokens, "decoded_logits": logits}


if __name__ == "__main__":
    pass
