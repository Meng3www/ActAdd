from utils_aa import prompts2tokens, tokens2resid_pre


# 1. linear map the steering vector to see what token(s) it maps to
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

def map_add_diff2tokens(prompt_add, prompt_sub, model, layer, top_k=1):
    """
    https://learn.arena.education/chapter1_transformer_interp/21_ioi/2-logit-attribution/
    get the add_diff between prompt_add and prompt_sub at layer, 
    apply the model’s final layer norm and unembed
    apply the softmax, get the top-k tokens
    """
    tokens_add, tokens_sub = prompts2tokens(prompt_add, prompt_sub, model)
    act_add = tokens2resid_pre(tokens_add, layer, model)
    act_sub = tokens2resid_pre(tokens_sub, layer, model)
    act_diff = act_add - act_sub
    """
    according to Gemini, unembed has the LayerNorm from the final layer operation included (cannot verify)
    and ChatGPT disagrees with it. it turns out that HookedTransformer init an Unembed object 
    with the model parameter and then can call it. it requires self.ln_final unless cfg.normalization_type is None
    """
    add_logits = model.unembed(model.)


if __name__ == '__main__':
    pass
