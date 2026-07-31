dir_input = "/scratch/fmeng/ActAdd/data/"
dir_output = "/scratch/fmeng/ActAdd/results/"
path_siebert = "/scratch/common_models/SiEBERT/"
path_all_MiniLM = "/scratch/common_models/all-MiniLM-L6-v2/"
path_llama = "/scratch/common_models/Meta-Llama-3-8B/"
path_opt = "/scratch/common_models/opt-6.7b/"
path_qwen_sentiment = "/scratch/common_models/Qwen2.5-7B-Instruct"
path_qwen_logprobs = "/scratch/common_models/Qwen2.5-7B/"

seed = 0
max_new_tokens = 64
max_coeff = 20  # for hyperparameter tuning
num_samples = 20  # for hyperparameter tuning
batch_size=25  # batch base generation
sampling_kwargs = dict(temperature=0, top_p=1.0, freq_penalty=1.0)  # sentiment
# sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)  # toxicity
