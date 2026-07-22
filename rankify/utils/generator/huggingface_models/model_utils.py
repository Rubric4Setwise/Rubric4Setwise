import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import login


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def load_model(model_name, **kwargs):
    model_parallelism = kwargs.get("model_parallelism", False)
    cache_dir = kwargs.get("cache_dir", None)
    auth_token = kwargs.get("auth_token", None)
    torch_dtype = kwargs.get("torch_dtype", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if torch_dtype is not None:
        model_args["torch_dtype"] = torch_dtype  # overload dtype if user specifies
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)

    return model
