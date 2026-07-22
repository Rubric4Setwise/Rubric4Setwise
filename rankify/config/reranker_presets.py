"""
Reranker preset configuration dictionary.

Each key is a short alias used for `RERANKER_CHOICE`. The value contains:
  - method:      the method name registered in METHOD_MAP
  - model_name:  a HuggingFace model id or a predefined alias
  - extra_kwargs: optional extra kwargs forwarded to the reranker constructor
"""

RERANKER_PRESETS = {
    # ==================== Cross-Encoder / Transformer ====================
    "bge-reranker-large": {"method": "transformer_ranker", "model_name": "bge-reranker-large"},
    "rankllama":          {"method": "rankllama", "model_name": "rankllama-v1-7b-lora-passage"},
    "rankvicuna":         {"method": "vicuna_reranker", "model_name": "rank_vicuna_7b_v1", "extra_kwargs": {"vllm_batched": True, "gpu_memory_utilization": 0.4}},
    "rankzephyr":         {"method": "zephyr_reranker", "model_name": "rank_zephyr_7b_v1_full", "extra_kwargs": {"vllm_batched": True, "gpu_memory_utilization": 0.4}},

    # ==================== Generative (T5 / GPT) ====================
    "rankgpt":  {"method": "rankgpt", "model_name": "llamav3.1-8b"},
    "monot5":   {"method": "monot5", "model_name": "monot5-3b-msmarco-10k"},
    "rankt5":   {"method": "rankt5", "model_name": "rankt5-3b"},

    # ==================== Rank4Gen / SetR ====================
    "rank4gen": {
        "method": "rank4gen", "model_name": "JohnnyFan/Rank4Gen-DPO-Qwen3-8B",
        "extra_kwargs": {"downstream_model": "Qwen2.5-7B-Instruct", "lang": "en", "mode": "index", "top_k": 0, "num_gpus": 1, "gpu_memory_utilization": 0.4, "vllm_batched": True, "cache_dir": "/cfs_cloud_code/jiangkailin/Rankify_model_data"},
    },
    # prompt_mode options: "selection_IRI" (3-step reasoning), "selection_woIRI" (CoT without IRI), "selection_only" (direct selection)
    "setr": {
        "method": "setr", "model_name": "JohnnyFan/SETR-Qwen3-8B",
        "extra_kwargs": {"prompt_mode": "selection_only", "top_k": 0, "max_passages": 20, "num_gpus": 1, "gpu_memory_utilization": 0.4, "vllm_batched": True, "context_size": 20480, "cache_dir": "/cfs_cloud_code/jiangkailin/Rankify_model_data"},
    },

    # ==================== Rubric4Setwise (rubric-guided set selection, one LLM call) ====================
    # The input JSONL must carry a `hybrid_rubrics` field (or a legacy `rubric` field).
    "rubric4setwise": {
        "method": "rubric4setwise", "model_name": "Qwen/Qwen3-8B",
        "extra_kwargs": {
            "max_k": 10, "top_k": 0, "max_passages": 20, "max_doc_tokens": 1500,
            "num_gpus": 1, "gpu_memory_utilization": 0.4, "vllm_batched": True,
            "context_size": 8192, "cache_dir": "/cfs_cloud_code/jiangkailin/Rankify_model_data",
        },
    },
    "rubric4setwise-llama8b": {
        "method": "rubric4setwise", "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "extra_kwargs": {
            "max_k": 10, "top_k": 0, "max_passages": 20, "max_doc_tokens": 1500,
            "num_gpus": 1, "gpu_memory_utilization": 0.4, "vllm_batched": True,
            "context_size": 8192, "cache_dir": "/cfs_cloud_code/jiangkailin/Rankify_model_data",
        },
    },

    # ==================== ReasonRank (Listwise Sliding Window + Reasoning) ====================
    "reasonrank-7b": {
        "method": "reasonrank", "model_name": "liuwenhan/reasonrank-7B",
        "extra_kwargs": {"prompt_mode": "reasoning", "window_size": 20, "step_size": 10, "max_passage_length": 100, "reasoning_max_tokens": 3172, "context_size": 16384, "num_gpus": 1, "vllm_batched": True, "gpu_memory_utilization": 0.85},
    },
    "reasonrank-32b": {
        "method": "reasonrank", "model_name": "liuwenhan/reasonrank-32B",
        "extra_kwargs": {"prompt_mode": "reasoning", "window_size": 20, "step_size": 10, "max_passage_length": 100, "reasoning_max_tokens": 3172, "context_size": 32768, "num_gpus": 4, "vllm_batched": True, "gpu_memory_utilization": 0.4},
    },

    # ==================== Rank-R1 (Setwise HeapSort + <think>/<answer>, LoRA on Qwen2.5) ====================
    "rankr1-7b": {
        "method": "rankr1", "model_name": "ielabgroup/Rank-R1-7B-v0.1",
        "extra_kwargs": {"num_child": 9, "k": 10, "sort_method": "heapsort", "num_permutation": 1, "max_tokens": 8000, "max_passage_length": 128, "num_gpus": 1, "gpu_memory_utilization": 0.4, "context_size": 32768},
    },
    "rankr1-14b": {
        "method": "rankr1", "model_name": "ielabgroup/Rank-R1-14B-v0.1",
        "extra_kwargs": {"num_child": 9, "k": 10, "sort_method": "heapsort", "num_permutation": 1, "max_tokens": 8000, "max_passage_length": 128, "num_gpus": 1, "gpu_memory_utilization": 0.7, "context_size": 32768},
    },

    # ==================== Setwise-SFT (Setwise HeapSort, no reasoning, LoRA on Qwen2.5) ====================
    "setwise-sft-7b": {
        "method": "rankr1", "model_name": "ielabgroup/Setwise-SFT-7B-v0.1",
        "extra_kwargs": {
            "num_child": 9, "k": 10, "sort_method": "heapsort", "num_permutation": 1,
            "max_tokens": 512, "max_passage_length": 128,
            "num_gpus": 1, "gpu_memory_utilization": 0.4, "context_size": 32768,
        },
    },
    "setwise-sft-14b": {
        "method": "rankr1", "model_name": "ielabgroup/Setwise-SFT-14B-v0.1",
        "extra_kwargs": {
            "num_child": 9, "k": 10, "sort_method": "heapsort", "num_permutation": 1,
            "max_tokens": 512, "max_passage_length": 128,
            "num_gpus": 2, "gpu_memory_utilization": 0.4, "context_size": 32768,
        },
    },

    # ==================== Rank1 (Pointwise + logprobs scoring) ====================
    "rank1-7b": {
        "method": "rank1", "model_name": "jhu-clsp/rank1-7b",
        "extra_kwargs": {"max_tokens": 8192, "max_passage_length": 512, "context_size": 16000, "num_gpus": 1, "gpu_memory_utilization": 0.4, "force_rethink": 0},
    },
    "rank1-32b": {
        "method": "rank1", "model_name": "jhu-clsp/rank1-32b",
        "extra_kwargs": {"max_tokens": 8192, "max_passage_length": 512, "context_size": 16000, "num_gpus": 4, "gpu_memory_utilization": 0.4, "force_rethink": 0},
    },

    # ==================== Rank-K (Listwise sliding window + ties) ====================
    "rankk-32b": {
        "method": "rankk", "model_name": "hltcoe/Rank-K-32B",
        "extra_kwargs": {"window_size": 20, "step_size": 10, "max_tokens": 4000, "max_passage_length": 300, "temperature": 0.7, "num_gpus": 4, "gpu_memory_utilization": 0.4, "context_size": 32768},
    },

    # ==================== Rearank (Listwise sliding window + think/answer) ====================
    "rearank-7b": {
        "method": "rearank", "model_name": "le723z/Rearank-7B",
        "extra_kwargs": {"window_size": 20, "step_size": 10, "max_tokens": 2048, "max_passage_length": 400, "num_gpus": 1, "gpu_memory_utilization": 0.4, "context_size": 32768, "enable_thinking": True},
    },
}

# Sorted list of available preset aliases (useful for logging / help text).
AVAILABLE_PRESETS = sorted(RERANKER_PRESETS.keys())
