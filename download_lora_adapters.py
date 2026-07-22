"""
Download the Rank-R1 and Setwise-SFT LoRA adapters (and their base models)
into the local HuggingFace cache.

The LoRA adapters are small (typically < 1GB) and download quickly. The
base model (Qwen2.5-7B-Instruct) is required for LoRA inference and is
downloaded on demand if it is not already cached.

Usage:
    python download_lora_adapters.py
"""

import os

os.environ["HF_HOME"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/huggingface"
os.environ["HF_HUB_CACHE"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/huggingface/hub"
# HF_TOKEN should be set in environment before running, e.g.: export HF_TOKEN="your_token_here"

from huggingface_hub import snapshot_download

# LoRA adapters to download
LORA_ADAPTERS = [
    "ielabgroup/Rank-R1-7B-v0.1",
    "ielabgroup/Rank-R1-14B-v0.1",
    "ielabgroup/Setwise-SFT-7B-v0.1",
    "ielabgroup/Setwise-SFT-14B-v0.1",
]

# Base models needed (download if not already cached)
BASE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",      # required by 7B adapters
    "Qwen/Qwen2.5-14B-Instruct",     # required by 14B adapters
]

# Other full models to download
FULL_MODELS = [
    "jhu-clsp/rank1-7b",
    "jhu-clsp/rank1-32b",
    "hltcoe/Rank-K-32B",
    "le723z/Rearank-7B",
    "liuwenhan/reasonrank-32B",
]

cache_dir = os.environ["HF_HUB_CACHE"]


def download_model(model_id: str, label: str = ""):
    """Download a model from HuggingFace Hub."""
    tag = f"[{label}] " if label else ""
    print(f"\n{'='*60}")
    print(f"{tag}Downloading: {model_id}")
    print(f"{'='*60}")
    try:
        local_path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"  ✓ Saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading LoRA Adapters and Models for Rankify")
    print(f"Cache directory: {cache_dir}")
    print("=" * 60)

    print("\n\n>>> Phase 1: LoRA Adapters (lightweight, ~100-500MB each)")
    for adapter in LORA_ADAPTERS:
        download_model(adapter, "LoRA")

    print("\n\n>>> Phase 2: Base Models (needed for LoRA inference)")
    for base in BASE_MODELS:
        cache_name = "models--" + base.replace("/", "--")
        if os.path.isdir(os.path.join(cache_dir, cache_name)):
            print(f"\n  ✓ Already cached: {base}")
        else:
            download_model(base, "Base")

    print("\n\n>>> Phase 3: Full Models (larger downloads)")
    for model in FULL_MODELS:
        cache_name = "models--" + model.replace("/", "--")
        if os.path.isdir(os.path.join(cache_dir, cache_name)):
            print(f"\n  ✓ Already cached: {model}")
        else:
            download_model(model, "Full")

    print("\n\n" + "=" * 60)
    print("Done! All models downloaded.")
    print("=" * 60)
