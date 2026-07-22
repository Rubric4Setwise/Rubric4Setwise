"""
Download Rank4Gen and SetR model weights from ModelScope.

This script uses the same resolve_model_path utility as the rerankers,
so downloaded models will be automatically found at runtime.

Usage:
    # Download to default cache (~/.cache/rankify/models):
    python download_models_modelscope.py

    # Download to custom directory:
    RANKIFY_MODEL_CACHE_DIR=/your/path python download_models_modelscope.py
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rankify.utils.model_downloader import resolve_model_path, DEFAULT_CACHE_DIR


def main():
    # Allow override via environment variable or command-line argument
    cache_dir = os.environ.get("RANKIFY_MODEL_CACHE_DIR", DEFAULT_CACHE_DIR)

    print("=" * 60)
    print("Rankify Model Downloader (ModelScope)")
    print(f"Cache directory: {cache_dir}")
    print("=" * 60)

    # Download Rank4Gen-DPO-Qwen3-8B
    print("\n[1/2] Downloading JohnnyFan/Rank4Gen-DPO-Qwen3-8B ...")
    rank4gen_path = resolve_model_path(
        "JohnnyFan/Rank4Gen-DPO-Qwen3-8B",
        cache_dir=cache_dir,
    )
    print(f"  -> Rank4Gen ready at: {rank4gen_path}")

    # Download SETR-Qwen3-8B
    print("\n[2/2] Downloading JohnnyFan/SETR-Qwen3-8B ...")
    setr_path = resolve_model_path(
        "JohnnyFan/SETR-Qwen3-8B",
        cache_dir=cache_dir,
    )
    print(f"  -> SetR ready at: {setr_path}")

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print(f"  Rank4Gen: {rank4gen_path}")
    print(f"  SetR:     {setr_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
