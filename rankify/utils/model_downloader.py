"""
Model Downloader Utility

Provides automatic model downloading from HuggingFace or ModelScope with local caching.
If a model_name is not a local path, it will be downloaded to the appropriate cache directory.

Usage:
    from rankify.utils.model_downloader import resolve_model_path

    # Returns local path (downloads if needed)
    local_path = resolve_model_path("ielabgroup/Rank-R1-7B-v0.1")
    local_path = resolve_model_path("JohnnyFan/Rank4Gen-DPO-Qwen3-8B", source="modelscope")
"""

import os
from pathlib import Path
from typing import Optional


# Default cache directory for downloaded models (ModelScope style).
# Can be overridden via environment variable RANKIFY_MODEL_CACHE_DIR.
DEFAULT_CACHE_DIR = os.environ.get(
    "RANKIFY_MODEL_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "rankify", "models"),
)

# Known model aliases -> model IDs
MODEL_ALIASES = {
    # Rank4Gen (ModelScope)
    "rank4gen": "JohnnyFan/Rank4Gen-DPO-Qwen3-8B",
    "rank4gen-dpo-qwen3-8b": "JohnnyFan/Rank4Gen-DPO-Qwen3-8B",
    "Rank4Gen-DPO-Qwen3-8B": "JohnnyFan/Rank4Gen-DPO-Qwen3-8B",
    # SetR (ModelScope)
    "setr": "JohnnyFan/SETR-Qwen3-8B",
    "setr-qwen3-8b": "JohnnyFan/SETR-Qwen3-8B",
    "SETR-Qwen3-8B": "JohnnyFan/SETR-Qwen3-8B",
}

# Models that should be downloaded from ModelScope (not HuggingFace)
MODELSCOPE_MODELS = {
    "JohnnyFan/Rank4Gen-DPO-Qwen3-8B",
    "JohnnyFan/SETR-Qwen3-8B",
}


def resolve_model_path(
    model_name: str,
    cache_dir: Optional[str] = None,
    source: str = "auto",
) -> str:
    """
    Resolve a model name to a local directory path.

    If model_name is already a valid local directory, return it directly.
    Otherwise, download it from HuggingFace or ModelScope.

    Args:
        model_name: Model name, alias, or model ID (e.g. "ielabgroup/Rank-R1-7B-v0.1")
                    or a local path.
        cache_dir: Directory to store downloaded models.
                   For ModelScope: defaults to RANKIFY_MODEL_CACHE_DIR.
                   For HuggingFace: defaults to HF_HUB_CACHE or HF_HOME/hub.
        source: Download source. Options:
                - "auto": Try to detect; use ModelScope for known models, HuggingFace for others.
                - "huggingface" or "hf": Download from HuggingFace Hub.
                - "modelscope" or "ms": Download from ModelScope.

    Returns:
        Local path to the model directory.

    Raises:
        RuntimeError: If download fails from all sources.
        ValueError: If source is not supported.
    """
    # If it's already a local directory with model files, use directly
    if os.path.isdir(model_name):
        if _is_valid_model_dir(model_name):
            print(f"[ModelDownloader] Using local model directory: {model_name}")
            return model_name
        # Even if no config.json found, trust the user
        return model_name

    # Resolve alias
    model_id = MODEL_ALIASES.get(model_name, model_name)

    # Determine source automatically
    if source == "auto":
        if model_id in MODELSCOPE_MODELS:
            source = "modelscope"
        else:
            source = "huggingface"

    # Normalize source name
    source = source.lower()
    if source in ("hf", "huggingface"):
        return _resolve_huggingface(model_id, cache_dir)
    elif source in ("ms", "modelscope"):
        return _resolve_modelscope(model_id, cache_dir)
    else:
        raise ValueError(f"Unsupported download source: '{source}'. Supported: 'auto', 'huggingface', 'modelscope'")


def _resolve_huggingface(model_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Resolve model from HuggingFace Hub.
    First checks local cache, then downloads if not found.
    """
    # Determine HF cache directory
    hf_cache_dir = cache_dir
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
        )

    # Check if already in HF cache
    local_path = _find_in_hf_cache(model_id, hf_cache_dir)
    if local_path:
        print(f"[ModelDownloader] Found cached model in HF cache: {local_path}")
        return local_path

    # Try to download from HuggingFace
    print(f"[ModelDownloader] Model '{model_id}' not found in local cache.")
    print(f"[ModelDownloader] Attempting to download from HuggingFace Hub...")
    print(f"[ModelDownloader] HF cache directory: {hf_cache_dir}")

    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            model_id,
            cache_dir=hf_cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"[ModelDownloader] Download complete: {local_path}")
        return local_path
    except Exception as e:
        print(f"[ModelDownloader] HuggingFace download failed: {e}")
        # Fallback: try ModelScope
        print(f"[ModelDownloader] Falling back to ModelScope...")
        try:
            return _download_from_modelscope(model_id, cache_dir or DEFAULT_CACHE_DIR)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to download model '{model_id}' from both HuggingFace and ModelScope.\n"
                f"  HuggingFace error: {e}\n"
                f"  ModelScope error: {e2}\n"
                f"Please download the model manually and provide the local path."
            ) from e2


def _resolve_modelscope(model_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Resolve model from ModelScope.
    First checks local cache, then downloads if not found.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    os.makedirs(cache_dir, exist_ok=True)

    # Check if already downloaded (ModelScope style)
    local_model_dir = _get_local_model_dir_modelscope(model_id, cache_dir)
    if local_model_dir and _is_valid_model_dir(local_model_dir):
        print(f"[ModelDownloader] Found cached model: {local_model_dir}")
        return local_model_dir

    # Download from ModelScope
    return _download_from_modelscope(model_id, cache_dir)


def _find_in_hf_cache(model_id: str, hf_cache_dir: str) -> Optional[str]:
    """
    Find a model in the HuggingFace cache directory.

    HF cache structure:
        hub/models--{namespace}--{model_name}/snapshots/{commit_hash}/
    """
    # Convert model_id to HF cache directory name
    # e.g. "ielabgroup/Rank-R1-7B-v0.1" -> "models--ielabgroup--Rank-R1-7B-v0.1"
    cache_name = "models--" + model_id.replace("/", "--")
    model_cache_dir = os.path.join(hf_cache_dir, cache_name)

    if not os.path.isdir(model_cache_dir):
        return None

    # Find the latest snapshot
    snapshots_dir = os.path.join(model_cache_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    # Get all snapshot directories
    snapshots = [
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    ]

    if not snapshots:
        return None

    # Use refs/main to find the current snapshot, or use the latest one
    refs_main = os.path.join(model_cache_dir, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            target_hash = f.read().strip()
        target_path = os.path.join(snapshots_dir, target_hash)
        if os.path.isdir(target_path) and _is_valid_model_dir(target_path):
            return target_path

    # Fallback: use the first valid snapshot
    for snapshot in sorted(snapshots):
        candidate = os.path.join(snapshots_dir, snapshot)
        if _is_valid_model_dir(candidate):
            return candidate

    # Last resort: if any snapshot directory exists at all, return it
    # (some LoRA adapters may only have small files)
    if snapshots:
        return os.path.join(snapshots_dir, sorted(snapshots)[0])

    return None


def _is_valid_model_dir(path: str) -> bool:
    """Check if a directory looks like a valid model directory (full model or LoRA adapter)."""
    p = Path(path)
    if not p.is_dir():
        return False
    # Check for common model files (full models + LoRA adapters)
    indicators = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "tokenizer_config.json",
        # LoRA adapter indicators
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    return any((p / f).exists() for f in indicators)


def _get_local_model_dir_modelscope(model_id: str, cache_dir: str) -> Optional[str]:
    """
    Try to find an already-downloaded model in the ModelScope cache directory.

    ModelScope snapshot_download stores models at:
        cache_dir/<namespace>/<model_name>/
    """
    parts = model_id.split("/")
    if len(parts) == 2:
        candidate = os.path.join(cache_dir, parts[0], parts[1])
        if os.path.isdir(candidate):
            return candidate
    return None


def _download_from_modelscope(model_id: str, cache_dir: str) -> str:
    """Download model from ModelScope and return local path."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise RuntimeError(
            "modelscope package is required for model downloading. "
            "Install it with: pip install modelscope"
        )

    os.makedirs(cache_dir, exist_ok=True)

    print(f"[ModelDownloader] Downloading '{model_id}' from ModelScope...")
    print(f"[ModelDownloader] Cache directory: {cache_dir}")

    try:
        local_path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model '{model_id}' from ModelScope: {e}"
        ) from e

    print(f"[ModelDownloader] Download complete: {local_path}")
    return local_path
