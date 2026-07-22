"""
Rankify - A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and RAG

Simple usage:
    >>> from rankify import pipeline
    >>> rag = pipeline("rag")
    >>> answers = rag("What is machine learning?", documents)
"""

import os
from pathlib import Path

# Set up cache directory - all model weights and datasets go here
DEFAULT_CACHE_DIR = "/cfs_cloud_code/jiangkailin/Rankify_model_data"
os.environ.setdefault("RERANKING_CACHE_DIR", DEFAULT_CACHE_DIR)

# Also redirect HuggingFace model downloads (from_pretrained) to the same base directory
HF_CACHE_DIR = os.path.join(DEFAULT_CACHE_DIR, "huggingface")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(DEFAULT_CACHE_DIR, "sentence_transformers"))

try:
    Path(os.environ["RERANKING_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
except (FileExistsError, OSError):
    pass

# Main pipeline interface
from rankify.pipeline import (
    Pipeline,
    pipeline,
    search_pipeline,
    rerank_pipeline,
    rag_pipeline,
    PipelineResult,
)

# Version
__version__ = "0.1.5"

__all__ = [
    # Pipeline (main interface)
    "Pipeline",
    "pipeline",
    "search_pipeline",
    "rerank_pipeline",
    "rag_pipeline",
    "PipelineResult",
    # Version
    "__version__",
]