"""
Rankify Reranker MCP API.

Wraps the Rankify library to provide reranking capabilities via MCP tools.
Supports all rankers: bge-reranker-large, rankllama, monot5, rankvicuna, rankzephyr,
rankgpt, rankt5, rankr1, rank1, rearank, reasonrank, setwise-sft, setr, rank4gen.
"""

import sys
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# Add Rankify to path
RANKIFY_PATH = os.environ.get(
    "RANKIFY_PATH",
    "../short/Rankify_only_ranker",
)
if RANKIFY_PATH not in sys.path:
    sys.path.insert(0, RANKIFY_PATH)


class RankifyRerankResultItem(BaseModel):
    """Single reranked document result."""

    index: int
    relevance_score: float
    text: str


class RankifyRerankerResult(BaseModel):
    """Complete reranker result."""

    method: str
    model_name: str
    preset: str
    result_type: str  # "ranking" or "selection"
    results: List[RankifyRerankResultItem]


# Global cache for loaded ranker instances to avoid reloading models
_ranker_cache: Dict[str, Any] = {}


def _get_or_create_ranker(preset_name: str):
    """
    Get a cached ranker instance or create a new one.
    Models are loaded lazily and cached for reuse.
    """
    if preset_name in _ranker_cache:
        return _ranker_cache[preset_name]

    from rankify.config.reranker_presets import RERANKER_PRESETS
    from rankify.models.reranking import Reranking

    if preset_name not in RERANKER_PRESETS:
        raise ValueError(
            f"Unknown ranker preset: '{preset_name}'. "
            f"Available presets: {sorted(RERANKER_PRESETS.keys())}"
        )

    preset = RERANKER_PRESETS[preset_name]
    extra_kwargs = preset.get("extra_kwargs", {})

    ranker = Reranking(
        method=preset["method"],
        model_name=preset["model_name"],
        **extra_kwargs,
    )

    _ranker_cache[preset_name] = ranker
    return ranker


# Set selection methods that return a subset rather than a full ranking
SET_SELECTION_PRESETS = {
    "setr",
    "rank4gen",
    "rubric4setwise",
    "rubric4setwise-llama8b",
}


def rankify_rerank(
    query: str,
    documents: List[str],
    preset: str,
    top_n: int = -1,
) -> RankifyRerankerResult:
    """
    Rerank documents using a Rankify ranker model.

    Args:
        query: The query string to rank documents against.
        documents: List of document texts to rerank.
        preset: Name of the ranker preset (e.g., "rankr1-7b", "bge-reranker-large").
        top_n: Number of top documents to return. -1 returns all.
               Ignored for set selection methods (setr, rank4gen) which select their own subset.

    Returns:
        RankifyRerankerResult with ranked/selected documents.
    """
    from rankify.config.reranker_presets import RERANKER_PRESETS
    from rankify.dataset.dataset import Answer, Context, Document, Question

    # Load or retrieve cached ranker
    ranker = _get_or_create_ranker(preset)
    preset_config = RERANKER_PRESETS[preset]

    # Determine if this is a set selection method
    is_set_selection = preset in SET_SELECTION_PRESETS

    # Build Rankify Document
    contexts = [
        Context(text=doc_text, id=str(i), score=0.0)
        for i, doc_text in enumerate(documents)
    ]
    rankify_doc = Document(
        question=Question(query),
        answers=Answer(),
        contexts=contexts,
    )

    # Execute ranking
    results = ranker.rank([rankify_doc])
    ranked_doc = results[0]

    # Extract results
    result_items = []

    if is_set_selection:
        # For set selection methods, use reorder_contexts (selected subset)
        selected_contexts = ranked_doc.reorder_contexts or []
        for rank_idx, ctx in enumerate(selected_contexts):
            original_index = int(ctx.id) if ctx.id is not None else rank_idx
            result_items.append(
                RankifyRerankResultItem(
                    index=original_index,
                    relevance_score=1.0 - (rank_idx * 0.01),  # Assign decreasing scores by position
                    text=ctx.text,
                )
            )
    else:
        # For ranking methods, use reorder_contexts with position-based scores
        ranked_contexts = ranked_doc.reorder_contexts or []
        total = len(ranked_contexts)
        for rank_idx, ctx in enumerate(ranked_contexts):
            original_index = int(ctx.id) if ctx.id is not None else rank_idx
            # Score: higher rank = higher score (normalized)
            score = (total - rank_idx) / total if total > 0 else 0.0
            # If context has a non-zero score from the ranker, use it
            if hasattr(ctx, "score") and ctx.score and ctx.score != 0.0:
                score = ctx.score
            result_items.append(
                RankifyRerankResultItem(
                    index=original_index,
                    relevance_score=score,
                    text=ctx.text,
                )
            )

    # Apply top_n filtering (only for ranking methods)
    if not is_set_selection and top_n > 0:
        result_items = result_items[:top_n]

    return RankifyRerankerResult(
        method=preset_config["method"],
        model_name=preset_config["model_name"],
        preset=preset,
        result_type="selection" if is_set_selection else "ranking",
        results=result_items,
    )


def get_available_presets() -> Dict[str, Dict[str, str]]:
    """Return available ranker presets with their info."""
    from rankify.config.reranker_presets import RERANKER_PRESETS

    info = {}
    for name, config in RERANKER_PRESETS.items():
        info[name] = {
            "method": config["method"],
            "model_name": config["model_name"],
            "type": "selection" if name in SET_SELECTION_PRESETS else "ranking",
        }
    return info
