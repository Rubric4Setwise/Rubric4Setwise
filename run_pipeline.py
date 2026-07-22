"""
Rankify JSONL Pipeline v3 - Multi-GPU Data-Parallel Scheduling
==============================================================
Core Design:
1. Ranking and Generation are strictly separated into phases
2. Ranking phase: for each ranker, split data into N shards (N = num GPUs),
   each GPU runs the same ranker on its own shard in parallel, then merge
3. Generation phase: after all ranking tasks complete, schedule generation
   tasks per GPU
4. Driven via CLI arguments + bash script invocations

Supported Rankers (14 total):
    bge-reranker-large, rankllama, rankvicuna, rankzephyr,
    rankgpt, monot5, rankt5,
    rank4gen, setr, reasonrank-7b, rearank-7b,
    setwise-sft-7b, rank1-7b, rankr1-7b

Usage:
    # Single ranking task (scheduled by bash script)
    python run_pipeline.py rank --reranker rankllama --gpu 0

    # Data-parallel ranking (shard mode)
    python run_pipeline.py rank --reranker rankllama --gpu 0 --shard-id 0 --num-shards 4

    # Merge shards after all parallel ranking completes
    python run_pipeline.py merge-shards --reranker rankllama --num-shards 4

    # Single generation task
    python run_pipeline.py generate --reranker rankllama --generator meta-llama/Llama-3.1-8B-Instruct --gpu 0

    # Summarize results
    python run_pipeline.py summarize
"""

import argparse
import json
import math
import os
import gc
import sys
import time
from typing import List, Dict, Any

# ============================================================
# CRITICAL: Set CUDA_VISIBLE_DEVICES before importing torch/vllm
# Parse --gpu argument early to ensure GPU isolation
# ============================================================
def _early_set_gpu():
    """Parse --gpu from sys.argv before any CUDA library is loaded."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            # Multi-GPU mode: ensure NCCL communication config
            if "," in gpu_id:
                os.environ.setdefault("NCCL_P2P_DISABLE", "1")
                os.environ.setdefault("NCCL_IB_DISABLE", "1")
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
            return

_early_set_gpu()

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.models.reranking import Reranking
from rankify.generator.generator import Generator
from rankify.metrics.metrics import Metrics

# ============================================================
# HuggingFace cache directory & token authentication
# ============================================================
os.environ["HF_HOME"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/huggingface"
os.environ["HF_HUB_CACHE"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/huggingface/hub"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["VLLM_USE_V1"] = "0"

# ============================================================
# Default configuration (can be overridden via CLI arguments)
# ============================================================
INPUT_FILE = "/cfs_cloud_code/jiangkailin/Setwise/processed_data/short_closed/rankify_short_closed_1000case.jsonl"
OUTPUT_DIR = "/cfs_cloud_code/jiangkailin/Setwise/ranker_output/baseline_ranker_all"
OUTPUT_FILENAME = "output.jsonl"
NUM_ENTRIES = "all"  # "all" or integer

# Reranker config
from rankify.config.reranker_presets import RERANKER_PRESETS

ALL_RERANKERS = [
    "bge-reranker-large", "rankllama", "rankvicuna", "rankzephyr",
    "rankgpt", "monot5", "rankt5",
    "rank4gen", "setr", "reasonrank-7b", "rearank-7b",
    "setwise-sft-7b", "rank1-7b", "rankr1-7b",
    "rubric4setwise",
]

ALL_GENERATORS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B"
]

# Generator config (defaults, overridable via CLI)
DEFAULT_GENERATOR_METHOD = "basic-rag"
DEFAULT_GENERATOR_BACKEND = "vllm"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTILIZATION = 0.4

# Pipeline config (defaults, overridable via CLI)
DEFAULT_TOP_K = 5
SET_SELECTION_METHODS = {"rank4gen", "setr", "rubric4setwise"}
BM25_BASELINE = "bm25-baseline"

# Evaluation config
DEFAULT_EVALUATE_RANKER = True
DEFAULT_TOP_K_FOR_EVAL = [1, 3, 5, 10]
DEFAULT_NDCG_CUTS = [3, 5, 10]
DEFAULT_EVALUATE_METRICS = True
DEFAULT_SAVE_INDIVIDUAL_SCORES = True

# vLLM sampling params (defaults, overridable via CLI)
DEFAULT_SAMPLING_PARAMS = {
    "max_tokens": 20,
    "temperature": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.3,
    "stop": ["\n", "(", "Note:", "Based on", "."],
}


# ============================================================
# Utility functions
# ============================================================
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, cls=NumpyEncoder) + "\n")


def convert_entry_to_document(entry: Dict[str, Any]) -> Document:
    """Convert a JSONL entry to a Rankify Document object."""
    question_text = entry["question"]
    # `answer` may be a string (single gold), a list of strings, or missing.
    raw_answer = entry.get("answer", entry.get("answers", ""))
    if isinstance(raw_answer, str):
        answer_text = [raw_answer] if raw_answer else []
    elif isinstance(raw_answer, list):
        answer_text = raw_answer
    else:
        answer_text = [str(raw_answer)]

    docs_list = entry["docs"]
    contexts = []
    for doc_item in docs_list:
        contexts.append(Context(
            id=str(doc_item.get("original_id", doc_item.get("id", ""))),
            title=doc_item.get("title", ""),
            text=doc_item.get("doc", doc_item.get("text", "")),
            score=float(doc_item.get("bm25_score", doc_item.get("score", 0.0)) or 0.0),
        ))

    # Legacy fallback: some datasets store titles at entry["context"]["title"].
    if "context" in entry and isinstance(entry["context"], dict) and "title" in entry["context"]:
        titles = entry["context"]["title"]
        for i, ctx in enumerate(contexts):
            if i < len(titles) and not ctx.title:
                ctx.title = titles[i]

    doc = Document(
        question=Question(question_text),
        answers=Answer(answer_text),
        contexts=contexts
    )
    return doc


def get_process_data(input_file: str, num_entries: str) -> List[Dict[str, Any]]:
    """Load and truncate data."""
    raw_data = load_jsonl(input_file)
    if num_entries == "all":
        return raw_data
    else:
        return raw_data[:int(num_entries)]


def resolve_reranker_meta(reranker_choice: str):
    """Resolve reranker method metadata.

    bm25-baseline is not a real reranker model: it uses BM25 top-K docs directly.
    """
    if reranker_choice == BM25_BASELINE:
        return BM25_BASELINE, False
    _reranker_cfg = RERANKER_PRESETS[reranker_choice]
    reranker_method = _reranker_cfg["method"]
    return reranker_method, reranker_method in SET_SELECTION_METHODS


def apply_bm25_baseline(entry: Dict[str, Any], top_k: int):
    """Use original BM25 order as ranking result; top-K docs go to generation."""
    docs = entry.get("docs", [])
    entry["ranked_docs"] = []
    for rank, doc in enumerate(docs, start=1):
        entry["ranked_docs"].append({
            "rank": rank,
            "original_id": doc.get("original_id"),
            "score": doc.get("bm25_score", 0),
            "doc": doc.get("doc", ""),
        })
    entry["context_for_generator"] = [
        {"original_id": doc.get("original_id"), "doc": doc.get("doc", "")}
        for doc in docs[:top_k]
    ]


def cleanup_gpu():
    """Thoroughly clean up GPU memory."""
    gc.collect()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# Ranking Phase
# ============================================================
def run_ranking(reranker_choice: str, gpu_id, input_file: str,
                output_dir: str, num_entries: str, top_k: int = None,
                evaluate_ranker: bool = None, top_k_for_eval: List[int] = None,
                ndcg_cuts: List[int] = None,
                shard_id: int = None, num_shards: int = None):
    """
    Execute a single reranker ranking task on the specified GPU.
    Results are saved as ranker_cache.jsonl.

    Supports data-parallel mode:
        When both shard_id and num_shards are specified, only processes
        the [shard_id/num_shards] data partition.
        Results are saved to ranker_cache_shard_{shard_id}.jsonl.
        After all shards complete, use merge-shards to combine them.

    Args:
        gpu_id: Single GPU id (int) or multiple GPU ids (str, e.g. "0,1")
        shard_id: Data shard ID (0-indexed)
        num_shards: Total number of shards
    """
    # Apply defaults
    top_k = top_k if top_k is not None else DEFAULT_TOP_K
    evaluate_ranker = evaluate_ranker if evaluate_ranker is not None else DEFAULT_EVALUATE_RANKER
    top_k_for_eval = top_k_for_eval if top_k_for_eval is not None else DEFAULT_TOP_K_FOR_EVAL
    ndcg_cuts = ndcg_cuts if ndcg_cuts is not None else DEFAULT_NDCG_CUTS

    # Set GPU
    gpu_str = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    num_visible_gpus = len(gpu_str.split(","))
    print(f"\n{'=' * 60}")
    print(f"[RANK] Reranker: {reranker_choice} | GPU: {gpu_str} ({num_visible_gpus} GPU(s))")
    print(f"[RANK] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"{'=' * 60}")

    reranker_method, is_set_selection = resolve_reranker_meta(reranker_choice)
    is_bm25_baseline = reranker_choice == BM25_BASELINE

    # Output paths
    ranker_cache_dir = os.path.join(output_dir, reranker_choice)
    os.makedirs(ranker_cache_dir, exist_ok=True)

    # Data-parallel mode: use shard-specific filename
    is_sharded = (shard_id is not None and num_shards is not None)
    if is_sharded:
        ranker_cache_file = os.path.join(ranker_cache_dir, f"ranker_cache_shard_{shard_id}.jsonl")
        print(f"  [DATA-PARALLEL] Shard {shard_id}/{num_shards}, cache: {ranker_cache_file}")
    else:
        ranker_cache_file = os.path.join(ranker_cache_dir, "ranker_cache.jsonl")

    # Check if cache already exists
    if os.path.exists(ranker_cache_file):
        print(f"  [SKIP] Cache already exists: {ranker_cache_file}")
        print(f"  To re-run ranking, delete this file first.")
        return

    # Load data
    print(f"  Loading data: {input_file}")
    process_data = get_process_data(input_file, num_entries)
    total_data_count = len(process_data)
    print(f"  Total {total_data_count} entries")

    # Data sharding: only take the portion belonging to this shard
    if is_sharded:
        shard_size = math.ceil(total_data_count / num_shards)
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_data_count)
        process_data = process_data[start_idx:end_idx]
        print(f"  [DATA-PARALLEL] Processing [{start_idx}:{end_idx}] = {len(process_data)} entries (Shard {shard_id})")

    if is_bm25_baseline:
        print(f"  [BM25-BASELINE] Using original BM25 order, top-{top_k} docs for generation (no reranker)")
        for entry in process_data:
            apply_bm25_baseline(entry, top_k)
        reranked_documents = None
    else:
        _reranker_cfg = RERANKER_PRESETS[reranker_choice]
        reranker_model = _reranker_cfg["model_name"]

        # Convert to Document objects
        print(f"  Converting to Document objects...")
        documents = [convert_entry_to_document(entry) for entry in process_data]
        print(f"  Converted {len(documents)} documents")

        # Attach pre-computed rubric to documents. Only `rubric4setwise` consumes
        # `document.rubric`; every other reranker ignores it, so we simply skip
        # this step for them (their input JSONL is not required to carry rubrics).
        if reranker_method == "rubric4setwise":
            L1_TYPES = {"Relevance", "Authenticity", "Quality"}
            L2_TYPES = {"Complementarity", "Redundancy", "Conflict"}
            L3_TYPES = {"Completeness", "Density", "Reachability"}
            rubric_count = 0
            for doc, entry in zip(documents, process_data):
                if "hybrid_rubrics" in entry and isinstance(entry["hybrid_rubrics"], dict):
                    rubric_list = []
                    for dim_name, dim_data in entry["hybrid_rubrics"].items():
                        rubrics = dim_data.get("rubrics", []) if isinstance(dim_data, dict) else []
                        if dim_name in L1_TYPES:
                            level = "L1"
                        elif dim_name in L2_TYPES:
                            level = "L2"
                        elif dim_name in L3_TYPES:
                            level = "L3"
                        else:
                            level = "L1"
                        for r in rubrics:
                            if r and len(r.strip()) > 5:
                                rubric_list.append({"level": level, "type": dim_name, "item": r.strip()})
                    doc.rubric = rubric_list
                    rubric_count += 1
                elif "rubric" in entry:
                    doc.rubric = entry["rubric"]
                    rubric_count += 1
            print(f"  [rubric4setwise] Attached rubric to {rubric_count}/{len(documents)} documents")
            if rubric_count == 0:
                print("  [rubric4setwise] WARNING: no `hybrid_rubrics` / `rubric` field found in the input JSONL.")

        # Initialize reranker and execute
        print(f"  Initializing Reranker: {reranker_method} / {reranker_model}")
        _extra = dict(_reranker_cfg.get("extra_kwargs", {}))

        # Auto-adapt multi-GPU: if CUDA_VISIBLE_DEVICES has multiple GPUs, override num_gpus
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            actual_num_gpus = len(cuda_visible.split(","))
            preset_num_gpus = _extra.get("num_gpus", 1)
            if actual_num_gpus != preset_num_gpus:
                print(f"  [GPU] Auto-adapt: preset num_gpus={preset_num_gpus} -> actual visible GPUs={actual_num_gpus}")
                _extra["num_gpus"] = actual_num_gpus

        reranker = Reranking(
            method=reranker_method,
            model_name=reranker_model,
            **_extra
        )

        print(f"  Reranking in progress...")
        start_time = time.time()
        reranked_documents = reranker.rank(documents)
        elapsed = time.time() - start_time
        print(f"  Reranking done! Elapsed: {elapsed:.1f}s ({elapsed/len(documents):.2f}s/sample)")

        # Release reranker
        del reranker
        cleanup_gpu()
        print(f"  Released Reranker GPU memory.")

        # Save ranking results
        print(f"  Saving ranking results...")
        for entry, doc in zip(process_data, reranked_documents):
            ranked_ctxs = doc.reorder_contexts if doc.reorder_contexts else doc.contexts

            if is_set_selection:
                entry["selected_docs"] = []
                for rank_pos, ctx in enumerate(ranked_ctxs):
                    entry["selected_docs"].append({
                        "rank": rank_pos + 1,
                        "original_id": int(ctx.id) if ctx.id is not None else None,
                        "score": ctx.score,
                        "doc": ctx.text
                    })
                entry["context_for_generator"] = [
                    {"original_id": int(ctx.id) if ctx.id is not None else None, "doc": ctx.text}
                    for ctx in ranked_ctxs
                ]
            else:
                entry["ranked_docs"] = []
                for rank_pos, ctx in enumerate(ranked_ctxs):
                    entry["ranked_docs"].append({
                        "rank": rank_pos + 1,
                        "original_id": int(ctx.id) if ctx.id is not None else None,
                        "score": ctx.score,
                        "doc": ctx.text
                    })
                entry["context_for_generator"] = [
                    {"original_id": int(ctx.id) if ctx.id is not None else None, "doc": ctx.text}
                    for ctx in ranked_ctxs[:top_k]
                ]

            # Save ranker LLM raw outputs (if any)
            if hasattr(doc, 'ranker_raw_outputs') and doc.ranker_raw_outputs:
                entry["ranker_raw_outputs"] = doc.ranker_raw_outputs

    # ============ Ranker Evaluation ============
    # In shard mode, skip evaluation (will be done after merging)
    if is_sharded:
        evaluate_ranker = False
        print(f"  [DATA-PARALLEL] Shard mode, skipping evaluation (will evaluate after merge)")
    elif is_bm25_baseline and evaluate_ranker:
        evaluate_ranker = False
        print(f"  [BM25-BASELINE] Skipping in-process evaluation (use merge-shards --evaluate-ranker for metrics)")

    if evaluate_ranker:
        print(f"  Computing Ranker evaluation metrics...")
        # Prepare relevance annotations
        sample_relevance = []
        for entry in process_data:
            supporting_titles = set()
            if "supporting_facts" in entry and "title" in entry["supporting_facts"]:
                supporting_titles = set(entry["supporting_facts"]["title"])
            context_titles = []
            if "context" in entry and "title" in entry["context"]:
                context_titles = entry["context"]["title"]

            relevance_map = {}
            for doc_item in entry["docs"]:
                original_idx = doc_item["original_id"]
                doc_text = doc_item["doc"]
                has_answer = False

                if supporting_titles and context_titles:
                    if original_idx - 1 < len(context_titles) and original_idx >= 1:
                        doc_title = context_titles[original_idx - 1]
                        doc_title_lower = doc_title.lower()
                        for sf_title in supporting_titles:
                            sf_title_lower = sf_title.lower()
                            if sf_title_lower == doc_title_lower or \
                               sf_title_lower in doc_title_lower or \
                               doc_title_lower in sf_title_lower:
                                has_answer = True
                                break

                if not supporting_titles:
                    correct_answer = entry.get("answer", "")
                    if correct_answer:
                        answer_lower = correct_answer.lower()
                        doc_lower = doc_text.lower()
                        if answer_lower in doc_lower:
                            has_answer = True
                        elif len(answer_lower) > 3:
                            answer_words = answer_lower.split()
                            for word in answer_words:
                                if len(word) > 3 and word in doc_lower:
                                    has_answer = True
                                    break

                relevance_map[original_idx] = has_answer
            sample_relevance.append(relevance_map)

        if is_set_selection:
            all_precisions, all_recalls, all_f1s, all_hit_rates, all_num_selected = [], [], [], [], []
            for i, (entry, doc) in enumerate(zip(process_data, reranked_documents)):
                relevance_map = sample_relevance[i]
                ranked_ctxs = doc.reorder_contexts if doc.reorder_contexts else doc.contexts
                selected_ids = set()
                for ctx in ranked_ctxs:
                    if ctx.id is not None:
                        selected_ids.add(int(ctx.id))
                relevant_ids = set(doc_id for doc_id, rel in relevance_map.items() if rel)

                if len(selected_ids) > 0:
                    true_positives = len(selected_ids & relevant_ids)
                    precision = true_positives / len(selected_ids)
                else:
                    true_positives = 0
                    precision = 0.0

                recall = true_positives / len(relevant_ids) if len(relevant_ids) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                hit = 1.0 if true_positives > 0 else 0.0

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                all_hit_rates.append(hit)
                all_num_selected.append(len(selected_ids))

                entry["ranker_evaluation"] = {
                    "method_type": "set_selection",
                    "num_selected": len(selected_ids),
                    "num_relevant": len(relevant_ids),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "hit": hit == 1.0,
                }

            # Count docs used per entry (set-selection mode uses len(selected_docs)).
            all_doc_counts = []
            for entry in process_data:
                doc_count = len(entry.get("context_for_generator", []))
                all_doc_counts.append(doc_count)

            n = len(process_data)
            eval_summary = {
                "method": reranker_method,
                "method_type": "set_selection",
                "avg_precision": sum(all_precisions) / n,
                "avg_recall": sum(all_recalls) / n,
                "avg_f1": sum(all_f1s) / n,
                "hit_rate": sum(all_hit_rates) / n,
                "avg_num_selected": sum(all_num_selected) / n,
                "avg_doc_count": sum(all_doc_counts) / n if n > 0 else 0,
            }
            print(f"    Precision={eval_summary['avg_precision']:.4f}, Recall={eval_summary['avg_recall']:.4f}, "
                  f"F1={eval_summary['avg_f1']:.4f}, HitRate={eval_summary['hit_rate']:.4f}, "
                  f"AvgDocCount={eval_summary['avg_doc_count']:.2f}")
        else:
            # Traditional ranking evaluation (simplified NDCG)
            ndcg_results = {}
            for k in ndcg_cuts:
                ndcg_scores_per_k = []
                for i, (entry, doc) in enumerate(zip(process_data, reranked_documents)):
                    relevance_map = sample_relevance[i]
                    reranked_ctxs = doc.reorder_contexts if doc.reorder_contexts else doc.contexts
                    all_ctxs = documents[i].contexts

                    actual_dcg = 0.0
                    for pos, ctx in enumerate(reranked_ctxs[:k]):
                        rel = 1.0 if relevance_map.get(int(ctx.id) if ctx.id else -1, False) else 0.0
                        actual_dcg += rel / math.log2(pos + 2)

                    ideal_sorted = sorted(all_ctxs, key=lambda x: relevance_map.get(int(x.id) if x.id else -1, False), reverse=True)
                    ideal_dcg = 0.0
                    for pos, ctx in enumerate(ideal_sorted[:k]):
                        rel = 1.0 if relevance_map.get(int(ctx.id) if ctx.id else -1, False) else 0.0
                        ideal_dcg += rel / math.log2(pos + 2)

                    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                    ndcg_scores_per_k.append(ndcg)

                ndcg_results[f"ndcg@{k}"] = sum(ndcg_scores_per_k) / len(ndcg_scores_per_k)

            # Top-K accuracy
            top_k_after = {}
            for k_val in top_k_for_eval:
                hits = 0
                for i, doc in enumerate(reranked_documents):
                    reranked_ctxs = doc.reorder_contexts if doc.reorder_contexts else doc.contexts
                    for ctx in reranked_ctxs[:k_val]:
                        if sample_relevance[i].get(int(ctx.id) if ctx.id else -1, False):
                            hits += 1
                            break
                top_k_after[f"top_{k_val}"] = hits / len(process_data) * 100

            # Count docs used per entry (ranking mode uses len(context_for_generator)).
            all_doc_counts = []
            for entry_item in process_data:
                doc_count = len(entry_item.get("context_for_generator", []))
                all_doc_counts.append(doc_count)
            n_docs = len(all_doc_counts)

            for i, entry in enumerate(process_data):
                entry["ranker_evaluation"] = {
                    "method_type": "ranking",
                    "ndcg": {metric: ndcg_results[metric] for metric in ndcg_results},
                    "top_k_accuracy": top_k_after,
                }

            eval_summary = {
                "method": reranker_method,
                "method_type": "ranking",
                "ndcg": ndcg_results,
                "top_k_accuracy": top_k_after,
                "avg_doc_count": sum(all_doc_counts) / n_docs if n_docs > 0 else 0,
            }
            print(f"    NDCG: {ndcg_results}")
            print(f"    Top-K: {top_k_after}")
            print(f"    AvgDocCount: {eval_summary['avg_doc_count']:.2f}")

        # Save evaluation file
        eval_file = os.path.join(ranker_cache_dir, "ranker_eval.json")
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=2)

    # Save ranker cache
    ranker_cache_data = []
    for entry in process_data:
        cache_entry = {
            "question": entry["question"],
            "answer": entry.get("answer", ""),
        }
        if is_set_selection:
            cache_entry["selected_docs"] = entry.get("selected_docs", [])
        else:
            cache_entry["ranked_docs"] = entry.get("ranked_docs", [])
        cache_entry["context_for_generator"] = entry.get("context_for_generator", [])
        if "ranker_raw_outputs" in entry:
            cache_entry["ranker_raw_outputs"] = entry["ranker_raw_outputs"]
        if "ranker_evaluation" in entry:
            cache_entry["ranker_evaluation"] = entry["ranker_evaluation"]
        ranker_cache_data.append(cache_entry)

    save_jsonl(ranker_cache_data, ranker_cache_file)
    print(f"  ✓ Saved {len(ranker_cache_data)} ranker cache entries to: {ranker_cache_file}")
    print(f"  [DONE] {reranker_choice} ranking complete!")


# ============================================================
# Merge Shards
# ============================================================
def merge_shards(reranker_choice: str, output_dir: str, num_shards: int,
                 cleanup: bool = True, input_file: str = None,
                 num_entries: str = None,
                 evaluate_ranker: bool = None,
                 top_k_for_eval: List[int] = None,
                 ndcg_cuts: List[int] = None,
                 top_k: int = None):
    """
    Merge data-parallel shard results into a complete ranker_cache.jsonl.
    After merging, run ranker evaluation on the full dataset.

    Args:
        reranker_choice: Reranker name
        output_dir: Output directory
        num_shards: Total number of shards
        cleanup: Whether to delete shard files after merging
        input_file: Input data file (needed for evaluation)
        num_entries: Number of entries
        evaluate_ranker: Whether to run evaluation after merge
        top_k_for_eval: Top-K values for evaluation
        ndcg_cuts: NDCG cut values
        top_k: Top-K for context selection
    """
    # Apply defaults
    evaluate_ranker = evaluate_ranker if evaluate_ranker is not None else DEFAULT_EVALUATE_RANKER
    top_k_for_eval = top_k_for_eval if top_k_for_eval is not None else DEFAULT_TOP_K_FOR_EVAL
    ndcg_cuts = ndcg_cuts if ndcg_cuts is not None else DEFAULT_NDCG_CUTS
    top_k = top_k if top_k is not None else DEFAULT_TOP_K
    input_file = input_file or INPUT_FILE
    num_entries = num_entries or NUM_ENTRIES

    ranker_cache_dir = os.path.join(output_dir, reranker_choice)
    ranker_cache_file = os.path.join(ranker_cache_dir, "ranker_cache.jsonl")

    if os.path.exists(ranker_cache_file):
        print(f"  [SKIP] Merged file already exists: {ranker_cache_file}")
        return

    print(f"\n{'=' * 60}")
    print(f"[MERGE] Merging {num_shards} shards -> {ranker_cache_file}")
    print(f"{'=' * 60}")

    merged_data = []
    missing_shards = []
    for sid in range(num_shards):
        shard_file = os.path.join(ranker_cache_dir, f"ranker_cache_shard_{sid}.jsonl")
        if not os.path.exists(shard_file):
            missing_shards.append(sid)
            continue
        shard_data = load_jsonl(shard_file)
        print(f"  Shard {sid}: {len(shard_data)} entries")
        merged_data.extend(shard_data)

    if missing_shards:
        print(f"  [ERROR] Missing shards: {missing_shards}")
        print(f"  Please wait for all shards to complete before merging.")
        return

    # Run evaluation on merged data
    reranker_method, is_set_selection = resolve_reranker_meta(reranker_choice)

    if evaluate_ranker:
        print(f"  Running evaluation on merged data ({len(merged_data)} entries)...")
        # Load original data for relevance annotations
        process_data = get_process_data(input_file, num_entries)

        # Prepare relevance annotations
        sample_relevance = []
        for entry in process_data:
            supporting_titles = set()
            if "supporting_facts" in entry and "title" in entry["supporting_facts"]:
                supporting_titles = set(entry["supporting_facts"]["title"])
            context_titles = []
            if "context" in entry and "title" in entry["context"]:
                context_titles = entry["context"]["title"]

            relevance_map = {}
            for doc_item in entry["docs"]:
                original_idx = doc_item["original_id"]
                doc_text = doc_item["doc"]
                has_answer = False

                if supporting_titles and context_titles:
                    if original_idx - 1 < len(context_titles) and original_idx >= 1:
                        doc_title = context_titles[original_idx - 1]
                        doc_title_lower = doc_title.lower()
                        for sf_title in supporting_titles:
                            sf_title_lower = sf_title.lower()
                            if sf_title_lower == doc_title_lower or \
                               sf_title_lower in doc_title_lower or \
                               doc_title_lower in sf_title_lower:
                                has_answer = True
                                break

                if not supporting_titles:
                    correct_answer = entry.get("answer", "")
                    if correct_answer:
                        answer_lower = correct_answer.lower()
                        doc_lower = doc_text.lower()
                        if answer_lower in doc_lower:
                            has_answer = True
                        elif len(answer_lower) > 3:
                            answer_words = answer_lower.split()
                            for word in answer_words:
                                if len(word) > 3 and word in doc_lower:
                                    has_answer = True
                                    break

                relevance_map[original_idx] = has_answer
            sample_relevance.append(relevance_map)

        if is_set_selection:
            all_precisions, all_recalls, all_f1s, all_hit_rates, all_num_selected = [], [], [], [], []
            for i, cached_entry in enumerate(merged_data):
                if i >= len(sample_relevance):
                    break
                relevance_map = sample_relevance[i]
                selected = cached_entry.get("selected_docs", [])
                selected_ids = set()
                for d in selected:
                    if d.get("original_id") is not None:
                        selected_ids.add(int(d["original_id"]))
                relevant_ids = set(doc_id for doc_id, rel in relevance_map.items() if rel)

                if len(selected_ids) > 0:
                    true_positives = len(selected_ids & relevant_ids)
                    precision = true_positives / len(selected_ids)
                else:
                    true_positives = 0
                    precision = 0.0

                recall = true_positives / len(relevant_ids) if len(relevant_ids) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                hit = 1.0 if true_positives > 0 else 0.0

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                all_hit_rates.append(hit)
                all_num_selected.append(len(selected_ids))

                merged_data[i]["ranker_evaluation"] = {
                    "method_type": "set_selection",
                    "num_selected": len(selected_ids),
                    "num_relevant": len(relevant_ids),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "hit": hit == 1.0,
                }

            # Count docs used per entry.
            all_merge_doc_counts = []
            for cached_entry in merged_data:
                doc_count = len(cached_entry.get("context_for_generator", cached_entry.get("selected_docs", [])))
                all_merge_doc_counts.append(doc_count)

            n = len(all_precisions)
            eval_summary = {
                "method": reranker_method,
                "method_type": "set_selection",
                "avg_precision": sum(all_precisions) / n if n > 0 else 0,
                "avg_recall": sum(all_recalls) / n if n > 0 else 0,
                "avg_f1": sum(all_f1s) / n if n > 0 else 0,
                "hit_rate": sum(all_hit_rates) / n if n > 0 else 0,
                "avg_num_selected": sum(all_num_selected) / n if n > 0 else 0,
                "avg_doc_count": sum(all_merge_doc_counts) / len(all_merge_doc_counts) if all_merge_doc_counts else 0,
            }
            print(f"    Precision={eval_summary['avg_precision']:.4f}, Recall={eval_summary['avg_recall']:.4f}, "
                  f"F1={eval_summary['avg_f1']:.4f}, HitRate={eval_summary['hit_rate']:.4f}, "
                  f"AvgDocCount={eval_summary['avg_doc_count']:.2f}")
        else:
            # Traditional ranking evaluation
            ndcg_results = {}
            for k in ndcg_cuts:
                ndcg_scores_per_k = []
                for i, cached_entry in enumerate(merged_data):
                    if i >= len(sample_relevance):
                        break
                    relevance_map = sample_relevance[i]
                    ranked = cached_entry.get("ranked_docs", [])

                    actual_dcg = 0.0
                    for pos, d in enumerate(ranked[:k]):
                        doc_id = d.get("original_id", -1)
                        rel = 1.0 if relevance_map.get(doc_id, False) else 0.0
                        actual_dcg += rel / math.log2(pos + 2)

                    # Ideal DCG from original data
                    if i < len(process_data):
                        all_doc_ids = [d["original_id"] for d in process_data[i]["docs"]]
                        ideal_sorted = sorted(all_doc_ids, key=lambda x: relevance_map.get(x, False), reverse=True)
                        ideal_dcg = 0.0
                        for pos, doc_id in enumerate(ideal_sorted[:k]):
                            rel = 1.0 if relevance_map.get(doc_id, False) else 0.0
                            ideal_dcg += rel / math.log2(pos + 2)
                    else:
                        ideal_dcg = 0.0

                    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                    ndcg_scores_per_k.append(ndcg)

                ndcg_results[f"ndcg@{k}"] = sum(ndcg_scores_per_k) / len(ndcg_scores_per_k) if ndcg_scores_per_k else 0

            top_k_after = {}
            for k_val in top_k_for_eval:
                hits = 0
                for i, cached_entry in enumerate(merged_data):
                    if i >= len(sample_relevance):
                        break
                    ranked = cached_entry.get("ranked_docs", [])
                    for d in ranked[:k_val]:
                        doc_id = d.get("original_id", -1)
                        if sample_relevance[i].get(doc_id, False):
                            hits += 1
                            break
                n = min(len(merged_data), len(sample_relevance))
                top_k_after[f"top_{k_val}"] = hits / n * 100 if n > 0 else 0

            # Count docs used per entry.
            all_merge_doc_counts = []
            for cached_entry in merged_data:
                doc_count = len(cached_entry.get("context_for_generator", []))
                all_merge_doc_counts.append(doc_count)

            for i, cached_entry in enumerate(merged_data):
                cached_entry["ranker_evaluation"] = {
                    "method_type": "ranking",
                    "ndcg": {metric: ndcg_results[metric] for metric in ndcg_results},
                    "top_k_accuracy": top_k_after,
                }

            eval_summary = {
                "method": reranker_method,
                "method_type": "ranking",
                "ndcg": ndcg_results,
                "top_k_accuracy": top_k_after,
                "avg_doc_count": sum(all_merge_doc_counts) / len(all_merge_doc_counts) if all_merge_doc_counts else 0,
            }
            print(f"    NDCG: {ndcg_results}")
            print(f"    Top-K: {top_k_after}")
            print(f"    AvgDocCount: {eval_summary['avg_doc_count']:.2f}")

        # Save evaluation file
        eval_file = os.path.join(ranker_cache_dir, "ranker_eval.json")
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved evaluation: {eval_file}")

    save_jsonl(merged_data, ranker_cache_file)
    print(f"  ✓ Merge complete! {len(merged_data)} entries -> {ranker_cache_file}")

    # Cleanup shard files
    if cleanup:
        for sid in range(num_shards):
            shard_file = os.path.join(ranker_cache_dir, f"ranker_cache_shard_{sid}.jsonl")
            if os.path.exists(shard_file):
                os.remove(shard_file)
        print(f"  ✓ Cleaned up shard files")


# ============================================================
# Generation Phase
# ============================================================
def run_generation(reranker_choice: str, generator_model: str, gpu_id,
                   input_file: str, output_dir: str, num_entries: str,
                   generator_method: str = None, generator_backend: str = None,
                   max_model_len: int = None, gpu_memory_utilization: float = None,
                   top_k: int = None, evaluate_metrics: bool = None,
                   save_individual_scores: bool = None,
                   max_tokens: int = None, temperature: float = None,
                   top_p: float = None, repetition_penalty: float = None):
    """
    Execute a single reranker+generator combination generation task on the specified GPU.
    Supports multi-GPU format gpu_id="0,1".
    """
    # Apply defaults
    generator_method = generator_method or DEFAULT_GENERATOR_METHOD
    generator_backend = generator_backend or DEFAULT_GENERATOR_BACKEND
    max_model_len = max_model_len if max_model_len is not None else DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization = gpu_memory_utilization if gpu_memory_utilization is not None else DEFAULT_GPU_MEMORY_UTILIZATION
    top_k = top_k if top_k is not None else DEFAULT_TOP_K
    evaluate_metrics = evaluate_metrics if evaluate_metrics is not None else DEFAULT_EVALUATE_METRICS
    save_individual_scores = save_individual_scores if save_individual_scores is not None else DEFAULT_SAVE_INDIVIDUAL_SCORES

    # Build sampling params
    sampling_params = dict(DEFAULT_SAMPLING_PARAMS)
    if max_tokens is not None:
        sampling_params["max_tokens"] = max_tokens
    if temperature is not None:
        sampling_params["temperature"] = temperature
    if top_p is not None:
        sampling_params["top_p"] = top_p
    if repetition_penalty is not None:
        sampling_params["repetition_penalty"] = repetition_penalty

    # Set GPU
    gpu_str = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    num_visible_gpus = len(gpu_str.split(","))
    print(f"\n{'=' * 60}")
    print(f"[GEN] Reranker: {reranker_choice} | Generator: {generator_model}")
    print(f"[GEN] GPU: {gpu_str} ({num_visible_gpus} GPU(s))")
    print(f"{'=' * 60}")

    # Support non-preset rankers (e.g., bm25-baseline)
    reranker_method, is_set_selection = resolve_reranker_meta(reranker_choice)

    # Paths
    ranker_cache_dir = os.path.join(output_dir, reranker_choice)
    ranker_cache_file = os.path.join(ranker_cache_dir, "ranker_cache.jsonl")
    generator_dir_name = generator_model.replace("/", "_")
    output_subdir = os.path.join(output_dir, reranker_choice, generator_dir_name)
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, OUTPUT_FILENAME)

    # Check if already completed
    if os.path.exists(output_file):
        print(f"  [SKIP] Output already exists: {output_file}")
        return

    # Check ranker cache exists
    if not os.path.exists(ranker_cache_file):
        print(f"  [ERROR] Ranker cache not found: {ranker_cache_file}")
        print(f"  Please run ranking phase first.")
        sys.exit(1)

    # Load original data and ranker cache
    process_data = get_process_data(input_file, num_entries)
    cached_data = load_jsonl(ranker_cache_file)
    cached_data = cached_data[:len(process_data)]

    print(f"  Loaded {len(process_data)} data entries, {len(cached_data)} ranker cache entries")

    # Merge ranker results into process_data
    for entry, cached_entry in zip(process_data, cached_data):
        if is_set_selection:
            entry["selected_docs"] = cached_entry.get("selected_docs", [])
        else:
            entry["ranked_docs"] = cached_entry.get("ranked_docs", [])
        entry["context_for_generator"] = cached_entry.get("context_for_generator", [])
        if "ranker_evaluation" in cached_entry:
            entry["ranker_evaluation"] = cached_entry["ranker_evaluation"]

    # Rebuild reranked_documents from cache
    documents = [convert_entry_to_document(entry) for entry in process_data]
    reranked_documents = documents
    for doc, entry in zip(reranked_documents, process_data):
        if is_set_selection:
            selected = entry.get("selected_docs", [])
            new_ctxs = []
            for d in selected:
                new_ctxs.append(Context(
                    id=str(d["original_id"]) if d["original_id"] is not None else None,
                    title="",
                    text=d["doc"],
                    score=d.get("score", 0.0)
                ))
            doc.contexts = new_ctxs
            doc.reorder_contexts = new_ctxs
        else:
            ctx_for_gen = entry.get("context_for_generator", [])
            new_ctxs = []
            for d in ctx_for_gen:
                new_ctxs.append(Context(
                    id=str(d["original_id"]) if d["original_id"] is not None else None,
                    title="",
                    text=d["doc"],
                    score=0.0
                ))
            doc.contexts = new_ctxs
            ranked = entry.get("ranked_docs", [])
            full_ctxs = []
            for d in ranked:
                full_ctxs.append(Context(
                    id=str(d["original_id"]) if d["original_id"] is not None else None,
                    title="",
                    text=d["doc"],
                    score=d.get("score", 0.0)
                ))
            doc.reorder_contexts = full_ctxs

    # Initialize Generator
    print(f"  Initializing Generator: {generator_model} ({generator_backend})")
    generator_kwargs = {
        "method": generator_method,
        "model_name": generator_model,
        "backend": generator_backend,
    }
    if generator_backend == "huggingface":
        generator_kwargs["torch_dtype"] = torch.float16
    elif generator_backend == "vllm":
        generator_kwargs["dtype"] = "float16"
        generator_kwargs["max_model_len"] = max_model_len
        generator_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    generator = Generator(**generator_kwargs)

    print(f"  Generating answers...")
    start_time = time.time()
    answers = generator.generate(reranked_documents, sampling_params=sampling_params)
    elapsed = time.time() - start_time
    print(f"  Generation complete! Elapsed: {elapsed:.1f}s ({elapsed/len(documents):.2f}s/sample)")

    # Release Generator
    del generator
    cleanup_gpu()

    # Evaluate
    if evaluate_metrics:
        print(f"  Computing generation metrics...")
        eval_documents = []
        for entry, doc in zip(process_data, reranked_documents):
            if is_set_selection:
                eval_contexts = doc.contexts
            else:
                eval_contexts = doc.contexts[:top_k]
            eval_doc = Document(
                question=doc.question,
                answers=Answer([entry.get("answer", "")]),
                contexts=eval_contexts
            )
            eval_documents.append(eval_doc)

        metrics = Metrics(eval_documents)
        evaluation_results = metrics.calculate_generation_metrics(answers)

        print(f"  Generation evaluation results:")
        for metric, score in evaluation_results.items():
            print(f"    {metric}: {score:.2f}%")

        if save_individual_scores:
            data_obj = type("Data", (object,), {
                "documents": eval_documents,
                "predictions": answers
            })()

            from rankify.metrics.metrics import ExactMatch, F1Score, ContainsMatch

            em_metric = ExactMatch({"dataset_name": "QA_Evaluation"})
            _, individual_em = em_metric.calculate_metric(data_obj)

            f1_metric = F1Score({"dataset_name": "QA_Evaluation"})
            _, individual_f1 = f1_metric.calculate_metric(data_obj)

            contains_metric = ContainsMatch({"dataset_name": "QA_Evaluation"})
            _, individual_contains = contains_metric.calculate_metric(data_obj)

            for i, entry in enumerate(process_data):
                entry["evaluation_scores"] = {
                    "exact_match": float(individual_em[i]) * 100,
                    "f1_score": float(individual_f1[i]) * 100,
                    "contains_match": float(individual_contains[i]) * 100
                }

        # Save evaluation summary
        eval_summary_file = output_file.replace(".jsonl", "_eval_summary.json")
        with open(eval_summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": len(process_data),
                "evaluation_metrics": evaluation_results,
            }, f, ensure_ascii=False, indent=2)

    # Save results
    for entry, answer in zip(process_data, answers):
        entry["generated_answer"] = answer

    save_jsonl(process_data, output_file)
    print(f"  ✓ Saved {len(process_data)} entries to: {output_file}")
    print(f"  [DONE] {reranker_choice} + {generator_model} generation complete!")


# ============================================================
# Summarize Phase
# ============================================================
def run_summarize(input_file: str, output_dir: str, num_entries: str,
                  rerankers: List[str], generators: List[str]):
    """Summarize all ranker results into a comprehensive JSONL"""
    print(f"\n{'=' * 60}")
    print(f"[SUMMARIZE] Summarizing all Ranker results")
    print(f"{'=' * 60}")

    process_data = get_process_data(input_file, num_entries)

    for entry in process_data:
        entry["selected_docs"] = {}
        entry["ranker_generations"] = {}
        entry["ranker_scores"] = {}
        entry["ranker_doc_counts"] = {}

    # Build a (question, answer) -> index mapping so matching is content-based.
    qa_to_idx = {}
    for idx, entry in enumerate(process_data):
        key = (entry.get("question", ""), entry.get("answer", ""))
        qa_to_idx[key] = idx

    rankers_found = []
    for _reranker in rerankers:
        ranker_dir = os.path.join(output_dir, _reranker)
        cache_file = os.path.join(ranker_dir, "ranker_cache.jsonl")

        if not os.path.exists(cache_file):
            print(f"  [SKIP] {_reranker}: cache file not found")
            continue

        rankers_found.append(_reranker)
        cache_data = load_jsonl(cache_file)

        print(f"  [OK] {_reranker}: loaded {len(cache_data)} cache entries")

        matched_count = 0
        for cached in cache_data:
            key = (cached.get("question", ""), cached.get("answer", ""))
            idx = qa_to_idx.get(key)
            if idx is None:
                continue
            matched_count += 1
            entry = process_data[idx]

            if "selected_docs" in cached and cached["selected_docs"]:
                doc_ids = [d["original_id"] for d in cached["selected_docs"]]
            elif "ranked_docs" in cached and cached["ranked_docs"]:
                ctx_for_gen = cached.get("context_for_generator", [])
                doc_ids = [d["original_id"] for d in ctx_for_gen]
            else:
                doc_ids = []

            entry["selected_docs"][_reranker] = doc_ids
            entry["ranker_doc_counts"][_reranker] = len(doc_ids)

        if matched_count < len(cache_data):
            print(f"    [WARN] {_reranker}: {len(cache_data) - matched_count} entries not matched")

        # Iterate generator outputs
        for _generator in generators:
            gen_dir_name = _generator.replace("/", "_")
            gen_output_file = os.path.join(ranker_dir, gen_dir_name, OUTPUT_FILENAME)

            if not os.path.exists(gen_output_file):
                continue

            gen_data = load_jsonl(gen_output_file)
            gen_short_name = _generator.split("/")[-1] if "/" in _generator else _generator

            for gen_entry in gen_data:
                key = (gen_entry.get("question", ""), gen_entry.get("answer", ""))
                idx = qa_to_idx.get(key)
                if idx is None:
                    continue
                entry = process_data[idx]

                generated_answer = gen_entry.get("generated_answer", "")
                if _reranker not in entry["ranker_generations"]:
                    entry["ranker_generations"][_reranker] = {}
                entry["ranker_generations"][_reranker][gen_short_name] = generated_answer

                eval_scores = gen_entry.get("evaluation_scores", {})
                if eval_scores:
                    if _reranker not in entry["ranker_scores"]:
                        entry["ranker_scores"][_reranker] = {}
                    entry["ranker_scores"][_reranker][gen_short_name] = eval_scores

    # Save summary
    summary_file = os.path.join(output_dir, "all_rankers_summary.jsonl")
    save_jsonl(process_data, summary_file)

    print(f"\n  Summary complete!")
    print(f"  Summarized {len(rankers_found)} rankers: {rankers_found}")
    print(f"  Output file: {summary_file}")
    print(f"  Total {len(process_data)} entries")


# ============================================================
# CLI Entry Point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rankify JSONL Pipeline v3 - Multi-GPU Data-Parallel Scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ranking (single GPU)
  python run_pipeline.py rank --reranker rankllama --gpu 0

  # Data-parallel ranking (4 GPUs, 4 shards)
  python run_pipeline.py rank --reranker rankllama --gpu 0 --shard-id 0 --num-shards 4
  python run_pipeline.py rank --reranker rankllama --gpu 1 --shard-id 1 --num-shards 4
  python run_pipeline.py rank --reranker rankllama --gpu 2 --shard-id 2 --num-shards 4
  python run_pipeline.py rank --reranker rankllama --gpu 3 --shard-id 3 --num-shards 4

  # Merge shards
  python run_pipeline.py merge-shards --reranker rankllama --num-shards 4

  # Generation
  python run_pipeline.py generate --reranker rankllama --generator meta-llama/Llama-3.1-8B-Instruct --gpu 0

  # Summarize
  python run_pipeline.py summarize
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # rank sub-command
    rank_parser = subparsers.add_parser("rank", help="Execute a single reranker ranking task")
    rank_parser.add_argument("--reranker", required=True, help="Reranker name")
    rank_parser.add_argument("--gpu", type=str, required=True, help="GPU ID (supports multi-GPU: '0,1')")
    rank_parser.add_argument("--input", default=INPUT_FILE, help="Input file path")
    rank_parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    rank_parser.add_argument("--num-entries", default=NUM_ENTRIES, help="Number of entries to process")
    rank_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K for context selection")
    rank_parser.add_argument("--evaluate-ranker", action="store_true", default=DEFAULT_EVALUATE_RANKER, help="Whether to evaluate ranker")
    rank_parser.add_argument("--no-evaluate-ranker", action="store_false", dest="evaluate_ranker", help="Disable ranker evaluation")
    rank_parser.add_argument("--top-k-for-eval", type=int, nargs="+", default=DEFAULT_TOP_K_FOR_EVAL, help="Top-K values for evaluation")
    rank_parser.add_argument("--ndcg-cuts", type=int, nargs="+", default=DEFAULT_NDCG_CUTS, help="NDCG cut values")
    rank_parser.add_argument("--shard-id", type=int, default=None, help="Data shard ID (0-indexed) for data-parallel mode")
    rank_parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards for data-parallel mode")

    # generate sub-command
    gen_parser = subparsers.add_parser("generate", help="Execute a single generation task")
    gen_parser.add_argument("--reranker", required=True, help="Reranker name")
    gen_parser.add_argument("--generator", required=True, help="Generator model name")
    gen_parser.add_argument("--gpu", type=str, required=True, help="GPU ID (supports multi-GPU: '0,1')")
    gen_parser.add_argument("--input", default=INPUT_FILE, help="Input file path")
    gen_parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    gen_parser.add_argument("--num-entries", default=NUM_ENTRIES, help="Number of entries to process")
    gen_parser.add_argument("--generator-method", default=DEFAULT_GENERATOR_METHOD, help="Generator method (e.g. basic-rag)")
    gen_parser.add_argument("--generator-backend", default=DEFAULT_GENERATOR_BACKEND, help="Generator backend (vllm/huggingface)")
    gen_parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN, help="Max model length for vLLM")
    gen_parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION, help="GPU memory utilization for vLLM")
    gen_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K contexts for generation")
    gen_parser.add_argument("--evaluate-metrics", action="store_true", default=DEFAULT_EVALUATE_METRICS, help="Whether to evaluate generation metrics")
    gen_parser.add_argument("--no-evaluate-metrics", action="store_false", dest="evaluate_metrics", help="Disable generation metrics evaluation")
    gen_parser.add_argument("--save-individual-scores", action="store_true", default=DEFAULT_SAVE_INDIVIDUAL_SCORES, help="Save per-sample scores")
    gen_parser.add_argument("--no-save-individual-scores", action="store_false", dest="save_individual_scores", help="Disable per-sample scores")
    gen_parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for generation sampling")
    gen_parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation sampling")
    gen_parser.add_argument("--top-p", type=float, default=None, help="Top-p for generation sampling")
    gen_parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for generation sampling")

    # merge-shards sub-command
    merge_parser = subparsers.add_parser("merge-shards", help="Merge data-parallel shard results")
    merge_parser.add_argument("--reranker", required=True, help="Reranker name")
    merge_parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    merge_parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    merge_parser.add_argument("--no-cleanup", action="store_true", help="Do not delete shard files after merging")
    merge_parser.add_argument("--input", default=INPUT_FILE, help="Input file path (for evaluation)")
    merge_parser.add_argument("--num-entries", default=NUM_ENTRIES, help="Number of entries")
    merge_parser.add_argument("--evaluate-ranker", action="store_true", default=DEFAULT_EVALUATE_RANKER, help="Run evaluation after merge")
    merge_parser.add_argument("--no-evaluate-ranker", action="store_false", dest="evaluate_ranker", help="Skip evaluation after merge")
    merge_parser.add_argument("--top-k-for-eval", type=int, nargs="+", default=DEFAULT_TOP_K_FOR_EVAL, help="Top-K values for evaluation")
    merge_parser.add_argument("--ndcg-cuts", type=int, nargs="+", default=DEFAULT_NDCG_CUTS, help="NDCG cut values")
    merge_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K for context selection")

    # summarize sub-command
    sum_parser = subparsers.add_parser("summarize", help="Summarize all results")
    sum_parser.add_argument("--input", default=INPUT_FILE, help="Input file path")
    sum_parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    sum_parser.add_argument("--num-entries", default=NUM_ENTRIES, help="Number of entries to process")
    sum_parser.add_argument("--rerankers", nargs="+", default=ALL_RERANKERS, help="Reranker list")
    sum_parser.add_argument("--generators", nargs="+", default=ALL_GENERATORS, help="Generator list")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "rank":
        run_ranking(
            reranker_choice=args.reranker,
            gpu_id=args.gpu,
            input_file=args.input,
            output_dir=args.output_dir,
            num_entries=args.num_entries,
            top_k=args.top_k,
            evaluate_ranker=args.evaluate_ranker,
            top_k_for_eval=args.top_k_for_eval,
            ndcg_cuts=args.ndcg_cuts,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    elif args.command == "generate":
        run_generation(
            reranker_choice=args.reranker,
            generator_model=args.generator,
            gpu_id=args.gpu,
            input_file=args.input,
            output_dir=args.output_dir,
            num_entries=args.num_entries,
            generator_method=args.generator_method,
            generator_backend=args.generator_backend,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            top_k=args.top_k,
            evaluate_metrics=args.evaluate_metrics,
            save_individual_scores=args.save_individual_scores,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    elif args.command == "merge-shards":
        merge_shards(
            reranker_choice=args.reranker,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            cleanup=not args.no_cleanup,
            input_file=args.input,
            num_entries=args.num_entries,
            evaluate_ranker=args.evaluate_ranker,
            top_k_for_eval=args.top_k_for_eval,
            ndcg_cuts=args.ndcg_cuts,
            top_k=args.top_k,
        )
    elif args.command == "summarize":
        run_summarize(
            input_file=args.input,
            output_dir=args.output_dir,
            num_entries=args.num_entries,
            rerankers=args.rerankers,
            generators=args.generators,
        )
    else:
        print("Please specify a sub-command: rank / generate / merge-shards / summarize")
        print("Use --help for details")
        sys.exit(1)
