"""
BM25 Retrieval Preprocessing Script
====================================
Reads a JSONL file containing questions, uses BM25 to retrieve top-K documents
from the Wikipedia corpus, and outputs a new JSONL file with the same format
as the original data (compatible with run_pipeline.py).

Usage:
    python run_bm25_retrieval.py \
        --input /cfs_cloud_code/jiangkailin/Setwise/processed_data/short_closed/rankify_short_closed_1000case.jsonl \
        --output /cfs_cloud_code/jiangkailin/Setwise/processed_data/short_closed/rankify_short_closed_1000case_bm25top20.jsonl \
        --n-docs 20 \
        --index-type wiki \
        --batch-size 50



python run_bm25_retrieval.py --input /cfs_cloud_code/jiangkailin/Setwise/processed_data_v2/short_closed/short_closed_v2.jsonl --output /cfs_cloud_code/jiangkailin/Setwise/processed_data_v2/short_closed/short_closed_v2_bm25top20.jsonl --n-docs 20 --index-type wiki --batch-size 50



python run_bm25_retrieval.py --input /cfs_cloud_code/jiangkailin/Setwise/processed_data_v3/short_closed/v3_6125data.jsonl --output /cfs_cloud_code/jiangkailin/Setwise/processed_data_v3/short_closed/v3_6125data_bm25top20.jsonl --n-docs 20 --index-type wiki --batch-size 50 --index-folder /cfs_cloud_code/jiangkailin/Rankify_model_data/bm25_wiki_index






The output JSONL preserves the original fields (id, question, answer, type, level,
supporting_facts, context) and replaces the `docs` field with BM25-retrieved documents.
"""


import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

# ============================================================
# Java 11 configuration (required by Pyserini/Lucene)
# Must be set BEFORE importing pyserini/jnius
# ============================================================
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["JVM_PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")





# HuggingFace cache (consistent with pipeline)
os.environ["HF_HOME"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/huggingface"
os.environ["HF_HUB_CACHE"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/huggingface/hub"

# Pyserini cache: store prebuilt indexes under Rankify_model_data
os.environ["PYSERINI_CACHE"] = "/cfs_cloud_code/jiangkailin/Rankify_model_data/pyserini_cache"

# Placeholder to prevent pyserini's OpenAI encoder from failing on import
os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")

from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.retrievers.retriever import Retriever


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_bm25_retrieval(
    input_file: str,
    output_file: str,
    n_docs: int = 20,
    index_type: str = "wiki",
    index_folder: str = None,
    batch_size: int = 50,
    num_entries: str = "all",
):
    """
    Run BM25 retrieval for each query and produce a new JSONL file.

    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        n_docs: Number of documents to retrieve per query (default: 20)
        index_type: Index type ("wiki" or "msmarco")
        index_folder: Custom index folder (optional, overrides index_type)
        batch_size: Number of queries to process per batch
        num_entries: "all" or integer limit
    """
    print(f"{'=' * 60}")
    print(f"[BM25 Retrieval] Starting...")
    print(f"  Input:      {input_file}")
    print(f"  Output:     {output_file}")
    print(f"  Top-K:      {n_docs}")
    print(f"  Index:      {index_folder if index_folder else index_type}")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}")

    # Load input data
    print(f"\n  Loading input data...")
    raw_data = load_jsonl(input_file)
    if num_entries != "all":
        raw_data = raw_data[:int(num_entries)]
    print(f"  Loaded {len(raw_data)} entries")

    # Initialize BM25 Retriever
    print(f"\n  Initializing BM25 Retriever (index_type={index_type})...")
    retriever_kwargs = {
        "method": "bm25",
        "n_docs": n_docs,
    }
    if index_folder:
        retriever_kwargs["index_folder"] = index_folder
    else:
        retriever_kwargs["index_type"] = index_type

    retriever = Retriever(**retriever_kwargs)
    print(f"  Retriever initialized!")

    # Process in batches
    total = len(raw_data)
    all_results = []
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_data = raw_data[batch_start:batch_end]

        # Convert to Document objects for retrieval
        batch_documents = []
        for entry in batch_data:
            doc = Document(
                question=Question(entry["question"]),
                answers=Answer(entry.get("answer", "")),
                contexts=[]
            )
            batch_documents.append(doc)

        # Retrieve
        retrieved_documents = retriever.retrieve(batch_documents)

        # Convert results back to output format
        for i, (entry, ret_doc) in enumerate(zip(batch_data, retrieved_documents)):
            # Build new docs list from retrieved contexts
            new_docs = []
            for rank, ctx in enumerate(ret_doc.contexts, start=1):
                new_docs.append({
                    "original_id": rank,
                    "doc": ctx.text,
                    "title": ctx.title if ctx.title else "",
                    "bm25_score": ctx.score,
                })

            # Build output entry: preserve all original fields, add/replace docs
            output_entry = dict(entry)
            output_entry["docs"] = new_docs

            all_results.append(output_entry)

        elapsed = time.time() - start_time
        speed = batch_end / elapsed if elapsed > 0 else 0
        print(f"  [{batch_end}/{total}] {elapsed:.1f}s elapsed, {speed:.1f} queries/s")

    # Save output
    save_jsonl(all_results, output_file)
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"[DONE] BM25 Retrieval complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/total:.2f}s/query)")
    print(f"  Output: {output_file}")
    print(f"  {len(all_results)} entries, each with top-{n_docs} BM25 docs")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BM25 Retrieval Preprocessing - retrieve top-K docs for reranking pipeline"
    )
    parser.add_argument(
        "--input", type=str,
        default="/cfs_cloud_code/jiangkailin/Setwise/processed_data/short_closed/rankify_short_closed_1000case.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output", type=str,
        default="/cfs_cloud_code/jiangkailin/Setwise/processed_data/short_closed/rankify_short_closed_1000case_bm25top20.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument("--n-docs", type=int, default=20, help="Number of documents to retrieve per query")
    parser.add_argument("--index-type", type=str, default="wiki", help="Index type: wiki or msmarco")
    parser.add_argument("--index-folder", type=str, default=None, help="Custom index folder (overrides index-type)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for retrieval")
    parser.add_argument("--num-entries", type=str, default="all", help="Number of entries to process ('all' or integer)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bm25_retrieval(
        input_file=args.input,
        output_file=args.output,
        n_docs=args.n_docs,
        index_type=args.index_type,
        index_folder=args.index_folder,
        batch_size=args.batch_size,
        num_entries=args.num_entries,
    )
