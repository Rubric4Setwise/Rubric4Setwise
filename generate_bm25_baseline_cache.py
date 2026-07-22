"""
Generate BM25 Baseline ranker_cache.jsonl
==========================================
Takes the BM25-retrieved docs (already in rank order by BM25 score)
and directly uses top-5 as context_for_generator, without any reranking.

This creates a "bm25-baseline" ranker entry that can be used directly
in the generation phase of run_pipeline.py.

Usage:
    python generate_bm25_baseline_cache.py

Or with custom parameters:
    python generate_bm25_baseline_cache.py --input <input.jsonl> --output-dir <dir> --top-k 5
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Generate BM25 baseline ranker cache")
    parser.add_argument(
        "--input", type=str,
        default="/cfs_cloud_code/jiangkailin/Setwise/processed_data_v2/short_closed/short_closed_v2_bm25top20.jsonl",
        help="Input JSONL file (BM25 top-20 retrieved)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/cfs_cloud_code/jiangkailin/Setwise/ranker_output/baseline_ranker_all_v2/bm25-baseline",
        help="Output directory for ranker cache"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K docs for generation context")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "ranker_cache.jsonl")

    print(f"Input: {args.input}")
    print(f"Output: {output_file}")
    print(f"Top-K: {args.top_k}")

    count = 0
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            entry = json.loads(line)
            docs = entry.get("docs", [])

            # ranked_docs: all 20 docs in BM25 order (already ranked)
            ranked_docs = []
            for rank, doc in enumerate(docs, start=1):
                ranked_docs.append({
                    "rank": rank,
                    "original_id": doc.get("original_id"),
                    "score": doc.get("bm25_score", 0),
                    "doc": doc.get("doc", ""),
                })

            # context_for_generator: top-K
            context_for_generator = []
            for doc in docs[:args.top_k]:
                context_for_generator.append({
                    "original_id": doc.get("original_id"),
                    "doc": doc.get("doc", ""),
                })

            cache_entry = {
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "ranked_docs": ranked_docs,
                "context_for_generator": context_for_generator,
                "ranker_evaluation": {
                    "method_type": "ranking",
                    "note": "BM25 baseline - no reranking applied"
                },
            }
            f_out.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done! {count} entries saved to {output_file}")


if __name__ == "__main__":
    main()
