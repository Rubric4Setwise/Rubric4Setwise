# rankify/cli/cli.py

import argparse
import logging
from pathlib import Path

from rankify.indexing.lucene_indexer import LuceneIndexer
from rankify.indexing.dpr_indexer import DPRIndexer
from rankify.indexing.ance_indexer import ANCEIndexer
from rankify.indexing.contriever_indexer import ContrieverIndexer
from rankify.indexing.colbert_indexer import ColBERTIndexer
from rankify.indexing.bge_indexer import BGEIndexer

SUPPORTED_RETRIEVERS = ["bm25", "dpr", "ance", "contriever", "colbert", "bge"]
logger = logging.getLogger("rankify.cli")

def handle_output_directory(output_dir: str) -> str:
    """Ensure the output directory exists."""
    out = Path(output_dir or "rankify_indices")
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {out}")
    return str(out)

def get_indexer_args(args: argparse.Namespace) -> dict:
    """
    Map CLI names -> indexer constructor parameters, dropping None values.
    """
    rename = {
        "output": "output_dir",
        "retriever": "retriever_name",
        "encoder": "encoder_name",
    }
    skip = {"command"}
    out = {}
    for k, v in vars(args).items():
        if v is None or k in skip:
            continue
        out[rename.get(k, k)] = v
    return out

def _build_and_load(indexer, label: str):
    logger.info(f"Building {label} index…")
    indexer.build_index()
    indexer.load_index()
    logger.info(f"{label} indexing complete. Index at: {indexer.index_dir}")

def main():
    # App-level logging (safe here; this is the CLI, not the library)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(prog="rankify-index", description="Rankify Indexer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index subcommand
    p = subparsers.add_parser("index", help="Index a corpus")
    p.add_argument("corpus_path", type=str, help="Path to a JSONL corpus")
    p.add_argument("--retriever", required=True, choices=SUPPORTED_RETRIEVERS,
                   help=f"Which retriever to use: {', '.join(SUPPORTED_RETRIEVERS)}")
    p.add_argument("--output", default="rankify_indices", help="Root output directory (default: rankify_indices)")
    p.add_argument("--index_type", default="wiki", help="Index type label used for folder names (default: wiki)")
    p.add_argument("--threads", type=int, default=32, help="Threads for preprocessing (default: 32)")
    p.add_argument("--chunk_size", type=int, default=1024, help="Lines per chunk when converting corpora (default: 1024)")

    # dense options
    p.add_argument("--encoder", help="Encoder model name (dense indexers)")
    p.add_argument("--batch_size", type=int, help="Batch size for encoding/indexing")
    p.add_argument("--embedding_batch_size", type=int,
                   help="Per-forward batch size for Contriever embedding (optional)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                   help="Device for dense models (default: cuda)")

    args = parser.parse_args()

    if args.command == "index":
        # validate corpus
        corpus = Path(args.corpus_path)
        if not corpus.exists():
            parser.error(f"Corpus not found: {corpus}")

        # output folder
        args.output = handle_output_directory(args.output)

        retriever = args.retriever.lower()
        # sensible defaults per retriever
        if retriever == "bm25":
            indexer = LuceneIndexer(**get_indexer_args(args))
            _build_and_load(indexer, "BM25")

        elif retriever == "dpr":
            if not args.encoder:
                args.encoder = "facebook/dpr-ctx_encoder-single-nq-base"
                logger.info(f"No --encoder provided; using {args.encoder}")
            indexer = DPRIndexer(**get_indexer_args(args))
            _build_and_load(indexer, "DPR")

        elif retriever == "ance":
            if not args.encoder:
                args.encoder = "castorini/ance-dpr-context-multi"
                logger.info(f"No --encoder provided; using {args.encoder}")
            indexer = ANCEIndexer(**get_indexer_args(args))
            _build_and_load(indexer, "ANCE")

        elif retriever == "contriever":
            if not args.encoder:
                args.encoder = "facebook/contriever"
                logger.info(f"No --encoder provided; using {args.encoder}")
            # pass embedding_batch_size only if provided (constructor has a default)
            kwargs = get_indexer_args(args)
            if args.embedding_batch_size is not None:
                kwargs["embedding_batch_size"] = args.embedding_batch_size
            indexer = ContrieverIndexer(**kwargs)
            _build_and_load(indexer, "Contriever")

        elif retriever == "colbert":
            if not args.encoder:
                args.encoder = "colbert-ir/colbertv2.0"
                logger.info(f"No --encoder provided; using {args.encoder}")
            indexer = ColBERTIndexer(**get_indexer_args(args))
            _build_and_load(indexer, "ColBERT")

        elif retriever == "bge":
            if not args.encoder:
                args.encoder = "BAAI/bge-large-en-v1.5"
                logger.info(f"No --encoder provided; using {args.encoder}")
            indexer = BGEIndexer(**get_indexer_args(args))
            _build_and_load(indexer, "BGE")

        else:
            parser.error(f"Unknown retriever: {retriever}")

if __name__ == "__main__":
    main()
