#format_converters.py
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from rankify.retrievers.utils import process_chunk, generate_chunks, count_file_lines, process_chunk_tabbed
from multiprocessing.dummy import Pool
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json
logging.basicConfig(level=logging.INFO)

def to_tsv(input_path, output_dir, chunk_size, threads) -> str:
    """
    Convert a corpus file to TSV format.
    :param input_path: Path to the input corpus file.
    :param output_dir: Directory to save the output TSV file.
    :param chunk_size: Number of lines to process in each chunk.
    :param threads: Number of threads to use for processing.
    :return: Path to the output TSV file.
    """
    output_path = output_dir / "passages.tsv"

    total_lines = count_file_lines(input_path)
    logging.info("Streaming and processing corpus with " + str(threads) + " threads...")
    with open(input_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out, \
            Pool(processes=threads) as pool:

            pbar = tqdm(total=total_lines, desc="Processing")

            for result in pool.imap_unordered(lambda args: process_chunk_tabbed(*args), generate_chunks(f_in, chunk_size)):
                for i, doc_json in enumerate(result):
                        f_out.write(doc_json + "\n")
                pbar.update(len(result))

            pbar.close()

    logging.info(f"Done saving corpus to {output_path}")
    return output_path


def to_pyserini_jsonl(input_path, output_dir, chunk_size, threads) -> str:
    """
    Convert the corpus to Pyserini JSONL format: one input JSON line -> one output JSON line.
    Keeps your original string IDs and pairs them with the *same line's* contents.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "corpus.jsonl"

    # Count for progress only
    total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_lines, desc="Rewriting corpus"):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                logging.warning(f"Skipping malformed line: {e}")
                continue

            doc_id = str(rec.get("id", ""))  # keep the original string id
            title  = rec.get("title") or ""
            # Prefer explicit 'text', then 'contents', then fallback to empty
            body   = rec.get("text")
            if body is None:
                body = rec.get("contents", "")

            # Build a Pyserini-friendly contents field: "title\nbody" if title exists
            contents = f"{title}\n{body}" if title else body

            out = {"id": doc_id, "contents": contents}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    logging.info(f"✅ Done saving corpus to {output_path}")
    return str(output_path)


def to_pyserini_jsonl_dense(input_path, output_dir, chunk_size, threads) -> str:
    """
    Create a JSONL where each line is: {"id": "...", "title": "...", "text": "..."}
    Accepts:
      - existing JSONL (dict per line)
      - TSV ("id<TAB>text")
      - raw text (one passage per line)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "corpus.jsonl"

    # First pass: peek to detect format
    def _peek_nonempty_line(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return line
        return ""

    first = _peek_nonempty_line(input_path)
    is_jsonl = first.lstrip().startswith("{")
    is_tsv = ("\t" in first) and not is_jsonl

    total = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(fin, total=total, desc="Rewriting corpus"), 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            rec = None
            if is_jsonl:
                try:
                    rec_in = json.loads(line)
                    # normalize fields
                    text = rec_in.get("text") or rec_in.get("contents") or rec_in.get("paragraph") or rec_in.get("content") or ""
                    title = rec_in.get("title")
                    if not title and isinstance(rec_in.get("titles"), list) and rec_in["titles"]:
                        title = rec_in["titles"][0]
                    doc_id = rec_in.get("id") or f"doc{i}"
                    rec = {"id": str(doc_id), "title": title or "", "text": str(text)}
                except Exception:
                    # Fall through to raw handling
                    pass

            if rec is None and is_tsv:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    doc_id, text = parts
                else:
                    doc_id, text = f"doc{i}", line
                rec = {"id": str(doc_id), "title": "", "text": text}

            if rec is None:
                # raw line
                rec = {"id": f"doc{i}", "title": "", "text": line}

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info(f"✅ Done saving dense corpus to {output_path}")
    return output_path

def to_contriever_embedding_chunks(jsonl_path, output_dir, chunk_size, model_name, batch_size, device):
    """
    Converts a corpus to Contriever embeddings in .pkl chunks.
    :param jsonl_path: Path to the input JSONL file containing documents.
    :param output_dir: Directory to save the output .pkl files.
    :param chunk_size: Number of documents to process in each chunk.
    :param model_name: Name of the Contriever model to use.
    :param batch_size: Number of documents to process in each batch for encoding.
    :param device: Device to run the model on (e.g., "cpu" or "cuda").
    Returns:
        List[Path]: Paths to all .pkl shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    shard_paths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunk = []
        shard_id = 0

        for idx, line in enumerate(f):
            doc = json.loads(line.strip())
            text = doc.get("contents", doc.get("text", "")).strip()
            if not text:
                continue
            doc_id = doc.get("id", f"doc{idx}")
            chunk.append((doc_id, text))

            if len(chunk) >= chunk_size:
                ids, embeddings = encode_contriever_chunk(chunk, tokenizer, model, batch_size, device)
                path = output_dir / f"embeddings_{shard_id}.pkl"
                with open(path, "wb") as fout:
                    pickle.dump((ids, embeddings), fout)
                shard_paths.append(path)
                shard_id += 1
                chunk = []

        if chunk:
            ids, embeddings = encode_contriever_chunk(chunk, tokenizer, model, batch_size, device)
            path = output_dir / f"embeddings_{shard_id}.pkl"
            with open(path, "wb") as fout:
                pickle.dump((ids, embeddings), fout)
            shard_paths.append(path)

    return shard_paths


def encode_contriever_chunk(chunk, tokenizer, model, batch_size, device) -> tuple:
    """
    Encodes a chunk of text using Contriever.
    :param chunk: List of tuples (doc_id, text) to encode.
    :param tokenizer: Tokenizer for the Contriever model.
    :param model: Contriever model to use for encoding.
    :param batch_size: Number of documents to process in each batch.
    :param device: Device to run the model on (e.g., "cpu" or "cuda").
    :return: Tuple of lists (ids, embeddings) where ids are document IDs and embeddings are their corresponding embeddings.
    """
    ids = []
    all_embeddings = []

    for i in tqdm(range(0, len(chunk), batch_size), desc="Encoding chunk"):
        batch = chunk[i:i+batch_size]
        batch_ids = [doc_id for doc_id, _ in batch]
        texts = [text for _, text in batch]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        all_embeddings.append(embeddings)
        ids.extend(batch_ids)

    embeddings = np.vstack(all_embeddings)
    return ids, embeddings
