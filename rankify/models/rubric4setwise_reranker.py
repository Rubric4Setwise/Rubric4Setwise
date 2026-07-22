"""
Rubric4Setwise Reranker
=======================

Rubric-guided set-selection reranker (formerly Rubric-MaxCov v5 / Agent V5).

Pipeline (a single LLM call per query):

    Step 1: Selection — LLM selects the minimum passage set guided by a
                        pre-computed rubric attached to `document.rubric`.
            Output format: '### Final Selection: [id1] [id2] ...'

Rubric format expected on `document.rubric`:
    [
        {"level": "L1", "type": "Relevance",     "item": "..."},
        {"level": "L2", "type": "Complementarity","item": "..."},
        {"level": "L3", "type": "Completeness",  "item": "..."},
        ...
    ]

Usage:
    from rankify.models.reranking import Reranking

    model = Reranking(
        method='rubric4setwise',
        model_name='Qwen/Qwen3-8B',
        max_k=10,
        num_gpus=1,
    )
    # Ensure each document has `document.rubric` set before calling rank()
    results = model.rank(documents)
"""

import copy
import math
import re
from typing import List, Optional, Dict

from tqdm import tqdm

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document


# =========================================================
# Prompt Template
# =========================================================

SELECTION_PROMPT = """Your task is to select the minimum set of passages that can fully support answering the search query below.

Search Query: {query}

Rubric — A structured checklist of information requirements. Each rubric item has three fields:
- level: evaluation granularity (L1 = per-document quality, L2 = cross-document set quality, L3 = end-to-end answerability).
- type: the specific quality dimension being checked (e.g., Relevance, Complementarity, Completeness, Density, Reachability).
- item: a yes/no question that the selected passage set should satisfy.

{rubric}

Candidate Passages (each indicated by a numerical identifier []):
{context}

Please follow the steps below:
Step 1. For each rubric item, list which passage(s) contain information that satisfies it. If no passage satisfies a rubric item, write "NOT FOUND".
Step 2. Select the minimum set of passages that together satisfy ALL rubric items. Each selected passage must contribute at least one rubric item that other selected passages do not cover.
Step 3. If some rubric items are "NOT FOUND", still include the closest matching passages that partially address those items.
Step 4. Output the selected passages. The format of final output should be '### Final Selection: [] [].\n', e.g., '### Final Selection: [2] [1].\n'.
"""


# =========================================================
# Parsing Utilities
# =========================================================

def _parse_selection(raw_output: str, N: int) -> List[int]:
    """Parse the selection output into a list of 0-indexed passage indices.

    Takes the LAST '### Final Selection' match so that a step-by-step
    analysis-then-conclusion output is respected.
    """
    matches = list(re.finditer(r'###\s*Final\s*Selection\s*:\s*(.*)', raw_output, re.IGNORECASE))
    if matches:
        selection_str = matches[-1].group(1)
        ids = re.findall(r'\[(\d+)\]', selection_str)
        indices = []
        for id_str in ids:
            idx = int(id_str) - 1
            if 0 <= idx < N:
                indices.append(idx)
        if indices:
            return indices

    # Fallback: scan from the last line upwards for any [id] pattern
    for line in reversed(raw_output.strip().split('\n')):
        ids = re.findall(r'\[(\d+)\]', line)
        if ids:
            indices = []
            for id_str in ids:
                idx = int(id_str) - 1
                if 0 <= idx < N:
                    indices.append(idx)
            if indices:
                return indices

    return []


def _compute_min_k(num_rubric: int) -> int:
    """Fallback minimum number of passages when parsing fails."""
    return max(2, math.ceil(num_rubric / 2))


def _extract_rubric_items(rubric_data) -> List[str]:
    """Extract rubric item strings from `document.rubric`.

    Supported formats:
    - List[dict] with an 'item' key
    - List[str]
    - str (split by newline)
    """
    if not rubric_data:
        return ["What is the answer to the query?"]

    if isinstance(rubric_data, list):
        items = []
        for entry in rubric_data:
            if isinstance(entry, dict):
                item_text = entry.get("item", "")
                if item_text and len(item_text) > 5:
                    items.append(item_text)
            elif isinstance(entry, str) and len(entry) > 5:
                items.append(entry)
        return items if items else ["What is the answer to the query?"]

    if isinstance(rubric_data, str):
        items = [line.strip() for line in rubric_data.strip().split('\n')
                 if line.strip() and len(line.strip()) > 5]
        return items if items else ["What is the answer to the query?"]

    return ["What is the answer to the query?"]


# =========================================================
# Main Reranker Class
# =========================================================

class Rubric4SetwiseReranker(BaseRanking):
    """Rubric-guided set-selection reranker (one LLM call per query).

    Args:
        method: Method name (defaults to 'rubric4setwise').
        model_name: HuggingFace model id (default: Qwen/Qwen3-8B).
        max_k: Maximum number of passages to select (default: 10).
        top_k: 0 = LLM decides the size, >0 = force top-K.
        max_passages: Maximum input passages to consider (default: 20).
        max_doc_tokens: Max tokens per document text (default: 1500).
        num_gpus: Number of GPUs for vLLM (default: 1).
        gpu_memory_utilization: vLLM GPU memory fraction (default: 0.4).
        vllm_batched: Whether to run batched vLLM inference (default: True).
        context_size: Max model context length (default: 8192).
        max_tokens_selection: Max output tokens for the LLM call (default: 1024).
    """

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "rubric4setwise"
        model_name = model_name or "Qwen/Qwen3-8B"

        from rankify.utils.model_downloader import resolve_model_path
        cache_dir = kwargs.get("cache_dir", None)
        self.model_name = resolve_model_path(model_name, cache_dir=cache_dir)

        self.max_k = kwargs.get("max_k", 10)
        self.top_k = kwargs.get("top_k", 0)
        self.max_passages = kwargs.get("max_passages", 20)
        self.max_doc_tokens = kwargs.get("max_doc_tokens", 1500)

        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.4)
        self.vllm_batched = kwargs.get("vllm_batched", True)
        self.context_size = kwargs.get("context_size", 8192)
        self.max_tokens_selection = kwargs.get("max_tokens_selection", 1024)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )

        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams

        print(f"[Rubric4Setwise] Loading LLM: {self.model_name}")
        print(f"[Rubric4Setwise] num_gpus={self.num_gpus}, gpu_mem={self.gpu_memory_utilization}")

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            max_model_len=self.context_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

        self.selection_sampling = SamplingParams(
            temperature=0.1, max_tokens=self.max_tokens_selection, top_p=0.9,
        )

        print(f"[Rubric4Setwise] Initialized: model={self.model_name}, max_k={self.max_k}")

    # ==================== Utility ====================

    def _truncate_doc(self, doc_text: str) -> str:
        tokens = self.tokenizer.encode(doc_text, add_special_tokens=False)
        if len(tokens) <= self.max_doc_tokens:
            return doc_text
        head_len = int(self.max_doc_tokens * 0.8)
        tail_len = self.max_doc_tokens - head_len
        head = self.tokenizer.decode(tokens[:head_len], skip_special_tokens=True)
        tail = self.tokenizer.decode(tokens[-tail_len:], skip_special_tokens=True)
        return head + "\n[...omitted...]\n" + tail

    def _messages_to_prompt(self, messages: list) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _format_context(self, doc_texts: List[str]) -> str:
        max_tokens = self.context_size - 2000
        lines = []
        total_tokens = 0
        for j, doc_text in enumerate(doc_texts):
            truncated = self._truncate_doc(doc_text)
            line = f"[{j+1}] {truncated}"
            line_tokens = len(self.tokenizer.encode(line, add_special_tokens=False))
            if total_tokens + line_tokens > max_tokens and lines:
                break
            lines.append(line)
            total_tokens += line_tokens
        return '\n'.join(lines)

    def _build_selection_prompt(self, query: str, doc_texts: List[str],
                                rubric_items: List[str]) -> str:
        context = self._format_context(doc_texts)
        rubric_text = '\n'.join([f"{i+1}. {q}" for i, q in enumerate(rubric_items)])
        messages = [
            {"role": "system", "content": "You are a helpful assistant that selects relevant passages to answer queries."},
            {"role": "user", "content": SELECTION_PROMPT.format(
                query=query, rubric=rubric_text, context=context,
            )},
        ]
        return self._messages_to_prompt(messages)

    # ==================== Main Rank Method ====================

    def rank(self, documents: List[Document]) -> List[Document]:
        if self.vllm_batched:
            return self._rank_batched(documents)
        return self._rank_sequential(documents)

    def _rank_sequential(self, documents: List[Document]) -> List[Document]:
        for document in tqdm(documents, desc="Rubric4Setwise reranking"):
            if not document.contexts:
                document.reorder_contexts = []
                continue

            query = document.question.question
            contexts = document.contexts
            N = min(len(contexts), self.max_passages)
            doc_texts = [(ctx.text or "").strip() for ctx in contexts[:N]]

            try:
                rubric_data = getattr(document, 'rubric', None)
                rubric_items = _extract_rubric_items(rubric_data)

                sel_prompt = self._build_selection_prompt(query, doc_texts, rubric_items)
                sel_output = self.llm.generate([sel_prompt], self.selection_sampling)
                sel_raw = sel_output[0].outputs[0].text
                selected_indices = _parse_selection(sel_raw, N)

                if not selected_indices:
                    min_k = _compute_min_k(len(rubric_items))
                    selected_indices = list(range(min(min_k, N)))

                if len(selected_indices) > self.max_k:
                    selected_indices = selected_indices[:self.max_k]

                self._finalize_document(document, contexts, selected_indices, rubric_items, {
                    "step1_selection": {"raw": sel_raw, "parsed": selected_indices},
                })

            except Exception as e:
                print(f"[Rubric4Setwise] Error: '{query[:50]}...': {e}")
                import traceback
                traceback.print_exc()
                document.reorder_contexts = copy.deepcopy(contexts[:3])

        return documents

    def _rank_batched(self, documents: List[Document]) -> List[Document]:
        print(f"[Rubric4Setwise] Preparing batch of {len(documents)} documents...")

        valid_docs = []
        for doc_idx, document in enumerate(documents):
            if not document.contexts:
                document.reorder_contexts = []
                continue
            valid_docs.append((doc_idx, document))

        if not valid_docs:
            return documents

        print(f"[Rubric4Setwise] Loading pre-computed rubrics for {len(valid_docs)} queries...")
        queries = [doc.question.question for _, doc in valid_docs]
        all_rubric_items = []
        for _, doc in valid_docs:
            rubric_data = getattr(doc, 'rubric', None)
            all_rubric_items.append(_extract_rubric_items(rubric_data))

        rubric_counts = [len(r) for r in all_rubric_items]
        print(f"[Rubric4Setwise] Rubric stats: min={min(rubric_counts)}, max={max(rubric_counts)}, "
              f"avg={sum(rubric_counts)/len(rubric_counts):.1f}")

        print(f"[Rubric4Setwise] Building selection prompts...")
        doc_texts_per_query = []
        selection_prompts = []
        for batch_idx, (doc_idx, document) in enumerate(valid_docs):
            contexts = document.contexts
            N = min(len(contexts), self.max_passages)
            doc_texts = [(ctx.text or "").strip() for ctx in contexts[:N]]
            doc_texts_per_query.append(doc_texts)
            selection_prompts.append(self._build_selection_prompt(
                queries[batch_idx], doc_texts, all_rubric_items[batch_idx]
            ))

        print(f"[Rubric4Setwise] Running selection inference ({len(selection_prompts)} prompts)...")
        selection_outputs = self.llm.generate(selection_prompts, self.selection_sampling)

        print(f"[Rubric4Setwise] Finalizing results...")
        for batch_idx, (doc_idx, document) in enumerate(valid_docs):
            try:
                raw = selection_outputs[batch_idx].outputs[0].text
                N = len(doc_texts_per_query[batch_idx])
                selected_indices = _parse_selection(raw, N)

                if not selected_indices:
                    min_k = _compute_min_k(len(all_rubric_items[batch_idx]))
                    selected_indices = list(range(min(min_k, N)))

                if len(selected_indices) > self.max_k:
                    selected_indices = selected_indices[:self.max_k]

                contexts = document.contexts
                self._finalize_document(document, contexts, selected_indices,
                                        all_rubric_items[batch_idx], {
                    "step1_selection": {"raw": raw, "parsed": selected_indices},
                })
            except Exception as e:
                query = document.question.question
                print(f"[Rubric4Setwise] Error: '{query[:50]}...': {e}")
                import traceback
                traceback.print_exc()
                document.reorder_contexts = copy.deepcopy(document.contexts[:3])

        print(f"[Rubric4Setwise] Batch inference complete!")
        return documents

    def _finalize_document(self, document: Document, contexts: List,
                           selected_indices: List[int], rubric_items: List[str],
                           trace: Dict) -> None:
        """Build reorder_contexts and record per-step trace."""
        import json as _json

        reordered = []
        for rank_pos, ctx_idx in enumerate(selected_indices):
            if 0 <= ctx_idx < len(contexts):
                ctx_copy = copy.deepcopy(contexts[ctx_idx])
                ctx_copy.score = float(len(selected_indices) - rank_pos)
                reordered.append(ctx_copy)

        document.reorder_contexts = reordered

        step1 = trace.get("step1_selection", {})

        def _indices_to_ids(indices):
            ids = []
            for idx in indices:
                if 0 <= idx < len(contexts):
                    ids.append(int(contexts[idx].id) if contexts[idx].id is not None else idx)
            return ids

        step1_indices = step1.get("parsed", [])

        step_summary = _json.dumps({
            "method": "rubric4setwise",
            "rubric_source": "precomputed",
            "rubric_items": rubric_items,
            "num_rubric": len(rubric_items),
            "min_k_hint": _compute_min_k(len(rubric_items)),
            "step1_selection": {
                "selected_doc_ids": _indices_to_ids(step1_indices),
                "num_selected": len(step1_indices),
            },
            "final_doc_ids": _indices_to_ids(selected_indices),
            "num_final": len(selected_indices),
        }, ensure_ascii=False)

        full_trace = _json.dumps({
            "step1_selection_raw": step1.get("raw", ""),
        }, ensure_ascii=False)

        document.ranker_raw_outputs = [step_summary, full_trace]
