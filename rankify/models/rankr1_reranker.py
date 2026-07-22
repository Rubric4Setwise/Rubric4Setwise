"""
Rank-R1 Reranker: Setwise Reranking with Reasoning (HeapSort/BubbleSort).

Based on: https://github.com/ielab/llm-rankers/tree/main/Rank-R1
Models:
  - ielabgroup/Rank-R1-7B-v0.1 (LoRA on Qwen2.5-7B-Instruct)
  - ielabgroup/Rank-R1-14B-v0.1 (LoRA on Qwen2.5-14B-Instruct)
  - ielabgroup/Setwise-SFT-7B-v0.1 (LoRA on Qwen2.5-7B-Instruct)
  - ielabgroup/Setwise-SFT-14B-v0.1 (LoRA on Qwen2.5-14B-Instruct)

Rank-R1 uses a setwise comparison approach:
- Each comparison presents a group of documents (num_child+1) to the LLM
- The LLM selects the MOST relevant document from the group
- HeapSort or BubbleSort algorithm uses these comparisons to rank
- Rank-R1: <think>/<answer> reasoning format
- Setwise-SFT: <answer> only (no reasoning)

Note: These models are LoRA adapters, loaded on top of Qwen2.5 base models via vLLM's LoRA support.
"""

import os
import re
import copy
import random
from typing import List, Optional, Dict
from collections import Counter

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM = None
    SamplingParams = None
    LoRARequest = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# Default prompt template (same as prompt_setwise-R1.toml)
DEFAULT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)

DEFAULT_USER_PROMPT = (
    'Given the query: "{query}", which of the following documents is most relevant?\n'
    '{docs}\n'
    'After completing the reasoning process, please provide only the label of the '
    'most relevant document to the query, enclosed in square brackets, within the '
    'answer tags. For example, if the third document is the most relevant, the answer '
    'should be: <think> reasoning process here </think> <answer>[3]</answer>.'
)

DEFAULT_PATTERN = r'<think>.*?</think>\s*<answer>(.*?)</answer>'
DEFAULT_DOC_PREFIX = "[{num}]: "
DEFAULT_DOC_SEPARATOR = "\n"

# ---- Setwise-SFT prompt (no reasoning, only <answer>) ----
SETWISE_SFT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant provides the user with the "
    "answer enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>."
)

SETWISE_SFT_USER_PROMPT = (
    'Given the query: "{query}", which of the following documents is most relevant?\n'
    '{docs}\n'
    'Please provide only the label of the most relevant document to the query, '
    'enclosed in square brackets, within the answer tags. For example, if the third '
    'document is the most relevant, the answer should be: <answer>[3]</answer>.'
)

SETWISE_SFT_PATTERN = r'<answer>(.*?)</answer>'

# ---- LoRA adapter → base model mapping ----
# These models are LoRA adapters that need a base model for inference.
LORA_BASE_MODEL_MAP = {
    "ielabgroup/Rank-R1-7B-v0.1": "Qwen/Qwen2.5-7B-Instruct",
    "ielabgroup/Rank-R1-14B-v0.1": "Qwen/Qwen2.5-14B-Instruct",
    "ielabgroup/Setwise-SFT-7B-v0.1": "Qwen/Qwen2.5-7B-Instruct",
    "ielabgroup/Setwise-SFT-14B-v0.1": "Qwen/Qwen2.5-14B-Instruct",
    "ielabgroup/Setwise-SFT-3B-v0.1": "Qwen/Qwen2.5-3B-Instruct",
}

# Setwise-SFT models use different prompts than Rank-R1
SETWISE_SFT_MODELS = {
    "ielabgroup/Setwise-SFT-7B-v0.1",
    "ielabgroup/Setwise-SFT-14B-v0.1",
    "ielabgroup/Setwise-SFT-3B-v0.1",
}


class RankR1Reranker(BaseRanking):
    """
    Rank-R1: Setwise Reranking with Reasoning via HeapSort/BubbleSort.

    This reranker uses an LLM to compare groups of documents and select the
    most relevant one, then uses sorting algorithms to produce a final ranking.

    Args:
        method (str): Method name ('rankr1').
        model_name (str): HuggingFace model ID or local path.
        api_key (str): Not used (local model).
        **kwargs: Additional parameters:
            - num_child (int): Number of children per comparison (default: 19, i.e. 20 docs per group)
            - k (int): Number of top documents to rank (default: 10)
            - sort_method (str): "heapsort" or "bubblesort" (default: "heapsort")
            - num_permutation (int): Number of random permutations for voting (default: 1)
            - max_tokens (int): Max generation tokens (default: 8000)
            - max_passage_length (int): Max tokens per passage (default: 128)
            - num_gpus (int): Number of GPUs for tensor parallelism (default: 1)
            - gpu_memory_utilization (float): GPU memory fraction (default: 0.9)
            - context_size (int): Model context window (default: 32768)

    Example:
        >>> reranker = Reranking(method='rankr1', model_name='ielabgroup/Rank-R1-7B-v0.1')
        >>> results = reranker.rank(documents)
    """

    CHARACTERS = [f'[{i+1}]' for i in range(20)]

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "rankr1"
        model_name = model_name or "ielabgroup/Rank-R1-7B-v0.1"

        # Resolve model path: download from HuggingFace/ModelScope if not local
        from rankify.utils.model_downloader import resolve_model_path

        # ---- LoRA detection ----
        # If model_name is a known LoRA adapter, resolve base model + LoRA path separately
        self.lora_path = None
        self.lora_request = None

        if model_name in LORA_BASE_MODEL_MAP:
            base_model_id = LORA_BASE_MODEL_MAP[model_name]
            self.model_name = resolve_model_path(base_model_id)
            # Resolve LoRA adapter path (download if needed)
            self.lora_path = resolve_model_path(model_name)
            self._is_setwise_sft = model_name in SETWISE_SFT_MODELS
        else:
            self.model_name = resolve_model_path(model_name)
            self._is_setwise_sft = False

        # Configuration
        self.num_child = kwargs.get("num_child", 19)  # 20 docs per comparison group
        self.k = kwargs.get("k", 10)  # rank top-k
        self.sort_method = kwargs.get("sort_method", "heapsort")
        self.num_permutation = kwargs.get("num_permutation", 1)
        self.max_tokens = kwargs.get("max_tokens", 8000)
        self.max_passage_length = kwargs.get("max_passage_length", 128)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.context_size = kwargs.get("context_size", 32768)

        # Prompt configuration: use Setwise-SFT prompts if applicable
        if self._is_setwise_sft:
            self.system_prompt = kwargs.get("system_prompt", SETWISE_SFT_SYSTEM_PROMPT)
            self.user_prompt = kwargs.get("user_prompt", SETWISE_SFT_USER_PROMPT)
            self.pattern = kwargs.get("pattern", SETWISE_SFT_PATTERN)
        else:
            self.system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            self.user_prompt = kwargs.get("user_prompt", DEFAULT_USER_PROMPT)
            self.pattern = kwargs.get("pattern", DEFAULT_PATTERN)
        self.doc_prefix = kwargs.get("doc_prefix", DEFAULT_DOC_PREFIX)
        self.doc_separator = kwargs.get("doc_separator", DEFAULT_DOC_SEPARATOR)

        # Validate
        if self.sort_method not in ["heapsort", "bubblesort"]:
            raise ValueError(f"sort_method must be 'heapsort' or 'bubblesort', got '{self.sort_method}'")

        if LLM is None:
            raise ImportError("vLLM is required for RankR1. Please install: pip install vllm")

        # Initialize vLLM engine
        lora_info = f", lora={self.lora_path}" if self.lora_path else ""
        print(
            f"[RankR1] Loading model: {self.model_name}{lora_info} "
            f"(sort={self.sort_method}, num_child={self.num_child}, k={self.k}, "
            f"num_gpus={self.num_gpus})"
        )

        llm_kwargs = dict(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.context_size,
            trust_remote_code=True,
        )
        # Enable LoRA in vLLM if we have an adapter
        if self.lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64

        self._llm = LLM(**llm_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Build LoRARequest for inference
        if self.lora_path and LoRARequest is not None:
            self.lora_request = LoRARequest("rankr1_adapter", 1, self.lora_path)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

        print(f"[RankR1] Model loaded successfully.")

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max_length tokens."""
        tokens = self._tokenizer.tokenize(text)[:max_length]
        return self._tokenizer.convert_tokens_to_string(tokens)

    def _compare(self, query: str, docs: List[Dict]) -> int:
        """
        Compare a group of documents and return the index of the most relevant one.
        Uses num_permutation random shuffles and majority voting.

        Args:
            query: The query string
            docs: List of dicts with 'text' key

        Returns:
            Index of the winning document in the original order
        """
        id_passage = list(enumerate(docs))
        labels = self.CHARACTERS[:len(docs)]

        batch_data = []
        for _ in range(self.num_permutation):
            batch_data.append(random.sample(id_passage, len(id_passage)))

        batch_ref = []
        input_texts = []

        for shuffled in batch_data:
            ref = [item[0] for item in shuffled]
            passages = [item[1]['text'] for item in shuffled]
            batch_ref.append(ref)

            # Build document text
            doc_texts = []
            for i, passage in enumerate(passages):
                prefix = self.doc_prefix.format(num=i + 1)
                doc_texts.append(f"{prefix}{passage}")
            docs_text = self.doc_separator.join(doc_texts)

            # Build messages
            user_content = self.user_prompt.format(query=query, docs=docs_text)
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_content},
            ]
            input_texts.append(messages)

        # Batch inference (with LoRA if configured)
        chat_kwargs = dict(
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        if self.lora_request is not None:
            chat_kwargs["lora_request"] = self.lora_request

        outputs = self._llm.chat(
            input_texts,
            **chat_kwargs,
        )

        # Parse results
        results = []
        for output in outputs:
            completion = output.outputs[0].text
            # Collect raw LLM outputs for this comparison
            self._current_raw_outputs.append(completion)
            match = re.search(self.pattern, completion.lower(), re.DOTALL)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(None)

        # Vote
        candidates = []
        for ref, result in zip(batch_ref, results):
            if result is None:
                continue
            # Find which label was selected
            for i, label in enumerate(labels):
                if label.lower() == result or f"[{i+1}]" == result:
                    # Map back to original index
                    candidates.append(ref[i])
                    break

        if not candidates:
            return 0  # fallback: return first doc

        # Majority voting
        candidate_counts = Counter(candidates)
        max_count = max(candidate_counts.values())
        most_common = [c for c, cnt in candidate_counts.items() if cnt == max_count]
        return most_common[0] if len(most_common) == 1 else random.choice(most_common)

    def _heapify(self, arr: List[Dict], n: int, i: int, query: str):
        """Heapify subtree rooted at index i."""
        if self.num_child * i + 1 < n:
            # Get root and its children
            children_start = self.num_child * i + 1
            children_end = min(self.num_child * (i + 1) + 1, n)
            docs = [arr[i]] + arr[children_start:children_end]
            inds = [i] + list(range(children_start, children_end))

            best_idx = self._compare(query, docs)
            largest = inds[best_idx]

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self._heapify(arr, n, largest, query)

    def _heap_sort(self, arr: List[Dict], query: str, k: int):
        """HeapSort: sort top-k elements."""
        n = len(arr)
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self._heapify(arr, n, i, query)
        # Extract top-k
        ranked = 0
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self._heapify(arr, i, 0, query)

    def _bubble_sort(self, arr: List[Dict], query: str, k: int):
        """BubbleSort: bubble top-k to front."""
        n = len(arr)
        for i in range(k):
            start_ind = n - (self.num_child + 1)
            end_ind = n
            while True:
                if start_ind < i:
                    start_ind = i
                docs = arr[start_ind:end_ind]
                best_idx = self._compare(query, docs)
                if best_idx != 0:
                    arr[start_ind], arr[start_ind + best_idx] = arr[start_ind + best_idx], arr[start_ind]
                if start_ind == i:
                    break
                start_ind -= self.num_child
                end_ind -= self.num_child

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using setwise comparison with HeapSort/BubbleSort.

        Args:
            documents: List of Document objects to rerank.

        Returns:
            List of Document objects with reorder_contexts set.
        """
        for doc in documents:
            contexts = doc.contexts
            if not contexts:
                continue

            # Initialize raw output collector for this document
            self._current_raw_outputs = []

            query = doc.question

            # Convert contexts to sortable list
            candidates = []
            for ctx in contexts:
                text = self._truncate(ctx.text, self.max_passage_length)
                candidates.append({
                    'text': text,
                    'ctx': ctx,
                })

            # Sort
            k = min(self.k, len(candidates))
            if self.sort_method == "heapsort":
                self._heap_sort(candidates, query, k)
                candidates = list(reversed(candidates))
            else:
                self._bubble_sort(candidates, query, k)

            # Build reorder_contexts: top-k sorted + remaining in original order
            top_k_ctxs = [c['ctx'] for c in candidates[:k]]
            remaining_ids = set()
            for c in candidates[:k]:
                remaining_ids.add(id(c['ctx']))
            remaining = [c['ctx'] for c in candidates[k:]]
            
            reorder_contexts = top_k_ctxs + remaining

            # Assign scores (descending)
            for i, ctx in enumerate(reorder_contexts):
                ctx.score = float(len(reorder_contexts) - i)

            doc.reorder_contexts = reorder_contexts
            # Save collected raw LLM outputs
            doc.ranker_raw_outputs = self._current_raw_outputs

        return documents
