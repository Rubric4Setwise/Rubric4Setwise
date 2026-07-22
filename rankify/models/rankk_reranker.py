"""
Rank-K Reranker: Listwise Sliding Window Reranking with Test-Time Reasoning.

Based on: https://github.com/hltcoe/rank-k
Models: hltcoe/Rank-K-32B

Rank-K uses a listwise sliding window approach:
- Each window presents multiple passages to the LLM
- LLM outputs a ranking like "[3] > [2] > [4] = [1] > [5]"
- Supports ties (= sign between equal-relevance docs)
- Sliding window moves from bottom to top (bubble-sort style)
- Uses non-zero temperature for test-time reasoning
"""

import os
import re
import copy
from typing import List, Optional, Dict

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


RANK_K_PROMPT = """Determine a ranking of the passages based on how relevant they are to the query. 
If the query is a question, how relevant a passage is depends on how well it answers the question. 
If not, try analyze the intent of the query and assess how well each passage satisfy the intent. 
The query may have typos and passages may contain contradicting information. 
However, we do not get into fact-checking. We just rank the passages based on they relevancy to the query. 

Sort them from the most relevant to the least. 
Answer with the passage number using a format of `[3] > [2] > [4] = [1] > [5]`. 
Ties are acceptable if they are equally relevant. 
I need you to be accurate but overthinking it is unnecessary.
Output only the ordering without any other text.

Query: {query}

{docs}"""


class RankKReranker(BaseRanking):
    """
    Rank-K: Listwise Sliding Window Reranking with Test-Time Reasoning.

    This reranker uses a sliding window to present groups of documents to an LLM,
    which produces a ranked list with support for ties.

    Args:
        method (str): Method name ('rankk').
        model_name (str): HuggingFace model ID or local path.
        api_key (str): Not used (local model).
        **kwargs: Additional parameters:
            - window_size (int): Sliding window size (default: 20)
            - step_size (int): Step size for sliding window (default: 10)
            - max_tokens (int): Max generation tokens (default: 4000)
            - max_passage_length (int): Max tokens per passage (default: 300)
            - temperature (float): Generation temperature (default: 0.7)
            - num_gpus (int): Number of GPUs (default: 1)
            - gpu_memory_utilization (float): GPU memory fraction (default: 0.95)
            - context_size (int): Model context window (default: 32768)

    Example:
        >>> reranker = Reranking(method='rankk', model_name='hltcoe/Rank-K-32B')
        >>> results = reranker.rank(documents)
    """

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "rankk"
        model_name = model_name or "hltcoe/Rank-K-32B"

        # Resolve model path: download from HuggingFace/ModelScope if not local
        from rankify.utils.model_downloader import resolve_model_path
        self.model_name = resolve_model_path(model_name)

        # Configuration
        self.window_size = kwargs.get("window_size", 20)
        self.step_size = kwargs.get("step_size", 10)
        self.max_tokens = kwargs.get("max_tokens", 4000)
        self.max_passage_length = kwargs.get("max_passage_length", 300)
        self.temperature = kwargs.get("temperature", 0.7)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.95)
        self.context_size = kwargs.get("context_size", 32768)

        if LLM is None:
            raise ImportError("vLLM is required for RankK. Please install: pip install vllm")

        # Initialize vLLM
        print(
            f"[RankK] Loading model: {self.model_name} "
            f"(window={self.window_size}, step={self.step_size}, "
            f"temp={self.temperature}, num_gpus={self.num_gpus})"
        )
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.context_size,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        print(f"[RankK] Model loaded successfully.")

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max_length tokens."""
        tokens = self._tokenizer.tokenize(text)[:max_length]
        return self._tokenizer.convert_tokens_to_string(tokens)

    def _combine_passages(self, passages: List[str]) -> str:
        """Format passages as [1] text\n\n[2] text\n\n..."""
        return "\n\n".join(f"[{i+1}] {text}" for i, text in enumerate(passages))

    def _parse_ranking(self, rank_string: str, num_docs: int) -> List[int]:
        """
        Parse ranking output like "[3] > [2] > [4] = [1] > [5]" into ordered indices.
        Supports ties (= sign). Returns 0-indexed list of doc indices.
        """
        groups = rank_string.split(">")
        ordered_indices = []
        seen = set()

        for group in groups:
            for item in group.split("="):
                item = item.strip().replace("[", "").replace("]", "")
                try:
                    idx = int(item) - 1  # Convert to 0-indexed
                    if 0 <= idx < num_docs and idx not in seen:
                        ordered_indices.append(idx)
                        seen.add(idx)
                except (ValueError, IndexError):
                    continue

        # Append any missing indices at the end
        for i in range(num_docs):
            if i not in seen:
                ordered_indices.append(i)

        return ordered_indices

    def _rerank_window(self, query: str, passages: List[str]) -> List[int]:
        """
        Rerank a single window of passages.
        Returns permutation indices (0-based).
        """
        docs_text = self._combine_passages(passages)
        content = RANK_K_PROMPT.format(query=query, docs=docs_text)

        messages = [{"role": "user", "content": content}]

        outputs = self._llm.chat(
            [messages],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        response = outputs[0].outputs[0].text.strip()
        # Take last line (model might have reasoning before)
        last_line = response.split("\n")[-1]

        return self._parse_ranking(last_line, len(passages))

    def _sliding_windows(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Apply sliding window reranking from bottom to top.
        """
        num_candidates = len(candidates)
        window_size = min(self.window_size, num_candidates)
        step_size = min(self.step_size, window_size)

        rank_end = num_candidates
        rank_start = 0

        end_pos = rank_end
        start_pos = end_pos - window_size

        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            
            # Get passages in this window
            window_passages = [c['text'] for c in candidates[start_pos:end_pos]]
            
            # Get reranked order
            permutation = self._rerank_window(query, window_passages)
            
            # Apply permutation
            window_candidates = candidates[start_pos:end_pos]
            reordered = [window_candidates[i] for i in permutation]
            candidates[start_pos:end_pos] = reordered

            # Slide window
            end_pos -= step_size
            start_pos -= step_size

            if start_pos <= rank_start and end_pos <= rank_start + window_size:
                break

        return candidates

    def _sliding_windows_batched(self, queries: List[str], all_candidates: List[List[Dict]]) -> List[List[Dict]]:
        """
        Apply sliding window reranking in batch mode across multiple queries.
        Each window position is batched across all queries.
        """
        # Calculate window positions for each query
        all_positions = []
        for candidates in all_candidates:
            num_candidates = len(candidates)
            window_size = min(self.window_size, num_candidates)
            step_size = min(self.step_size, window_size)

            positions = []
            end_pos = num_candidates
            start_pos = end_pos - window_size

            while end_pos > 0:
                actual_start = max(start_pos, 0)
                positions.append((actual_start, end_pos))
                end_pos -= step_size
                start_pos -= step_size
                if actual_start == 0:
                    break

            all_positions.append(positions)

        # Process window positions in batches
        max_windows = max(len(p) for p in all_positions)

        for window_idx in range(max_windows):
            batch_messages = []
            batch_metadata = []  # (query_idx, start, end)

            for q_idx, (candidates, positions) in enumerate(zip(all_candidates, all_positions)):
                if window_idx < len(positions):
                    start_pos, end_pos = positions[window_idx]
                    window_passages = [c['text'] for c in candidates[start_pos:end_pos]]
                    docs_text = self._combine_passages(window_passages)
                    content = RANK_K_PROMPT.format(query=queries[q_idx], docs=docs_text)
                    batch_messages.append([{"role": "user", "content": content}])
                    batch_metadata.append((q_idx, start_pos, end_pos))

            if not batch_messages:
                continue

            # Batch inference
            outputs = self._llm.chat(
                batch_messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # Apply permutations
            for (q_idx, start_pos, end_pos), output in zip(batch_metadata, outputs):
                response = output.outputs[0].text.strip()
                # Collect raw LLM output
                self._raw_outputs_per_query[q_idx].append(response)
                last_line = response.split("\n")[-1]
                window_size_actual = end_pos - start_pos
                permutation = self._parse_ranking(last_line, window_size_actual)

                window_candidates = all_candidates[q_idx][start_pos:end_pos]
                reordered = [window_candidates[i] for i in permutation]
                all_candidates[q_idx][start_pos:end_pos] = reordered

        return all_candidates

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using listwise sliding window.

        Args:
            documents: List of Document objects to rerank.

        Returns:
            List of Document objects with reorder_contexts set.
        """
        queries = []
        all_candidates = []

        for doc in documents:
            contexts = doc.contexts
            if not contexts:
                queries.append("")
                all_candidates.append([])
                continue

            query = doc.question
            queries.append(query)

            candidates = []
            for ctx in contexts:
                text = self._truncate(ctx.text, self.max_passage_length)
                candidates.append({'text': text, 'ctx': ctx})
            all_candidates.append(candidates)

        # Initialize raw output collectors
        self._raw_outputs_per_query = {i: [] for i in range(len(queries))}

        # Batch reranking
        all_candidates = self._sliding_windows_batched(queries, all_candidates)

        # Write results back
        for i, (doc, candidates) in enumerate(zip(documents, all_candidates)):
            if not candidates:
                continue

            reorder_contexts = []
            for j, c in enumerate(candidates):
                c['ctx'].score = float(len(candidates) - j)
                reorder_contexts.append(c['ctx'])

            doc.reorder_contexts = reorder_contexts
            # Save collected raw LLM outputs
            doc.ranker_raw_outputs = self._raw_outputs_per_query.get(i, [])

        return documents
