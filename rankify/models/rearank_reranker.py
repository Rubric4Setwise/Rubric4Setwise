"""
Rearank Reranker: Reasoning-Enhanced Listwise Reranking Agent.

Based on: https://github.com/lezhang7/Rearank
Models: le723z/Rearank-7B

Rearank uses a listwise sliding window approach with explicit reasoning:
- System prompt identifies it as "DeepRerank"
- Uses <think>/<answer> format for reasoning before ranking
- Each passage is presented individually with acknowledgment
- Final ranking in <answer> [1] > [2] > ... </answer> format
- Sliding window from bottom to top (batch mode supported)
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


REARANK_SYSTEM_PROMPT = (
    "You are DeepRerank, an intelligent assistant that can rank passages based on "
    "their relevancy to the search query. You first thinks about the reasoning "
    "process in the mind and then provides the user with the answer."
)


class RearankReranker(BaseRanking):
    """
    Rearank: Reasoning-Enhanced Listwise Reranking Agent.

    This reranker uses a multi-turn conversation format with explicit reasoning
    to perform listwise reranking with sliding windows.

    Args:
        method (str): Method name ('rearank').
        model_name (str): HuggingFace model ID or local path.
        api_key (str): Not used (local model).
        **kwargs: Additional parameters:
            - window_size (int): Sliding window size (default: 20)
            - step_size (int): Step size for sliding window (default: 10)
            - max_tokens (int): Max generation tokens (default: 2048)
            - max_passage_length (int): Max chars per passage (default: 400)
            - num_gpus (int): Number of GPUs (default: 1)
            - gpu_memory_utilization (float): GPU memory fraction (default: 0.9)
            - context_size (int): Model context window (default: 32768)
            - enable_thinking (bool): Enable thinking mode for Qwen3 (default: True)

    Example:
        >>> reranker = Reranking(method='rearank', model_name='le723z/Rearank-7B')
        >>> results = reranker.rank(documents)
    """

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "rearank"
        model_name = model_name or "le723z/Rearank-7B"

        # Resolve model path: download from HuggingFace/ModelScope if not local
        from rankify.utils.model_downloader import resolve_model_path
        self.model_name = resolve_model_path(model_name)

        # Configuration
        self.window_size = kwargs.get("window_size", 20)
        self.step_size = kwargs.get("step_size", 10)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.max_passage_length = kwargs.get("max_passage_length", 400)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.context_size = kwargs.get("context_size", 32768)
        self.enable_thinking = kwargs.get("enable_thinking", True)

        if LLM is None:
            raise ImportError("vLLM is required for Rearank. Please install: pip install vllm")

        # Initialize vLLM
        print(
            f"[Rearank] Loading model: {self.model_name} "
            f"(window={self.window_size}, step={self.step_size}, "
            f"num_gpus={self.num_gpus}, thinking={self.enable_thinking})"
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
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_tokens=self.max_tokens,
        )

        print(f"[Rearank] Model loaded successfully.")

    def _truncate_chars(self, text: str, max_chars: int) -> str:
        """Truncate text to max_chars characters (word-boundary aware)."""
        text = ' '.join(text.split())  # normalize whitespace
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _create_messages(self, query: str, passages: List[str]) -> List[Dict]:
        """
        Create the Rearank multi-turn conversation format.
        """
        num = len(passages)
        instruction = (
            f"I will provide you with passages, each indicated by number identifier []. "
            f"Rank the passages based on their relevance to the search query."
            f"Search Query: {query}. \n"
            f"Rank the {num} passages above based on their relevance to the search query."
            f"The passages should be listed in descending order using identifiers. "
            f"The most relevant passages should be listed first. "
            f"The output format should be <answer> [] > [] </answer>, "
            f"e.g., <answer> [1] > [2] </answer>."
        )

        messages = [
            {"role": "system", "content": REARANK_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

        # Add each passage with acknowledgment
        for rank, content in enumerate(passages, 1):
            content = content.replace('Title: Content: ', '').strip()
            content = ' '.join(content.split())
            messages.append({"role": "user", "content": f"[{rank}] {content}"})
            messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})

        # Final ranking request
        messages.append({
            "role": "user",
            "content": (
                f'Please rank these passages according to their relevance to the search query: "{query}"\n'
                f"Follow these steps exactly:\n"
                f"1. First, within <think> tags, analyze EACH passage individually:\n"
                f"- Evaluate how well it addresses the query\n"
                f"- Note specific relevant information or keywords\n\n"
                f"2. Then, within <answer> tags, provide ONLY the final ranking in "
                f"descending order of relevance using the format: [X] > [Y] > [Z]"
            ),
        })

        return messages

    def _clean_response(self, response: str) -> str:
        """Extract ranking from response, handling both <answer> tags and plain output."""
        # Try to extract from <answer> tags
        if "<answer>" in response:
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if match:
                response = match.group(1).strip()

        # Handle Qwen3 think tags without answer tags
        elif "<think>" in response:
            response = response.split("</think>")[-1]

        # Extract just the numbers (remove brackets, >, and other non-digit chars)
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        return new_response.strip()

    def _parse_permutation(self, response: str, num_docs: int) -> List[int]:
        """Parse the cleaned response into a permutation (0-indexed)."""
        cleaned = self._clean_response(response)

        indices = []
        seen = set()
        for x in cleaned.split():
            try:
                idx = int(x) - 1  # Convert to 0-indexed
                if 0 <= idx < num_docs and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
            except ValueError:
                continue

        # Remove duplicates while preserving order
        # Append missing indices
        for i in range(num_docs):
            if i not in seen:
                indices.append(i)

        return indices

    def _apply_permutation(self, candidates: List[Dict], start: int, end: int, permutation: List[int]):
        """Apply permutation to a window slice of candidates."""
        window = candidates[start:end]
        reordered = [copy.deepcopy(window[i]) for i in permutation]
        candidates[start:end] = reordered

    def _sliding_windows_batched(self, queries: List[str], all_candidates: List[List[Dict]]) -> List[List[Dict]]:
        """
        Apply sliding window reranking in batch mode.
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

            while start_pos >= 0:
                start_pos = max(start_pos, 0)
                positions.append((start_pos, end_pos))
                end_pos -= step_size
                start_pos -= step_size
                if positions[-1][0] == 0:
                    break

            all_positions.append(positions)

        # Process window positions in batches (synchronized across queries)
        max_windows = max(len(p) for p in all_positions) if all_positions else 0

        for window_idx in range(max_windows):
            batch_texts = []
            batch_metadata = []

            for q_idx, (candidates, positions) in enumerate(zip(all_candidates, all_positions)):
                if window_idx < len(positions):
                    start_pos, end_pos = positions[window_idx]
                    passages = [c['text'] for c in candidates[start_pos:end_pos]]
                    messages = self._create_messages(queries[q_idx], passages)

                    # Apply chat template
                    if 'Qwen3' in self.model_name:
                        text = self._tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=self.enable_thinking
                        )
                    else:
                        text = self._tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )

                    batch_texts.append(text)
                    batch_metadata.append((q_idx, start_pos, end_pos))

            if not batch_texts:
                continue

            # Batch inference
            outputs = self._llm.generate(batch_texts, self.sampling_params)

            # Apply permutations
            for (q_idx, start_pos, end_pos), output in zip(batch_metadata, outputs):
                response = output.outputs[0].text
                # Collect raw LLM output
                self._raw_outputs_per_query[q_idx].append(response)
                num_docs = end_pos - start_pos
                permutation = self._parse_permutation(response, num_docs)
                self._apply_permutation(all_candidates[q_idx], start_pos, end_pos, permutation)

        return all_candidates

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using Rearank's listwise sliding window with reasoning.

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
                text = self._truncate_chars(ctx.text, self.max_passage_length)
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
