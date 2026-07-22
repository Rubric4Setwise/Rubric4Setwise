"""
ReasonRank Reranker - Reasoning-Enhanced Listwise Document Reranking

ReasonRank (ACL 2025) enhances listwise ranking by injecting explicit reasoning
into the ranking process. The model first thinks step-by-step about the relevance
of each passage, then outputs the final ranking.

Key features:
- Reasoning-enhanced listwise ranking with <think>...</think><answer>...</answer> format
- Sliding window algorithm for handling long document lists
- Supports vLLM batched inference for high throughput
- Multiple prompt modes: reasoning (default), standard RankGPT, Qwen3-style

Reference:
    "ReasonRank: Reasoning-Enhanced Listwise Reranking"
    (ACL 2025)

Usage:
    from rankify.models.reranking import Reranking

    model = Reranking(
        method='reasonrank',
        model_name='reasonrank-7b',
        vllm_batched=True,
        num_gpus=4,
        window_size=20,
        step_size=10,
    )
    results = model.rank(documents)
"""

import os
import copy
import re
from typing import List, Optional, Tuple, Dict, Any

import torch
from tqdm import tqdm
from ftfy import fix_text

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


# =========================================================
# Prompt Templates (from ReasonRank paper)
# =========================================================

# System prompt for reasoning mode
REASONRANK_SYSTEM_PROMPT_REASONING = (
    "You are RankLLM, an intelligent assistant that can rank passages based on "
    "their relevance to the query. Given a query and a passage list, you first "
    "thinks about the reasoning process in the mind and then provides the answer "
    "(i.e., the reranked passage list). The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

# System prompt for standard (non-reasoning) mode
REASONRANK_SYSTEM_PROMPT_STANDARD = (
    "You are RankLLM, an intelligent assistant that can rank passages based on "
    "their relevance to the query."
)

# Pattern to extract answer from reasoning output
REASONING_PATTERN = r'<think>.*?</think>\s*<answer>(.*?)</answer>'


# =========================================================
# Output parsing
# =========================================================

def _clean_response_reasoning(response: str) -> str:
    """
    Parse ReasonRank model output in reasoning mode.
    Extracts the ranking from <think>...</think><answer>...</answer> format.
    """
    # Try to extract answer from <think>...<answer>...</answer> pattern
    match = re.search(REASONING_PATTERN, response.lower(), re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        # Fallback: try to find <answer> tag without proper <think> closure
        if '<answer>' in response.lower():
            response = response.lower().split('<answer>')[-1]
            if '</answer>' in response:
                response = response.split('</answer>')[0]
        else:
            # Reasoning might be too long; use full response
            pass

    # Extract only digits (passage numbers)
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    return new_response.strip()


def _clean_response_standard(response: str) -> str:
    """Parse standard RankGPT output (no reasoning tags)."""
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    return new_response.strip()


def _parse_permutation(cleaned_response: str, num_passages: int) -> List[int]:
    """
    Parse cleaned response into a permutation (0-based indices).
    
    Args:
        cleaned_response: Space-separated digits string
        num_passages: Number of passages in the window
    
    Returns:
        List of 0-based indices representing the ranking
    """
    try:
        response = [int(x) - 1 for x in cleaned_response.split()]
    except ValueError:
        return list(range(num_passages))

    # Remove duplicates while preserving order
    seen = set()
    unique_response = []
    for idx in response:
        if idx not in seen and 0 <= idx < num_passages:
            seen.add(idx)
            unique_response.append(idx)

    # Append any missing indices in original order
    for idx in range(num_passages):
        if idx not in seen:
            unique_response.append(idx)

    return unique_response


# =========================================================
# Main Reranker Class
# =========================================================

class ReasonRankReranker(BaseRanking):
    """
    ReasonRank: Reasoning-Enhanced Listwise Document Reranking.

    This reranker uses a fine-tuned LLM with explicit reasoning capabilities
    to perform listwise ranking using a sliding window approach.

    Args:
        method (str): Method name ('reasonrank').
        model_name (str): Path or HuggingFace model ID for the ranking model.
        api_key (str): Not used (local model). Kept for interface compatibility.
        **kwargs: Additional parameters:
            - prompt_mode (str): "reasoning" (default), "standard", or "qwen3"
            - window_size (int): Sliding window size (default: 20)
            - step_size (int): Step size for sliding window (default: 10)
            - max_passage_length (int): Max tokens per passage (default: 100)
            - reasoning_max_tokens (int): Max generation tokens for reasoning (default: 3172)
            - context_size (int): Model context window size (default: 32768)
            - num_gpus (int): Number of GPUs for tensor parallelism (default: 1)
            - vllm_batched (bool): Use vLLM batched inference (default: True)
            - gpu_memory_utilization (float): GPU memory fraction for vLLM (default: 0.9)
            - batch_size (int): Batch size for prompt construction (default: 32)

    Example:
        >>> from rankify.models.reranking import Reranking
        >>> reranker = Reranking(
        ...     method='reasonrank',
        ...     model_name='reasonrank-7b',
        ...     vllm_batched=True,
        ...     num_gpus=4,
        ... )
        >>> results = reranker.rank(documents)
    """

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "reasonrank"
        model_name = model_name or "liuwenhan/reasonrank-7B"

        # Resolve model path: download from HuggingFace/ModelScope if not local
        from rankify.utils.model_downloader import resolve_model_path
        self.model_name = resolve_model_path(model_name)

        # Configuration
        self.prompt_mode = kwargs.get("prompt_mode", "reasoning")
        self.window_size = kwargs.get("window_size", 20)
        self.step_size = kwargs.get("step_size", 10)
        self.max_passage_length = kwargs.get("max_passage_length", 100)
        self.reasoning_max_tokens = kwargs.get("reasoning_max_tokens", 3172)
        self.context_size = kwargs.get("context_size", 32768)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.vllm_batched = kwargs.get("vllm_batched", True)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.batch_size = kwargs.get("batch_size", 32)

        # Validate
        valid_modes = ["reasoning", "standard", "qwen3"]
        if self.prompt_mode not in valid_modes:
            raise ValueError(
                f"prompt_mode must be one of {valid_modes}, got '{self.prompt_mode}'"
            )

        if not self.vllm_batched:
            raise ValueError(
                "ReasonRank requires vllm_batched=True. "
                "Please install vLLM: pip install vllm"
            )

        if LLM is None:
            raise ImportError(
                "vLLM is required for ReasonRank. "
                "Please install: pip install vllm"
            )

        # Initialize vLLM engine
        print(
            f"[ReasonRank] Loading model: {self.model_name} "
            f"(prompt_mode={self.prompt_mode}, window_size={self.window_size}, "
            f"num_gpus={self.num_gpus})"
        )
        self._llm = LLM(
            self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=False,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.context_size,
        )
        self._tokenizer = self._llm.get_tokenizer()

        print(f"[ReasonRank] Model loaded successfully.")

    def _replace_number(self, s: str) -> str:
        """Replace [N] patterns in text to avoid confusion with passage IDs."""
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def _get_system_prompt(self) -> str:
        """Get system prompt based on prompt_mode."""
        if self.prompt_mode == "reasoning":
            return REASONRANK_SYSTEM_PROMPT_REASONING
        else:
            return REASONRANK_SYSTEM_PROMPT_STANDARD

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        """Generate prefix prompt."""
        return (
            f"I will provide you with {num} passages, each indicated by a "
            f"numerical identifier []. Rank the passages based on their "
            f"relevance to the search query: {query}.\n"
        )

    def _add_post_prompt(self, query: str, num: int) -> str:
        """Generate post prompt based on prompt_mode."""
        example_ordering = "[2] > [1]"
        if self.prompt_mode == "reasoning":
            return (
                f"Search Query: {query}.\n"
                f"Rank the {num} passages above based on their relevance to "
                f"the search query. All the passages should be included and "
                f"listed using identifiers, in descending order of relevance. "
                f"The format of the answer should be [] > [], e.g., {example_ordering}."
            )
        elif self.prompt_mode == "qwen3":
            return (
                f"Search Query: {query}.\n"
                f"Rank the {num} passages above based on their relevance to "
                f"the search query. All the passages should be included and "
                f"listed using identifiers, in descending order of relevance. "
                f"The final ranked list should be enclosed within <answer> </answer> "
                f"tags, i.e., <answer> ranked list here </answer>. "
                f"The format of ranked list should be [] > [], e.g., {example_ordering}. "
                f"Only respond with the ranking results, do not say any word or explain."
            )
        else:  # standard
            return (
                f"Search Query: {query}.\n"
                f"Rank the {num} passages above based on their relevance to "
                f"the search query. All the passages should be included and "
                f"listed using identifiers, in descending order of relevance. "
                f"The output format should be [] > [], e.g., {example_ordering}. "
                f"Only respond with the ranking results, do not say any word or explain."
            )

    def _convert_doc_to_prompt_content(self, text: str, max_length: int) -> str:
        """Truncate document content to max_length tokens."""
        content = (text or "").strip()
        content = fix_text(content)
        # Truncate by tokens
        tokens = self._tokenizer.tokenize(content)[:max_length]
        content = self._tokenizer.convert_tokens_to_string(tokens)
        return self._replace_number(content)

    def _create_prompt(self, query: str, passages: List[str], rank_start: int, rank_end: int) -> str:
        """
        Build the full prompt for a single window.
        
        Args:
            query: The search query
            passages: List of passage texts for the current window
            rank_start: Start index (for context, not directly used in prompt)
            rank_end: End index
            
        Returns:
            Formatted prompt string ready for the model
        """
        query = self._replace_number(query).strip()
        num = len(passages)

        messages = []
        # System message
        messages.append({"role": "system", "content": self._get_system_prompt()})

        # User message: prefix + passages + post prompt
        prefix = self._add_prefix_prompt(query, num)
        input_context = f"{prefix}\n"

        for rank, text in enumerate(passages, 1):
            content = self._convert_doc_to_prompt_content(text, self.max_passage_length)
            input_context += f"[{rank}] {content}\n"

        input_context += self._add_post_prompt(query, num)
        messages.append({"role": "user", "content": input_context})

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = fix_text(prompt)
        return prompt

    def _get_sampling_params(self, num_passages: int) -> 'SamplingParams':
        """Get sampling parameters based on prompt mode."""
        if self.prompt_mode == "reasoning":
            # Reasoning mode: variable length output, no min_tokens
            return SamplingParams(
                temperature=0.0,
                max_tokens=self.reasoning_max_tokens,
            )
        elif self.prompt_mode == "qwen3":
            # Qwen3 thinking mode
            return SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                max_tokens=self.reasoning_max_tokens,
            )
        else:
            # Standard mode: fixed-length output (like RankGPT)
            output_tokens = self._estimate_output_tokens(num_passages)
            return SamplingParams(
                temperature=0.0,
                max_tokens=output_tokens,
                min_tokens=output_tokens,
            )

    def _estimate_output_tokens(self, num_passages: int) -> int:
        """Estimate the number of output tokens for the ranking result."""
        token_str = " > ".join([f"[{i+1}]" for i in range(num_passages)])
        return len(self._tokenizer.encode(token_str))

    def _parse_output(self, raw_output: str) -> str:
        """Parse model output based on prompt mode."""
        if self.prompt_mode in ("reasoning", "qwen3"):
            return _clean_response_reasoning(raw_output)
        else:
            return _clean_response_standard(raw_output)

    def _apply_permutation(
        self,
        candidates: List[Dict],
        permutation_indices: List[int],
        rank_start: int,
        rank_end: int,
    ) -> List[Dict]:
        """
        Apply a permutation to a slice of the candidates list.
        
        Args:
            candidates: Full candidates list (will be modified in-place)
            permutation_indices: 0-based indices representing the new order within the window
            rank_start: Start of the window in candidates
            rank_end: End of the window in candidates
            
        Returns:
            Modified candidates list
        """
        window = copy.deepcopy(candidates[rank_start:rank_end])
        original_scores = [c["score"] for c in window]

        for j, perm_idx in enumerate(permutation_indices):
            if perm_idx < len(window):
                candidates[rank_start + j] = copy.deepcopy(window[perm_idx])
                candidates[rank_start + j]["score"] = original_scores[j]

        return candidates

    def _sliding_windows_batched(
        self,
        all_candidates: List[List[Dict]],
        queries: List[str],
        rank_start: int,
        rank_end: int,
    ) -> List[List[Dict]]:
        """
        Apply sliding window ranking in batch mode.
        
        Args:
            all_candidates: List of candidate lists (one per query)
            queries: List of query strings
            rank_start: Start rank
            rank_end: End rank
            
        Returns:
            Reranked candidate lists
        """
        # Auto-adjust window_size: shrink it when there are fewer candidates than window_size.
        num_candidates = rank_end - rank_start
        window_size = min(self.window_size, num_candidates)
        step = min(self.step_size, window_size)  # step must not exceed window_size

        # Initialize working copies
        working_candidates = [copy.deepcopy(cands) for cands in all_candidates]

        # Initialize raw output collectors per query
        if not hasattr(self, '_raw_outputs_per_query'):
            self._raw_outputs_per_query = {}
        for i in range(len(queries)):
            self._raw_outputs_per_query[i] = []

        # Sliding window loop
        windows_end = rank_end
        windows_start = rank_end - window_size

        while windows_end > rank_start and windows_start + step != rank_start:
            windows_start = max(windows_start, rank_start)
            actual_window_size = windows_end - windows_start

            # Build prompts for all queries at this window position
            prompts = []
            for i, (query, candidates) in enumerate(zip(queries, working_candidates)):
                passages = [
                    c["doc"]["text"]
                    for c in candidates[windows_start:windows_end]
                ]
                prompt = self._create_prompt(query, passages, windows_start, windows_end)
                prompts.append(prompt)

            # Batch inference
            sampling_params = self._get_sampling_params(actual_window_size)
            outputs = self._llm.generate(prompts, sampling_params, use_tqdm=False)

            # Process outputs and apply permutations
            for i, output in enumerate(outputs):
                raw_output = output.outputs[0].text
                # Collect raw LLM output
                self._raw_outputs_per_query[i].append(raw_output)
                cleaned = self._parse_output(raw_output)
                perm_indices = _parse_permutation(cleaned, actual_window_size)
                working_candidates[i] = self._apply_permutation(
                    working_candidates[i], perm_indices, windows_start, windows_end
                )

            # Slide window
            windows_end = windows_end - step
            windows_start = windows_start - step

        return working_candidates

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Perform ReasonRank listwise reranking with sliding windows.

        For each Document, applies sliding window listwise ranking using
        the ReasonRank model with explicit reasoning.

        Args:
            documents: List of Document objects to process.

        Returns:
            List of Document objects with reorder_contexts populated.
        """
        if not documents:
            return documents

        # Prepare data structures
        queries = []
        all_candidates = []
        # Initialize raw output collectors
        doc_raw_outputs = {i: [] for i in range(len(documents))}

        for doc in documents:
            queries.append(doc.question.question)
            candidates = [
                {
                    "docid": ctx.id,
                    "doc": {"text": ctx.text},
                    "score": ctx.score if ctx.score is not None else 0.0,
                }
                for ctx in doc.contexts
            ]
            all_candidates.append(candidates)

        # Determine rank range
        min_candidates = min(len(c) for c in all_candidates)
        rank_start = 0
        rank_end = min(min_candidates, 100)  # Cap at 100

        effective_window = min(self.window_size, rank_end - rank_start)
        effective_step = min(self.step_size, effective_window)
        print(
            f"[ReasonRank] Reranking {len(documents)} documents, "
            f"rank_range=[{rank_start}, {rank_end}], "
            f"window_size={effective_window} (cfg={self.window_size}), "
            f"step={effective_step} (cfg={self.step_size})"
        )

        # Check if all candidate lists have the same length (required for batched)
        candidate_lengths = set(len(c) for c in all_candidates)
        if len(candidate_lengths) == 1:
            # All same length: use batched sliding window
            reranked_candidates = self._sliding_windows_batched(
                all_candidates, queries, rank_start, rank_end
            )
        else:
            # Variable lengths: process one by one
            reranked_candidates = []
            for i in tqdm(range(len(documents)), desc="ReasonRank reranking"):
                doc_rank_end = min(len(all_candidates[i]), 100)
                result = self._sliding_windows_batched(
                    [all_candidates[i]], [queries[i]], rank_start, doc_rank_end
                )
                reranked_candidates.append(result[0])

        # Map results back to documents
        for i, (doc, reranked_cands) in enumerate(zip(documents, reranked_candidates)):
            contexts = copy.deepcopy(doc.contexts)
            docid_to_context = {str(ctx.id): ctx for ctx in contexts}

            reorder_contexts = []
            for rank_pos, candidate in enumerate(reranked_cands):
                ctx = docid_to_context.get(str(candidate["docid"]))
                if ctx is not None:
                    ctx.score = float(len(reranked_cands) - rank_pos)
                    reorder_contexts.append(ctx)

            doc.reorder_contexts = reorder_contexts
            # Save collected raw LLM outputs
            doc.ranker_raw_outputs = getattr(self, '_raw_outputs_per_query', {}).get(i, [])

        print(f"[ReasonRank] Reranking complete.")
        return documents
