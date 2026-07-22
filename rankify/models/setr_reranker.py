"""
SetR Reranker - Set Selection for Retrieval Augmented Generation

SetR (ACL 2025) shifts from ranking to set selection: instead of producing a
full permutation of documents, it selects a *subset* that maximally covers the
information requirements of a query.

Key features:
- Set Selection paradigm: selects a subset rather than ranking all documents
- IRI (Information Requirement Identification): decomposes query into sub-needs
- Three prompt strategies: selection_IRI, selection_woIRI, selection_only
- Local vLLM inference (no external API server needed)

Reference:
    "Shifting from Ranking to Set Selection for Retrieval Augmented Generation"
    (ACL 2025 Oral)

Usage:
    from rankify.models.reranking import Reranking

    model = Reranking(
        method='setr',
        model_name='/path/to/SETR-Qwen3-8B',
        prompt_mode='selection_IRI',
        top_k=10,
        num_gpus=4,
    )
    results = model.rank(documents)
"""

import copy
import re
from typing import List, Optional

from tqdm import tqdm

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document


# =========================================================
# Prompt Templates (from SetR paper / official code)
# =========================================================

SETR_SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank and select "
    "passages based on their relevancy to the query."
)

# --- selection_IRI: 3-step reasoning with Information Requirement Identification ---
SELECTION_IRI_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""

# --- selection_woIRI: CoT without explicit IRI ---
SELECTION_WOIRI_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Select the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1].
Let's think step by step."""

# --- selection_only: direct selection without reasoning ---
SELECTION_ONLY_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}

Search Query: {question}.
Select the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1]. Only respond with the selection results, do not say any word or explain."""


PROMPT_TEMPLATES = {
    "selection_IRI": SELECTION_IRI_PROMPT,
    "selection_woIRI": SELECTION_WOIRI_PROMPT,
    "selection_only": SELECTION_ONLY_PROMPT,
}


# =========================================================
# Output parsing
# =========================================================

def _parse_setr_output(raw_output: str, max_id: int = 20) -> List[int]:
    """
    Parse SetR model output to extract selected passage indices.

    Looks for '### Final Selection: [4] [2] [1]' or '## Final Selection: ...'
    pattern and extracts the 1-based indices.

    Args:
        raw_output: Raw model output string.
        max_id: Maximum valid passage ID (1-based).

    Returns:
        List of 0-based indices in selection order.
    """
    if not raw_output:
        return []

    # Try to find the "Final Selection" section
    # Handle variations: "### Final Selection:", "## Final Selection:", etc.
    selection_part = raw_output
    for marker in ["## Final Selection:", "### Final Selection:", "##Final Selection:", "###Final Selection:"]:
        if marker.lower() in raw_output.lower():
            idx = raw_output.lower().rfind(marker.lower())
            selection_part = raw_output[idx:]
            break

    # Extract all [N] patterns from the selection part
    indices = re.findall(r"\[(\d+)\]", selection_part)

    # Deduplicate while preserving order, convert to 0-based, filter valid range
    seen = set()
    result: List[int] = []
    for idx_str in indices:
        idx_val = int(idx_str)
        if 1 <= idx_val <= max_id and idx_val not in seen:
            seen.add(idx_val)
            result.append(idx_val - 1)  # Convert to 0-based

    return result


# =========================================================
# Main Reranker Class
# =========================================================

class SetRReranker(BaseRanking):
    """
    SetR: Set Selection for Retrieval Augmented Generation.

    This reranker uses a fine-tuned LLM (e.g., SETR-Qwen3-8B) to
    perform set selection — choosing a subset of passages that maximally
    covers the information requirements of a query.

    Uses local vLLM engine for inference (no external API server needed).

    Args:
        method (str): Method name ('setr').
        model_name (str): Local model path or HuggingFace model ID.
        prompt_mode (str): Selection strategy:
            - "selection_IRI" (default): 3-step reasoning with IRI
            - "selection_woIRI": CoT without explicit IRI
            - "selection_only": Direct selection, no reasoning
        top_k (int): Maximum number of passages to select (default: 10).
            Selected passages beyond top_k are trimmed. If model selects
            fewer than top_k, remaining slots are filled with unselected
            passages in original order.
        max_tokens (int): Maximum generation tokens (default: 4096 for IRI,
            512 for selection_only).
        max_passages (int): Maximum number of input passages (default: 20).
        num_gpus (int): Number of GPUs for tensor parallelism (default: 1).
        gpu_memory_utilization (float): GPU memory utilization for vLLM (default: 0.9).
        vllm_batched (bool): Whether to batch all queries for inference (default: True).
        context_size (int): Maximum model context length (default: 20480).

    Example:
        >>> from rankify.models.reranking import Reranking
        >>> reranker = Reranking(
        ...     method='setr',
        ...     model_name='/path/to/SETR-Qwen3-8B',
        ...     prompt_mode='selection_IRI',
        ...     top_k=10,
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
        self.method = method or "setr"
        model_name = model_name or "JohnnyFan/SETR-Qwen3-8B"

        # Resolve model path: download from ModelScope if not a local directory
        from rankify.utils.model_downloader import resolve_model_path
        cache_dir = kwargs.get("cache_dir", None)
        self.model_name = resolve_model_path(model_name, cache_dir=cache_dir)

        # Configuration
        self.prompt_mode = kwargs.get("prompt_mode", "selection_IRI")
        self.top_k = kwargs.get("top_k", 10)
        self.max_passages = kwargs.get("max_passages", 20)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.vllm_batched = kwargs.get("vllm_batched", True)
        self.context_size = kwargs.get("context_size", 20480)

        # Max tokens depends on prompt mode
        default_max_tokens = 512 if self.prompt_mode == "selection_only" else 4096
        self.max_tokens = kwargs.get("max_tokens", default_max_tokens)

        # Validate prompt_mode
        valid_modes = list(PROMPT_TEMPLATES.keys())
        if self.prompt_mode not in valid_modes:
            raise ValueError(
                f"prompt_mode must be one of {valid_modes}, got '{self.prompt_mode}'"
            )

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )

        # Initialize local vLLM engine
        from vllm import LLM, SamplingParams

        print(f"[SetR] Loading model locally with vLLM: {self.model_name}")
        print(f"[SetR] num_gpus={self.num_gpus}, gpu_memory_utilization={self.gpu_memory_utilization}")

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            max_model_len=self.context_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

        print(
            f"[SetR] Initialized: model={self.model_name}, "
            f"prompt_mode={self.prompt_mode}, top_k={self.top_k}"
        )

    def _format_contexts(self, contexts: list) -> str:
        """Format contexts into numbered passages for the prompt."""
        lines = []
        for idx, ctx in enumerate(contexts[: self.max_passages]):
            text = (ctx.text or "").strip()
            # Replace newlines within passage text
            text = re.sub(r"\n+", " ", text)
            # Include title if available
            title = getattr(ctx, "title", None) or ""
            if title:
                lines.append(f"[{idx + 1}] title: {title}\t{text}")
            else:
                lines.append(f"[{idx + 1}] {text}")
        return "\n\n\n".join(lines)

    def _build_messages(self, query: str, context_str: str, num_passages: int) -> list:
        """Build the chat messages for inference."""
        user_prompt = PROMPT_TEMPLATES[self.prompt_mode].format(
            question=query,
            context=context_str,
            num=num_passages,
        )
        return [
            {"role": "system", "content": SETR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert chat messages to a prompt string using the tokenizer's chat template."""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: disable thinking mode
            )
        except TypeError:
            # Fallback if tokenizer doesn't support enable_thinking kwarg
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _generate_single(self, messages: list) -> str:
        """Generate output for a single set of messages using local vLLM."""
        prompt = self._messages_to_prompt(messages)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def _generate_batch(self, messages_list: list) -> list:
        """Generate outputs for a batch of message sets using local vLLM."""
        prompts = [self._messages_to_prompt(msgs) for msgs in messages_list]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text for out in outputs]

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Perform SetR set selection on documents.

        For each Document, selects a subset of contexts based on information
        coverage, then fills remaining slots (up to top_k) with unselected
        passages in their original order.

        Args:
            documents: List of Document objects to process.

        Returns:
            List of Document objects with reorder_contexts populated.
        """
        if self.vllm_batched:
            return self._rank_batched(documents)
        else:
            return self._rank_sequential(documents)

    def _rank_sequential(self, documents: List[Document]) -> List[Document]:
        """Sequential ranking: process one document at a time."""
        for document in tqdm(documents, desc="SetR reranking"):
            if not document.contexts:
                document.reorder_contexts = []
                continue

            query = document.question.question
            contexts = document.contexts
            num_passages = min(len(contexts), self.max_passages)

            # Format context passages
            context_str = self._format_contexts(contexts)

            # Build messages
            messages = self._build_messages(query, context_str, num_passages)

            try:
                raw_output = self._generate_single(messages)
                # Save raw LLM output to document
                document.ranker_raw_outputs = [raw_output]

                # Parse selection result (0-based indices)
                selected_indices = _parse_setr_output(raw_output, max_id=num_passages)

                # Fix empty selection: fallback to top-5 original order
                if not selected_indices:
                    print(f"[SetR] Empty selection for query '{query[:50]}...', fallback to top-5")
                    selected_indices = list(range(min(5, num_passages)))

                # Apply top_k limit
                if self.top_k > 0:
                    selected_indices = selected_indices[: self.top_k]

                # Fill remaining slots with unselected passages in original order
                if self.top_k > 0 and len(selected_indices) < self.top_k:
                    selected_set = set(selected_indices)
                    for idx in range(num_passages):
                        if idx not in selected_set:
                            selected_indices.append(idx)
                        if len(selected_indices) >= self.top_k:
                            break

                # Build reorder_contexts
                reordered = []
                for rank_pos, ctx_idx in enumerate(selected_indices):
                    if 0 <= ctx_idx < len(contexts):
                        ctx_copy = copy.deepcopy(contexts[ctx_idx])
                        ctx_copy.score = float(len(selected_indices) - rank_pos)
                        reordered.append(ctx_copy)

                document.reorder_contexts = reordered

            except Exception as e:
                print(f"[SetR] Error processing query '{query[:50]}...': {e}")
                fallback = contexts[: self.top_k] if self.top_k > 0 else contexts[:5]
                document.reorder_contexts = copy.deepcopy(fallback)

        return documents

    def _rank_batched(self, documents: List[Document]) -> List[Document]:
        """Batched ranking: collect all prompts, run vLLM batch inference."""
        print(f"[SetR] Preparing batch of {len(documents)} documents...")

        # Phase 1: Build all prompts
        all_messages = []
        valid_indices = []
        num_passages_list = []

        for doc_idx, document in enumerate(documents):
            if not document.contexts:
                document.reorder_contexts = []
                continue

            query = document.question.question
            contexts = document.contexts
            num_passages = min(len(contexts), self.max_passages)

            context_str = self._format_contexts(contexts)
            messages = self._build_messages(query, context_str, num_passages)

            all_messages.append(messages)
            valid_indices.append(doc_idx)
            num_passages_list.append(num_passages)

        if not all_messages:
            return documents

        # Phase 2: Batch inference
        print(f"[SetR] Running vLLM batch inference on {len(all_messages)} queries...")
        raw_outputs = self._generate_batch(all_messages)

        # Phase 3: Parse outputs and build reorder_contexts
        for batch_idx, doc_idx in enumerate(valid_indices):
            document = documents[doc_idx]
            contexts = document.contexts
            num_passages = num_passages_list[batch_idx]

            try:
                raw_output = raw_outputs[batch_idx]
                # Save raw LLM output to document
                document.ranker_raw_outputs = [raw_output]
                selected_indices = _parse_setr_output(raw_output, max_id=num_passages)

                # Fix empty selection: fallback to top-5 original order
                if not selected_indices:
                    query = document.question.question
                    print(f"[SetR] Empty selection for query '{query[:50]}...', fallback to top-5")
                    selected_indices = list(range(min(5, num_passages)))

                if self.top_k > 0:
                    selected_indices = selected_indices[: self.top_k]

                if self.top_k > 0 and len(selected_indices) < self.top_k:
                    selected_set = set(selected_indices)
                    for idx in range(num_passages):
                        if idx not in selected_set:
                            selected_indices.append(idx)
                        if len(selected_indices) >= self.top_k:
                            break

                reordered = []
                for rank_pos, ctx_idx in enumerate(selected_indices):
                    if 0 <= ctx_idx < len(contexts):
                        ctx_copy = copy.deepcopy(contexts[ctx_idx])
                        ctx_copy.score = float(len(selected_indices) - rank_pos)
                        reordered.append(ctx_copy)

                document.reorder_contexts = reordered

            except Exception as e:
                query = document.question.question
                print(f"[SetR] Error processing query '{query[:50]}...': {e}")
                fallback = contexts[: self.top_k] if self.top_k > 0 else contexts[:5]
                document.reorder_contexts = copy.deepcopy(fallback)

        print(f"[SetR] Batch inference complete!")
        return documents
