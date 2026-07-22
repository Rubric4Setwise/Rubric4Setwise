"""
Rank1 Reranker: Pointwise Reasoning Reranker with Logprob Scoring.

Based on: https://github.com/orionw/rank1
Models: jhu-clsp/rank1-7b, jhu-clsp/rank1-32b

Rank1 uses a pointwise approach:
- For each (query, passage) pair, model generates a reasoning chain
- Model must end with "</think> true" or "</think> false"
- Score = P(true) / (P(true) + P(false)) from logprobs
- Incomplete responses are fixed by forcing the model to output true/false
"""

import os
import math
from typing import List, Optional, Tuple

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


class Rank1Reranker(BaseRanking):
    """
    Rank1: Pointwise Reasoning Reranker using test-time compute.

    The model generates a reasoning chain (<think>...</think>) then decides
    true/false. Score is computed from logprobs of true vs false tokens.

    Args:
        method (str): Method name ('rank1').
        model_name (str): HuggingFace model ID or local path.
        api_key (str): Not used (local model).
        **kwargs: Additional parameters:
            - max_tokens (int): Max generation tokens for reasoning (default: 8192)
            - max_passage_length (int): Max tokens per passage (default: 512)
            - context_size (int): Model context window (default: 16000)
            - num_gpus (int): Number of GPUs (default: 1)
            - gpu_memory_utilization (float): GPU memory fraction (default: 0.9)
            - force_rethink (int): Number of rethink iterations (default: 0)

    Example:
        >>> reranker = Reranking(method='rank1', model_name='jhu-clsp/rank1-7b')
        >>> results = reranker.rank(documents)
    """

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "rank1"
        model_name = model_name or "jhu-clsp/rank1-7b"

        # Resolve model path: download from HuggingFace/ModelScope if not local
        from rankify.utils.model_downloader import resolve_model_path
        self.model_name = resolve_model_path(model_name)

        # Configuration
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.max_passage_length = kwargs.get("max_passage_length", 512)
        self.context_size = kwargs.get("context_size", 16000)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.force_rethink = kwargs.get("force_rethink", 0)

        if LLM is None:
            raise ImportError("vLLM is required for Rank1. Please install: pip install vllm")

        # Initialize tokenizer
        print(f"[Rank1] Loading model locally: {self.model_name} (num_gpus={self.num_gpus})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Cache token IDs
        self.true_token = self._tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self._tokenizer(" false", add_special_tokens=False).input_ids[0]

        # Initialize vLLM
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            max_model_len=self.context_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype="float16",
        )

        # Sampling params: stop at "</think> true" or "</think> false"
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False,
        )

        print(f"[Rank1] Model loaded successfully.")

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max_length tokens."""
        tokens = self._tokenizer.tokenize(text)[:max_length]
        return self._tokenizer.convert_tokens_to_string(tokens)

    def _build_prompt(self, query: str, passage: str) -> str:
        """Build the pointwise prompt."""
        return (
            "Determine if the following passage is relevant to the query. "
            "Answer only with 'true' or 'false'.\n"
            f"Query: {query}\n"
            f"Passage: {passage}\n"
            "<think>"
        )

    def _fix_incomplete_responses(
        self,
        original_prompts: List[str],
        generated_texts: List[str],
    ) -> List[float]:
        """
        Fix incomplete responses by forcing true/false output.

        When the model doesn't end with </think> true/false, we:
        1. Truncate to last complete sentence
        2. Append </think>
        3. Force generate 1 token (true or false only)
        4. Compute score from logprobs
        """
        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()
            if not text.endswith(('.', '!', '?')):
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[:last_punct + 1]
            cleaned_texts.append(text.strip())

        forced_prompts = [
            f"{prompt}\n{cleaned}\n</think>"
            for prompt, cleaned in zip(original_prompts, cleaned_texts)
        ]

        fix_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False,
        )
        outputs = self._llm.generate(forced_prompts, fix_params)

        scores = []
        for output in outputs:
            try:
                final_logits = output.outputs[0].logprobs[-1]
                if self.true_token not in final_logits or self.false_token not in final_logits:
                    scores.append(0.5)
                    continue
                true_logit = final_logits[self.true_token].logprob
                false_logit = final_logits[self.false_token].logprob
                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                scores.append(true_score / (true_score + false_score))
            except Exception as e:
                print(f"[Rank1] Error fixing response: {e}")
                scores.append(0.5)

        return scores

    def _process_batch(self, prompts: List[str]) -> List[float]:
        """
        Process a batch of prompts and return relevance scores.
        Handles both complete and incomplete responses.
        """
        outputs = self._llm.generate(prompts, self.sampling_params)

        scores = [None] * len(prompts)
        incomplete_prompts = []
        incomplete_texts = []
        incomplete_indices = []

        # Collect raw LLM outputs
        self._current_raw_outputs = []

        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            self._current_raw_outputs.append(text)
            try:
                final_logits = output.outputs[0].logprobs[-1]
            except (IndexError, TypeError):
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue

            if self.true_token not in final_logits or self.false_token not in final_logits:
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue

            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            scores[i] = true_score / (true_score + false_score)

        # Fix incomplete responses
        if incomplete_indices:
            fixed_scores = self._fix_incomplete_responses(incomplete_prompts, incomplete_texts)
            for idx, score in zip(incomplete_indices, fixed_scores):
                scores[idx] = score

        # Ensure no None values
        return [s if s is not None else 0.5 for s in scores]

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using pointwise relevance scoring.

        Args:
            documents: List of Document objects to rerank.

        Returns:
            List of Document objects with reorder_contexts set.
        """
        for doc in documents:
            contexts = doc.contexts
            if not contexts:
                continue

            query = doc.question

            # Build prompts for all contexts
            prompts = []
            for ctx in contexts:
                passage = self._truncate(ctx.text, self.max_passage_length)
                prompts.append(self._build_prompt(query, passage))

            # Get scores (also collect raw outputs)
            scores = self._process_batch(prompts)

            # Save raw LLM outputs collected during _process_batch
            doc.ranker_raw_outputs = getattr(self, '_current_raw_outputs', [])

            # Handle rethink if configured
            rethink_count = self.force_rethink
            if rethink_count > 0:
                # For rethink, we'd need the generated texts - simplified version
                # Re-run with "Wait" appended (simplified)
                pass  # TODO: implement if needed

            # Assign scores and sort
            scored_contexts = list(zip(contexts, scores))
            scored_contexts.sort(key=lambda x: x[1], reverse=True)

            reorder_contexts = []
            for ctx, score in scored_contexts:
                ctx.score = score
                reorder_contexts.append(ctx)

            doc.reorder_contexts = reorder_contexts

        return documents
