import torch
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.utils.models.rank_llm.data import Request
from rankify.utils.models.rank_llm.rerank.listwise import RankListwiseOSLLM
from typing import List
from rankify.utils.models.rank_llm.rerank import PromptMode
from tqdm import tqdm  # Import tqdm for progress tracking
import copy
class ZephyrReranker(BaseRanking):
    """
    Implements **ZephyrReranker**, a **listwise ranking approach** designed for
    **zero-shot passage reranking** with strong **robustness and efficiency**.

    This method utilizes a **RankZephyr-based model** to score query-passage relevance 
    and **reorder retrieved documents** based on contextualized ranking predictions.

    References:
        - **Pradeep, R. et al. (2023)**: *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!*
          [Paper](https://arxiv.org/abs/2312.02724)

    Attributes:
        method (str): The name of the reranking method.
        model_name (str): The name of the **Zephyr** model used for reranking.
        window_size (int): The **window size** used for batch-wise ranking.
        _reranker (RankListwiseOSLLM): The Zephyr-based reranker instance.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the benefits of meditation?")
        contexts = [
            Context(text="Meditation reduces stress and improves focus.", id=0),
            Context(text="Excessive noise pollution affects mental health.", id=1),
            Context(text="Daily meditation can enhance emotional well-being.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize Zephyr Reranker
        model = Reranking(method='zephyr_reranker', model_name='rank_zephyr_7b_v1_full')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """
    def __init__(
        self,
        method: str = None,
        model_name: str = "castorini/rank_zephyr_7b_v1_full",
        **kwargs,
    ):
        """
        Initializes the ZephyrReranker for integration into the framework.

        Args:
            method (str, optional): The reranking method name.
            model_name (str, optional): Path to the Zephyr model.
                Defaults to `"castorini/rank_zephyr_7b_v1_full"`.
            **kwargs: Additional parameters:
                - `context_size` (int, default=4096): Maximum context size for the model.
                - `prompt_mode` (PromptMode, default=PromptMode.RANK_GPT): Defines the **prompt template**.
                - `num_few_shot_examples` (int, default=0): Number of **few-shot examples**.
                - `device` (str, default="cuda"): Computation device (`"cpu"` or `"cuda"`).
                - `num_gpus` (int, default=1): Number of GPUs to use.
                - `variable_passages` (bool, default=True): Whether to handle variable passage lengths.
                - `window_size` (int, default=20): Sliding window size for ranking.
                - `system_message` (str, optional): System message for the model.
        """
        context_size: int = kwargs.get("context_size", 4096)
        prompt_mode = kwargs.get("prompt_mode", PromptMode.RANK_GPT)
        num_few_shot_examples: int = kwargs.get("num_few_shot_examples", 0)
        device: str = kwargs.get("device", "cuda")
        num_gpus: int = kwargs.get("num_gpus", 1)
        variable_passages: bool = kwargs.get("variable_passages", True)
        window_size: int = kwargs.get("window_size", 20)
        system_message: str = kwargs.get(
            "system_message",
            "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
        )
        vllm_batched: bool = kwargs.get("vllm_batched", False)
        gpu_memory_utilization: float = kwargs.get("gpu_memory_utilization", 0.9)

        self.window_size = window_size
        self._vllm_batched = vllm_batched
        self._reranker = RankListwiseOSLLM(
            model=model_name,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
            vllm_batched=vllm_batched,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks each document's contexts using the Zephyr model.

        Args:
            documents (List[Document]): A list of **Document** instances to rerank.

        Returns:
            List[Document]: The reranked list of **Document** instances with updated `reorder_contexts`.
        """
        if self._vllm_batched:
            return self._rank_batched(documents)
        for document in tqdm(documents, desc="Reranking Documents"):
            document = self._rerank_document(document)
        return documents

    def _rank_batched(self, documents: List[Document]) -> List[Document]:
        """Batch reranking using vLLM for acceleration."""
        # Build all requests at once
        requests = []
        for doc in documents:
            request = Request(
                query={"text": doc.question.question},
                candidates=[
                    {"docid": ctx.id, "doc": {"text": ctx.text}, "score": ctx.score}
                    for ctx in doc.contexts
                ],
            )
            requests.append(request)

        # Batch rerank
        reranked_results = self._reranker.rerank_batch(
            requests=requests,
            rank_start=0,
            rank_end=100,
            window_size=self.window_size,
            step=10,
            shuffle_candidates=False,
            logging=False,
        )

        # Map results back to documents
        for doc, result in zip(documents, reranked_results):
            contexts = copy.deepcopy(doc.contexts)
            docid_to_context = {str(ctx.id): ctx for ctx in contexts}
            reorder_contexts = []
            for candidate in result.candidates:
                d = docid_to_context[str(candidate["docid"])]
                d.score = candidate["score"]
                reorder_contexts.append(d)
            doc.reorder_contexts = reorder_contexts

        return documents

    def _rerank_document(self, document: Document) -> Document:
        """
        Reranks a single document using the Zephyr model.

        Args:
            document (Document): A **Document** instance to rerank.

        Returns:
            Document: The reranked **Document** instance with updated `reorder_contexts`.
        """
        # Prepare request data structure for reranking
        request = Request(
            query={"text": document.question.question},  # Extract `text` from `Question`
            candidates=[
                {"docid": ctx.id, "doc": {"text": ctx.text}, "score": ctx.score}
                for ctx in document.contexts
            ],
        )

        rank_start = 0
        rank_end = 100
        window_size = self.window_size
        step = 10
        shuffle_candidates = False
        logging = False

        # Rerank using the Zephyr model
        reranked_result = self._reranker.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            window_size=window_size,
            step=step,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )[0]

        contexts = copy.deepcopy(document.contexts)

        # Create a mapping from docid to the original context
        docid_to_context = {str(ctx.id): ctx for ctx in contexts}

        # Reorder contexts based on reranked_result
        reorder_contexts = []
        for candidate in reranked_result.candidates:
            d = docid_to_context[str(candidate["docid"])]
            d.score = candidate["score"]
            reorder_contexts.append(d)
        document.reorder_contexts = reorder_contexts
        return document
