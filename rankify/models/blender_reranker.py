import llm_blender
from llm_blender.gpt_eval.utils import get_ranks_from_scores
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
from tqdm import tqdm  # Import tqdm for progress tracking
from llm_blender.pair_ranker.config import RankerConfig  # Adjust import path if needed

import copy

class BlenderReranker(BaseRanking):
    """
    A reranking model that utilizes LLM-Blender's PairRanker to reorder document contexts based on relevance.

    This reranker employs **pairwise ranking techniques** to compare context passages and determine the optimal ranking.
    The model is based on the **LLM-Blender approach for reranking**.

    Attributes:
        method (str, optional): The reranking method name.
        blender (llm_blender.Blender): The LLM-Blender model used for reranking.

    References:
        - Jiang, Dongfu, Xiang Ren, and Bill Yuchen Lin. *LLM-Blender: Ensembling large language models with pairwise ranking and generative fusion*.
          [arXiv preprint](https://arxiv.org/abs/2306.02561) (2023).

    See Also:
        - `Reranking`: Main interface for reranking models, including `BlenderReranker`.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.models.reranking import Reranking

        question = Question("When did Thomas Edison invent the light bulb?")
        answers = Answer(["1879"])
        contexts = [
            Context(text="Lightning strike at Seoul National University", id=1),
            Context(text="Thomas Edison invented the light bulb in 1879", id=4),
        ]
        document = Document(question=question, answers=answers, contexts=contexts)

        model = Reranking(method="blender_reranker", model_name="PairRM")
        model.rank([document])
        ```

    Notes:
        - LLM-Blender’s PairRanker uses **pairwise ranking** to compare contexts and determine the best order.
        - The model supports **flexible reranking** across multiple contexts.
        - It is **integrated into the `Reranking` class**, meaning users should call `Reranking` instead of directly instantiating `BlenderReranker`.
    """

    def __init__(self, method: str = None, model_name: str = "llm-blender/PairRM", **kwargs):
        """
        Initializes the BlenderReranker for document reranking using LLM-Blender's PairRanker.

        Args:
            method (str, optional): The reranking method name. Defaults to `None`.
            model_name (str, optional): The model name for LLM-Blender's PairRanker. Defaults to `"llm-blender/PairRM"`.
            **kwargs: Additional keyword arguments for configuration.

        Example:
            ```python
            model = BlenderReranker(method="blender_reranker", model_name="PairRM")
            ```
        """
        ranker_config = RankerConfig(
            device="cuda",
            fp16=True,
        )
        self.device = kwargs.get("device", "cuda") 
        self.method = method
        self.blender = llm_blender.Blender() #ranker_config=ranker_config
        self.blender.loadranker(model_name , fp16=True, device=self.device)  # Load the ranker model

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks the contexts within each document using LLM-Blender's PairRanker.

        Args:
            documents (List[Document]): A list of documents containing contexts to be reranked.

        Returns:
            List[Document]: The documents with reordered contexts.

        Raises:
            ValueError: If no contexts are provided in a document.
            ValueError: If the model returns an invalid ranking result.

        Example:
            ```python
            model = BlenderReranker(method="blender_reranker", model_name="PairRM")
            reranked_documents = model.rank(documents)
            ```
        """
        for document in tqdm(documents, desc="Reranking Documents"):
                document = self._rerank_document(document)
        return documents

    def _rerank_document(self, document: Document) -> Document:
        """
        Reranks a single document's contexts using LLM-Blender's PairRanker.

        Args:
            document (Document): The document instance whose contexts need to be reranked.

        Returns:
            Document: The reranked document with updated `reorder_contexts`.

        Raises:
            ValueError: If no contexts are provided.
            ValueError: If the API returns an invalid ranking.

        Example:
            ```python
            ranked_doc = model._rerank_document(document)
            ```
        """
        # Prepare inputs for LLM-Blender
        input_text = document.question.question  # The query text
        candidate_texts = [ctx.text for ctx in document.contexts]  # Candidate contexts

        # Ensure candidates are provided
        if not candidate_texts:
            raise ValueError("No Context Provide!!!!")

        # Perform reranking
        scores = self.blender.rank([input_text], [candidate_texts], return_scores=True, disable_tqdm=True)
        ranks =get_ranks_from_scores(scores)

        if ranks.size == 0 or len(ranks[0]) != len(candidate_texts):
            raise ValueError("Invalid ranks returned from LLM-Blender.")

        # Map ranks to contexts
        contexts = copy.deepcopy(document.contexts)
        ranked_contexts = []
        for score, idx in zip(scores[0], ranks[0]):
            ctx= contexts[idx-1]
            ctx.score= score
            ranked_contexts.append(ctx)


        # Update the document's reordered contexts
        document.reorder_contexts = ranked_contexts
        return document
    


