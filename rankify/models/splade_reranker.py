import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
from tqdm import tqdm  # Import tqdm for progress tracking

import copy
def splade_max_pooling(logits, attention_mask):
    """
    Perform Splade-style max pooling with log scaling.
    """
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    return max_val


class SpladeReranker(BaseRanking):
    """
    Implements **SpladeReranker**, a **sparse lexical and expansion model**
    for first-stage ranking using **masked language models (MLMs).**



    SPLADE employs **sparse representations** of queries and documents,
    performing lexical matching while **expanding terms** in an interpretable way.

    References:
        - **Formal et al. (2021)**: *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. [Paper](https://arxiv.org/abs/2107.05720)
        - **Formal et al. (2021)**: *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval*. [Paper](https://arxiv.org/abs/2109.10086)

    Attributes:
        method (str): The name of the reranking method.
        model_name (str): The name or path to the **SPLADE** model.
        device (str): The device (CPU/GPU) used for inference.
        query_max_length (int): The maximum token length for **queries**.
        document_max_length (int): The maximum token length for **documents**.
        batch_size (int): The batch size for document encoding.
        model (AutoModelForMaskedLM): The **Masked Language Model (MLM)** used for sparse representation.
        tokenizer (AutoTokenizer): The tokenizer for encoding query and document texts.

    Examples:
        **Basic Usage:**
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the advantages of renewable energy?")
        contexts = [
            Context(text="Renewable energy reduces carbon emissions and is sustainable.", id=0),
            Context(text="Fossil fuels have been the primary source of energy for centuries.", id=1),
            Context(text="Solar and wind power are prominent forms of renewable energy.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize SPLADE Reranker
        model = Reranking(method='splade', model_name='splade-cocondenser')
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
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        **kwargs
    ):
        """
        Initializes **SPLADE Reranker** for reranking tasks.

        Args:
            method (str): The name of the reranking method.
            model_name (str): The name or path to the **SPLADE** model.
            **kwargs: Additional parameters:
                - device (str, optional): Computation device (`"auto"`, `"cuda"`, `"cpu"`). Default is `"auto"`.
                - use_fp16 (bool, optional): Whether to use **FP16** inference. Default is `True`.
                - batch_size (int, optional): Batch size for document encoding. Default is `16`.
                - query_max_length (int, optional): Maximum token length for **queries**. Default is `512`.
                - document_max_length (int, optional): Maximum token length for **documents**. Default is `512`.
        """
        super().__init__(method)
        self.device = self._detect_device(kwargs.get("device", "auto"))
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        if kwargs.get("use_fp16", True) and "cuda" in self.device:
            self.model.half()

        self.query_max_length = kwargs.get("query_max_length", 512)
        self.document_max_length = kwargs.get("document_max_length", 512)
        self.batch_size = kwargs.get("batch_size", 16)

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks a list of **Document** instances using the **SPLADE** model.

        Args:
            documents (List[Document]): A list of `Document` instances to rerank.

        Returns:
            List[Document]: The documents with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            document = self._rerank_document(document)
        return documents

    def _rerank_document(self, document: Document) -> Document:
        """
        Reranks a single document's contexts using the **SPLADE** model.

        Args:
            document (Document): A **Document** instance to rerank.

        Returns:
            Document: The reranked **Document** with updated `reorder_contexts`.
        """
        query = document.question.question  # Extract query text
        contexts = copy.deepcopy(document.contexts)

        # Extract context texts
        context_texts = [ctx.text for ctx in contexts]
        if not context_texts:
            document.reorder_contexts = []
            return document

        # Compute query and document scores
        scores = self._rerank(query, context_texts)
        # Map scores back to contexts and update the scores
        for ctx, score in zip(contexts, scores):
            ctx.score = score

        # Map scores back to contexts
        scored_contexts = list(zip(contexts, scores))
        scored_contexts.sort(key=lambda x: x[1], reverse=True)

        # Reorder contexts in the document
        document.reorder_contexts = [ctx for ctx, _ in scored_contexts]
        return document

    def _compute_vector(self, texts: List[str], max_length: int) -> torch.Tensor:
        """
        Compute **SPLADE-style** sparse embeddings for a list of texts.

        Args:
            texts (List[str]): A list of **texts** to compute embeddings for.
            max_length (int): The **maximum length** for tokenization.

        Returns:
            torch.Tensor: Sparse **document representations**.
        """
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)
            logits, attention_mask = output.logits, tokens["attention_mask"]

        return splade_max_pooling(logits, attention_mask)

    def _rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Computes relevance scores between a **query** and **documents**.

        Args:
            query (str): The **query** text.
            documents (List[str]): A list of **document texts**.

        Returns:
            List[float]: Relevance scores for each document.
        """
        # Compute query embedding
        query_emb = self._compute_vector([query], max_length=self.query_max_length)[0]

        # Compute document embeddings
        doc_embs = []
        for i in range(0, len(documents), self.batch_size):
            doc_embs.append(
                self._compute_vector(
                    documents[i : i + self.batch_size],
                    max_length=self.document_max_length,
                )
            )
        doc_embs = torch.cat(doc_embs, dim=0)

        # Compute similarity scores
        scores = torch.matmul(query_emb.unsqueeze(0), doc_embs.t()).squeeze(0)
        return scores.tolist()

    @staticmethod
    def _detect_device(device: str) -> str:
        """
        Detects the appropriate device for computation.

        Args:
            device (str): Desired device (`"auto"`, `"cuda"`, or `"cpu"`).

        Returns:
            str: The detected device.
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
