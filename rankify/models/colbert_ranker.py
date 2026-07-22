import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from math import ceil
from typing import List, Optional, Union
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.utils.models.colbert import _insert_token, _colbert_score, ColBERTModel  # Assume you provide these helpers unchanged
from tqdm import tqdm  # Import tqdm for progress tracking
import copy

class ColBERTReranker(BaseRanking):
    """
    A reranking model that leverages **ColBERT (Contextualized Late Interaction over BERT)** to reorder document contexts based on semantic relevance.

    ColBERT introduces a **late interaction mechanism** that efficiently compares query and document token embeddings to determine ranking scores.

    Attributes:
        method (str, optional): The reranking method name.
        model_name (str): The pretrained ColBERT model to be used for reranking.
        api_key (str, optional): API key for external services (not used in this model).
        device (str): The device on which the model runs (`cuda` if available, otherwise `cpu`).
        batch_size (int): The batch size for encoding documents and queries (default: 32).
        normalize (bool): Whether to normalize embeddings for cosine similarity computation (default: True).
        query_token (str): Special token used to distinguish query embeddings (default: `[unused0]`).
        document_token (str): Special token used to distinguish document embeddings (default: `[unused1]`).
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to process queries and contexts.
        model (ColBERTModel): The ColBERT model for generating contextual embeddings.
        query_max_length (int): The maximum token length for queries (default: 32).
        doc_max_length (int): The maximum token length for documents (determined by model constraints).

    References:
        - **Khattab, Omar, and Matei Zaharia.** *ColBERT: Efficient and effective passage search via contextualized late interaction over BERT.*  
          [Paper](https://arxiv.org/abs/2004.12832)
        - **Santhanam, Keshav, et al.** *Colbertv2: Effective and efficient retrieval via lightweight late interaction.*  
          [Paper](https://aclanthology.org/2022.naacl-main.272/)

    See Also:
        - `Reranking`: Main interface for reranking models, including `ColBERTReranker`.

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

        # Initialize Reranking with ColBERTReranker
        model = Reranking(method='colbert_ranker', model_name='Colbert')
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
        model_name: str = None,
        api_key: str = None,
        **kwargs,
    ):        
        """
        Initializes a **ColBERTReranker** instance.

        Args:
            method (str, optional): The reranking method name.
            model_name (str): The pretrained ColBERT model.
            api_key (str, optional): API key (not used).
            **kwargs: Additional parameters.

        Example:
            ```python
            model = ColBERTReranker(method='colbert_ranker', model_name='Colbert')
            ```
        """
        self.method = method
        self.model_name = model_name
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = kwargs.get("batch_size", 32)
        self.normalize = kwargs.get("normalize", True)
        self.query_token = kwargs.get("query_token", "[unused0]")
        self.document_token = kwargs.get("document_token", "[unused1]")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ColBERTModel.from_pretrained(self.model_name).to(self.device)
        self.query_token_id = self.tokenizer.convert_tokens_to_ids(self.query_token)
        self.document_token_id = self.tokenizer.convert_tokens_to_ids(self.document_token)
        self.query_max_length = 32  # Lower bound
        self.doc_max_length = (
            self.model.config.max_position_embeddings - 2
        )  # Upper bound

        self.model.eval()

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks each document's contexts using **ColBERT scoring** and updates `reorder_contexts`.

        Args:
            documents (List[Document]): A list of Document instances to rerank.

        Returns:
            List[Document]: Documents with updated `reorder_contexts`.

        Example:
            ```python
            reranked_documents = model.rank(documents)
            ```
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            if not document.contexts:
                # Skip document if no valid contexts
                print(f"[SKIP] Document skipped — no valid contexts. Query: '{query}'")
                continue
            contexts = [copy.deepcopy(ctx.text) for ctx in document.contexts]

            # Compute scores
            scores = self._colbert_rank(query, contexts)

            # Reorder contexts based on scores
            document.reorder_contexts = []
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for index in ranked_indices:
                d =copy.deepcopy(document.contexts[index]) 
                d.score= scores[index]
                document.reorder_contexts.append(d)
        return documents
    
    @torch.inference_mode()
    def _colbert_rank(self, query: str, contexts: List[str]) -> List[float]:
        """
        Computes **relevance scores** for a query and its associated contexts.

        Args:
            query (str): The query string.
            contexts (List[str]): A list of context strings.

        Returns:
            List[float]: Relevance scores for each context.
        """
        query_encoding = self._query_encode([query])
        documents_encoding = self._document_encode(contexts)
        query_embeddings = self._to_embs(query_encoding)
        document_embeddings = self._to_embs(documents_encoding)
        scores = (
            _colbert_score(
                query_embeddings,
                document_embeddings,
                query_encoding["attention_mask"],
                documents_encoding["attention_mask"],
            )
            .cpu()
            .tolist()[0]
        )
        return scores

    def _query_encode(self, query: list[str]):
        """
        Encodes the query using **ColBERT tokenization**.

        Args:
            query (List[str]): List containing a single query.

        Returns:
            Dict[str, torch.Tensor]: Tokenized query encoding.
        """
        return self._encode(
            query, self.query_token_id, max_length=self.doc_max_length, is_query=True
        )

    def _document_encode(self, documents: list[str]):
        """
        Encodes document passages while ensuring alignment with **ColBERT constraints**.

        Args:
            documents (List[str]): List of document texts.

        Returns:
            Dict[str, torch.Tensor]: Tokenized document encoding.
        """
        tokenized_doc_lengths = [
            len(
                self.tokenizer.encode(
                    doc, max_length=self.doc_max_length, truncation=True
                )
            )
            for doc in documents
        ]
        max_length = max(tokenized_doc_lengths)
        max_length = (
            ceil(max_length / 32) * 32
        )  # Round up to the nearest multiple of 32
        max_length = max(
            max_length, self.query_max_length
        )  # Ensure not smaller than query_max_length
        max_length = int(
            min(max_length, self.doc_max_length)
        )  # Ensure not larger than doc_max_length
        return self._encode(documents, self.document_token_id, max_length)

    def _encode(
        self,
        texts: list[str],
        insert_token_id: int,
        max_length: int,
        is_query: bool = False,
    ):
        """
        Tokenizes and encodes text while inserting **ColBERT-specific tokens**.

        Args:
            texts (List[str]): List of input texts.
            insert_token_id (int): Special token ID for ColBERT.
            max_length (int): Maximum sequence length.
            is_query (bool, optional): Whether encoding is for a query.

        Returns:
            Dict[str, torch.Tensor]: Tokenized and padded encoding.
        """
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=max_length - 1,  # for insert token
            truncation=True,
        )
        encoding = _insert_token(encoding, insert_token_id)  # type: ignore

        if is_query:
            mask_token_id = self.tokenizer.mask_token_id

            new_encodings = {"input_ids": [], "attention_mask": []}

            for i, input_ids in enumerate(encoding["input_ids"]):
                original_length = (
                    (input_ids != self.tokenizer.pad_token_id).sum().item()
                )

                # Calculate QLEN dynamically for each query
                if original_length % 16 <= 8:
                    QLEN = original_length + 8
                else:
                    QLEN = ceil(original_length / 16) * 16

                if original_length < QLEN:
                    pad_length = QLEN - original_length
                    padded_input_ids = input_ids.tolist() + [mask_token_id] * pad_length
                    padded_attention_mask = (
                        encoding["attention_mask"][i].tolist() + [0] *  pad_length
                    )
                else:
                    padded_input_ids = input_ids[:QLEN].tolist()
                    padded_attention_mask = encoding["attention_mask"][i][
                        :QLEN
                    ].tolist()

                new_encodings["input_ids"].append(padded_input_ids)
                new_encodings["attention_mask"].append(padded_attention_mask)

            for key in new_encodings:
                new_encodings[key] = torch.tensor(
                    new_encodings[key], device=self.device
                )

            encoding = new_encodings

        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        return encoding

    def _to_embs(self, encoding) -> torch.Tensor:
        """
        Converts tokenized inputs into **ColBERT embeddings**.

        Args:
            encoding (Dict[str, torch.Tensor]): Tokenized input encoding.

        Returns:
            torch.Tensor: ColBERT embeddings.
        """
        with torch.inference_mode():
            batched_embs = []
            for i in range(0, encoding["input_ids"].size(0), self.batch_size):
                batch_encoding = {
                    key: val[i : i + self.batch_size] for key, val in encoding.items()
                }
                batch_embs = self.model(**batch_encoding)
                batched_embs.append(batch_embs)
            embs = torch.cat(batched_embs, dim=0)
        if self.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs
