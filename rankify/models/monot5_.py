import logging
import math
from typing import List, Tuple, Union, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context
from rankify.utils.helper import get_device, get_dtype
import torch
from rankify.utils.pre_defind_models import PREDICTION_TOKENS
logger = logging.getLogger(__name__)
from tqdm import tqdm  # Import tqdm for progress tracking
from math import ceil

import copy


class MonoT5(BaseRanking):
    """
    Implements **MonoT5 Reranking**, a **sequence-to-sequence (seq2seq) ranking approach** 
    using **T5** to assess document relevance in **zero-shot and fine-tuned settings**.


    MonoT5 **ranks passages** by generating binary relevance predictions ("true"/"false") for query-passage pairs, 
    leveraging a **pretrained T5 model**.

    References:
        - **Nogueira et al. (2020)**: *Document Ranking with a Pretrained Sequence-to-Sequence Model*.
          [Paper](https://arxiv.org/abs/2003.06713)

    Attributes:
        method (str, optional): The **name of the reranking method**.
        model_name (str, optional): The **name of the pre-trained MonoT5 model**.
        device (torch.device): The **device (CPU/GPU)** on which the model runs.
        _context_size (int): The **maximum sequence length** for encoding.
        batch_size (int): The **batch size** used for inference.
        tokenizer (T5Tokenizer): The **T5 tokenizer** for processing inputs.
        model (T5ForConditionalGeneration): The **pretrained T5 model** for ranking.
        token_true_id (int): **Token ID** corresponding to the **"true"** label.
        token_false_id (int): **Token ID** corresponding to the **"false"** label.

    Examples:
        **Basic Usage:**
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

        # Initialize MonoT5 Reranker
        model = Reranking(method='monot5', model_name='monot5-base-msmarco')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """

    def __init__(self, method=None, model_name=None, **kwargs):
        """
        Initializes **MonoT5** for reranking tasks.

        Args:
            method (str, optional): The **reranking method name**.
            model_name (str, optional): The **name of the pretrained MonoT5 model**.
            kwargs (dict): Additional parameters such as `batch_size` and `context_size`.
        """      
        # Convert device string to torch.device
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device) if isinstance(device, str) else device
        self._context_size = kwargs.get("context_size", 512) 
        self.batch_size = kwargs.get("batch_size", 32)
        #self._prompt_mode = kwargs.get("device", "monot5")
        self.inputs_template = kwargs.get("inputs_template", "Query: {query} Document: {text} Relevant:")
        self.return_logits = kwargs.get("return_logits", False) 
        # Load the T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self._device)

        # Set tokens for scoring based on the model
        token_false, token_true = self._get_output_tokens(model_name)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids(token_false)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids(token_true)
        logger.info(f"True token ID set to {self.token_true_id}, False token ID set to {self.token_false_id}")

    @torch.inference_mode()
    def _get_scores(
        self,
        query: str,
        docs: List[str],
        max_length: int = 512,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """
        Implementation from https://github.com/unicamp-dl/InRanker/blob/main/inranker/rankers.py.
        Lightly modified so only the positive logits are returned and renamed the chunking function.

        Given a query and a list of documents, return a list of scores.
        Args:
            query: The query string.
            docs: A list of document strings.
            max_length: The maximum length of the input sequence.
        """
        if self.return_logits:
            logits = []
        else:
            scores = []
        if batch_size is None:
            batch_size = self.batch_size
        for batch in tqdm(
            self._chunks(docs, batch_size),
            disable=True,
            desc="Scoring...",
            total=ceil(len(docs) / batch_size),
        ):
            queries_documents = [
                self.inputs_template.format(query=query, text=text)
                for text in batch
            ]
            tokenized = self.tokenizer(
                queries_documents,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=max_length,
            ).to(self._device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            _, batch_scores = self._greedy_decode(
                model=self.model,
                input_ids=input_ids,
                length=1,
                attention_mask=attention_mask,
                return_last_logits=True,
            )
            batch_scores = batch_scores[
                :, [self.token_false_id, self.token_true_id]
            ].cpu()
            if self.return_logits:
                logits.extend(batch_scores[:, 1].tolist())
            else:
                batch_scores = torch.log_softmax(batch_scores, dim=-1)
                batch_scores = torch.exp(batch_scores[:, 1])
                scores.extend(batch_scores.tolist())

        if self.return_logits:
            return logits
        return scores

    @torch.inference_mode()
    def _greedy_decode(
        self,
        model,
        input_ids: torch.Tensor,
        length: int,
        attention_mask: torch.Tensor = None,
        return_last_logits: bool = True,
    ):
        """Implementation from https://github.com/unicamp-dl/InRanker/blob/main/inranker/rankers.py"""
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        next_token_logits = None
        for _ in range(length):
            try:
                model_inputs = model.prepare_inputs_for_generation(
                    decode_ids,
                    encoder_outputs=encoder_outputs,
                    past=None,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                outputs = model(**model_inputs)
            except TypeError:
                # Newer transformers versions have deprecated `past`
                # Our aim is to maintain pipeline compatibility for as many people as possible
                # So currently, we maintain a forking path with this error. Might need to do it more elegantly later on (TODO).
                model_inputs = model.prepare_inputs_for_generation(
                    decode_ids,
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
            decode_ids = torch.cat(
                [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
            )
        if return_last_logits:
            return decode_ids, next_token_logits
        return decode_ids

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        **Ranks contexts** within each document based on **relevance** to the query.

        Args:
            documents (List[Document]): A **list of Document instances** to rerank.

        Returns:
            List[Document]: Documents with updated **`reorder_contexts`** after reranking.
        """
        
        for doc in tqdm(documents, desc="Reranking Documents"):
            query = doc.question.question
            prompts = [self.inputs_template.format(query=query, text=context.text) for context in doc.contexts]
            scores = self._get_scores(query, [context.text for context in doc.contexts])
            contexts = copy.deepcopy(doc.contexts)
            for context, score in zip(contexts, scores):
                context.score = score
            doc.reorder_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        return documents
    def _chunks(self, contexts: list[Context],batch_size: int):
        """
        Splits a **list of contexts** into smaller **batches** for processing.

        Args:
            contexts (List[Context]): The **list of contexts** to split.
            batch_size (int): The **batch size**.

        Yields:
            List[Context]: **Batches** of contexts.
        """
        for i in range(0, len(contexts), batch_size):
            yield contexts[i:i+batch_size]

    @staticmethod
    def _get_output_tokens(model_name: str, token_false: str = "auto", token_true: str = "auto"):
        """
        Retrieves the **true/false prediction tokens** based on the MonoT5 model.

        Args:
            model_name (str): The **name of the model**.
            token_false (str, optional): The **"false"** token (default: `"auto"`).
            token_true (str, optional): The **"true"** token (default: `"auto"`).

        Returns:
            Tuple[str, str]: **True and false output tokens**.
        """
        if token_false == "auto" and model_name in PREDICTION_TOKENS:
            token_false = PREDICTION_TOKENS[model_name][0]
        if token_true == "auto" and model_name in PREDICTION_TOKENS:
            token_true = PREDICTION_TOKENS[model_name][1]
        if token_false == "auto" or token_true == "auto":
            token_false, token_true = PREDICTION_TOKENS["default"]
            logger.warning(f"Model {model_name} not found in PREDICTION_TOKENS. Using default tokens.")
        return token_false, token_true