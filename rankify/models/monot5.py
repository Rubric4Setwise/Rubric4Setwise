import logging
import math
from typing import List, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context
from rankify.utils.helper import get_device
from rankify.utils.pre_defind_models import PREDICTION_TOKENS
import torch
from tqdm import tqdm
from math import ceil
import copy

logger = logging.getLogger(__name__)

class MonoT5(BaseRanking):
    def __init__(self, method=None, model_name=None, **kwargs):
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device) if isinstance(device, str) else device
        self._context_size = kwargs.get("context_size", 512)
        self.batch_size = kwargs.get("batch_size", 32)
        self.use_amp = kwargs.get("use_amp", torch.cuda.is_available())
        if "inputs_template" in kwargs:
            logger.warning("Custom inputs_template is ignored for MonoT5. Using default: 'Query: {query} Document: {text} Relevant:'")
        self.inputs_template = "Query: {query} Document: {text} Relevant:"
        self.return_logits = kwargs.get("return_logits", False)

        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        dtype = torch.float16 if self._device.type == "cuda" and self.use_amp else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(self._device).eval()

        # Set prediction tokens
        token_false, token_true = self._get_output_tokens(model_name)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids(token_false)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids(token_true)
        logger.info(f"True token ID: {self.token_true_id}, False token ID: {self.token_false_id}")

    @torch.inference_mode()
    def _get_scores(self, query: str, docs: List[str], max_length: int = 512, batch_size: Optional[int] = None) -> List[float]:
        scores = []
        if batch_size is None:
            batch_size = self.batch_size
        for batch in tqdm(self._chunks(docs, batch_size), disable=True, desc="Scoring...", total=ceil(len(docs) / batch_size)):
            queries_documents = [self.inputs_template.format(query=query, text=text) for text in batch]
            tokenized = self.tokenizer(
                queries_documents,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
                return_attention_mask=True,
            ).to(self._device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                _, batch_scores = self._greedy_decode(
                    model=self.model,
                    input_ids=input_ids,
                    length=1,
                    attention_mask=attention_mask,
                    return_last_logits=True,
                )
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
            batch_scores = torch.log_softmax(batch_scores, dim=-1)
            scores.extend(batch_scores[:, 1].tolist())
        return scores

    @torch.inference_mode()
    def _greedy_decode(self, model, input_ids: torch.Tensor, length: int, attention_mask: torch.Tensor = None, return_last_logits: bool = True):
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)
        next_token_logits = outputs.logits[:, -1, :]
        decode_ids = torch.cat([decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1)
        if return_last_logits:
            return decode_ids, next_token_logits
        return decode_ids

    def rank(self, documents: List[Document]) -> List[Document]:
        for doc in tqdm(documents, desc="Reranking Documents"):
            query = doc.question.question
            scores = self._get_scores(query, [context.text for context in doc.contexts], max_length=self._context_size)
            contexts = copy.deepcopy(doc.contexts)
            for context, score in zip(contexts, scores):
                context.score = score
            doc.reorder_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        return documents

    def _chunks(self, texts: List[str], batch_size: int):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

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