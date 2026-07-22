from math import ceil
from typing import List
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.dataset.dataset import Context
import copy
from tqdm import tqdm  # Import tqdm for progress tracking

class InRanker(BaseRanking):
    """
    Implements **InRanker**, a **distilled ranker** for **zero-shot information retrieval** based on **Seq2Seq models**.

    

    **InRanker** ranks passages by estimating their **relevance probability** using a **pre-trained language model**.  
    It **tokenizes** query-document pairs and **predicts relevance** via **binary classification**.

    The model assigns **softmax scores** between `"false"` and `"true"` tokens,  
    where the `"true"` probability determines **document relevance**.

    References:
        - **Laitz et al. (2024)**: *InRanker: Distilled Rankers for Zero-shot Information Retrieval*.  
          [Paper](https://arxiv.org/abs/2401.06910)

    Attributes:
        method (str, optional): The reranking method name.
        model_name (str): Name of the pre-trained **Seq2Seq model**.
        api_key (str, optional): API key for authentication (if needed).
        tokenizer (AutoTokenizer): The tokenizer for processing queries and documents.
        model (AutoModelForSeq2SeqLM): The **sequence-to-sequence model** used for reranking.
        precision (str): Model precision (`"bf16"`, `"fp16"`, or `"fp32"`).
        device (str): The device used for computation (`"cuda"` or `"cpu"`).
        batch_size (int): Number of query-document pairs processed in a batch.
        max_length (int): Maximum length of tokenized sequences.
        token_false_id (int): Token ID for `"false"` (indicating irrelevance).
        token_true_id (int): Token ID for `"true"` (indicating relevance).

    See Also:
        - `Reranking`: Main interface for reranking models, including `InRanker`.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        question = Question("What are the symptoms of COVID-19?")
        contexts = [
            Context(text="Fever and cough are common symptoms of COVID-19.", id=0),
            Context(text="Headache is a rare symptom.", id=1),
            Context(text="Fatigue and loss of taste are also common.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize Reranking with InRanker
        model = Reranking(method='inranker', model_name='inranker-small')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```

    Notes:
        - Uses a **Seq2Seq binary classification** approach (`"false"` vs `"true"`).
        - Supports **batch processing** for efficiency.
        - Works in **zero-shot retrieval** scenarios without fine-tuning.
    """
    def __init__(self, method: str= None, model_name: str= None, api_key: str= None, **kwargs) ->None:
        """
        Initializes **InRanker** for **zero-shot document reranking**.

        Args:
            method (str, optional): The reranking method name.
            model_name (str): Name of the pre-trained **Seq2Seq model**.
            api_key (str, optional): API key for authentication (if needed).
            **kwargs: Additional parameters, including:
                - `precision` (str): `"bf16"`, `"fp16"`, or `"fp32"` for model precision.
                - `device` (str): `"cuda"` or `"cpu"` (default: `"cuda"`).
                - `batch_size` (int): Number of query-document pairs per batch (default: `32`).
                - `max_length` (int): Maximum sequence length for tokenization (default: `512`).

        Example:
            ```python
            model = InRanker(method='inranker', model_name='inranker-small')
            ```
        """
        model_args = {}

        self.precision = kwargs.get("precision", "bf16")
        self.slient = kwargs.get("slient", True)
        self.batch_size = kwargs.get("batch_size", 32)
        self.device = kwargs.get("device" , "cuda")
        self.max_length = kwargs.get("max_length",512)
        if self.precision == "bf16":
            model_args["torch_dtype"] =  torch.bfloat16
        elif self.precision == "fp16":
            model_args["torch_dtype"] = torch.float16
        else:
            model_args["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,**model_args)
        self.model.to(self.device)
        self.model.eval()
        #token_false, token_true = ["▁false", "▁true"]
        token_false, token_true = ["▁false","▁true"]
        self.token_false_id = self.tokenizer.get_vocab()[token_false]
        self.token_true_id = self.tokenizer.get_vocab()[token_true]

    
    def rank(self, documents: list[Document])-> List[Document]:
        """
        Reranks documents using **InRanker's binary classification** approach.

        Each document's **contexts** are scored based on the probability of being **relevant** (`"true"` token).

        Args:
            documents (List[Document]): A list of `Document` instances to rerank.

        Returns:
            List[Document]: The reranked list of `Document` instances with updated `reorder_contexts`.

        Example:
            ```python
            reranked_docs = model.rank(documents)
            ```
        """
        

        for document in tqdm(documents, desc="Reranking Documents"):
            scores =[]
            logits = []
            context_copy= copy.deepcopy(document.contexts)
            for batch in self._chunks(document.contexts,self.batch_size):
                
                
                queries_documents = [ 
                    f"Query: {document.question.question} Document: {context.text} Relevant:" for context in batch
                ]
                tokenized = self.tokenizer(queries_documents,
                                           padding=True,
                                           truncation="longest_first",
                                           return_tensors="pt",
                                           max_length=self.max_length).to(self.device)
                
                
                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)
                _ , batch_scores = self._greedy_decode(
                    model = self.model,
                    input_ids = input_ids,
                    length= 1,
                    attention_mask= attention_mask,
                    return_last_logits=True
                )

                batch_scores = batch_scores[:,[self.token_false_id, self.token_true_id]]
                logits.extend(batch_scores.tolist())

                batch_scores = torch.log_softmax(batch_scores,dim=-1)
                batch_scores = torch.exp(batch_scores[:,1])
                batch_scores = batch_scores.tolist()

                scores.extend(batch_scores)
            
            for score, context in zip(scores, context_copy):
                 context.score = score
            
            context_copy.sort(key=lambda x:x.score, reverse=True)
            document.reorder_contexts = context_copy
        
        return documents

    
    @torch.no_grad()
    def _greedy_decode(self,model,
        input_ids: torch.Tensor,
        length: int,
        attention_mask: torch.Tensor = None,
        return_last_logits: bool = True):
        """
        Performs **greedy decoding** to generate the next token logits.

        Args:
            model (AutoModelForSeq2SeqLM): The **pre-trained Seq2Seq model**.
            input_ids (torch.Tensor): The **input token IDs**.
            length (int): The number of decoding steps.
            attention_mask (torch.Tensor, optional): The **attention mask** for input sequences.
            return_last_logits (bool, optional): Whether to return the last token's **logits**.

        Returns:
            tuple: A tuple containing:
                - **decoded token IDs** (torch.Tensor)
                - **last-step logits** (torch.Tensor)

        Example:
            ```python
            decoded_ids, logits = model._greedy_decode(model, input_ids, length=1)
            ```
        """
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        next_token_logits = None
        for _ in range(length):
            model_inputs = model.prepare_inputs_for_generation(
                decode_ids,
                encoder_outputs=encoder_outputs,
                past=None,
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
    @staticmethod
    def _chunks(contexts: list[Context],batch_size: int):
        """
        Splits a **list of contexts** into **smaller batches**.

        Args:
            contexts (List[Context]): The list of **contexts** to split.
            batch_size (int): The **batch size**.

        Yields:
            List[Context]: A **batch** of contexts.

        Example:
            ```python
            for batch in model._chunks(contexts, batch_size=8):
                process_batch(batch)
            ```
        """
        for i in range(0, len(contexts), batch_size):
            yield contexts[i:i+batch_size]
