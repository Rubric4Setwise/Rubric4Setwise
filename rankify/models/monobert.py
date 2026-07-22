import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
from copy import deepcopy
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from tqdm import tqdm  # Import tqdm for progress tracking


class MonoBERT(BaseRanking):
    """
    Implements **MonoBERT Reranking**, a **BERT-based multi-stage ranking approach**
    for **improving document retrieval** in information retrieval tasks.


    MonoBERT **re-ranks retrieved documents** based on query-passage **relevance scores** using a **pretrained BERT model**.

    References:
        - **Nogueira et al. (2019)**: *Multi-stage Document Ranking with BERT*.
          [Paper](https://arxiv.org/abs/1910.14424)

    Attributes:
        method (str): The **name of the reranking method**.
        model_name (str): The **name of the pre-trained MonoBERT model** used for reranking.
        device (torch.device): The **device (CPU/GPU)** on which the model runs.
        use_amp (bool): Whether to use **Automatic Mixed Precision (AMP)** for **faster inference**.
        model (AutoModelForSequenceClassification): The **pretrained MonoBERT model** for reranking.
        tokenizer (AutoTokenizer): The **tokenizer** for MonoBERT.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the health benefits of green tea?")
        contexts = [
            Context(text="Green tea contains antioxidants that promote heart health.", id=0),
            Context(text="Excessive caffeine intake can cause insomnia.", id=1),
            Context(text="Green tea consumption is linked to improved metabolism.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize MonoBERT Reranker
        model = Reranking(method='monobert', model_name='monobert-large')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """ 
    def __init__(self, method: str = None, model_name: str = None, api_key: str = None, **kwargs):
        """
        Initializes **MonoBERT** for reranking tasks.

        Args:
            method (str, optional): The **reranking method name**.
            model_name (str, optional): The **name of the pretrained MonoBERT model** 
                (default: `"castorini/monobert-large-msmarco"`).
            api_key (str, optional): **Not used**, but included for framework consistency.
            kwargs (dict): Additional parameters such as `use_amp` for **mixed precision inference**.
        """
        self.method = method
        self.model_name = model_name or "castorini/monobert-large-msmarco"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = kwargs.get("use_amp", False)
        self.model = self.get_model(self.model_name)
        self.tokenizer = self.get_tokenizer()

    @staticmethod
    def get_model(pretrained_model_name_or_path: str) -> AutoModelForSequenceClassification:
        """
        Loads the **MonoBERT model**.

        Args:
            pretrained_model_name_or_path (str): Path to the **pretrained MonoBERT model**.

        Returns:
            AutoModelForSequenceClassification: The **MonoBERT model**.
        """
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = "bert-large-uncased") -> AutoTokenizer:
        """
        Loads the **tokenizer** for MonoBERT.

        Args:
            pretrained_model_name_or_path (str): Path to the **pretrained tokenizer**.

        Returns:
            AutoTokenizer: The **MonoBERT tokenizer**.
        """
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)

    @torch.no_grad()
    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks each document's **contexts** using **MonoBERT** and updates `reorder_contexts`.

        Args:
            documents (List[Document]): A list of **Document** instances to rerank.

        Returns:
            List[Document]: Documents with updated **`reorder_contexts`** after reranking.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            contexts = deepcopy(document.contexts)

            # Rescore contexts using MonoBERT
            for context in contexts:
                inputs = self.tokenizer.encode_plus(
                    query,
                    context.text,
                    max_length=512,
                    truncation=True,
                    return_token_type_ids=True,
                    return_tensors="pt"
                )
                with autocast(enabled=self.use_amp):
                    input_ids = inputs["input_ids"].to(self.device)
                    token_type_ids = inputs["token_type_ids"].to(self.device)
                    outputs = self.model(input_ids, token_type_ids=token_type_ids, return_dict=False)
                    logits = outputs[0]

                    # Handle binary and multi-class classification
                    if logits.size(1) > 1:
                        context.score = torch.nn.functional.log_softmax(logits, dim=1)[0, -1].item()
                    else:
                        context.score = logits.item()

            # Sort contexts by score in descending order
            ranked_contexts = sorted(contexts, key=lambda ctx: ctx.score, reverse=True)

            # Update `reorder_contexts` in the document
            document.reorder_contexts = ranked_contexts

        return documents
