from rankify.models.base import BaseRanking
from typing import Union, List, Optional, Tuple
from rankify.dataset.dataset import Document, Context
import torch
import copy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from tqdm import tqdm  # Import tqdm for progress tracking

from rankify.utils.helper import get_device,get_dtype

class TransformerRanker(BaseRanking):
    """
    Implements **TransformerRanker**, a general **pretrained transformer-based** reranking model.

    References:
        - **MixedBread AI (2024)**: *Reranking Overview*. [Paper](https://www.mixedbread.ai/docs/reranking/overview)
        - **Xiao et al. (2024)**: *BGE-Reranker: A Packed Resource for General Chinese Embeddings*. [Paper](https://arxiv.org/abs/2309.07597)
        - **Günther et al. (2023)**: *Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models*. [Paper](https://arxiv.org/abs/2307.11224)
        - **mGTE (2024)**: *Generalized Long-Context Text Representation for Multilingual Retrieval*. [Paper](https://arxiv.org/abs/2407.19669)
        - **Martin et al. (2019)**: *CamemBERT: A Tasty French Language Model*. [Paper](https://arxiv.org/abs/1911.03894)
        - **BCEmbedding (2023)**: *Bilingual and Cross-lingual Embeddings for RAG*. [Paper](https://github.com/netease-youdao/BCEmbedding)

    Attributes:
        method (str): The name of the reranking method.
        model_name (str): The name or path to the **pretrained reranking model**.
        device (torch.device): The computation device (**CPU/GPU**).
        dtype (torch.dtype): The data type for model inference.
        tokenizer (AutoTokenizer): The tokenizer for encoding queries and documents.
        batch_size (int): The batch size for efficient inference.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the benefits of deep learning?")
        contexts = [
            Context(text="Deep learning allows models to extract hierarchical representations.", id=0),
            Context(text="Machine learning includes both supervised and unsupervised learning.", id=1),
            Context(text="Neural networks are a fundamental component of deep learning.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize Transformer Ranker (e.g., BGE-Reranker)
        model = Reranking(method='transformer_ranker', model_name='mxbai-rerank-xsmall')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """

    def __init__(self, method = None, model_name = None, api_key = None, **kwargs):
        """
        Initializes **TransformerRanker** for reranking tasks.

        Args:
            method (str, optional): The reranking method name.
            model_name (str): The name or path to the **pretrained reranker model**.
            api_key (str, optional): API key if required (default: None).
            **kwargs: Additional parameters:
                - batch_size (int, optional): Batch size for inference (default: `16`).
                - device (str, optional): Device (`"cpu"`, `"cuda"`, `"auto"`). Default is `"cuda"`.
                - dtype (torch.dtype, optional): Data type for inference (`torch.float32` or `torch.bfloat16`).
        """
        device = kwargs.get("device", "cuda")
        self.device = get_device(device)
        self.dtype = get_dtype( kwargs.get("dtype", torch.float32), self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1, 
            trust_remote_code=True,
            torch_dtype=self.dtype
            ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = kwargs.get("batch_size", 16)

    def tokenize(self, inputs: Union[str,List[str], List[Tuple[str, str]]]):
        """
        Tokenizes **queries and documents** for model input.

        Args:
            inputs (Union[str, List[str], List[Tuple[str, str]]]): 
                Query-document pairs to tokenize.

        Returns:
            dict: Tokenized inputs suitable for the reranking model.
        """
        return self.tokenizer(inputs,return_tensors="pt", padding=True, truncation=True).to(self.device)

    @torch.no_grad()
    def rank(self, documents: list[Document])-> List[Document]:
        """
        Reranks a list of **Document** instances using a **Transformer-based Reranker**.

        Args:
            documents (List[Document]): A list of **Document** instances to rerank.

        Returns:
            List[Document]: The reranked list of **Documents** with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            context_copy= copy.deepcopy(document.contexts)

            inputs = [(document.question.question, context.text) for context in document.contexts]

            batched_inputs = [
                inputs[i:i+self.batch_size] for i in range(0,len(inputs),self.batch_size)
            ]
            
            scores = []

            for batch in batched_inputs:
                tokenized_inputs = self.tokenize(batch)
                batch_scores = self.model(**tokenized_inputs).logits.squeeze()
                batch_scores = batch_scores.detach().cpu().numpy().tolist()

                if isinstance(batch_scores, float):
                    scores.append(batch_scores)
                else:
                    scores.extend(batch_scores)
            for score, context in zip(scores,context_copy):
                context.score = score
            context_copy.sort(key=lambda x:x.score, reverse=True)
            document.reorder_contexts = context_copy    
        return documents
                
            
    