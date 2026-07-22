import torch
from rankify.utils.models.llm2vec import LLM2Vec
from typing import List
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context 
import copy
from tqdm import tqdm  # Import tqdm for progress tracking

class LLM2VecReranker(BaseRanking):
    """
    Implements **LLM2Vec Reranking**, a **zero-shot ranking approach** using large language models (LLMs) as text encoders.


    This method leverages **LLM2Vec embeddings** to compute **cosine similarity** between **queries** and **passage contexts**, 
    allowing efficient **zero-shot reranking** based on **learned representations**.

    References:
        - **BehnamGhader et al. (2024)**: *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders*.
          [Paper](https://arxiv.org/abs/2404.05961)

    Attributes:
        method (str): The **name of the reranking method**.
        model_name (str): The **name of the pre-trained LLM2Vec model** used for encoding.
        device (torch.device): The **device (CPU/GPU)** on which the model runs.
        batch_size (int): The **batch size** for encoding query-context pairs.
        peft_model_name_or_path (str): The **path or name of the PEFT-tuned model variant**.
        instruction (str): The **instruction prompt** used for encoding query-context pairs.
        model (LLM2Vec): The **LLM2Vec model** for generating embeddings.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the benefits of renewable energy?")
        contexts = [
            Context(text="Renewable energy reduces greenhouse gas emissions.", id=0),
            Context(text="Fossil fuels contribute to climate change.", id=1),
            Context(text="Solar power is a sustainable source of electricity.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize LLM2Vec Reranker
        model = Reranking(method='llm2vec', model_name='Meta-Llama-31-8B')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """
    def __init__(self, method: str = None, model_name: str = None, api_key: str = None, **kwargs):
        """
        Initializes the **LLM2Vec Reranker** for reranking tasks.

        Args:
            method (str, optional): The **reranking method name**.
            model_name (str, optional): The **name of the pre-trained LLM2Vec model** 
                (default: `"McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"`).
            api_key (str, optional): **Not used**, but included for framework consistency.
            kwargs (dict): Additional parameters such as `batch_size` and `peft_model_name_or_path`.
        """
        self.method = method
        self.model_name = model_name or "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = kwargs.get("batch_size", 8)
        self.peft_model_name_or_path = kwargs.get("peft_model_name_or_path", "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised")
        self.instruction = kwargs.get("instruction", "Given a query, rank the relevant contexts:")
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the LLM2Vec model.

        Returns:
            LLM2Vec: The pretrained LLM2Vec model.
        """
        return LLM2Vec.from_pretrained(
            self.model_name,
            peft_model_name_or_path=self.peft_model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks each document's **contexts** using **LLM2Vec** embeddings and **cosine similarity**.

        Args:
            documents (List[Document]): A list of **Document** instances containing contexts to rerank.

        Returns:
            List[Document]: The reranked list of **Document** instances with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            # Encode the query
            query = document.question.question
            query_embedding = self._encode_queries([query])

            # Encode the contexts
            contexts = document.contexts
            context_embeddings = self._encode_contexts(contexts)

            # Compute cosine similarity scores
            scores = self._cosine_similarity(query_embedding, context_embeddings)
            copy_context = copy.deepcopy(contexts)
            # Assign scores to contexts and sort
            for i, context in enumerate(copy_context):
                context.score = scores[i].item()

            ranked_contexts = sorted(copy_context, key=lambda ctx: ctx.score, reverse=True)

            # Update `reorder_contexts` in the document
            document.reorder_contexts = ranked_contexts

        return documents

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Encodes a **list of queries** using **LLM2Vec**.

        Args:
            queries (List[str]): List of **query strings**.

        Returns:
            torch.Tensor: **Query embeddings**.
        """
        query_sentences = [[self.instruction, query, 0] for query in queries]
        return self.model.encode(query_sentences, batch_size=self.batch_size, convert_to_tensor=True)

    def _encode_contexts(self, contexts: List[Context]) -> torch.Tensor:
        """
        Encodes a **list of contexts** using **LLM2Vec**.

        Args:
            contexts (List[Context]): List of **context objects** from a **Document**.

        Returns:
            torch.Tensor: **Context embeddings**.
        """
        # Format each context as [instruction, context text, label (dummy)] for LLM2Vec
        context_sentences = [[self.instruction, ctx.text, 0] for ctx in contexts]

        # Pass the formatted sentences to the LLM2Vec model
        return self.model.encode(context_sentences, batch_size=self.batch_size, convert_to_tensor=True)

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Computes the **cosine similarity** between **query** and **context embeddings**.

        Args:
            a (torch.Tensor): The **query embedding tensor**.
            b (torch.Tensor): The **context embedding tensor**.

        Returns:
            torch.Tensor: **Cosine similarity scores** for each context.
        """
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.T).squeeze(0)
