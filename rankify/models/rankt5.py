import torch
import transformers
import numpy as np
import os
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List
import copy
from tqdm import tqdm  # Import tqdm for progress tracking
import copy

class RankT5(BaseRanking):
    """
    Implements **MonoRankT5**, a **T5-based re-ranking method** that fine-tunes **T5** with ranking losses.


    RankT5 supports both **MonoT5** and **RankT5** re-ranking models:
    
    - **MonoT5** applies a **binary classification loss** to determine query-document relevance.
    - **RankT5** uses **ranking losses** to improve ranking effectiveness.

    References:
        - **Zhuang, H. et al. (2023)**: *RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses*.
          [Paper](https://arxiv.org/abs/2210.10634)

    Attributes:
        method (str): The ranking method (`"monot5"` or `"rankt5"`).
        model_name (str): The name of the **T5 model**.
        model (T5ForConditionalGeneration): The **T5 model** for ranking.
        tokenizer (T5Tokenizer): The tokenizer for encoding input texts.
        max_input_length (int): Maximum sequence length for input encoding.
        batch_size (int): The batch size for processing documents.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What is the impact of climate change?")
        contexts = [
            Context(text="Climate change causes rising sea levels and extreme weather.", id=0),
            Context(text="The stock market fluctuates due to various economic factors.", id=1),
            Context(text="Global warming contributes to increased wildfires and heatwaves.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize MonoRankT5
        model = Reranking(method='rankt5', model_name='rankt5-base')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """
    def __init__(self, method: str = None, model_name: str = 'castorini/monot5-base-msmarco', api_key: str = None , **kwargs):
        """
        Initializes the **MonoRankT5** class.

        Args:
            method (str): The ranking method (`"monot5"` or `"rankt5"`).
            model_name (str): The name of the **T5 model**.
            api_key (str, optional): Not used, included for framework consistency.

        Raises:
            ValueError: If the `method` is not `"monot5"` or `"rankt5"`.
        """
        self.method = method
        self.model_name = model_name
        self.mode = method.lower()
        if self.mode not in ['monot5', 'rankt5']:
            raise ValueError("Mode must be either 'monot5' or 'rankt5'")
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.max_input_length = 512  # Default maximum input length
        self.batch_size = 10         # Default batch size for inference

    def load_model_and_tokenizer(self):
        """
        Loads the pre-trained **T5 model** and tokenizer.

        Returns:
            Tuple[T5ForConditionalGeneration, T5Tokenizer]: The loaded model and tokenizer.
        """
        print("Loading T5 model and tokenizer...")
        model = T5ForConditionalGeneration.from_pretrained(self.model_name).to('cuda')
        model.eval()
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer

    def run_inference(self, input_tensors):
        """
        Runs inference using the **T5 model**.

        Args:
            input_tensors (dict): The input tensors for model inference.

        Returns:
            dict: The model outputs.
        """
        with torch.no_grad():
            output = self.model.generate(
                **input_tensors, 
                max_length=2,
                return_dict_in_generate=True,
                output_scores=True
            )
        return output

    def rank(self, documents: List[Document])-> List[Document]:
        """
        Reranks the passages for each document using **T5-based re-ranking**.

        Args:
            documents (List[Document]): List of documents containing queries and passages.

        Returns:
            List[Document]: The documents with updated `reorder_contexts`.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            passages = copy.deepcopy(document.contexts)

            # Prepare input texts based on the selected mode
            if self.mode == 'monot5':
                input_texts = [f"Query: {query} Document: {x.text} Relevant:" for x in passages]
            elif self.mode == 'rankt5':
                input_texts = [f"Query: {query} Document: {x.text}" for x in passages]

            # Group inputs into batches
            grouped_input_texts = list(self.group2chunks(input_texts, n=self.batch_size))

            scores_holder = []
            for batch_input_texts in grouped_input_texts:
                input_tensors = self.tokenizer(
                    batch_input_texts, 
                    return_tensors='pt',
                    padding='max_length', 
                    max_length=self.max_input_length, 
                    truncation=True
                ).to('cuda')

                outputs = self.run_inference(input_tensors)
                del input_tensors

                scores = torch.stack(outputs.scores)

                # Extract scores based on the mode
                if self.mode == 'monot5':
                    # Extract scores for 'yes' (true) and 'no' (false)
                    yesno_softmax_scores = torch.nn.functional.log_softmax(
                        scores[0][:, [1176, 6136]], dim=1
                    )[:, 0].tolist()  # true, false
                    scores_holder += yesno_softmax_scores
                elif self.mode == 'rankt5':
                    # Extract the score associated with <extra_id_10>
                    rankt5_scores = scores[0][:, 32089].tolist()  # <extra_id_10>
                    scores_holder += rankt5_scores

            # Rank passages based on scores
            all_scores_tensor = torch.tensor(scores_holder)
            rank = torch.argsort(all_scores_tensor, descending=True).tolist()

            # Prepare the reordered list of contexts
            #reranked_passages = [copy.deepcopy(passages[rank_id]) for rank_id in rank]
            reranked_passages = []
            for new_idx in rank:
                context = copy.deepcopy(passages[new_idx])
                context.score = scores_holder[new_idx]
                reranked_passages.append(context)
            # Assign the reranked passages to reorder_contexts
            document.reorder_contexts = reranked_passages
        return documents
    def group2chunks(self, lst, n=5):
        """
        Groups a list into chunks of size **n**.

        Args:
            lst (list): The list to be chunked.
            n (int, optional): The chunk size (default is `5`).

        Yields:
            list: A chunk of `n` elements.
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
