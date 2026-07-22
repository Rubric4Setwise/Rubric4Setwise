import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
from rankify.dataset.dataset import Document

import os
import zipfile
import requests
from tqdm import tqdm
from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
import collections
from typing import Optional, List, Dict, Any
import logging
from rankify.models.base import BaseRanking
from tqdm import tqdm  # Import tqdm for progress tracking

import copy

class FlashRanker(BaseRanking):
    """
    Implements FlashRank, a fast and efficient reranking model supporting pairwise cross-encoding (ONNX)
    and listwise ranking with LLMs (GGUF models).

    FlashRank efficiently reranks passages for a given query using either:
    - ONNX-based pairwise reranking (cross-encoder) for fast inference.
    - LLM-based listwise reranking using RankGPT.

    This method is optimized for speed and accuracy while maintaining scalability.

    Attributes:
        method (str): The reranking method name.
        model_name (str): The name of the model used for reranking.
        api_key (str, optional): API key for accessing remote models (if applicable).
        cache_dir (Path): Directory where models are cached.
        model_dir (Path): Directory containing the specific model.
        session (ort.InferenceSession, optional): The ONNX runtime session for inference (used for pairwise reranking).
        tokenizer (Tokenizer, optional): The tokenizer for text processing.
        llm_model (Llama, optional): If using an LLM-based reranker, this holds the model instance.

    References:
        - Damodaran, P. (2023). FlashRank, Lightest and Fastest 2nd Stage Reranker for search pipelines.  
          [Paper](https://doi.org/10.5281/zenodo.10426927)

    Examples:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.models.reranking import Reranking

        question = Question("What is the capital of France?")
        answers = Answer(["Paris is the capital of France."])
        contexts = [
            Context(text="Berlin is the capital of Germany.", id=0),
            Context(text="Paris is the capital of France.", id=1),
            Context(text="Madrid is the capital of Spain.", id=2),
        ]
        document = Document(question=question, answers=answers, contexts=contexts)

        # Initialize Reranking with FlashRanker
        model = Reranking(method="flashrank", model_name="ms-marco-TinyBERT-L-2-v2")
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```

    Notes:
        - FlashRank supports ONNX models for cross-encoder-based reranking.
        - LLM-based reranking is supported using GGUF models.
        - Integrated into the `Reranking` class, so use `Reranking` instead of `FlashRanker` directly.
    """

    def __init__(self, method: str = None, model_name: str = None, api_key: str = None, **kwargs):
        """
        Initializes the FlashRanker model for reranking.

        Args:
            method (str, optional): The reranking method name.
            model_name (str, optional): The name of the reranking model to be used.
            model_dir (str, optional): Path to a custom model directory if the user provides their own model.
            api_key (str, optional): API key for remote access (if applicable).

        Raises:
            ValueError: If an invalid model name is provided or model files are missing.
        """
        max_length: int = 512
        log_level: str = "INFO"
        
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        self.logger = logging.getLogger(__name__)
        model_dir = kwargs.get("model_dir", None)
        # Predefined FlashRank models
        available_models = HF_PRE_DEFIND_MODELS['flashrank']
        
        if model_name:
            if model_name not in available_models:
                raise ValueError(
                    f"Invalid model name '{model_name}'. Choose from: {list(available_models.keys())} or specify a custom model directory."
                )
            model_file = HF_PRE_DEFIND_MODELS['flashrank-model-file'].get(model_name)
        elif model_dir:
            model_name = "custom_model"
            model_file =kwargs.get("model_file")  # User must provide the model file name
            if not model_file:
                raise ValueError("When using a custom model directory, you must specify 'model_dir' and 'model_file'.")

        else:
            raise ValueError("Either 'model_name' or 'model_dir' must be provided.")

        # Set paths
        self.cache_dir: Path = Path(os.environ.get('RERANKING_CACHE_DIR', './cache')) / 'models'
        self.model_dir: Path = Path(model_dir) if model_dir else self.cache_dir / model_name

        # Ensure model directory exists
        if not self.model_dir.exists():
            if model_dir:
                raise FileNotFoundError(f"Custom model directory '{self.model_dir}' not found.")
            self.logger.info(f"Downloading model '{model_name}'...")
            self._download_model_files(model_name)

        listwise_rankers = {'rank_zephyr_7b_v1_full'}
        self.llm_model = None

        if model_name in listwise_rankers:
            try:
                from llama_cpp import Llama
                self.llm_model = Llama(
                    model_path=str(self.model_dir / model_file),
                    n_ctx=max_length,
                    n_threads=8,
                )
            except ImportError:
                raise ImportError("Please install 'pip install flashrank[listwise]' to use listwise rerankers.")

        else:
            self.session = ort.InferenceSession(str(self.model_dir / model_file))
            self.tokenizer: Tokenizer = self._get_tokenizer(max_length)

    def _prepare_model_dir(self, model_name: str):
        """
        Ensures the model directory is prepared by downloading and extracting the model if not present.

        Args:
            model_name (str): The name of the model to be prepared.
        """
        if not self.cache_dir.exists():
            self.logger.debug(f"Cache directory {self.cache_dir} not found. Creating it..")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.model_dir.exists():
            self.logger.info(f"Downloading {model_name}...")
            self._download_model_files(model_name)

    def _download_model_files(self, model_name: str):
        """
        Downloads and extracts the model files from a specified URL.

        Args:
            model_name (str): The name of the model to download.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        local_zip_file = self.cache_dir / f"{model_name}.zip"
        model_url = "https://huggingface.co/prithivida/flashrank/resolve/main/{}.zip"
        formatted_model_url = model_url.format(model_name)
        
        with requests.get(formatted_model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_zip_file, 'wb') as f, tqdm(desc=local_zip_file.name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        os.remove(local_zip_file)

    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """
        Initializes and configures the tokenizer with padding and truncation.

        Args:
            max_length (int): The maximum token length for truncation.

        Returns:
            Tokenizer: Configured tokenizer for text processing.
        """
        config = json.load(open(str(self.model_dir / "config.json")))
        tokenizer_config = json.load(open(str(self.model_dir / "tokenizer_config.json")))
        tokens_map = json.load(open(str(self.model_dir / "special_tokens_map.json")))
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        #print(tokenizer_config["model_max_length"] , max_length)
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        vocab_file = self.model_dir / "vocab.txt"
        if vocab_file.exists():
            tokenizer.vocab = self._load_vocab(vocab_file)
            tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in tokenizer.vocab.items()])
        return tokenizer

    def _load_vocab(self, vocab_file: Path) -> Dict[str, int]:
        """
        Loads the vocabulary from a file and returns it as an ordered dictionary.

        Args:
            vocab_file (Path): The file path to the vocabulary.

        Returns:
            Dict[str, int]: An ordered dictionary mapping tokens to their respective indices.
        """
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
    
    def _get_prefix_prompt(self, query, num):
        """
        Generates the prefix for prompting the LLM.

        Args:
            query (str): The query to rank contexts against.
            num (int): The number of passages to rank.

        Returns:
            List[Dict[str, str]]: The initial ranking prompt.
        """
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_postfix_prompt(self, query, num):
        """
        Generates the final prompt for the LLM to request ranking output.

        Args:
            query (str): The query to rank contexts against.
            num (int): The number of passages to be ranked.

        Returns:
            Dict[str, str]: The prompt to finalize the ranking request.
        """
        example_ordering = "[2] > [1]"
        return {
            "role": "user",
            "content": f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain.",
        }



    def rank(self, documents: List[Document] ) -> List[Document]:
        """
        Reranks a list of documents using FlashRank.

        Args:
            documents (List[Document]): A list of Document instances to rerank.

        Returns:
            List[Document]: Documents with updated `reorder_contexts` after reranking.
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            if len(document.contexts) == 0:
                continue
            query = document.question.question
            passages = document.contexts
            if self.llm_model is not None:
                document.reorder_contexts=self._listwisellm(query,passages)
            else:
                document.reorder_contexts=self._pairwisecrossencoding(query,passages)
        return documents
    def _listwisellm(self,query,passages):
        """
        Performs listwise reranking using an LLM model.

        Args:
            query (str): The search query.
            passages (List[Document]): The list of passages to be ranked.

        Returns:
            List[Document]: Passages sorted in descending order of relevance.

        Raises:
            ImportError: If `llama_cpp` is not installed for GGUF-based listwise reranking.
        """
        if self.llm_model is not None:
            self.logger.debug("Running listwise ranking..")
            num_of_passages = len(passages)
            messages = self._get_prefix_prompt(query, num_of_passages)

            result_map = {}
            for rank, passage in enumerate(passages):
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{rank + 1}] {passage.text}",
                    }
                )
                messages.append(
                        {
                            "role": "assistant", 
                            "content": f"Received passage [{rank + 1}]."
                        }
                )
                
                result_map[rank + 1] = passage

            messages.append(self._get_postfix_prompt(query, num_of_passages))
            raw_ranks = self.llm_model.create_chat_completion(messages)
            results = []
            for rank in raw_ranks["choices"][0]["message"]["content"].split(" > "):
                results.append(result_map[int(rank.strip("[]"))])
            return results    

        # self.session will be instantiated for ONNX based pairwise CE models
    def _pairwisecrossencoding(self,query,passages):
        """
        Performs pairwise cross-encoding reranking using an ONNX-based model.

        Args:
            query (str): The search query.
            passages (List[Document]): The list of passages to be ranked.

        Returns:
            List[Document]: Passages sorted in descending order of relevance.
        """
        passages_copy = copy.deepcopy(passages)
        self.logger.debug("Running pairwise ranking..")
        query_passage_pairs = [[query, passage.text] for passage in passages_copy]

        input_text = self.tokenizer.encode_batch(query_passage_pairs)
        input_ids = np.array([e.ids for e in input_text])
        token_type_ids = np.array([e.type_ids for e in input_text])
        attention_mask = np.array([e.attention_mask for e in input_text])

        use_token_type_ids = token_type_ids is not None and not np.all(token_type_ids == 0)

        onnx_input = {"input_ids": input_ids.astype(np.int64), "attention_mask": attention_mask.astype(np.int64)}
        if use_token_type_ids:
            onnx_input["token_type_ids"] = token_type_ids.astype(np.int64)

        outputs = self.session.run(None, onnx_input)

        logits = outputs[0]

        if logits.shape[1] == 1:
            scores = 1 / (1 + np.exp(-logits.flatten()))
        else:
            exp_logits = np.exp(logits)
            scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)

        for score, passage in zip(scores, passages_copy):
            passage.score = score

        passages_copy.sort(key=lambda x: x.score, reverse=True)
        return passages_copy