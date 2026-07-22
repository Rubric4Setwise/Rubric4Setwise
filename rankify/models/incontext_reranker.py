import math
import torch
from transformers import AutoTokenizer
from typing import List
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from rankify.utils.models.incontext_reranker.custom_cache import DynamicCache, DynamicCacheWithQuery
from rankify.utils.models.incontext_reranker.custom_modeling_mistral import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import repeat_kv
from rankify.utils.models.incontext_reranker.custom_modeling_llama import LlamaForCausalLM
import transformers
import gc
import copy
from tqdm import tqdm  # Import tqdm for progress tracking

class InContextReranker(BaseRanking):
    """
    Implements **In-Context Reranking (ICR)** using **Large Language Models (LLMs)**, specifically **Mistral** and **Llama**.


    **ICR** reranks passages using query-aware **attention scores** computed across multiple LLM layers.  
    It employs a **sliding window strategy** for efficient ranking and supports different retrieval types:

    - **QA Mode**: Answers a given question using retrieved contexts.
    - **IE Mode**: Extracts information relevant to a query.

    The available **scoring strategies** include:
    - **`query_last`**: Default method, uses query token attention scores.
    - **`attention_sorting`**: Ranks contexts based on attention weights.
    - **`NA_only`**: Estimates the model's intrinsic bias.
    - **`NA_calibration_no_agg`**: Uses a neutral `"N/A"` query for bias correction.
    - **`masked_NA_calibration`**: Default ICR method with token-level masking.

    References:
        - **Chen et al. (2024)** *Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers.*  
          [Paper](https://dl.acm.org/doi/10.1145/3626772.3657855)

    Attributes:
        method (str, optional): The reranking method name.
        model_name (str): Name of the LLM used for reranking.
        tokenizer (AutoTokenizer): The tokenizer for text processing.
        llm (torch.nn.Module): The pre-trained LLM model (Mistral or Llama).
        prompt_template (str): Template for constructing query-context prompts.
        scoring_strategy (str): The attention-based scoring strategy used for reranking.
        retrieval_type (str): The retrieval type, either `"QA"` (question answering) or `"IE"` (information extraction).
        sliding_window_size (int): Size of the **sliding window** for reranking.
        sliding_window_stride (int): Step size for moving the **sliding window**.
        reverse_doc_order (bool): Whether to reverse document order before reranking.

    See Also:
        - `Reranking`: Main interface for reranking models, including `InContextReranker`.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Answer, Context
        from rankify.models.reranking import Reranking

        question = Question("What are the symptoms of COVID-19?")
        answers = Answer(["COVID-19 symptoms include fever and cough."])
        contexts = [
            Context(text="Fever and cough are common symptoms of COVID-19.", id=0),
            Context(text="Headache is a rare symptom.", id=1),
            Context(text="Fatigue and loss of taste are also common.", id=2),
        ]
        document = Document(question=question, answers=answers, contexts=contexts)

        # Initialize Reranking with InContextReranker
        model = Reranking(method='incontext_reranker', model_name='llamav3.1-8b')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```

    Notes:
        - Requires a **Mistral** or **Llama** model.
        - Uses **sliding windows** for efficient ranking.
        - Supports **attention-based scoring** for **zero-shot ranking**.
    """
    def __init__(self, method=None, model_name=None, **kwargs):
        """
        Initializes the In-Context Reranker.

        Args:
            method (str, optional): The reranking method name.
            model_name (str): Name of the LLM model (e.g., `"meta-llama/Meta-Llama-3.1-8B-Instruct"`).
            **kwargs: Additional customization parameters, including:
                - `scoring_strategy` (str): Strategy for computing attention scores (`"query_last"`, `"attention_sorting"`, etc.).
                - `retrieval_type` (str): Type of retrieval (`"QA"` or `"IE"`).
                - `sliding_window_size` (int): Number of documents considered per window.

        Example:
            ```python
            model = InContextReranker(method='incontext_reranker', model_name='llamav3.1-8b')
            ```
        """
        # Setup the base LLM
        self._base_llm_name =model_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
        print(f"initialized tokenizer for [{model_name}]")
        
        if any([x in model_name.lower() for x in ['mistralai/mistral', ]]):
            BaseLLMClass = MistralForCausalLM
        elif any([x in model_name.lower() for x in ['llama']]):
            BaseLLMClass = LlamaForCausalLM
        else:
            print(f"Warning: The model family for [{model_name}] is not supported by InContextRAGModel!")
            raise NotImplementedError

        prompt_template = kwargs.get("prompt_template", "instruct")

        prompt_template, prompt_prefix, prompt_suffix, = self._setup_llm_prompts(prompt_template, model_name)
        
        self.use_fa2 = kwargs.get("use_fa2", True)
        if self.use_fa2:
            _attn_implementation = "flash_attention_2"
        else:
            _attn_implementation = "eager"

        llm = BaseLLMClass.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                attn_implementation=_attn_implementation,
                device_map='auto'
            )
        self.llm = llm
        self.llm.config.pad_token_id = self.llm.config.eos_token_id
        
        # Setup prompts for ICR
        assert prompt_template in ['instruct', 'simple', 'simple_instruct'], "Invalid prompt template!"
        
        self.prompt_template = prompt_template
        self.prompt_prefix = prompt_prefix

        
        self.prompt_suffix = prompt_suffix
        self.scoring_strategy  = kwargs.get("scoring_strategy", "query_last")
        retrieval_type = kwargs.get("retrieval_type", "QA")

        if retrieval_type == 'QA':
            print('[ICR is using QA prompt type]')
            self.retrieval_instruction = ' Here are some paragraphs:'
            self.retrieval_instruction_late = 'Please answer the following question based on the information in the paragraphs above.'
        elif retrieval_type == 'IE':
            print('[ICR is using IE prompt type]')
            self.retrieval_instruction = ' Here are some paragraphs:'
            self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.'
        else:
            raise NotImplementedError('Invalid retrieval type! Should be one of [QA, IE]')

        assert self.scoring_strategy in ['query_last', 'attention_sorting', 'NA_only', 'NA_calibration_no_agg', 'masked_NA_calibration'], "Invalid scoring strategy!"
        
        self._use_fa2 = self.use_fa2
        if self.use_fa2:
            print('Using FA2 for retrieval score computation.')
        else:
            print('Using eager attention weights for retrieval score computation.')
        self.num_layers = self.llm.config.num_hidden_layers
        

        self.start_layer = 0
        self.end_layer = self.num_layers - 1
        
        print('[ICR is using layers from {} to {}.]'.format(self.start_layer, self.end_layer))


        # The following settings are for constructing the input prompt.
        self.prompt_bos_length=1
        if any(x in self._base_llm_name.lower() for x in ['mistral-']):
            self.additional_prompt_offset = 1 # for models that adds a ' ' at the beginning when tokenizing the prompt. e.g. '\n\n' -> [<s>, ' ', '\n\n']
            self.prompt_separator = '\n\n'
        elif any([x in self._base_llm_name.lower() for x in ['llama']]):
            self.additional_prompt_offset = 0
            self.prompt_separator = ' \n\n'
        else:
            self.additional_prompt_offset = 0
            self.prompt_separator = '\n\n'

        kwargs.get("reverse_doc_order", False)
        # Setup sliding window.
        # ICR typically works worse with sliding window, especially with smaller window sizes. Try to fit all documents to be re-ranked in the context as much as possible. 
        self.reverse_doc_order = kwargs.get("reverse_doc_order", False)
        self.sliding_window_size = kwargs.get("sliding_window_size", 10)
        if self.sliding_window_size is None:
            self.sliding_window_stride = self.sliding_window_size//2
        else:
            self.sliding_window_stride = self.sliding_window_size
        
    def _setup_llm_prompts(self, prompt_template, base_llm_name):
        """
        Configures the LLM prompt format based on the model type.

        Args:
            prompt_template (str): The prompt template type (`"instruct"`, `"simple"`, `"simple_instruct"`).
            base_llm_name (str): The base model name.

        Returns:
            tuple: The `prompt_template`, `prompt_prefix`, and `prompt_suffix`.

        Example:
            ```python
            prompt_template, prompt_prefix, prompt_suffix = model._setup_llm_prompts("instruct", "llamav3.1-8b")
            ```
        """
        if prompt_template == '':
            prompt_template='instruct' if any(x in base_llm_name.lower() for x in ['instruct']) else 'simple'
        else:
            assert prompt_template in ['instruct', 'simple', 'simple_instruct']
        print('ICR is using prompt template [{}] for in-context retrieval'.format(prompt_template))
        
        if  'mistral' in base_llm_name.lower():
            prompt_prefix = '[INST]'
            prompt_suffix = '[/INST]'
        elif 'llama-3' in base_llm_name.lower():
            prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        else:
            raise NotImplementedError("Prompt prefix and suffix not defined for the model family of {}.".format(base_llm_name))
        
        return prompt_template, prompt_prefix, prompt_suffix

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks documents using **In-Context Reranking (ICR)**.

        Args:
            documents (List[Document]): A list of `Document` instances containing query and contexts.

        Returns:
            List[Document]: Documents with reordered contexts based on reranking.

        Example:
            ```python
            reranked_docs = model.rank(documents)
            ```
        """
        max_length = 300
        for document in tqdm(documents, desc="Reranking Documents"):
            query = document.question.question
            contexts = [ctx.text.split()[:int(max_length)] for ctx in document.contexts]
            
            # Perform reranking with sliding windows
            #print(contexts)
            sorted_doc_ids, sorted_doc_scores = self.rerank(query, contexts, order="desc")[0]
            #print(sorted_doc_ids, sorted_doc_scores)
            # Assign scores and reorder contexts
            copy_context = copy.deepcopy(document.contexts)
            for idx, ctx_id in enumerate(sorted_doc_ids):
                copy_context[ctx_id].score = sorted_doc_scores[idx]

            ranked_contexts = sorted(copy_context, key=lambda x: x.score, reverse=True)
            document.reorder_contexts = ranked_contexts

            
        return documents

    def rerank(self, query, documents,  return_per_doc_results=False, order="desc"):
        """
        Applies **In-Context Reranking** using a **sliding window** approach.

        Args:
            query (str): The query for which documents need to be ranked.
            documents (List[str]): List of document texts.
            return_per_doc_results (bool, optional): Whether to return detailed per-document scores.
            order (str, optional): Sorting order (`"desc"` or `"asc"`).

        Returns:
            tuple: A tuple containing sorted document IDs and their scores.

        Example:
            ```python
            sorted_ids, scores = model.rerank("What is quantum computing?", documents)
            ```
        """
        
        N_docs = len(documents)

        if self.sliding_window_size < 0:
            self.sliding_window_size = N_docs
            
        sorted_doc_ids = list(range(N_docs))
        sorted_doc_ids.reverse()
        
        sorted_doc_scores = []
        if return_per_doc_results == 'tok':
            per_doc_results = []
        else:
            per_doc_results = None
        
        _i = 0
        _j = min(self.sliding_window_size, N_docs)
        while True:
            
            ids = [sorted_doc_ids[i] for i in range(_i, _j)]
            if not self.reverse_doc_order:
                # Put the most relevant documents at the front of document list.
                ids.reverse()

            docs = [documents[i] for i in ids]
            (_sorted_doc_ids, _sorted_doc_scores), _per_doc_results = self.get_sorted_docs(query, docs, return_per_doc_results=return_per_doc_results, order='asc')

            __sorted_doc_ids = [ids[i] for i in _sorted_doc_ids]
            for i in range(_i, _j):
                sorted_doc_ids[i] = __sorted_doc_ids[i-_i]

            if _j < N_docs:
                sorted_doc_scores.extend(_sorted_doc_scores[:self.sliding_window_stride])
                if return_per_doc_results == 'tok':
                    per_doc_results.extend(_per_doc_results[:self.sliding_window_stride])
            else:
                sorted_doc_scores.extend(_sorted_doc_scores)
                if return_per_doc_results == 'tok':
                    per_doc_results.extend(_per_doc_results)
                break

            _i += self.sliding_window_stride
            _j += self.sliding_window_stride
            _j = min(_j, N_docs)
            
        
        if order == 'desc':
            sorted_doc_ids.reverse()
            sorted_doc_scores.reverse()
            if return_per_doc_results == 'tok':
                per_doc_results.reverse()

        assert len(sorted_doc_ids) == len(sorted_doc_scores), "Length mismatch between sorted doc ids ({}) and scores({})!".format(len(sorted_doc_ids), len(sorted_doc_scores))
        return (sorted_doc_ids, sorted_doc_scores), per_doc_results

    def get_sorted_docs(self, query, retrieval_doc_pool, return_per_doc_results=False, prompt_prefix='', order='desc'):
        """
        Scores and sorts documents using **attention-based scoring**.

        Args:
            query (str): The query text.
            retrieval_doc_pool (List[str]): List of documents to be ranked.
            order (str, optional): Sort order (`"desc"` or `"asc"`).

        Returns:
            tuple: Sorted document IDs and their scores.

        Example:
            ```python
            sorted_ids, scores = model.get_sorted_docs("What is climate change?", documents)
            ```
        """
        kv_cache = None
        
        if self.scoring_strategy == 'query_last':
            # ICR without calibration.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            doc_scores, perdoc_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results)
        
        elif self.scoring_strategy == 'attention_sorting':
            # ICR without both calibration and attention aggregation.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            query_start_idx = query_end_idx # Only using last query token (i.e. attention sorting).
            doc_scores, perdoc_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results)
    
        elif self.scoring_strategy == 'NA_only':
            # For analyzing the intrinsic bias captured by calibration scores.
            query = 'N/A'

            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            doc_scores, perdoc_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results)
        
        elif self.scoring_strategy == 'NA_calibration_no_agg':
            # ICR without attention aggregation.
            
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            query_start_idx = query_end_idx
            doc_scores_query, perdoc_result, kv_cache = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results, return_cache=True)
            
            
            calibration_query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(calibration_query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')

            # Use kv_cache from first query to speed up forward() for the calibration query.
            # query_start_idx should be the same for both queries.
            for i in range(len(kv_cache.key_cache)):
                kv_cache.key_cache[i] = kv_cache.key_cache[i][:,:,:query_start_idx,:]
                kv_cache.value_cache[i] = kv_cache.value_cache[i][:,:,:query_start_idx,:]
            kv_cache._seen_tokens = query_start_idx
            
            
            if kv_cache is not None:
                context_start_idx=query_start_idx
            else:
                context_start_idx=0

            query_start_idx = query_end_idx
            doc_scores_calib, doc_tok_scores_calib_na,_ = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  kv_cache=kv_cache, context_start_idx=context_start_idx)

            doc_scores = doc_scores_query - doc_scores_calib
            
            if return_per_doc_results != 'none':
                for i in range(len(perdoc_result)):
                    perdoc_result[i][1] -= doc_tok_scores_calib_na[i][1]
        
        elif self.scoring_strategy == 'masked_NA_calibration':
            return_per_doc_results = 'tok'
            # The default ICR method
            
            # FP with calibration query
            calibration_query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(calibration_query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')

            doc_scores_calib, doc_tok_scores_calib_na, kv_cache = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  return_cache=True)
            
            # Use kv_cache from first query to speed up forward() for the calibration query.
            # query_start_idx should be the same for both queries.
            for i in range(len(kv_cache.key_cache)):
                kv_cache.key_cache[i] = kv_cache.key_cache[i][:,:,:query_start_idx,:]
                kv_cache.value_cache[i] = kv_cache.value_cache[i][:,:,:query_start_idx,:]
            kv_cache._seen_tokens = query_start_idx
            
            if kv_cache is not None:
                context_start_idx=query_start_idx
            else:
                context_start_idx=0

            # FP with the actual query            
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
        
            doc_scores_query, perdoc_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  kv_cache=kv_cache, context_start_idx=context_start_idx)

            
            _i = 0
            doc_scores = torch.zeros(len(retrieval_doc_pool))

            for doc_tok_score, doc_tok_score_na in zip(perdoc_result, doc_tok_scores_calib_na):
                doc_tok_score[1] = doc_tok_score[1].to(doc_tok_score_na[1].device)
                calibrated_scores = doc_tok_score[1] - doc_tok_score_na[1]
                
                mean_bias = calibrated_scores.mean()
                std_bias = calibrated_scores.std()
                threshold = mean_bias - 2*std_bias
                tok_mask = (calibrated_scores>threshold)
                
                doc_tok_score[1] = doc_tok_score[1] * tok_mask
                doc_tok_score_na[1] = doc_tok_score_na[1] * tok_mask
                doc_tok_score[1] = doc_tok_score[1] - doc_tok_score_na[1]
                doc_scores[_i] = doc_tok_score[1].sum()
                _i+=1

        per_doc_result = None
        if order in ['desc', 'asc']:
            sorted_results = torch.sort(doc_scores, descending=(order=='desc'))
            if return_per_doc_results != 'none':
                per_doc_result = [(perdoc_result[i][0], perdoc_result[i][1]) for i in sorted_results.indices]
            
            return (sorted_results.indices.tolist(), sorted_results.values.tolist()), per_doc_result
        elif order=='none':
            # Only return the scores and the per-doc results for documents in the input order.
            # Used during development.
            return list(range(len(retrieval_doc_pool))), doc_scores, per_doc_result
        else:
            print(f"Invalid order: {order}. Please use 'desc', 'asc' or 'none")
            raise NotImplementedError
    def score_documents(
            self,
            llm_input,
            doc_tok_idx_spans,
            query_start_tok_idx,
            query_end_tok_idx,
            context_start_idx=0,
            return_per_doc_results=False,
            long_prompt=False,
            return_cache=False,
            kv_cache=None,
        ):
        """
        Computes **attention-based document scores**.

        Args:
            llm_input (str): The full LLM input prompt.
            doc_tok_idx_spans (List[tuple]): Token spans for each document.
            query_start_tok_idx (int): Start index of query tokens.
            query_end_tok_idx (int): End index of query tokens.

        Returns:
            List[float]: Document relevance scores.

        Example:
            ```python
            scores = model.score_documents(prompt, spans, 10, 20)
            ```
        """
        tokenized_input = self.tokenizer(llm_input,return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids[:, context_start_idx:]
        _query_indices = list(range(query_start_tok_idx-context_start_idx, query_end_tok_idx-context_start_idx+1))
        
        if kv_cache is None:
            if self._use_fa2:
                kv_cache=DynamicCacheWithQuery(query_indices=_query_indices)
            else:
                kv_cache=DynamicCache()
        else:
            kv_cache.query_cache = []
            _query_indices = _query_indices
            kv_cache._query_indices = _query_indices

        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
                )

        if self._use_fa2:
            # Extract key and query vectors from FA2. Then recompute attention scores for re-ranking.
            kv_cache = output.past_key_values

            long_prompt = False
            if len(_input_ids[0]) > 40000:
                # For sequences that are too long, compute scores on CPU to void GPU OOM.
                # Adjust the limit here depending on your system configuration.
                print('Long sequence of more than 40K tokens detected. Computing attention scores on CPU.')
                long_prompt = True
            
            attention_weights = []
            doc_tok_weights = []
            
            if long_prompt:
                _device = 'cpu'
            else:
                _device = 'cuda:0'
            
            # loop through all layers and compute attention scores
            for i in range(self.start_layer, self.end_layer+1):                     
                attn_weights = self._get_attn_weights(kv_cache.key_cache[i][:,:,:query_end_tok_idx+1], kv_cache.query_cache[i],  use_cpu=long_prompt).to(_device).squeeze(0)
                attn_weights = attn_weights.mean(1) # average over query tokens
                attention_weights.append(attn_weights.squeeze(0))
                
        else:
            # Directly extract attention weights from the attention layers of the LLM.
            attention_weights = [attn[0][:,query_start_tok_idx:query_end_tok_idx+1,:].mean(1) for attn in output.attentions]

        attention_weights = torch.stack(attention_weights, dim=0)
        
        if return_per_doc_results != 'none':
            per_doc_results = [[None, None] for _ in range(len(doc_tok_idx_spans))]
        else:
            per_doc_results = None
    
        attention_weights = attention_weights.sum(0) # sum attention scores across layers            
        attention_weights = attention_weights.sum(0) # sum attention scores across attention heads
        doc_scores = []
        
        for i, doc_span in enumerate(doc_tok_idx_spans): 
            _tok_score = attention_weights[doc_span[0]:doc_span[1]]
            doc_scores.append(_tok_score.sum())

            if return_per_doc_results != 'none':
                _doc_tok_ids = tokenized_input.input_ids[0][doc_span[0]:doc_span[1]]
                _doc_toks = self.tokenizer.convert_ids_to_tokens(_doc_tok_ids)
                per_doc_results[i][0] = _doc_toks
                per_doc_results[i][1] = _tok_score.clone().detach() # sum over layers
            
        doc_scores = torch.tensor(doc_scores)
        gc.collect()
        torch.cuda.empty_cache()

        if return_cache:
            return doc_scores, per_doc_results, kv_cache
        else:
            return doc_scores, per_doc_results
    def _prepare_input_for_document_retrieval(self, query, documents, system_prompt='', query_position='last'):
        """
        Constructs the **ICR prompt** with query and document contexts.

        Args:
            query (str): Query text.
            documents (List[str]): List of documents.
            system_prompt (str, optional): Custom system prompt.
            query_position (str, optional): Query placement (`"first"` or `"last"`).

        Returns:
            tuple: The formatted prompt and document spans.

        Example:
            ```python
            prompt, spans, query_start, query_end = model._prepare_input_for_document_retrieval("What is AI?", docs)
            ```
        """
        llm_prompt = ''
        document_span_intervals = []
        

        if self.prompt_template == 'simple':
            system_prompt = ''
        elif self.prompt_template == 'simple_instruct':
            system_prompt = system_prompt
        elif self.prompt_template == 'instruct':
            if system_prompt != '':
                system_prompt = self.retrieval_instruction.format(len(documents), query) + self.prompt_separator + system_prompt
            else:
                system_prompt = self.retrieval_instruction.format(len(documents), query)
        
        system_prompt = self.prompt_prefix + system_prompt

        query_start_idx = None
        query_end_idx = None
        
        
        separator_length = self.tokenizer(self.prompt_separator, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset # remove the leading ['<s>', '_'] tokens
        
        llm_prompt = system_prompt
        
        
        prompt_length = self.tokenizer(llm_prompt+self.prompt_separator, return_tensors='pt').input_ids.size(1)-separator_length # add and subtract separator tokens for accurate prefix length
        
        if query_position == 'first':
            if self.prompt_template in ['simple', 'instruct']:
                instruction_prompt = f'Query:'

                llm_prompt += self.prompt_separator + instruction_prompt 
                prompt_length += separator_length
                prompt_length += self.tokenizer(self.prompt_separator + instruction_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
                query_start_idx = prompt_length - 1 # The ':' after 'Query'    
            else:
                llm_prompt += self.prompt_separator
                prompt_length += separator_length
                query_start_idx = prompt_length # The start of the query context
            
            if self.prompt_template == 'simple':
                query_prompt = f' {query.strip()}{self.prompt_separator}Answer:'
            elif self.prompt_template in ['instruct', 'simple_instruct']:
                query_prompt = f' {query.strip()}'
            
            llm_prompt += query_prompt
            prompt_length += self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
            query_end_idx = prompt_length - 1 
            
        
        if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch!')
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                print('-'*30)
                self.__show_tokens(llm_prompt)
                raise Exception('ICR prompt length mismatch before adding docs.')

        _doc_separator_length = separator_length

        for i, doc in enumerate(documents):
            
            doc = f'[{i+1}] {doc}'
            prompt_length += _doc_separator_length
            llm_prompt += self.prompt_separator + doc
            doc_length = self.tokenizer(self.prompt_separator + doc, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - _doc_separator_length - self.additional_prompt_offset # - bos_length for the leading ['<s>'] token, -additional for the potential extra tokens, e.g. the '_' token between <s> and <0x0A> when <0x0A> is the first token for mistral models.
            
            document_span_intervals.append((prompt_length, prompt_length + doc_length))
            prompt_length += doc_length

            if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch @ doc {}!'.format(i))
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                print('-'*30)
                self.__show_tokens(llm_prompt)
                print('-'*30)
                print('doc length:', doc_length)
                self.__show_tokens(self.prompt_separator+doc)
                raise Exception('ICR prompt length mismatch after adding docs.')


        if query_position == 'last':
            query_start_idx = prompt_length + separator_length
            if self.prompt_template in ['simple', 'instruct']:
                instruction_prompt = self.retrieval_instruction_late + self.prompt_separator + 'Query:'
                llm_prompt += self.prompt_separator + instruction_prompt 
                prompt_length += separator_length
                prompt_length += self.tokenizer(self.prompt_separator + instruction_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
                
            else:
                llm_prompt += self.prompt_separator
                prompt_length += separator_length

        if self.prompt_template == 'simple':
            query_prompt = f' {query.strip()}{self.prompt_separator}Answer:'
        elif self.prompt_template in ['instruct', 'simple_instruct']:
            query_prompt = f' {query.strip()}'
            if query_position == 'last':
                query_prompt += self.prompt_suffix.format(len(documents))

        llm_prompt += query_prompt
        prompt_length += self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
        if query_position == 'last':
            query_end_idx = prompt_length - 1
        return llm_prompt, document_span_intervals, query_start_idx, query_end_idx
    
    @classmethod
    def __show_tokens(self, string):
        # Shows tokenized string.
        # Mainly used for debugging prompt construction for document retrieval.
        tokenized_string_ids = self.tokenizer(string, return_tensors='pt').input_ids[0]
        print(self.tokenizer.convert_ids_to_tokens(tokenized_string_ids), tokenized_string_ids.size(0))
                      
    @classmethod
    def _get_attn_weights(cls, key_states, query_states, use_cpu=False):

        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        if use_cpu:
            query_states = query_states.cpu()
            key_states = key_states.cpu()

        key_states = repeat_kv(key_states, num_key_value_groups)


        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Make causal mask and add it to attention weights.
        causal_mask = cls._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True) # Log-sum-exp of attention weights for numerical stability in softmax.
        attn_weights = torch.exp(attn_weights - attn_lses) # softmax
        
        return attn_weights
    
    @classmethod
    def _get_causal_mask(cls, attn_weights):
        # Make causal mask for attention weights.
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(0))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask