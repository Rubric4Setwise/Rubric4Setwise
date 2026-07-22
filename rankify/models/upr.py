# The following code is modified from https://github.com/DevSinghSachan/unsupervised-passage-reranking/blob/main/upr.py
# For more details, refer to the paper: https://arxiv.org/abs/2204.07496
import random
import numpy
import json
import os
import shutil
import torch
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from rankify.dataset.dataset import Document
from typing import List
from transformers import logging
logging.set_verbosity_error()
from rankify.models.base import BaseRanking
from tqdm import tqdm  # Import tqdm for progress tracking
import copy


def set_random_seed(seed):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value for random number generators.

    Example:
        ```python
        set_random_seed(42)
        ```
    """
    random.seed(seed)
    numpy.seed(seed)
    torch.manual_seed(seed)



class UPR(BaseRanking):
    """
    Implements Unsupervised Passage Reranking (UPR) using models like GPT and T5.

    This class follows the approach described in UPR paper for unsupervised reranking
    of retrieved documents by leveraging question generation techniques. The model improves
    passage retrieval by estimating the likelihood of a query given a passage.

    Attributes:
        model_name (str): The name of the model used for reranking.
        method (str): The reranking method.
        model (transformers.PreTrainedModel): The pre-trained model utilized for reranking.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        use_bf16 (bool): Whether to use bfloat16 precision for computations.
        use_gpu (bool): Whether GPU acceleration is enabled.
        batch_size (int): Batch size for processing contexts.
        shard_size (int): Shard size for processing contexts in chunks.
        include_eos_token (bool): Whether to include the end-of-sequence token during ranking.
        verbalizer_head (str): Prefix to prepend before the passage during input construction.
        verbalizer (str): A prompt instructing the model to generate a question based on the passage.

    References:
        Devendra Singh Sachan et al. "Improving Passage Retrieval with Zero-Shot Question Generation."
        Proceedings of the 2022 Annual Conference of the Association for Computational Linguistics (ACL), 2022.
        [Paper](https://arxiv.org/abs/2204.07496) 
    """
    def __init__(self, method: str= None, model_name: str= 'google/t5-small-lm-adapt', api_key: str=None, **kwargs) -> None:
        """
        Initializes a UPR instance.

        Args:
            method (str, optional): The name of the reranking method. Defaults to None.
            model_name (str, optional): The name of the model to be used for reranking.
                Defaults to `"google/t5-small-lm-adapt"`.
            api_key (str, optional): API key for remote model access. Defaults to None.
            **kwargs: Additional parameters for model configuration.

        Example:
            ```python
            model = Reranking(method="upr", model_name="t5-base")
            model.rank([document])
            ```
        """
        self.model_name = model_name
        self.method = method
        self.model =  None
        self.tokenizer = None
        self.use_bf16 = kwargs.get("use_bf16", True) 
        self.use_gpu = kwargs.get("use_gpu", True)  
        self.batch_size= kwargs.get("batch_size", 1)  
        self.shard_size = kwargs.get("shard_size", 128)   
        self.include_eos_token = kwargs.get("include_eos_token", True)    
        self.verbalizer_head= kwargs.get("verbalizer_head", "Passage: ")     
        self.verbalizer = kwargs.get("verbalizer", "Please write a question based on this passage.") 
        self._load()


    def _load(self) -> None:
        """
        Loads the pre-trained model and tokenizer.

        Depending on the model name, this method loads either a T5 or GPT-based model.
        """
        if 'gpt' in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_bf16 else torch.float32)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype=torch.bfloat16 if self.use_bf16 else torch.float32
            )

        for param in self.model.parameters():
            param.required_grad = False
        
        if self.use_gpu:
            self.model = self.model.cuda()
        
        self.model.eval()
    @classmethod
    def rank_t5(cls, document , model, tokenizer, verbalizer_head, verbalizer, use_gpu, shard_size):
        """
        Reranks contexts in the document using a T5 model.

        Args:
            document (Document): The document containing contexts to be reranked.
            model (T5ForConditionalGeneration): The pre-trained T5 model.
            tokenizer (T5Tokenizer): The tokenizer associated with the T5 model.
            verbalizer_head (str): Prefix to use before the passage during input construction.
            verbalizer (str): A prompt that requests the model to generate a question based on the passage.
            use_gpu (bool): Whether to use GPU for computations.
            shard_size (int): Shard size for processing contexts in chunks.

        Returns:
            List[Document]: The reordered list of contexts based on their relevance.

        Example:
            ```python
            reranked_contexts = UPR.rank_t5(document, model, tokenizer, "Passage:", "Generate a question.", True, 128)
            ```
        """
        all_ids=[]
        has_answer_list=[]
        for context in document.contexts:
            text,title,has_answer = context.text,context.title,context.has_answer
            if title is not None:
                ids="{} {} {}. {}".format(verbalizer_head, title, text,verbalizer)
            else:
                ids="{} {}. {}".format(verbalizer_head, text,verbalizer)

            all_ids.append(ids)
            has_answer_list.append(has_answer)
        
        input_encoding = tokenizer(all_ids, 
                                        padding='longest',
                                        max_length=512,
                                        pad_to_multiple_of=8,
                                        truncation=True,
                                        return_tensors='pt')
        context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

        if use_gpu:
            context_tensor = context_tensor.cuda()
            attention_mask= attention_mask.cuda()
        
        decoder_prefix = document.question.question

        target_encoding = tokenizer(decoder_prefix, 
                                            max_length=128, 
                                            truncation=True,
                                            return_tensors='pt')
        
        decoder_prefix_tensor = target_encoding.input_ids

        if use_gpu:
            decoder_prefix_tensor = decoder_prefix_tensor.cuda()
        
        decoder_prefix_tensor = torch.repeat_interleave(decoder_prefix_tensor,
                                                        len(context_tensor),
                                                        dim=0)
        shared_nll_list = []
        for i in range(0,len(context_tensor), shard_size):
            encoder_tensor_view = context_tensor[i:i+shard_size]
            attention_mask_view = attention_mask[i:i+shard_size]
            decoder_tensor_view = decoder_prefix_tensor[i:i+shard_size]

            with torch.no_grad():
                logits = model(input_ids = encoder_tensor_view, 
                                    attention_mask=attention_mask_view,
                                    labels=decoder_tensor_view).logits
            log_softmax = torch.nn.functional.log_softmax(logits,dim=-1)
            nll = - log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

            avg_null = torch.sum(nll,dim=1)
            shared_nll_list.append(avg_null)

        topk_scores,indexes = torch.topk(-torch.cat(shared_nll_list),k=len(context_tensor))
        reordered_context = [copy.deepcopy(document.contexts[i]) for i in indexes]

        for i, ctx in enumerate(reordered_context):
            ctx.score = topk_scores[i].item()
        return reordered_context
    
    @classmethod    
    def rank_gpt(cls, document , model, tokenizer, verbalizer_head, verbalizer, use_gpu, shard_size, include_eos_token):
        """
        Reranks contexts in the document using a GPT model.

        Args:
            document (Document): The document containing contexts to be reranked.
            model (AutoModelForCausalLM): The pre-trained GPT model.
            tokenizer (AutoTokenizer): The tokenizer associated with the GPT model.
            verbalizer_head (str): Prefix to use before the passage during input construction.
            verbalizer (str): A prompt that requests the model to generate a question based on the passage.
            use_gpu (bool): Whether to use GPU for computations.
            shard_size (int): Shard size for processing contexts in chunks.
            include_eos_token (bool): Whether to include the end-of-sequence token during ranking.

        Returns:
            List[Document]: The reordered list of contexts based on their relevance.

        Example:
            ```python
            reranked_contexts = UPR.rank_gpt(document, model, tokenizer, "Passage:", "Generate a question.", True, 128, True)
            ```
        """
        all_ids , all_labels = [], []
        has_answer_list = []
        max_input_size = -1
        for context in document.contexts:
            text,title,has_answer = context.text,context.title,context.has_answer
            if title is not None:
                passage ="{} {} {}. {}".format(verbalizer_head, title, text,verbalizer)
            else:
                passage ="{} {}. {}".format(verbalizer_head, text,verbalizer)

            cids = tokenizer(passage,
                             max_length=512,
                             truncation=True).input_ids
            
            clabel = [-100]*len(cids)

            question = document.question.question

            qids= tokenizer(question,
                            max_length=128,
                            truncation=True).input_ids
            
            qlabel= qids

            ids = cids + qids
        
            labels = clabel + qlabel #+ [tokenizer.eos_token_id]

            if include_eos_token:
                ids = ids + [tokenizer.eos_token_id]
                labels = labels +[tokenizer.eos_token_id]
            
            all_ids.append(ids)
            all_labels.append(labels)

            if len(ids)> max_input_size:
                max_input_size = len(ids)
            
            has_answer_list.append(has_answer)
        
        padded_labels, padded_ids = [], []
        for ids, label in zip(all_ids, all_labels):
            assert len(ids) == len(label)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            #print(tokenizer.pad_token_id, tokenizer.eos_token_id)
            if len(label) < max_input_size:
                label = label + [-100] * (max_input_size - len(label))
                ids = ids + [tokenizer.pad_token_id] * (max_input_size - len(ids))

            padded_labels.append(label)
            padded_ids.append(ids)

        #print(padded_ids)
        padded_labels = torch.LongTensor(padded_labels)
        padded_ids = torch.LongTensor(padded_ids)


        if use_gpu:
            context_tensor = padded_ids.cuda()
            padded_labels = padded_labels.cuda()
        else:
            context_tensor = padded_ids
        
        sharded_nll_list = []

        for i in range(0, len(context_tensor), shard_size):
            encoder_tensor_view = context_tensor[i: i + shard_size]
            labels_view = padded_labels[i: i + shard_size]

            with torch.no_grad():
                logits = model(input_ids= encoder_tensor_view).logits
                shift_logits = logits[...,:-1,:].contiguous()
                shift_labels = labels_view[..., 1:].contiguous()
                loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nll = nll.view(shift_labels.size())
                avg_nll = torch.sum(nll, dim=1)
            sharded_nll_list.append(avg_nll)

        topk_scores,indexes = torch.topk(-torch.cat(sharded_nll_list),k=len(context_tensor))
        reordered_context = [copy.deepcopy(document.contexts[i]) for i in indexes]

        for i, ctx in enumerate(reordered_context):
            ctx.score = topk_scores[i].item()
        return reordered_context


    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Reranks the contexts in each document using the appropriate model (GPT or T5).

        Args:
            documents (List[Document]): A list of documents whose contexts need to be reranked.

        Returns:
            List[Document]: The reranked list of documents.

        Notes:
            - Uses T5 for sequence-to-sequence reranking and GPT for generative reranking.
            - Scores passages based on the probability of generating the input query.

        Example:
            ```python
            model = Reranking(method="upr", model_name="t5-base")
            model.rank([document])
            ```
        """
        for document in tqdm(documents, desc="Reranking Documents"):
            if len(document.contexts) == 0:
                continue
            if 'gpt' in self.model_name:
                reordered_context = self.rank_gpt(document,  self.model, self.tokenizer, self.verbalizer_head, self.verbalizer, self.use_gpu , self.shard_size, self.include_eos_token)
            else:
                reordered_context = self.rank_t5(document,  self.model, self.tokenizer, self.verbalizer_head, self.verbalizer, self.use_gpu , self.shard_size)
            document.reorder_contexts= reordered_context
        return documents
            












    
