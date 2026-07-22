import torch
from transformers import pipeline
from typing import List
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from tqdm import tqdm
import copy
import math

class EchoRankReranker(BaseRanking):
    def __init__(self, method=None, model_name=None, **kwargs):
        self.method = method
        self.cheap_model_name = model_name or kwargs.get("cheap_modelcard", "google/flan-t5-large")
        self.exp_model_name = kwargs.get("exp_modelcard", "google/flan-t5-xl")

        self.budget_tokens = kwargs.get("budget_tokens", 4000)
        self.budget_split_x = kwargs.get("budget_split_x", 0.5)
        self.budget_split_y = kwargs.get("budget_split_y", 0.5)
        self.total_passages = kwargs.get("total_passages", 50)

        self.binary_prompt_head_len = kwargs.get("binary_prompt_head_len", 15)
        self.binary_output_possible_len = kwargs.get("binary_output_possible_len", 1)
        self.prp_prompt_head_len = kwargs.get("prp_prompt_head_len", 25)
        self.prp_output_possible_len = kwargs.get("prp_output_possible_len", 2)

        self.device = 0 if torch.cuda.is_available() else -1
        self.cheap_model = pipeline("text2text-generation", model=self.cheap_model_name, device=self.device)
        self.exp_model = pipeline("text2text-generation", model=self.exp_model_name, device=self.device)

    def _get_binary_response(self, passage, query):
        prompt = f"Is the following passage related to the query?\npassage: {passage}\nquery: {query}\nAnswer in yes or no."
        try:
            ans = self.exp_model(prompt)[0]["generated_text"].strip().lower()
        except Exception:
            ans = ""
        return ans

    def _get_pairwise_response(self, query, passage_a, passage_b):
        prompt = f"""Given a query "{query}", which of the following two passages is more relevant to the query?
Passage A: {passage_a}
Passage B: {passage_b}
Output Passage A or Passage B."""
        try:
            ans = self.cheap_model(prompt)[0]["generated_text"].strip().lower()
        except Exception:
            ans = ""
        return ans

    def _count_pairwise_top_l(self, query_len, contexts):
        tokens = 0
        howmany = 0
        budget = int(self.budget_split_y * self.budget_tokens)

        for i in range(len(contexts) - 1):
            t1 = len(contexts[i].text.split())
            t2 = len(contexts[i + 1].text.split())
            possible = self.prp_prompt_head_len + self.prp_output_possible_len + query_len + t1 + t2

            if tokens + possible < budget:
                tokens += possible
                howmany += 1
            else:
                break

        return howmany


    def rank(self, documents: List[Document]) -> List[Document]:
        for document in tqdm(documents, desc="Reranking Documents"):
            #print(document.question.question)
            query = document.question.question
            query_len = len(query.split())
            contexts = document.contexts[:self.total_passages]

            # Stage 1: Binary Classification
            binary_token_limit = int(self.budget_split_x * self.budget_tokens)
            binary_running_token = 0
            yes_ctxs, no_ctxs = [], []

            for ctx in contexts:
                #print("context")
                text_len = len(ctx.text.split())
                estimated = self.binary_prompt_head_len + self.binary_output_possible_len + text_len + query_len

                if binary_running_token + estimated < binary_token_limit:
                    binary_running_token += estimated
                    response = self._get_binary_response(ctx.text, query)
                    if "yes" in response:
                        yes_ctxs.append(ctx)
                    elif "no" in response:
                        no_ctxs.append(ctx)
                    else:
                        yes_ctxs.append(ctx)  # ambiguous results are considered "yes"
                    binary_running_token += len(response.split())
                else:
                    break
            #print("after context")         
            reranked_list = yes_ctxs + no_ctxs

            # Stage 2: Pairwise Bubble Sort Based on Token Budget
            top_l = self._count_pairwise_top_l(query_len, reranked_list)
            full_passes = math.ceil(top_l / self.total_passages)
            #print(full_passes,len(reranked_list))

            for _ in range(full_passes):
                for i in range(len(reranked_list) - 1):
                    a, b = reranked_list[i], reranked_list[i + 1]
                    response = self._get_pairwise_response(query, a.text, b.text)
                    if "passage b" in response:
                        reranked_list[i], reranked_list[i + 1] = b, a
            #print("aaaaaaaaaaa")
            # Final Output
            for idx, ctx in enumerate(reranked_list):
                ctx.score = 1.0 - idx / len(reranked_list)  # higher score = more relevant
            #print("aaaaaaaaaaa")
            document.reorder_contexts = reranked_list

        return documents

