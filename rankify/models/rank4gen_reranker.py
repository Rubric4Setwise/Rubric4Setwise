"""
Rank4Gen Reranker - Setwise Document Selection and Ranking for RAG

Rank4Gen is a "RAG-Preference-Aligned Document Set Selection and Ranking" method
that uses a fine-tuned LLM (Qwen3-8B) to select and rank documents based on
downstream generator preferences.

Key features:
- Setwise paradigm: selects a subset of documents AND ranks them
- Generator-aware: adapts selection based on downstream model characteristics
- Supports both index mode and snapshot mode
- Token-budget-aware truncation for long document handling
- Local vLLM inference (no external API server needed)

Usage:
    from rankify.models.reranking import Reranking

    # Using local vLLM (recommended)
    model = Reranking(
        method='rank4gen',
        model_name='/path/to/Rank4Gen-DPO-Qwen3-8B',
        downstream_model='default',  # or 'Qwen3-8B', 'Llama-3.1-8B-Instruct', etc.
        lang='en',                   # 'en' or 'zh'
        mode='index',                # 'index' or 'snapshot'
        num_gpus=4,
    )
    results = model.rank(documents)
"""

import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document, Context


# =========================================================
# Constants
# =========================================================

TRUNCATION_MARK = " <TRUNCATED>"

# Ranker budgeting
RANKER_MAX_CONTEXT_TOKENS = 40960
RANKER_MAX_OUTPUT_TOKENS = 4096
RANKER_PROMPT_TOKEN_BUDGET = RANKER_MAX_CONTEXT_TOKENS - RANKER_MAX_OUTPUT_TOKENS

IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"

# =========================================================
# Prompt Templates
# =========================================================

EN_SYSTEM_PROMPT_TEMPLATE = """You are **Rank4Gen**, a **Ranker** designed for retrieval-augmented generation tasks.
Given a **Query (<Query>)** and **Candidate Documents (<Documents>)**, you need to **select and rank** the documents from a set of candidate documents that are most suitable for the downstream generator to answer the query, based on the characteristics and preferences of **Downstream Generator Information**.

When the downstream generator is `default`, it indicates a default mode with no specific preferences. In this case, you should **select and rank** the candidate documents that are **most helpful for the query** and **most directly support answering it**.

Please **strictly follow** the **Instructions (<Instruct>)** below for document selection and ranking.

---

## Downstream Generator Information

The downstream generator you serve is: `{downstream_model}`  
Generator description: `{description}`

---

## Output Mode

### 1. Index Mode
If the instruction contains **`/index`**, output only the **document index**, one per line, without additional text or explanation.

**Example:**
[<doc_index_1>]
[<doc_index_2>]
[<doc_index_3>]

### 2. Snapshot Mode
If the instruction contains **`/snapshot`**, output the selected documents **line by line** using *snapshot format*.  
Each line must include:

- **Document index**  
- **Preview of the first 100 characters** of the document content  

**Example:**
[<doc_index_1>] <first_100_characters_of_document>...
[<doc_index_2>] <first_100_characters_of_document>...
[<doc_index_3>] <first_100_characters_of_document>..."""

EN_USER_PROMPT_TEMPLATE = """<Instruct>: I will provide you with {num} documents, each indicated by a numerical identifier []. Select the documents based on their relevance to the search query "{question}".

<Query>: {question}

<Documents>: 
{context}

Select the documents that mostly cover clear and diverse information to answer the query.

Please output the final document selection and sorting results according to the format constraints of the **"Output Mode"**.

<Output>:"""

ZH_SYSTEM_PROMPT_TEMPLATE = """你是**Rank4Gen**，一个检索增强生成任务的**Ranker**。  
给定**查询 (<Query>)**与**候选文档 (<Documents>)**，你需要根据**下游生成器信息**的特点和偏好，从候选文档中**筛选并排序**出最适合该生成器回答的文档。

当下游生成器为`default`时，代表无偏好的默认模式，你需要从候选文档中**选择并排序**出**对该查询最有帮助**、**最能直接支持回答**的文档。

请**严格按照**下方的**指令 (<Instruct>)**进行文档选择与排序。

---

## 下游生成器信息

你所服务的下游生成器是：`{downstream_model}`
生成器描述：`{description}`

---

## 输出模式

### 1. Index 模式
如果指令中包含 ** `/index`**，则仅输出 **文档索引**，每行一个，不添加任何解释或额外文本。

**示例:**
[<doc_index_1>]
[<doc_index_2>]
[<doc_index_3>]

### 2. Snapshot 模式
如果指令中包含 **`/snapshot`**，请使用 *snapshot 格式* **逐行输出**所选文档。  
每行必须包括：

- **文档索引**  
- **文档内容前 100 个字符的预览**

**示例：**
[<doc_index_1>] <first_100_characters_of_document>...
[<doc_index_2>] <first_100_characters_of_document>...
[<doc_index_3>] <first_100_characters_of_document>..."""

ZH_USER_PROMPT_TEMPLATE = """<Instruct>: 我将向你提供 {num} 个文档，每个文档都有一个数字标识符 []。请根据它们与搜索查询"{question}"的相关性选择段落。

<Query>: {question}

<Documents>:
{context}

请选择那些能够提供清晰且多样信息、最能回答查询的文档。

请根据 "输出模式" 的格式要求输出最终的文档选择和排序结果。

<Output>: """

SYSTEM_TEMPLATES = {"en": EN_SYSTEM_PROMPT_TEMPLATE, "zh": ZH_SYSTEM_PROMPT_TEMPLATE}
USER_TEMPLATES = {"en": EN_USER_PROMPT_TEMPLATE, "zh": ZH_USER_PROMPT_TEMPLATE}

# Default model descriptions (fallback when description file is not found)
DEFAULT_DESCRIPTIONS = {
    "default": [
        "The Default model is a versatile Large Language Model(LLM) trained on massive datasets. "
        "It possesses cross-task comprehension and generation capabilities, enabling it to reason, "
        "learn, and make decisions across diverse complex scenarios."
    ]
}


# =========================================================
# Token helpers
# =========================================================

def _encode_no_special(tok, s: str) -> List[int]:
    return tok(s, add_special_tokens=False)["input_ids"]


def _init_token_constants(tok) -> Dict[str, Any]:
    return {
        "im_end_ids": _encode_no_special(tok, IM_END),
        "im_start_role_ids": {
            "system": _encode_no_special(tok, f"{IM_START}system\n"),
            "user": _encode_no_special(tok, f"{IM_START}user\n"),
            "assistant": _encode_no_special(tok, f"{IM_START}assistant\n"),
        },
        "trunc_ids": _encode_no_special(tok, TRUNCATION_MARK),
        "newline_ids": _encode_no_special(tok, "\n"),
    }


def _count_chat_tokens(tok, consts: Dict[str, Any], messages: List[Dict[str, str]]) -> int:
    if not messages:
        return 0
    batch = [m.get("content", "") for m in messages]
    roles = [m.get("role", "") for m in messages]
    enc = tok(batch, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)

    total = 0
    for role, ids in zip(roles, enc["input_ids"]):
        total += len(consts["im_start_role_ids"].get(role, []))
        total += len(ids)
        total += len(consts["im_end_ids"])
    return total


def _truncate_ids_to_text(tok, ids: List[int], max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if len(ids) <= max_tokens:
        return tok.decode(ids, skip_special_tokens=True)
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


# =========================================================
# Prompt construction with truncation
# =========================================================

def _preprocess_documents(tok, documents: List[Dict[str, Any]]) -> Tuple[List[List[int]], List[List[int]], List[str], List[str]]:
    prefixes: List[str] = []
    all_texts: List[str] = []
    doc_id_list: List[str] = []
    original_texts: List[str] = []

    for doc in documents:
        doc_id = str(doc.get("id"))
        doc_id_list.append(doc_id)
        prefixes.append(f"[{doc_id}] ")

        txt = str(doc.get("text", "")).strip()
        # Avoid collisions with [n] patterns inside doc text
        txt = re.sub(r"\[(\d+)\]", r"(\1)", txt).strip()
        original_texts.append(txt)
        all_texts.append(txt)

    prefix_encoded = tok(prefixes, add_special_tokens=False)["input_ids"]
    text_encoded = tok(all_texts, add_special_tokens=False)["input_ids"]
    return prefix_encoded, text_encoded, doc_id_list, original_texts


def _build_user_prompt_with_truncation(
    tok,
    consts: Dict[str, Any],
    user_template: str,
    system_prompt: str,
    query: str,
    documents: List[Dict[str, Any]],
    mode: str,
) -> str:
    mode_tag = "/index" if mode == "index" else "/snapshot"
    suffix = "\n" + mode_tag

    placeholder_context = "<CTX>"
    skeleton = user_template.format(num=len(documents), question=query, context=placeholder_context)
    fixed_user_body = skeleton.replace(placeholder_context, "")

    fixed_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fixed_user_body + suffix},
    ]
    fixed_tokens = _count_chat_tokens(tok, consts, fixed_messages)
    remaining_for_context = RANKER_PROMPT_TOKEN_BUDGET - fixed_tokens

    if remaining_for_context <= 0 or not documents:
        return user_template.format(num=len(documents), question=query, context="") + suffix

    prefix_ids_list, doc_text_ids_list, doc_id_list, original_texts = _preprocess_documents(tok, documents)
    nl_len = len(consts["newline_ids"])

    context_tokens = sum(len(p) + len(t) + nl_len for p, t in zip(prefix_ids_list, doc_text_ids_list))
    if context_tokens <= remaining_for_context:
        lines = [f"[{doc_id}] {txt}" for doc_id, txt in zip(doc_id_list, original_texts)]
        return user_template.format(num=len(documents), question=query, context="\n".join(lines)) + suffix

    # Need truncation
    n = len(documents)
    trunc_mark_tokens = len(consts["trunc_ids"])
    overhead_total = sum(len(p) + trunc_mark_tokens + nl_len for p in prefix_ids_list)

    remaining_for_text_only = remaining_for_context - overhead_total
    if remaining_for_text_only <= 0:
        mark = TRUNCATION_MARK.strip()
        lines = [f"[{doc_id}] {mark}" for doc_id in doc_id_list]
        return user_template.format(num=len(documents), question=query, context="\n".join(lines)) + suffix

    avg_tokens_per_doc_text = max(1, remaining_for_text_only // n)

    lines: List[str] = []
    for doc_id, txt_ids in zip(doc_id_list, doc_text_ids_list):
        if len(txt_ids) <= avg_tokens_per_doc_text:
            txt = tok.decode(txt_ids, skip_special_tokens=True)
            lines.append(f"[{doc_id}] {txt}")
        else:
            keep_tokens = max(1, avg_tokens_per_doc_text - trunc_mark_tokens)
            shortened = _truncate_ids_to_text(tok, txt_ids, keep_tokens).rstrip()
            lines.append(f"[{doc_id}] {shortened}{TRUNCATION_MARK}")

    return user_template.format(num=len(documents), question=query, context="\n".join(lines)) + suffix


# =========================================================
# Output parsing
# =========================================================

def _extract_visible_output_after_think(raw_output: str) -> str:
    if not raw_output:
        return ""
    marker = "</think>"
    idx = raw_output.rfind(marker)
    if idx == -1:
        return raw_output
    return raw_output[idx + len(marker):]


def _parse_ranker_output(raw_output: str) -> List[str]:
    visible = _extract_visible_output_after_think(raw_output)
    if not visible:
        return []
    indices = re.findall(r"\[(\d+)\]", visible)

    seen = set()
    ordered: List[str] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


# =========================================================
# Main Reranker Class
# =========================================================

class Rank4GenReranker(BaseRanking):
    """
    Rank4Gen: RAG-Preference-Aligned Document Set Selection and Ranking.

    This reranker uses a fine-tuned LLM to perform setwise document selection
    and ranking. It selects a subset of documents most suitable for the
    downstream generator and ranks them by relevance.

    Uses local vLLM engine for inference (no external API server needed).

    Args:
        method (str): Method name ('rank4gen').
        model_name (str): Local model path or HuggingFace model ID.
        tokenizer_path (str): Path to the tokenizer (defaults to model_name).
        downstream_model (str): Downstream generator name for preference alignment.
            Options: "default", "Qwen3-8B", "Qwen2.5-7B-Instruct",
                     "Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Qwen-7B", etc.
        lang (str): Prompt language, "en" or "zh".
        mode (str): Output mode, "index" or "snapshot".
        top_k (int): Maximum number of selected documents (0 = keep all selected by model).
        desc_file (str): Path to model descriptions JSON file.
        seed (int): Random seed for description selection.
        num_gpus (int): Number of GPUs for tensor parallelism (default: 1).
        gpu_memory_utilization (float): GPU memory utilization for vLLM (default: 0.9).
        vllm_batched (bool): Whether to batch all queries for inference (default: True).

    Example:
        >>> from rankify.models.reranking import Reranking
        >>> reranker = Reranking(
        ...     method='rank4gen',
        ...     model_name='/path/to/Rank4Gen-DPO-Qwen3-8B',
        ...     downstream_model='default',
        ... )
        >>> results = reranker.rank(documents)
    """

    def __init__(
        self,
        method: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.method = method or "rank4gen"
        model_name = model_name or "JohnnyFan/Rank4Gen-DPO-Qwen3-8B"

        # Resolve model path: download from ModelScope if not a local directory
        from rankify.utils.model_downloader import resolve_model_path
        cache_dir = kwargs.get("cache_dir", None)
        self.model_name = resolve_model_path(model_name, cache_dir=cache_dir)

        # Configuration from kwargs
        self.tokenizer_path = kwargs.get("tokenizer_path", self.model_name)
        self.downstream_model = kwargs.get("downstream_model", "default")
        self.lang = kwargs.get("lang", "en")
        self.mode = kwargs.get("mode", "index")
        self.top_k = kwargs.get("top_k", 0)
        self.seed = kwargs.get("seed", 42)
        self.num_gpus = kwargs.get("num_gpus", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        self.vllm_batched = kwargs.get("vllm_batched", True)

        assert self.lang in ("en", "zh"), f"lang must be 'en' or 'zh', got '{self.lang}'"
        assert self.mode in ("index", "snapshot"), f"mode must be 'index' or 'snapshot', got '{self.mode}'"

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True, use_fast=False
        )
        self.token_consts = _init_token_constants(self.tokenizer)

        # Load model descriptions
        desc_file = kwargs.get("desc_file", None)
        self.descriptions = self._load_descriptions(desc_file)

        # Build system prompt
        random.seed(self.seed)
        if self.downstream_model not in self.descriptions:
            print(f"[Rank4Gen] Warning: downstream_model '{self.downstream_model}' not found in descriptions, using 'default'")
            self.downstream_model = "default"

        description = random.choice(self.descriptions[self.downstream_model])
        self.system_prompt = SYSTEM_TEMPLATES[self.lang].format(
            downstream_model=self.downstream_model,
            description=description,
        )
        self.user_template = USER_TEMPLATES[self.lang]

        # Initialize local vLLM engine
        from vllm import LLM, SamplingParams

        print(f"[Rank4Gen] Loading model locally with vLLM: {self.model_name}")
        print(f"[Rank4Gen] num_gpus={self.num_gpus}, gpu_memory_utilization={self.gpu_memory_utilization}")

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            max_model_len=RANKER_MAX_CONTEXT_TOKENS,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=RANKER_MAX_OUTPUT_TOKENS,
        )

        print(f"[Rank4Gen] Initialized: model={self.model_name}, "
              f"downstream={self.downstream_model}, lang={self.lang}, mode={self.mode}")

    def _load_descriptions(self, desc_file: Optional[str]) -> Dict[str, List[str]]:
        """Load model description file, with fallback to defaults."""
        if desc_file and os.path.exists(desc_file):
            with open(desc_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try to find description file relative to this module
        module_dir = Path(__file__).parent
        candidates = [
            module_dir / "rank4gen_descriptions" / "model_descriptions.json",
            module_dir / "rank4gen_descriptions" / "model_descriptions_zh.json",
        ]

        if self.lang == "zh":
            candidates = list(reversed(candidates))

        for candidate in candidates:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)

        # Fallback to default descriptions
        print("[Rank4Gen] Warning: No description file found, using default descriptions.")
        return DEFAULT_DESCRIPTIONS

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a prompt string using the tokenizer's chat template."""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: disable thinking mode
            )
        except TypeError:
            # Fallback if tokenizer doesn't support enable_thinking kwarg
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _generate_single(self, messages: List[Dict[str, str]]) -> str:
        """Generate output for a single set of messages using local vLLM."""
        prompt = self._messages_to_prompt(messages)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def _generate_batch(self, messages_list: List[List[Dict[str, str]]]) -> List[str]:
        """Generate outputs for a batch of message sets using local vLLM."""
        prompts = [self._messages_to_prompt(msgs) for msgs in messages_list]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text for out in outputs]

    def rank(self, documents: List[Document]) -> List[Document]:
        """
        Perform Rank4Gen setwise document selection and ranking.

        For each Document in the list, selects a subset of contexts and
        re-orders them based on the model's output.

        Args:
            documents: List of Document objects to rerank.

        Returns:
            List of Document objects with reorder_contexts populated.
        """
        if self.vllm_batched:
            return self._rank_batched(documents)
        else:
            return self._rank_sequential(documents)

    def _rank_sequential(self, documents: List[Document]) -> List[Document]:
        """Sequential ranking: process one document at a time."""
        for document in tqdm(documents, desc="Rank4Gen reranking"):
            if not document.contexts:
                document.reorder_contexts = []
                continue

            query = document.question.question
            contexts = document.contexts

            # Prepare documents in Rank4Gen format
            rank4gen_docs = []
            for idx, ctx in enumerate(contexts):
                rank4gen_docs.append({
                    "id": idx + 1,  # 1-based indexing
                    "text": ctx.text or "",
                })

            # Build prompt with token-budget-aware truncation
            user_prompt = _build_user_prompt_with_truncation(
                tok=self.tokenizer,
                consts=self.token_consts,
                user_template=self.user_template,
                system_prompt=self.system_prompt,
                query=query,
                documents=rank4gen_docs,
                mode=self.mode,
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw_output = self._generate_single(messages)
                # Save raw LLM output to document
                document.ranker_raw_outputs = [raw_output]
                ranked_ids = _parse_ranker_output(raw_output)

                # Apply top_k if specified
                if self.top_k > 0:
                    ranked_ids = ranked_ids[:self.top_k]

                # Map ranked IDs back to contexts
                reordered = []
                n_contexts = len(contexts)
                for rank_pos, doc_id_str in enumerate(ranked_ids):
                    try:
                        doc_idx = int(doc_id_str) - 1  # Convert back to 0-based
                        if 0 <= doc_idx < n_contexts:
                            ctx_copy = copy.deepcopy(contexts[doc_idx])
                            ctx_copy.score = float(len(ranked_ids) - rank_pos)
                            reordered.append(ctx_copy)
                    except (ValueError, IndexError):
                        continue

                document.reorder_contexts = reordered

            except Exception as e:
                print(f"[Rank4Gen] Error processing query '{query[:50]}...': {e}")
                document.reorder_contexts = copy.deepcopy(contexts)

        return documents

    def _rank_batched(self, documents: List[Document]) -> List[Document]:
        """Batched ranking: collect all prompts, run vLLM batch inference."""
        print(f"[Rank4Gen] Preparing batch of {len(documents)} documents...")

        # Phase 1: Build all prompts
        all_messages = []
        valid_indices = []  # Track which documents have contexts

        for doc_idx, document in enumerate(documents):
            if not document.contexts:
                document.reorder_contexts = []
                continue

            query = document.question.question
            contexts = document.contexts

            rank4gen_docs = []
            for idx, ctx in enumerate(contexts):
                rank4gen_docs.append({
                    "id": idx + 1,
                    "text": ctx.text or "",
                })

            user_prompt = _build_user_prompt_with_truncation(
                tok=self.tokenizer,
                consts=self.token_consts,
                user_template=self.user_template,
                system_prompt=self.system_prompt,
                query=query,
                documents=rank4gen_docs,
                mode=self.mode,
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            all_messages.append(messages)
            valid_indices.append(doc_idx)

        if not all_messages:
            return documents

        # Phase 2: Batch inference
        print(f"[Rank4Gen] Running vLLM batch inference on {len(all_messages)} queries...")
        raw_outputs = self._generate_batch(all_messages)

        # Phase 3: Parse outputs and build reorder_contexts
        for batch_idx, doc_idx in enumerate(valid_indices):
            document = documents[doc_idx]
            contexts = document.contexts

            try:
                raw_output = raw_outputs[batch_idx]
                # Save raw LLM output to document
                document.ranker_raw_outputs = [raw_output]
                ranked_ids = _parse_ranker_output(raw_output)

                if self.top_k > 0:
                    ranked_ids = ranked_ids[:self.top_k]

                reordered = []
                n_contexts = len(contexts)
                for rank_pos, doc_id_str in enumerate(ranked_ids):
                    try:
                        ctx_idx = int(doc_id_str) - 1
                        if 0 <= ctx_idx < n_contexts:
                            ctx_copy = copy.deepcopy(contexts[ctx_idx])
                            ctx_copy.score = float(len(ranked_ids) - rank_pos)
                            reordered.append(ctx_copy)
                    except (ValueError, IndexError):
                        continue

                document.reorder_contexts = reordered

            except Exception as e:
                query = document.question.question
                print(f"[Rank4Gen] Error processing query '{query[:50]}...': {e}")
                document.reorder_contexts = copy.deepcopy(contexts)

        print(f"[Rank4Gen] Batch inference complete!")
        return documents
