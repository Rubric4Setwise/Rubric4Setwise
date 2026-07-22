# rankify/metrics/ragas_bridge.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import warnings

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall
)
from ragas.run_config import RunConfig

# Import LLM/Embeddings wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def _to_str_question(doc) -> str:
    """Extract question string from document"""
    q = getattr(doc, "question", None)
    if hasattr(q, "question"):
        return q.question
    return str(q or "")


def _ctx_texts(doc, use_reordered: bool) -> List[str]:
    """Extract context texts from document"""
    arr = (doc.reorder_contexts if use_reordered and getattr(doc, "reorder_contexts", None) 
           else getattr(doc, "contexts", []) or [])
    out = []
    for c in arr:
        t = getattr(c, "text", None)
        out.append(t if isinstance(t, str) else str(t or ""))
    return out


def _gold_ref(doc) -> Optional[str]:
    """Extract gold reference answer from document"""
    ans = getattr(doc, "answers", None)
    if hasattr(ans, "answers"):
        vals = [str(a) for a in ans.answers]
        return " ".join(vals) if vals else None
    if isinstance(ans, (list, tuple)):
        vals = [str(a) for a in ans]
        return " ".join(vals) if vals else None
    if isinstance(ans, str):
        return ans
    return None


def build_ragas_dataset(documents, predictions: List[str], use_reordered: bool = True) -> EvaluationDataset:
    """Build RAGAS evaluation dataset from Rankify documents"""
    samples = []
    for doc, pred in zip(documents, predictions):
        samples.append(
            SingleTurnSample(
                user_input=_to_str_question(doc),
                response=str(pred or ""),
                retrieved_contexts=_ctx_texts(doc, use_reordered),
                reference=_gold_ref(doc),
            )
        )
    return EvaluationDataset(samples=samples)


@dataclass
class RagasModels:
    """
    Configuration for RAGAS LLM and embeddings.
    
    Examples:
        # OpenAI (fast)
        ragas = RagasModels(
            llm_kind="openai",
            llm_name="gpt-4o-mini",
            llm_api_key="your-key"
        )
        
        # HuggingFace (local models, no API key needed)
        ragas = RagasModels(
            llm_kind="hf",
            llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            embeddings_kind="hf",
            embeddings_name="sentence-transformers/all-MiniLM-L6-v2",
            max_retries=1,  # Reduce retries for HF
            timeout=300,  # 5 minutes for HF models
        )
    """
    llm_kind: str = "openai"  # "openai" or "hf"
    llm_name: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    embeddings_kind: str = "openai"  # "openai" or "hf"
    embeddings_name: str = "text-embedding-3-small"
    embeddings_api_key: Optional[str] = None
    
    # HuggingFace specific options
    torch_dtype: Optional[str] = "float16"  # "float16", "bfloat16", or "float32"
    device: Optional[str] = None  # "cuda", "cpu", or None (auto)
    max_new_tokens: int = 512  # Max tokens for HF generation
    
    # RAGAS execution config
    timeout: int = 60  # Timeout per metric call (increase for HF models)
    max_retries: int = 2  # Number of retries on failure
    max_workers: int = 4  # Parallel workers

    def build(self):
        """Build and return wrapped LLM and embeddings for RAGAS"""
        if self.llm_kind == "openai":
            llm, emb = self._build_openai()
        elif self.llm_kind == "hf":
            llm, emb = self._build_huggingface()
        else:
            raise ValueError(f"Unsupported llm_kind: {self.llm_kind}. Use 'openai' or 'hf'")
        
        # Wrap for RAGAS
        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_emb = LangchainEmbeddingsWrapper(emb)
        
        return wrapped_llm, wrapped_emb

    def _build_openai(self):
        """Build OpenAI LLM and embeddings"""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")
        
        llm = ChatOpenAI(
            model=self.llm_name,
            api_key=self.llm_api_key,
            temperature=0,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        emb = OpenAIEmbeddings(
            model=self.embeddings_name,
            api_key=self.embeddings_api_key or self.llm_api_key
        )
        
        return llm, emb

    def _build_huggingface(self):
        """Build HuggingFace LLM and embeddings"""
        try:
            from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except ImportError:
            raise ImportError(
                "Install required packages: pip install langchain-huggingface transformers torch"
            )
        
        # Determine device
        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, torch.float16)
        
        print(f"Loading HuggingFace LLM: {self.llm_name} on {device} with {self.torch_dtype}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        if device == "cpu":
            model = model.to(device)
        
        # Optimized pipeline settings for RAGAS (shorter outputs, faster generation)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,  # Reduced for faster generation
            temperature=0.1,
            do_sample=False,  # Greedy decoding for speed
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print(f"Loading HuggingFace Embeddings: {self.embeddings_name}")
        
        # Initialize embeddings
        emb = HuggingFaceEmbeddings(
            model_name=self.embeddings_name,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16,  # Smaller batch for stability
                'show_progress_bar': False
            }
        )
        
        return llm, emb

    def get_run_config(self) -> RunConfig:
        """Get RAGAS RunConfig with appropriate timeouts"""
        return RunConfig(
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_workers=self.max_workers,
            max_wait=self.timeout * 2,  # Max wait time
        )


def run_ragas_eval(
    dataset: EvaluationDataset,
    models: RagasModels,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on dataset.
    
    Args:
        dataset: RAGAS EvaluationDataset
        models: RagasModels configuration
        metrics: List of metric names to compute. Options:
            - "faithfulness"
            - "response_relevancy" 
            - "context_precision"
            - "context_recall"
            If None, computes all metrics (excluding context_recall which needs reference)
        
    Returns:
        Dictionary of metric scores
    """
    print(f"Building RAGAS models ({models.llm_kind})...")
    llm, emb = models.build()
    
    # Get run config with appropriate timeouts
    run_config = models.get_run_config()
    
    # Initialize metrics with proper parameters
    available_metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "response_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
    }
    
    # Check if we have references for context_recall
    has_references = any(
        sample.reference is not None and sample.reference.strip()
        for sample in dataset.samples
    )
    
    if has_references:
        available_metrics["context_recall"] = LLMContextRecall(llm=llm)
    
    # Select metrics
    if metrics is None:
        # By default, use faster metrics
        if models.llm_kind == "hf":
            # For HF, use only faithfulness and context_precision (faster)
            metrics_list = [
                available_metrics["faithfulness"],
                available_metrics["context_precision"]
            ]
            print("Using fast metrics for HuggingFace: faithfulness, context_precision")
        else:
            metrics_list = list(available_metrics.values())
    else:
        metrics_list = []
        for m in metrics:
            if m in available_metrics:
                metrics_list.append(available_metrics[m])
            else:
                warnings.warn(f"Metric '{m}' not available. Available: {list(available_metrics.keys())}")
    
    if not metrics_list:
        raise ValueError("No valid metrics selected!")
    
    print(f"Running RAGAS evaluation with {len(metrics_list)} metrics...")
    print(f"Timeout: {run_config.timeout}s, Max retries: {run_config.max_retries}")
    
    # Run evaluation with custom config
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            run_config=run_config,
            raise_exceptions=False  # Don't crash on individual failures
        )
        
        # Convert to dict - get mean scores
        scores_df = result.to_pandas()
        scores = {}
        
        for metric in metrics_list:
            metric_name = metric.name
            if metric_name in scores_df.columns:
                col_values = scores_df[metric_name]
                # Filter out NaN values before computing mean
                valid_values = col_values.dropna()
                if len(valid_values) > 0:
                    scores[metric_name] = float(valid_values.mean())
                else:
                    warnings.warn(f"All values for {metric_name} are NaN")
                    scores[metric_name] = 0.0
        
        return scores
    
    except Exception as e:
        warnings.warn(f"RAGAS evaluation error: {e}")
        import traceback
        traceback.print_exc()
        # Return zero scores for requested metrics
        return {m.name: 0.0 for m in metrics_list}