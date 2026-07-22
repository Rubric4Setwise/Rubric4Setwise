"""
Rankify Pipeline - Simple One-Line Interface

Provides HuggingFace-style simplicity for creating retrieval, reranking, and RAG pipelines.

Example:
    ```python
    from rankify import pipeline

    # Simple RAG pipeline
    rag = pipeline("rag")
    answers = rag("What is machine learning?", documents)

    # Custom configuration
    search = pipeline("search", retriever="bge", reranker="flashrank")
    results = search.run("Find documents about AI", corpus)
    ```
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from rankify.dataset.dataset import Document, Question, Answer, Context


@dataclass
class PipelineResult:
    """Result from a pipeline execution."""
    query: str
    documents: List[Document]
    answers: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        if self.answers:
            return f"PipelineResult(query='{self.query[:50]}...', answers={len(self.answers)}, docs={len(self.documents)})"
        return f"PipelineResult(query='{self.query[:50]}...', docs={len(self.documents)})"


class Pipeline:
    """
    Unified pipeline interface for Rankify.
    
    Supports three modes:
    - "retrieval" or "search": Retrieve documents
    - "rerank": Retrieve and rerank documents  
    - "rag": Full RAG pipeline (retrieve + rerank + generate)
    
    Example:
        ```python
        # Search pipeline
        search = Pipeline("search", retriever="bm25")
        results = search("What is Python?", corpus)
        
        # RAG pipeline
        rag = Pipeline("rag", retriever="bge", reranker="monot5", generator="basic-rag")
        answers = rag("Explain transformers", documents)
        ```
    """
    
    # Default configurations for different tasks
    DEFAULTS = {
        "retrieval": {
            "retriever": "bm25",
            "n_docs": 100,
        },
        "search": {
            "retriever": "bm25",
            "n_docs": 100,
        },
        "rerank": {
            "retriever": "bm25",
            "reranker": "flashrank",
            "reranker_model": "ms-marco-MiniLM-L-12-v2",
            "n_docs": 100,
            "top_k": 10,
        },
        "rag": {
            "retriever": "bm25",
            "reranker": "flashrank",
            "reranker_model": "ms-marco-MiniLM-L-12-v2",
            "generator": "basic-rag",
            "generator_model": "gpt-4o-mini",
            "generator_backend": "openai",
            "n_docs": 100,
            "top_k": 5,
        },
    }
    
    def __init__(
        self,
        task: str = "rag",
        retriever: Optional[str] = None,
        reranker: Optional[str] = None,
        generator: Optional[str] = None,
        retriever_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        generator_model: Optional[str] = None,
        generator_backend: Optional[str] = None,
        n_docs: int = 100,
        top_k: int = 10,
        index_type: str = "wiki",
        index_folder: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a Rankify pipeline.
        
        Args:
            task: Pipeline type - "retrieval", "search", "rerank", or "rag"
            retriever: Retriever method (bm25, dpr, bge, colbert, etc.)
            reranker: Reranker method (flashrank, monot5, rankgpt, etc.)
            generator: RAG method (basic-rag, chain-of-thought-rag, etc.)
            retriever_model: Specific retriever model name
            reranker_model: Specific reranker model name
            generator_model: LLM model name for generation
            generator_backend: LLM backend (openai, azure, litellm, etc.)
            n_docs: Number of documents to retrieve
            top_k: Number of top documents to use for generation
            index_type: Index type (wiki, custom, etc.)
            index_folder: Path to custom index
            **kwargs: Additional configuration
        """
        self.task = task.lower()
        if self.task not in self.DEFAULTS:
            raise ValueError(f"Unknown task: {task}. Choose from: {list(self.DEFAULTS.keys())}")
        
        # Merge defaults with user config
        defaults = self.DEFAULTS[self.task].copy()
        
        self.config = {
            "retriever": retriever or defaults.get("retriever"),
            "reranker": reranker or defaults.get("reranker"),
            "generator": generator or defaults.get("generator"),
            "retriever_model": retriever_model,
            "reranker_model": reranker_model or defaults.get("reranker_model"),
            "generator_model": generator_model or defaults.get("generator_model"),
            "generator_backend": generator_backend or defaults.get("generator_backend"),
            "n_docs": n_docs,
            "top_k": top_k,
            "index_type": index_type,
            "index_folder": index_folder,
            **kwargs,
        }
        
        # Lazy initialization
        self._retriever = None
        self._reranker = None
        self._generator = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of components."""
        if self._initialized:
            return
        
        # Initialize retriever
        if self.config["retriever"]:
            from rankify.retrievers.retriever import Retriever
            retriever_kwargs = {
                "method": self.config["retriever"],
                "n_docs": self.config["n_docs"],
                "index_type": self.config["index_type"],
            }
            if self.config["index_folder"]:
                retriever_kwargs["index_folder"] = self.config["index_folder"]
            if self.config["retriever_model"]:
                retriever_kwargs["encoder_name"] = self.config["retriever_model"]
            
            self._retriever = Retriever(**retriever_kwargs)
        
        # Initialize reranker
        if self.config["reranker"] and self.task in ["rerank", "rag"]:
            from rankify.models.reranking import Reranking
            self._reranker = Reranking(
                method=self.config["reranker"],
                model_name=self.config["reranker_model"],
            )
        
        # Initialize generator
        if self.config["generator"] and self.task == "rag":
            from rankify.generator.generator import Generator
            self._generator = Generator(
                method=self.config["generator"],
                model_name=self.config["generator_model"],
                backend=self.config["generator_backend"],
                n_contexts=self.config["top_k"],
            )
        
        self._initialized = True
    
    def __call__(
        self,
        query: Union[str, List[str]],
        documents: Optional[List[Document]] = None,
        corpus: Optional[List[Dict[str, str]]] = None,
    ) -> Union[PipelineResult, List[PipelineResult]]:
        """
        Run the pipeline on a query.
        
        Args:
            query: Single query string or list of queries
            documents: Pre-loaded Document objects (optional)
            corpus: Raw corpus for retrieval (list of {"text": ..., "title": ...})
            
        Returns:
            PipelineResult or list of PipelineResults
        """
        # Handle single query
        if isinstance(query, str):
            return self._run_single(query, documents, corpus)
        
        # Handle batch queries
        return [self._run_single(q, documents, corpus) for q in query]
    
    def _run_single(
        self,
        query: str,
        documents: Optional[List[Document]] = None,
        corpus: Optional[List[Dict[str, str]]] = None,
    ) -> PipelineResult:
        """Run pipeline on a single query."""
        self._initialize()
        
        # Create document if not provided
        if documents is None:
            doc = Document(
                question=Question(query),
                answers=Answer([]),
                contexts=[],
            )
            documents = [doc]
        else:
            # Update query in first document
            if documents and documents[0].question.question != query:
                doc = Document(
                    question=Question(query),
                    answers=documents[0].answers if documents else Answer([]),
                    contexts=documents[0].contexts if documents else [],
                )
                documents = [doc] + documents[1:]
        
        # Step 1: Retrieve
        if self._retriever:
            documents = self._retriever.retrieve(documents)
        
        # Step 2: Rerank
        if self._reranker:
            documents = self._reranker.rank(documents)
        
        # Step 3: Generate (RAG only)
        answers = None
        if self._generator:
            answers = self._generator.generate(documents)
        
        return PipelineResult(
            query=query,
            documents=documents,
            answers=answers,
            metadata={
                "task": self.task,
                "config": self.config,
            }
        )
    
    def run(
        self,
        query: Union[str, List[str]],
        documents: Optional[List[Document]] = None,
        corpus: Optional[List[Dict[str, str]]] = None,
    ) -> Union[PipelineResult, List[PipelineResult]]:
        """Alias for __call__."""
        return self(query, documents, corpus)
    
    def retrieve(self, query: str, n_docs: Optional[int] = None) -> List[Context]:
        """Retrieve documents for a query."""
        self._initialize()
        if not self._retriever:
            raise ValueError("No retriever configured")
        
        doc = Document(
            question=Question(query),
            answers=Answer([]),
            contexts=[],
        )
        result = self._retriever.retrieve([doc])
        return result[0].contexts[:n_docs] if n_docs else result[0].contexts
    
    def rerank(
        self,
        query: str,
        contexts: List[Union[str, Context, Dict[str, str]]],
        top_k: Optional[int] = None,
    ) -> List[Context]:
        """Rerank a list of contexts for a query."""
        self._initialize()
        if not self._reranker:
            raise ValueError("No reranker configured. Use task='rerank' or task='rag'")
        
        # Convert to Context objects
        ctx_objects = []
        for i, ctx in enumerate(contexts):
            if isinstance(ctx, str):
                ctx_objects.append(Context(text=ctx, id=str(i)))
            elif isinstance(ctx, dict):
                ctx_objects.append(Context(
                    text=ctx.get("text", ""),
                    id=ctx.get("id", str(i)),
                    title=ctx.get("title", ""),
                ))
            else:
                ctx_objects.append(ctx)
        
        doc = Document(
            question=Question(query),
            answers=Answer([]),
            contexts=ctx_objects,
        )
        result = self._reranker.rank([doc])
        reranked = result[0].reorder_contexts or result[0].contexts
        return reranked[:top_k] if top_k else reranked
    
    def generate(
        self,
        query: str,
        contexts: List[Union[str, Context]],
    ) -> str:
        """Generate an answer using RAG."""
        self._initialize()
        if not self._generator:
            raise ValueError("No generator configured. Use task='rag'")
        
        # Convert to Context objects
        ctx_objects = []
        for i, ctx in enumerate(contexts):
            if isinstance(ctx, str):
                ctx_objects.append(Context(text=ctx, id=str(i)))
            else:
                ctx_objects.append(ctx)
        
        doc = Document(
            question=Question(query),
            answers=Answer([]),
            contexts=ctx_objects,
        )
        answers = self._generator.generate([doc])
        return answers[0] if answers else ""
    
    def __repr__(self):
        return f"Pipeline(task='{self.task}', retriever='{self.config['retriever']}', reranker='{self.config.get('reranker')}', generator='{self.config.get('generator')}')"


def pipeline(
    task: str = "rag",
    retriever: Optional[str] = None,
    reranker: Optional[str] = None,
    generator: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """
    Create a Rankify pipeline with one line.
    
    This is the main entry point for simple usage of Rankify.
    
    Args:
        task: Pipeline type
            - "retrieval" or "search": Just retrieve documents
            - "rerank": Retrieve and rerank
            - "rag": Full RAG (retrieve + rerank + generate)
        retriever: Retriever method (bm25, dpr, bge, colbert, contriever, ance, hyde)
        reranker: Reranker method (flashrank, monot5, rankgpt, inranker, etc.)
        generator: RAG method (basic-rag, chain-of-thought-rag, self-consistency-rag, etc.)
        **kwargs: Additional configuration
        
    Returns:
        Pipeline object
        
    Examples:
        ```python
        from rankify import pipeline
        
        # Simple search
        search = pipeline("search")
        results = search("What is Python?")
        
        # Custom RAG
        rag = pipeline("rag", retriever="bge", reranker="monot5")
        answers = rag("Explain machine learning", documents)
        
        # Fast production pipeline
        prod = pipeline("rerank", retriever="bm25", reranker="flashrank")
        results = prod.run("Find AI papers")
        ```
    """
    return Pipeline(
        task=task,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        **kwargs,
    )


# Convenience aliases
def search_pipeline(retriever: str = "bm25", **kwargs) -> Pipeline:
    """Create a search-only pipeline."""
    return pipeline("search", retriever=retriever, **kwargs)


def rerank_pipeline(
    retriever: str = "bm25",
    reranker: str = "flashrank",
    **kwargs
) -> Pipeline:
    """Create a retrieve + rerank pipeline."""
    return pipeline("rerank", retriever=retriever, reranker=reranker, **kwargs)


def rag_pipeline(
    retriever: str = "bm25",
    reranker: str = "flashrank",
    generator: str = "basic-rag",
    **kwargs
) -> Pipeline:
    """Create a full RAG pipeline."""
    return pipeline("rag", retriever=retriever, reranker=reranker, generator=generator, **kwargs)
