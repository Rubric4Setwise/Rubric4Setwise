"""
Hybrid Retriever - Combines Sparse and Dense Retrieval

Provides state-of-the-art retrieval by combining BM25 (sparse) with dense retrievers
using fusion strategies like Reciprocal Rank Fusion (RRF).

Example:
    ```python
    from rankify.retrievers import HybridRetriever

    # Combine BM25 + BGE with RRF
    retriever = HybridRetriever(
        sparse="bm25",
        dense="bge",
        fusion="rrf",
        weights=[0.4, 0.6],
    )
    
    documents = retriever.retrieve(documents)
    ```
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

from rankify.dataset.dataset import Document, Context


@dataclass
class RetrievalResult:
    """Result from a single retriever."""
    retriever_name: str
    contexts: List[Context]
    scores: List[float]


class HybridRetriever:
    """
    Hybrid Retriever combining sparse and dense retrieval methods.
    
    Supports multiple fusion strategies:
    - "rrf": Reciprocal Rank Fusion (default, best for most cases)
    - "weighted": Weighted score combination
    - "interleave": Alternate results from each retriever
    
    Example:
        ```python
        # Basic usage
        retriever = HybridRetriever(sparse="bm25", dense="bge")
        results = retriever.retrieve(documents)
        
        # Custom weights
        retriever = HybridRetriever(
            sparse="bm25",
            dense="colbert",
            fusion="weighted",
            weights=[0.3, 0.7],  # 30% sparse, 70% dense
        )
        
        # Multiple dense retrievers
        retriever = HybridRetriever(
            sparse="bm25",
            dense=["dpr", "bge"],
            fusion="rrf",
        )
        ```
    """
    
    def __init__(
        self,
        sparse: str = "bm25",
        dense: Union[str, List[str]] = "bge",
        fusion: str = "rrf",
        weights: Optional[List[float]] = None,
        n_docs: int = 100,
        rrf_k: int = 60,
        index_type: str = "wiki",
        sparse_index_folder: Optional[str] = None,
        dense_index_folder: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse: Sparse retrieval method (bm25)
            dense: Dense retrieval method(s) (dpr, bge, ance, colbert, contriever)
            fusion: Fusion strategy - "rrf", "weighted", or "interleave"
            weights: Optional weights for each retriever [sparse_weight, dense_weight(s)...]
            n_docs: Number of documents to retrieve from each method
            rrf_k: RRF parameter k (default 60, higher = more emphasis on top ranks)
            index_type: Index type (wiki, custom, etc.)
            sparse_index_folder: Path to sparse index
            dense_index_folder: Path to dense index
            **kwargs: Additional configuration
        """
        self.sparse_method = sparse
        self.dense_methods = [dense] if isinstance(dense, str) else dense
        self.fusion = fusion.lower()
        self.n_docs = n_docs
        self.rrf_k = rrf_k
        self.index_type = index_type
        self.sparse_index_folder = sparse_index_folder
        self.dense_index_folder = dense_index_folder
        self.kwargs = kwargs
        
        # Calculate number of retrievers
        num_retrievers = 1 + len(self.dense_methods)  # 1 sparse + N dense
        
        # Set up weights
        if weights:
            if len(weights) != num_retrievers:
                raise ValueError(f"Expected {num_retrievers} weights, got {len(weights)}")
            self.weights = weights
        else:
            # Equal weights by default
            self.weights = [1.0 / num_retrievers] * num_retrievers
        
        # Lazy initialization
        self._sparse_retriever = None
        self._dense_retrievers = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of retrievers."""
        if self._initialized:
            return
        
        from rankify.retrievers.retriever import Retriever
        
        # Initialize sparse retriever
        sparse_kwargs = {
            "method": self.sparse_method,
            "n_docs": self.n_docs,
            "index_type": self.index_type,
        }
        if self.sparse_index_folder:
            sparse_kwargs["index_folder"] = self.sparse_index_folder
        
        self._sparse_retriever = Retriever(**sparse_kwargs)
        
        # Initialize dense retrievers
        self._dense_retrievers = []
        for method in self.dense_methods:
            dense_kwargs = {
                "method": method,
                "n_docs": self.n_docs,
                "index_type": self.index_type,
            }
            if self.dense_index_folder:
                dense_kwargs["index_folder"] = self.dense_index_folder
            
            self._dense_retrievers.append(Retriever(**dense_kwargs))
        
        self._initialized = True
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve documents using hybrid strategy.
        
        Args:
            documents: List of Document objects with questions
            
        Returns:
            Documents with fused contexts
        """
        self._initialize()
        
        results = []
        for doc in documents:
            fused_contexts = self._retrieve_single(doc)
            new_doc = Document(
                question=doc.question,
                answers=doc.answers,
                contexts=fused_contexts,
            )
            results.append(new_doc)
        
        return results
    
    def _retrieve_single(self, document: Document) -> List[Context]:
        """Retrieve for a single document/query."""
        all_results: List[RetrievalResult] = []
        
        # Get sparse results
        sparse_docs = self._sparse_retriever.retrieve([document])
        sparse_contexts = sparse_docs[0].contexts if sparse_docs else []
        all_results.append(RetrievalResult(
            retriever_name=self.sparse_method,
            contexts=sparse_contexts,
            scores=self._extract_scores(sparse_contexts),
        ))
        
        # Get dense results
        for i, dense_retriever in enumerate(self._dense_retrievers):
            dense_docs = dense_retriever.retrieve([document])
            dense_contexts = dense_docs[0].contexts if dense_docs else []
            all_results.append(RetrievalResult(
                retriever_name=self.dense_methods[i],
                contexts=dense_contexts,
                scores=self._extract_scores(dense_contexts),
            ))
        
        # Apply fusion
        if self.fusion == "rrf":
            return self._rrf_fusion(all_results)
        elif self.fusion == "weighted":
            return self._weighted_fusion(all_results)
        elif self.fusion == "interleave":
            return self._interleave_fusion(all_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")
    
    def _extract_scores(self, contexts: List[Context]) -> List[float]:
        """Extract scores from contexts, using rank if no score available."""
        scores = []
        for i, ctx in enumerate(contexts):
            if hasattr(ctx, 'score') and ctx.score is not None:
                scores.append(float(ctx.score))
            else:
                # Use inverse rank as score
                scores.append(1.0 / (i + 1))
        return scores
    
    def _rrf_fusion(self, results: List[RetrievalResult]) -> List[Context]:
        """
        Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each retriever
        Higher k values give more weight to top-ranked results.
        """
        # Track RRF scores by document ID
        rrf_scores: Dict[str, float] = defaultdict(float)
        context_map: Dict[str, Context] = {}
        
        for result in results:
            for rank, ctx in enumerate(result.contexts):
                doc_id = ctx.id if ctx.id else ctx.text[:100]  # Use text prefix if no ID
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                rrf_scores[doc_id] += rrf_score
                context_map[doc_id] = ctx
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build result list
        fused = []
        for doc_id in sorted_ids[:self.n_docs]:
            ctx = context_map[doc_id]
            # Update context with fused score
            new_ctx = Context(
                text=ctx.text,
                id=ctx.id,
                title=ctx.title if hasattr(ctx, 'title') else "",
                score=rrf_scores[doc_id],
            )
            fused.append(new_ctx)
        
        return fused
    
    def _weighted_fusion(self, results: List[RetrievalResult]) -> List[Context]:
        """
        Weighted score combination.
        
        Final score = sum(weight_i * normalized_score_i) for each retriever
        """
        # Track weighted scores by document ID
        weighted_scores: Dict[str, float] = defaultdict(float)
        context_map: Dict[str, Context] = {}
        
        for i, result in enumerate(results):
            # Normalize scores to [0, 1]
            if result.scores:
                max_score = max(result.scores) if result.scores else 1.0
                min_score = min(result.scores) if result.scores else 0.0
                score_range = max_score - min_score if max_score != min_score else 1.0
            
            for j, ctx in enumerate(result.contexts):
                doc_id = ctx.id if ctx.id else ctx.text[:100]
                
                # Normalize and weight the score
                if result.scores and j < len(result.scores):
                    normalized = (result.scores[j] - min_score) / score_range
                else:
                    normalized = 1.0 / (j + 1)
                
                weighted_scores[doc_id] += self.weights[i] * normalized
                context_map[doc_id] = ctx
        
        # Sort by weighted score
        sorted_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
        
        # Build result list
        fused = []
        for doc_id in sorted_ids[:self.n_docs]:
            ctx = context_map[doc_id]
            new_ctx = Context(
                text=ctx.text,
                id=ctx.id,
                title=ctx.title if hasattr(ctx, 'title') else "",
                score=weighted_scores[doc_id],
            )
            fused.append(new_ctx)
        
        return fused
    
    def _interleave_fusion(self, results: List[RetrievalResult]) -> List[Context]:
        """
        Round-robin interleaving of results.
        
        Alternates between retrievers, skipping duplicates.
        """
        seen_ids = set()
        fused = []
        max_len = max(len(r.contexts) for r in results)
        
        for rank in range(max_len):
            for result in results:
                if rank < len(result.contexts):
                    ctx = result.contexts[rank]
                    doc_id = ctx.id if ctx.id else ctx.text[:100]
                    
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        fused.append(ctx)
                        
                        if len(fused) >= self.n_docs:
                            return fused
        
        return fused
    
    def __repr__(self):
        return (
            f"HybridRetriever(sparse='{self.sparse_method}', "
            f"dense={self.dense_methods}, fusion='{self.fusion}')"
        )
