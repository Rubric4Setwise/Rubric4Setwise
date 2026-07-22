"""
Rankify Integrations - Connect with LangChain, LlamaIndex, and Vector DBs

Provides seamless integration with popular frameworks:
- LangChain: Use Rankify as a LangChain retriever
- LlamaIndex: Use Rankify as a LlamaIndex retriever
- Vector DBs: Pinecone, Weaviate, Milvus, Qdrant, Chroma

Example:
    ```python
    # LangChain integration
    from rankify.integrations import LangChainRetriever
    retriever = LangChainRetriever(method="bge", reranker="flashrank")
    chain = RetrievalQA.from_chain_type(retriever=retriever)
    
    # LlamaIndex integration
    from rankify.integrations import LlamaIndexRetriever
    retriever = LlamaIndexRetriever(method="colbert")
    ```
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass


# =============================================================================
# LANGCHAIN INTEGRATION
# =============================================================================

class LangChainRetriever:
    """
    LangChain-compatible retriever using Rankify.
    
    Usage:
        ```python
        from rankify.integrations import LangChainRetriever
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        
        retriever = LangChainRetriever(
            method="bge",
            reranker="flashrank",
            n_docs=100,
            top_k=10,
        )
        
        llm = ChatOpenAI()
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
        )
        
        result = chain.invoke({"query": "What is machine learning?"})
        ```
    """
    
    def __init__(
        self,
        method: str = "bm25",
        reranker: Optional[str] = None,
        reranker_model: Optional[str] = None,
        n_docs: int = 100,
        top_k: int = 10,
        index_type: str = "wiki",
        **kwargs,
    ):
        """
        Initialize LangChain retriever.
        
        Args:
            method: Rankify retriever method (bm25, bge, colbert, etc.)
            reranker: Optional reranker method
            reranker_model: Optional reranker model name
            n_docs: Number of docs to retrieve
            top_k: Number of docs to return after reranking
            index_type: Index type
        """
        self.method = method
        self.reranker_method = reranker
        self.reranker_model = reranker_model
        self.n_docs = n_docs
        self.top_k = top_k
        self.index_type = index_type
        self.kwargs = kwargs
        
        # Lazy initialization
        self._retriever = None
        self._reranker = None
    
    def _initialize(self):
        """Initialize Rankify components."""
        if self._retriever is not None:
            return
        
        from rankify.retrievers.retriever import Retriever
        
        self._retriever = Retriever(
            method=self.method,
            n_docs=self.n_docs,
            index_type=self.index_type,
            **self.kwargs,
        )
        
        if self.reranker_method:
            from rankify.models.reranking import Reranking
            self._reranker = Reranking(
                method=self.reranker_method,
                model_name=self.reranker_model,
            )
    
    def _get_relevant_documents(self, query: str) -> List[Any]:
        """
        Get relevant documents for LangChain.
        
        Returns LangChain Document objects.
        """
        self._initialize()
        
        try:
            from langchain.schema import Document as LCDocument
        except ImportError:
            raise ImportError("LangChain is required. Install with: pip install langchain")
        
        from rankify.dataset.dataset import Document, Question, Answer
        
        doc = Document(
            question=Question(query),
            answers=Answer([]),
            contexts=[],
        )
        
        # Retrieve
        results = self._retriever.retrieve([doc])
        
        # Rerank if configured
        if self._reranker:
            results = self._reranker.rank(results)
            contexts = (results[0].reorder_contexts or results[0].contexts)[:self.top_k]
        else:
            contexts = results[0].contexts[:self.top_k]
        
        # Convert to LangChain documents
        return [
            LCDocument(
                page_content=ctx.text,
                metadata={
                    "id": ctx.id,
                    "title": getattr(ctx, 'title', ''),
                    "score": getattr(ctx, 'score', None),
                }
            )
            for ctx in contexts
        ]
    
    def get_relevant_documents(self, query: str) -> List[Any]:
        """Sync method for LangChain compatibility."""
        return self._get_relevant_documents(query)
    
    async def aget_relevant_documents(self, query: str) -> List[Any]:
        """Async method for LangChain compatibility."""
        return self._get_relevant_documents(query)
    
    # For newer LangChain versions
    def invoke(self, input: str, config: Optional[Dict] = None) -> List[Any]:
        """Invoke method for LangChain LCEL."""
        return self._get_relevant_documents(input)
    
    async def ainvoke(self, input: str, config: Optional[Dict] = None) -> List[Any]:
        """Async invoke method for LangChain LCEL."""
        return self._get_relevant_documents(input)


# =============================================================================
# LLAMAINDEX INTEGRATION
# =============================================================================

class LlamaIndexRetriever:
    """
    LlamaIndex-compatible retriever using Rankify.
    
    Usage:
        ```python
        from rankify.integrations import LlamaIndexRetriever
        from llama_index.core import VectorStoreIndex
        
        retriever = LlamaIndexRetriever(
            method="colbert",
            reranker="monot5",
        )
        
        # Use with query engine
        nodes = retriever.retrieve("What is transformers?")
        ```
    """
    
    def __init__(
        self,
        method: str = "bm25",
        reranker: Optional[str] = None,
        reranker_model: Optional[str] = None,
        n_docs: int = 100,
        top_k: int = 10,
        **kwargs,
    ):
        """
        Initialize LlamaIndex retriever.
        
        Args:
            method: Rankify retriever method
            reranker: Optional reranker method
            reranker_model: Optional reranker model
            n_docs: Number of docs to retrieve
            top_k: Number of docs to return
        """
        self.method = method
        self.reranker_method = reranker
        self.reranker_model = reranker_model
        self.n_docs = n_docs
        self.top_k = top_k
        self.kwargs = kwargs
        
        self._retriever = None
        self._reranker = None
    
    def _initialize(self):
        """Initialize Rankify components."""
        if self._retriever is not None:
            return
        
        from rankify.retrievers.retriever import Retriever
        
        self._retriever = Retriever(
            method=self.method,
            n_docs=self.n_docs,
            **self.kwargs,
        )
        
        if self.reranker_method:
            from rankify.models.reranking import Reranking
            self._reranker = Reranking(
                method=self.reranker_method,
                model_name=self.reranker_model,
            )
    
    def retrieve(self, query: str) -> List[Any]:
        """
        Retrieve nodes for LlamaIndex.
        
        Returns LlamaIndex NodeWithScore objects.
        """
        self._initialize()
        
        try:
            from llama_index.core.schema import TextNode, NodeWithScore
        except ImportError:
            raise ImportError("LlamaIndex is required. Install with: pip install llama-index")
        
        from rankify.dataset.dataset import Document, Question, Answer
        
        doc = Document(
            question=Question(query),
            answers=Answer([]),
            contexts=[],
        )
        
        # Retrieve
        results = self._retriever.retrieve([doc])
        
        # Rerank if configured
        if self._reranker:
            results = self._reranker.rank(results)
            contexts = (results[0].reorder_contexts or results[0].contexts)[:self.top_k]
        else:
            contexts = results[0].contexts[:self.top_k]
        
        # Convert to LlamaIndex nodes
        return [
            NodeWithScore(
                node=TextNode(
                    text=ctx.text,
                    id_=ctx.id,
                    metadata={
                        "title": getattr(ctx, 'title', ''),
                    }
                ),
                score=getattr(ctx, 'score', 1.0 / (i + 1)),
            )
            for i, ctx in enumerate(contexts)
        ]
    
    def _retrieve(self, query_bundle: Any) -> List[Any]:
        """For LlamaIndex BaseRetriever compatibility."""
        query = query_bundle.query_str if hasattr(query_bundle, 'query_str') else str(query_bundle)
        return self.retrieve(query)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_langchain_retriever(
    method: str = "bm25",
    reranker: Optional[str] = "flashrank",
    **kwargs,
) -> LangChainRetriever:
    """Create a LangChain-compatible retriever."""
    return LangChainRetriever(method=method, reranker=reranker, **kwargs)


def create_llamaindex_retriever(
    method: str = "bm25",
    reranker: Optional[str] = "flashrank",
    **kwargs,
) -> LlamaIndexRetriever:
    """Create a LlamaIndex-compatible retriever."""
    return LlamaIndexRetriever(method=method, reranker=reranker, **kwargs)
