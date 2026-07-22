# rankify/retrievers/retriever.py - LAZY IMPORT VERSION
from typing import List, Dict, Type
from rankify.dataset.dataset import Document
from .base_retriever import BaseRetriever

# All retrievers (including BM25) are lazy-loaded to avoid pulling in heavy/missing dependencies


# Method mapping - uses strings for lazy loading
METHOD_MAP: Dict[str, str] = {
    "bm25": "BM25Retriever",
    "dpr-multi": "DenseRetriever",
    "dpr-single": "DenseRetriever",
    "ance-multi": "ANCERetriever",
    "bpr-single": "DenseRetriever",
    "bge": "BGERetriever",
    "colbert": "ColBERTRetriever",
    "contriever": "ContrieverRetriever",
    "online": "OnlineRetriever",
    "hyde": "HydeRetriever",
    "diver-dense": "DiverDenseRetriever",
    "diver-bm25": "DiverBM25Retriever",
    "reasonir": "ReasonIRRetriever",
    "reason-embed": "ReasonEmbedRetriever",
    "bge-reasoner-embed": "BgeReasonerRetriever",
    "unicoil": "UniCOILRetriever",
    "unicoil-noexp": "UniCOILRetriever",
    "splade-v2": "SpladeV2Retriever",
    "openai-embedding": "APIEmbeddingRetriever",
    "cohere-embedding": "APIEmbeddingRetriever",
    "voyage-embedding": "APIEmbeddingRetriever",
}


def _get_retriever_class(class_name: str) -> Type[BaseRetriever]:
    """Lazy-load retriever class by name to avoid importing all dependencies at module level."""
    if class_name == "BM25Retriever":
        from .bm25_retriever import BM25Retriever
        return BM25Retriever
    elif class_name == "DenseRetriever":
        from .dense_retriever import DenseRetriever
        return DenseRetriever
    elif class_name == "ANCERetriever":
        from .ance_retriever import ANCERetriever
        return ANCERetriever
    elif class_name == "BGERetriever":
        from .bge_retriever import BGERetriever
        return BGERetriever
    elif class_name == "ColBERTRetriever":
        from .colbert_retriever import ColBERTRetriever
        return ColBERTRetriever
    elif class_name == "ContrieverRetriever":
        from .contriever_retriever import ContrieverRetriever
        return ContrieverRetriever
    elif class_name == "OnlineRetriever":
        from .online_retriever import OnlineRetriever
        return OnlineRetriever
    elif class_name == "HydeRetriever":
        from .hyde_retriever import HydeRetriever
        return HydeRetriever
    elif class_name == "DiverDenseRetriever":
        from .diver_dense_retriever import DiverDenseRetriever
        return DiverDenseRetriever
    elif class_name == "DiverBM25Retriever":
        from .diver_bm25_retriever import DiverBM25Retriever
        return DiverBM25Retriever
    elif class_name == "ReasonIRRetriever":
        from .reasonir_retriever import ReasonIRRetriever
        return ReasonIRRetriever
    elif class_name == "ReasonEmbedRetriever":
        from .reasonembed_retriever import ReasonEmbedRetriever
        return ReasonEmbedRetriever
    elif class_name == "BgeReasonerRetriever":
        from .bge_reasoner_retriever import BgeReasonerRetriever
        return BgeReasonerRetriever
    elif class_name == "UniCOILRetriever":
        from .unicoil_retriever import UniCOILRetriever
        return UniCOILRetriever
    elif class_name == "SpladeV2Retriever":
        from .splade_v2_retriever import SpladeV2Retriever
        return SpladeV2Retriever
    elif class_name == "APIEmbeddingRetriever":
        from .api_embedding_retriever import APIEmbeddingRetriever
        return APIEmbeddingRetriever
    else:
        raise ValueError(f"Unknown retriever class: {class_name}")


class Retriever:
    """
    Unified retriever interface for the rankify framework.
    
    Provides a simple interface to access different retrieval methods
    (BM25, DPR, ANCE, BPR) with consistent parameters.
    
    Example:
        ```python
        # Initialize with BM25
        retriever = Retriever(method="bm25", n_docs=10, index_type="wiki")
        
        # Initialize with DPR
        retriever = Retriever(method="dpr-multi", n_docs=5, index_type="msmarco")
        
        # Retrieve documents
        retrieved_documents = retriever.retrieve(documents)
        ```
    """
    
    def __init__(self, method: str, n_docs: int = 10, index_type: str = "wiki", 
                 index_folder: str = None, encoder_name: str = None, **kwargs):
        """
        Initialize the retriever.
        
        Args:
            method (str): Retrieval method ('bm25', 'dpr-multi', 'dpr-single', 'ance-multi', 'bpr-single', etc.)
            n_docs (int): Number of documents to retrieve per query
            index_type (str): Index type ('wiki', 'msmarco') - ignored if index_folder is provided
            index_folder (str): Path to custom index folder (optional)
            encoder_name (str): Model name for encoding (method-specific)
            **kwargs: Additional parameters passed to the specific retriever
        """
        self.method = method.lower()
        self.n_docs = n_docs
        self.index_type = index_type.lower()
        self.index_folder = index_folder
        self.encoder_name = encoder_name
        self.kwargs = kwargs
        
        # Initialize the specific retriever
        self.retriever = self._initialize_retriever()
    
    def _initialize_retriever(self) -> BaseRetriever:
        """Initialize the specific retriever based on the method."""
        if self.method not in METHOD_MAP:
            supported_methods = ", ".join(METHOD_MAP.keys())
            raise ValueError(f"Unsupported method '{self.method}'. "
                           f"Supported methods: {supported_methods}")
        
        class_name = METHOD_MAP[self.method]
        retriever_class = _get_retriever_class(class_name)
        
        # Prepare initialization parameters
        init_params = {
            "n_docs": self.n_docs,
            **self.kwargs
        }

        # No-index retrievers do NOT use index_type or index_folder
        NO_INDEX_METHODS = {
            "diver-dense",
            "diver-bm25",
            "reasonir",
            "reason-embed",
            "bge-reasoner-embed",
            "unicoil",
            "unicoil-noexp",
            "splade-v2",
            "openai-embedding",
            "cohere-embedding",
            "voyage-embedding",
        }
        if self.method in NO_INDEX_METHODS:
            return retriever_class(**init_params)
        
        # Handle ANCE variants
        if self.method in ["ance", "ance-msmarco", "ance-multi"]:
            init_params["index_type"] = self.index_type
            if self.index_folder:
                init_params["index_folder"] = self.index_folder
            if self.encoder_name:
                init_params["encoder_name"] = self.encoder_name
        else:
            init_params["index_type"] = self.index_type
            if self.index_folder:
                init_params["index_folder"] = self.index_folder
            if self.method in ["dpr-multi", "dpr-single", "bpr-single"]:
                init_params["method"] = self.method
        
        return retriever_class(**init_params)
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        """
        Retrieve relevant contexts for the given documents.
        
        Args:
            documents (List[Document]): List of documents containing queries
            
        Returns:
            List[Document]: Documents updated with retrieved contexts
        """
        return self.retriever.retrieve(documents)
    
    @classmethod
    def supported_methods(cls) -> List[str]:
        """Get list of supported retrieval methods."""
        return list(METHOD_MAP.keys())
    
    def __repr__(self) -> str:
        index_info = f"index_folder='{self.index_folder}'" if self.index_folder else f"index_type='{self.index_type}'"
        return (f"Retriever(method='{self.method}', n_docs={self.n_docs}, "
                f"{index_info})")