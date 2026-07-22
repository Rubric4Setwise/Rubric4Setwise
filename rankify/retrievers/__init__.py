# rankify/retrievers/__init__.py - LAZY IMPORT VERSION
# Only import Retriever at top level; all specific retrievers are lazy-loaded

from .base_retriever import BaseRetriever
from .retriever import Retriever


__all__ = [
    "Retriever",
    "BaseRetriever",
]