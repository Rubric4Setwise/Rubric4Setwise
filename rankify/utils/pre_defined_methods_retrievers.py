from rankify.retrievers.OnlineRetriever import OnlineRetriever
from rankify.retrievers.dpr import DenseRetriever
from rankify.retrievers.bm25 import BM25Retriever
from rankify.retrievers.contriever import ContrieverRetriever
from rankify.retrievers.BGERetriever import BGERetriever
from rankify.retrievers.colbert import ColBERTRetriever
from rankify.retrievers.hyde import HydeRetreiver


METHOD_MAP ={
    'bm25': BM25Retriever,
    'dpr': DenseRetriever,
    'bpr': DenseRetriever,
    'contriever': ContrieverRetriever,
    'ance': DenseRetriever,
    'bge': BGERetriever,
    'colbert': ColBERTRetriever,
    'hyde': HydeRetreiver,
    'online_retriever': OnlineRetriever,
}