"""
Rankify Server Package

Provides REST API for production deployments.

Start server:
    >>> from rankify.server import RankifyServer
    >>> server = RankifyServer(retriever="bge", reranker="flashrank")
    >>> server.start(port=8000)
    
Or via CLI:
    >>> rankify serve --port 8000 --retriever bge --reranker flashrank
"""

from rankify.server.server import (
    RankifyServer,
    create_server,
)

__all__ = [
    "RankifyServer",
    "create_server",
]
