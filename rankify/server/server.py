"""
Rankify Server - REST API for Production Deployments

Provides a FastAPI-based server with endpoints for:
- /retrieve - Document retrieval
- /rerank - Rerank existing documents
- /rag - Full RAG pipeline

Start server:
    >>> rankify serve --port 8000
    
Or in Python:
    >>> from rankify.server import RankifyServer
    >>> server = RankifyServer(retriever="bge", reranker="flashrank")
    >>> server.start(port=8000)
"""

import os
import json
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import logging

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class RetrieveRequest(BaseModel):
        """Request for document retrieval."""
        query: str = Field(..., description="Search query")
        n_docs: int = Field(100, description="Number of documents to retrieve")
        
    class RerankRequest(BaseModel):
        """Request for reranking."""
        query: str = Field(..., description="Query for reranking")
        documents: List[Dict[str, Any]] = Field(..., description="Documents to rerank")
        top_k: int = Field(10, description="Number of top documents to return")
        
    class RAGRequest(BaseModel):
        """Request for RAG generation."""
        query: str = Field(..., description="Question for RAG")
        n_contexts: int = Field(5, description="Number of contexts for generation")
        stream: bool = Field(False, description="Enable streaming response")
        
    class DocumentResponse(BaseModel):
        """Single document in response."""
        id: str
        text: str
        title: Optional[str] = None
        score: Optional[float] = None
        
    class RetrieveResponse(BaseModel):
        """Response from retrieval."""
        query: str
        documents: List[DocumentResponse]
        latency_ms: float
        
    class RerankResponse(BaseModel):
        """Response from reranking."""
        query: str
        documents: List[DocumentResponse]
        latency_ms: float
        
    class RAGResponse(BaseModel):
        """Response from RAG."""
        query: str
        answer: str
        contexts: List[DocumentResponse]
        latency_ms: float
        
    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        version: str
        retriever: Optional[str] = None
        reranker: Optional[str] = None
        generator: Optional[str] = None


class RankifyServer:
    """
    FastAPI-based server for Rankify.
    
    Example:
        ```python
        from rankify.server import RankifyServer
        
        # Create server with configuration
        server = RankifyServer(
            retriever="bge",
            reranker="flashrank",
            generator="basic-rag",
        )
        
        # Start server
        server.start(host="0.0.0.0", port=8000)
        ```
    """
    
    def __init__(
        self,
        retriever: str = "bm25",
        reranker: str = "flashrank",
        generator: Optional[str] = None,
        retriever_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        generator_model: str = "gpt-4o-mini",
        generator_backend: str = "openai",
        index_type: str = "wiki",
        n_docs: int = 100,
        **kwargs,
    ):
        """
        Initialize Rankify server.
        
        Args:
            retriever: Retriever method
            reranker: Reranker method
            generator: RAG method (None for search-only)
            retriever_model: Specific retriever model
            reranker_model: Specific reranker model
            generator_model: LLM model for generation
            generator_backend: LLM backend
            index_type: Index type
            n_docs: Default documents to retrieve
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for the server. "
                "Install with: pip install fastapi uvicorn"
            )
        
        self.config = {
            "retriever": retriever,
            "reranker": reranker,
            "generator": generator,
            "retriever_model": retriever_model,
            "reranker_model": reranker_model or "ms-marco-MiniLM-L-12-v2",
            "generator_model": generator_model,
            "generator_backend": generator_backend,
            "index_type": index_type,
            "n_docs": n_docs,
            **kwargs,
        }
        
        # Components (lazy loaded)
        self._retriever = None
        self._reranker = None
        self._generator = None
        self._app = None
    
    def _initialize_components(self):
        """Initialize ML components."""
        from rankify.retrievers.retriever import Retriever
        from rankify.models.reranking import Reranking
        
        logger.info(f"Loading retriever: {self.config['retriever']}")
        self._retriever = Retriever(
            method=self.config["retriever"],
            n_docs=self.config["n_docs"],
            index_type=self.config["index_type"],
        )
        
        logger.info(f"Loading reranker: {self.config['reranker']}")
        self._reranker = Reranking(
            method=self.config["reranker"],
            model_name=self.config["reranker_model"],
        )
        
        if self.config["generator"]:
            logger.info(f"Loading generator: {self.config['generator']}")
            from rankify.generator.generator import Generator
            self._generator = Generator(
                method=self.config["generator"],
                model_name=self.config["generator_model"],
                backend=self.config["generator_backend"],
            )
    
    def create_app(self) -> "FastAPI":
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Initializing Rankify components...")
            self._initialize_components()
            logger.info("Rankify server ready!")
            yield
            # Shutdown
            logger.info("Shutting down Rankify server...")
        
        app = FastAPI(
            title="Rankify API",
            description="REST API for Retrieval, Reranking, and RAG",
            version="0.1.0",
            lifespan=lifespan,
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check
        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                version="0.1.0",
                retriever=self.config["retriever"],
                reranker=self.config["reranker"],
                generator=self.config["generator"],
            )
        
        # Retrieve endpoint
        @app.post("/retrieve", response_model=RetrieveResponse)
        async def retrieve(request: RetrieveRequest):
            """Retrieve documents for a query."""
            start = time.time()
            
            from rankify.dataset.dataset import Document, Question, Answer
            
            doc = Document(
                question=Question(request.query),
                answers=Answer([]),
                contexts=[],
            )
            
            results = self._retriever.retrieve([doc])
            contexts = results[0].contexts[:request.n_docs]
            
            latency = (time.time() - start) * 1000
            
            return RetrieveResponse(
                query=request.query,
                documents=[
                    DocumentResponse(
                        id=ctx.id or str(i),
                        text=ctx.text,
                        title=ctx.title if hasattr(ctx, 'title') else None,
                        score=ctx.score if hasattr(ctx, 'score') else None,
                    )
                    for i, ctx in enumerate(contexts)
                ],
                latency_ms=round(latency, 2),
            )
        
        # Rerank endpoint
        @app.post("/rerank", response_model=RerankResponse)
        async def rerank(request: RerankRequest):
            """Rerank documents for a query."""
            start = time.time()
            
            from rankify.dataset.dataset import Document, Question, Answer, Context
            
            # Convert request documents to Context objects
            contexts = [
                Context(
                    text=d.get("text", ""),
                    id=d.get("id", str(i)),
                    title=d.get("title", ""),
                )
                for i, d in enumerate(request.documents)
            ]
            
            doc = Document(
                question=Question(request.query),
                answers=Answer([]),
                contexts=contexts,
            )
            
            results = self._reranker.rank([doc])
            reranked = results[0].reorder_contexts or results[0].contexts
            top_k = reranked[:request.top_k]
            
            latency = (time.time() - start) * 1000
            
            return RerankResponse(
                query=request.query,
                documents=[
                    DocumentResponse(
                        id=ctx.id or str(i),
                        text=ctx.text,
                        title=ctx.title if hasattr(ctx, 'title') else None,
                        score=ctx.score if hasattr(ctx, 'score') else None,
                    )
                    for i, ctx in enumerate(top_k)
                ],
                latency_ms=round(latency, 2),
            )
        
        # RAG endpoint
        @app.post("/rag", response_model=RAGResponse)
        async def rag(request: RAGRequest):
            """Generate answer using RAG."""
            if not self._generator:
                raise HTTPException(
                    status_code=400,
                    detail="Generator not configured. Start server with generator parameter."
                )
            
            start = time.time()
            
            from rankify.dataset.dataset import Document, Question, Answer
            
            doc = Document(
                question=Question(request.query),
                answers=Answer([]),
                contexts=[],
            )
            
            # Retrieve
            results = self._retriever.retrieve([doc])
            doc = results[0]
            
            # Rerank
            results = self._reranker.rank([doc])
            doc = results[0]
            
            # Get top contexts
            contexts = (doc.reorder_contexts or doc.contexts)[:request.n_contexts]
            doc.contexts = contexts
            
            # Generate
            answers = self._generator.generate([doc])
            answer = answers[0] if answers else ""
            
            latency = (time.time() - start) * 1000
            
            return RAGResponse(
                query=request.query,
                answer=answer,
                contexts=[
                    DocumentResponse(
                        id=ctx.id or str(i),
                        text=ctx.text[:500],  # Truncate for response
                        title=ctx.title if hasattr(ctx, 'title') else None,
                        score=ctx.score if hasattr(ctx, 'score') else None,
                    )
                    for i, ctx in enumerate(contexts)
                ],
                latency_ms=round(latency, 2),
            )
        
        # Batch retrieve
        @app.post("/retrieve/batch")
        async def retrieve_batch(queries: List[str], n_docs: int = 100):
            """Batch retrieve for multiple queries."""
            from rankify.dataset.dataset import Document, Question, Answer
            
            docs = [
                Document(question=Question(q), answers=Answer([]), contexts=[])
                for q in queries
            ]
            
            results = self._retriever.retrieve(docs)
            
            return {
                "results": [
                    {
                        "query": q,
                        "documents": [
                            {"id": ctx.id, "text": ctx.text}
                            for ctx in r.contexts[:n_docs]
                        ]
                    }
                    for q, r in zip(queries, results)
                ]
            }
        
        self._app = app
        return app
    
    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
    ):
        """
        Start the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload for development
            workers: Number of worker processes
        """
        app = self.create_app()
        
        logger.info(f"Starting Rankify server at http://{host}:{port}")
        logger.info(f"OpenAPI docs at http://{host}:{port}/docs")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
        )


def create_server(
    retriever: str = "bm25",
    reranker: str = "flashrank",
    generator: Optional[str] = None,
    **kwargs,
) -> RankifyServer:
    """
    Create a Rankify server instance.
    
    Example:
        ```python
        from rankify.server import create_server
        
        server = create_server(retriever="bge", reranker="flashrank")
        server.start(port=8000)
        ```
    """
    return RankifyServer(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        **kwargs,
    )


# CLI support
def main():
    """CLI entry point for rankify serve."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Rankify server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--retriever", default="bm25", help="Retriever method")
    parser.add_argument("--reranker", default="flashrank", help="Reranker method")
    parser.add_argument("--generator", default=None, help="Generator method")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    server = RankifyServer(
        retriever=args.retriever,
        reranker=args.reranker,
        generator=args.generator,
    )
    server.start(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
