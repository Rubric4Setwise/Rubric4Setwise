"""
RankifyAgent - Intelligent Model Recommendation System for Rankify.

This module provides:
- RankifyAgent: Conversational AI assistant for model selection
- RankifyRecommender: Programmatic recommendation API
- Model Registry: Metadata for all available models

Example Usage:
    ```python
    from rankify.agent import RankifyAgent, recommend

    # Quick recommendation
    result = recommend(task="qa", gpu=True)
    print(result.retriever.name)
    print(result.reranker.name)

    # Conversational interface
    agent = RankifyAgent(backend="azure")
    response = agent.chat("I need a fast search system for production")
    print(response.message)
    print(response.code_snippet)
    ```
"""

from rankify.agent.model_registry import (
    ModelMetadata,
    TaskType,
    Speed,
    Accuracy,
    RETRIEVER_REGISTRY,
    RERANKER_REGISTRY,
    RAG_METHOD_REGISTRY,
    get_all_retrievers,
    get_all_rerankers,
    get_all_rag_methods,
    get_model,
    filter_models,
)

from rankify.agent.recommender import (
    RankifyRecommender,
    RecommendationResult,
    PipelineConfig,
    recommend,
)

from rankify.agent.agent import (
    RankifyAgent,
    AgentResponse,
)


__all__ = [
    # Agent
    "RankifyAgent",
    "AgentResponse",
    
    # Recommender
    "RankifyRecommender",
    "RecommendationResult",
    "PipelineConfig",
    "recommend",
    
    # Registry
    "ModelMetadata",
    "TaskType",
    "Speed",
    "Accuracy",
    "RETRIEVER_REGISTRY",
    "RERANKER_REGISTRY",
    "RAG_METHOD_REGISTRY",
    "get_all_retrievers",
    "get_all_rerankers",
    "get_all_rag_methods",
    "get_model",
    "filter_models",
]
