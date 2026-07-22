"""
RankifyAgent Recommender Engine.

Provides intelligent model recommendations based on user constraints and task requirements.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from rankify.agent.model_registry import (
    ModelMetadata,
    TaskType,
    Speed,
    Accuracy,
    RETRIEVER_REGISTRY,
    RERANKER_REGISTRY,
    RAG_METHOD_REGISTRY,
    filter_models,
)


@dataclass
class PipelineConfig:
    """Configuration for a complete RAG pipeline."""
    retriever: str
    retriever_params: Dict[str, Any] = field(default_factory=dict)
    reranker: Optional[str] = None
    reranker_params: Dict[str, Any] = field(default_factory=dict)
    rag_method: Optional[str] = None
    rag_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retriever": self.retriever,
            "retriever_params": self.retriever_params,
            "reranker": self.reranker,
            "reranker_params": self.reranker_params,
            "rag_method": self.rag_method,
            "rag_params": self.rag_params,
        }


@dataclass
class RecommendationResult:
    """Result of a model recommendation."""
    retriever: ModelMetadata
    reranker: Optional[ModelMetadata] = None
    rag_method: Optional[ModelMetadata] = None
    pipeline_config: Optional[PipelineConfig] = None
    explanation: str = ""
    alternatives: Dict[str, List[ModelMetadata]] = field(default_factory=dict)
    
    def build_pipeline(self):
        """Build pipeline from recommendation."""
        from rankify.agent.pipeline_builder import PipelineBuilder
        builder = PipelineBuilder()
        return builder.build(self.pipeline_config)


class RankifyRecommender:
    """
    Intelligent model recommender for Rankify.
    
    Recommends optimal retrievers, rerankers, and RAG methods
    based on user constraints and task requirements.
    """
    
    def __init__(self):
        self.retrievers = RETRIEVER_REGISTRY
        self.rerankers = RERANKER_REGISTRY
        self.rag_methods = RAG_METHOD_REGISTRY
    
    def recommend(
        self,
        task: str = "qa",
        constraints: Optional[Dict[str, Any]] = None,
        include_reranker: bool = True,
        include_rag: bool = False,
        top_k: int = 3,
    ) -> RecommendationResult:
        """
        Get recommended models for a given task and constraints.
        
        Args:
            task: Task type - "qa", "search", "summarization", "conversational", "domain_specific"
            constraints: Optional constraints dict with keys like:
                - gpu: bool - Whether GPU is available
                - max_memory_mb: int - Maximum memory budget
                - max_latency_ms: int - Maximum latency budget
                - api_only: bool - Only recommend API-based models
                - no_api: bool - Only recommend local models
                - language: str - Required language support
            include_reranker: Whether to recommend a reranker
            include_rag: Whether to recommend a RAG method
            top_k: Number of alternatives to include
            
        Returns:
            RecommendationResult with recommended models and alternatives
        """
        constraints = constraints or {}
        task_type = self._parse_task_type(task)
        
        # Get best retriever
        retrievers = self._filter_and_score(
            self.retrievers, constraints, task_type
        )
        best_retriever = retrievers[0] if retrievers else None
        
        # Get best reranker
        best_reranker = None
        rerankers = []
        if include_reranker:
            rerankers = self._filter_and_score(
                self.rerankers, constraints, task_type
            )
            best_reranker = rerankers[0] if rerankers else None
        
        # Get best RAG method
        best_rag = None
        rag_methods = []
        if include_rag:
            rag_methods = self._filter_and_score(
                self.rag_methods, constraints, task_type
            )
            best_rag = rag_methods[0] if rag_methods else None
        
        # Build pipeline config
        pipeline_config = None
        if best_retriever:
            pipeline_config = PipelineConfig(
                retriever=best_retriever.method,
                retriever_params={"model_name": best_retriever.model_path} if best_retriever.model_path else {},
                reranker=best_reranker.method if best_reranker else None,
                reranker_params={"model_name": best_reranker.model_path} if best_reranker and best_reranker.model_path else {},
                rag_method=best_rag.method if best_rag else None,
            )
        
        # Generate explanation
        explanation = self._generate_explanation(
            task_type, constraints, best_retriever, best_reranker, best_rag
        )
        
        return RecommendationResult(
            retriever=best_retriever,
            reranker=best_reranker,
            rag_method=best_rag,
            pipeline_config=pipeline_config,
            explanation=explanation,
            alternatives={
                "retrievers": retrievers[1:top_k+1] if len(retrievers) > 1 else [],
                "rerankers": rerankers[1:top_k+1] if len(rerankers) > 1 else [],
                "rag_methods": rag_methods[1:top_k+1] if len(rag_methods) > 1 else [],
            }
        )
    
    def recommend_for_latency(
        self,
        max_latency_ms: int,
        task: str = "qa",
        include_reranker: bool = True,
    ) -> RecommendationResult:
        """Recommend models optimized for latency budget."""
        speed_constraint = self._latency_to_speed(max_latency_ms)
        constraints = {"max_speed": speed_constraint}
        return self.recommend(task=task, constraints=constraints, include_reranker=include_reranker)
    
    def recommend_for_accuracy(
        self,
        task: str = "qa",
        gpu_available: bool = True,
    ) -> RecommendationResult:
        """Recommend highest accuracy models regardless of speed."""
        constraints = {"gpu": gpu_available, "prefer_accuracy": True}
        return self.recommend(task=task, constraints=constraints, include_reranker=True, include_rag=True)
    
    def recommend_for_production(
        self,
        gpu_available: bool = False,
        api_allowed: bool = True,
    ) -> RecommendationResult:
        """Recommend production-ready models (fast, reliable)."""
        constraints = {
            "gpu": gpu_available,
            "api_only": api_allowed,
            "prefer_speed": True,
        }
        return self.recommend(task="search", constraints=constraints, include_reranker=True)
    
    def _parse_task_type(self, task: str) -> TaskType:
        """Parse task string to TaskType enum."""
        task_map = {
            "qa": TaskType.QUESTION_ANSWERING,
            "question_answering": TaskType.QUESTION_ANSWERING,
            "search": TaskType.SEARCH,
            "retrieval": TaskType.SEARCH,
            "summarization": TaskType.SUMMARIZATION,
            "summarize": TaskType.SUMMARIZATION,
            "conversational": TaskType.CONVERSATIONAL,
            "chat": TaskType.CONVERSATIONAL,
            "domain": TaskType.DOMAIN_SPECIFIC,
            "domain_specific": TaskType.DOMAIN_SPECIFIC,
        }
        return task_map.get(task.lower(), TaskType.QUESTION_ANSWERING)
    
    def _filter_and_score(
        self,
        registry: Dict[str, ModelMetadata],
        constraints: Dict[str, Any],
        task: TaskType,
    ) -> List[ModelMetadata]:
        """Filter models by constraints and score by task suitability."""
        results = []
        
        for name, model in registry.items():
            if model.matches_constraints(constraints):
                score = self._compute_score(model, constraints, task)
                results.append((score, model))
        
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [model for _, model in results]
    
    def _compute_score(
        self,
        model: ModelMetadata,
        constraints: Dict[str, Any],
        task: TaskType,
    ) -> float:
        """Compute overall score for a model."""
        score = 0.0
        
        # Task suitability (0-1)
        score += model.score_for_task(task) * 0.4
        
        # Accuracy score (0-1)
        accuracy_scores = {
            Accuracy.BASIC: 0.2,
            Accuracy.GOOD: 0.4,
            Accuracy.VERY_GOOD: 0.6,
            Accuracy.EXCELLENT: 0.8,
            Accuracy.STATE_OF_THE_ART: 1.0,
        }
        score += accuracy_scores.get(model.accuracy, 0.5) * 0.3
        
        # Speed score (0-1), inverse if prefer_accuracy
        speed_scores = {
            Speed.VERY_FAST: 1.0,
            Speed.FAST: 0.8,
            Speed.MEDIUM: 0.5,
            Speed.SLOW: 0.3,
            Speed.VERY_SLOW: 0.1,
        }
        speed_weight = 0.2 if constraints.get("prefer_accuracy") else 0.3
        if constraints.get("prefer_speed"):
            speed_weight = 0.4
        score += speed_scores.get(model.speed, 0.5) * speed_weight
        
        return score
    
    def _latency_to_speed(self, latency_ms: int) -> Speed:
        """Convert latency budget to speed requirement."""
        if latency_ms < 20:
            return Speed.VERY_FAST
        elif latency_ms < 100:
            return Speed.FAST
        elif latency_ms < 500:
            return Speed.MEDIUM
        elif latency_ms < 2000:
            return Speed.SLOW
        else:
            return Speed.VERY_SLOW
    
    def _generate_explanation(
        self,
        task: TaskType,
        constraints: Dict[str, Any],
        retriever: Optional[ModelMetadata],
        reranker: Optional[ModelMetadata],
        rag: Optional[ModelMetadata],
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        parts = []
        
        if retriever:
            parts.append(f"**Retriever: {retriever.name}**")
            parts.append(f"  - {retriever.description}")
            parts.append(f"  - Speed: {retriever.speed.value}, Accuracy: {retriever.accuracy.value}")
            if retriever.gpu_required:
                parts.append("  - ⚠️ Requires GPU")
        
        if reranker:
            parts.append(f"\n**Reranker: {reranker.name}**")
            parts.append(f"  - {reranker.description}")
            parts.append(f"  - Speed: {reranker.speed.value}, Accuracy: {reranker.accuracy.value}")
            if reranker.api_required:
                parts.append(f"  - 🔌 Requires API ({reranker.api_provider})")
        
        if rag:
            parts.append(f"\n**RAG Method: {rag.name}**")
            parts.append(f"  - {rag.description}")
        
        return "\n".join(parts)


# Convenience function for quick recommendations
def recommend(
    task: str = "qa",
    gpu: bool = True,
    api_allowed: bool = True,
    include_rag: bool = False,
) -> RecommendationResult:
    """
    Quick recommendation function.
    
    Args:
        task: "qa", "search", "summarization", "conversational"
        gpu: Whether GPU is available
        api_allowed: Whether API-based models are allowed
        include_rag: Whether to include RAG method in recommendation
        
    Returns:
        RecommendationResult with best models for the use case
    """
    recommender = RankifyRecommender()
    constraints = {"gpu": gpu}
    if not api_allowed:
        constraints["no_api"] = True
    return recommender.recommend(
        task=task,
        constraints=constraints,
        include_rag=include_rag,
    )
