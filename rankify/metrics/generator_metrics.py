# rankify/metrics/generator_metrics.py
from typing import Dict, List, Optional
import warnings

from .base_metrics import ExactMatch, F1Score, PrecisionScore, RecallScore, ContainsMatch, BLEUScore
from .ragas_bridge import RagasModels, build_ragas_dataset, run_ragas_eval


class GeneratorMetrics:
    """
    Comprehensive generation metrics including:
    - Classic: EM, F1, Precision, Recall, Contains, BLEU
    - ROUGE-L
    - BERTScore  
    - RAGAS: Faithfulness, Response Relevancy, Context Precision, Context Recall
    """

    def __init__(self, documents, config: Optional[Dict] = None):
        self.documents = documents
        self.config = config or {"dataset_name": "QA_Evaluation"}

    def classic(self, predictions: List[str]) -> Dict[str, float]:
        """Compute classic metrics (EM, F1, etc.)"""
        data = type("Data", (object,), {
            "documents": self.documents,
            "predictions": predictions
        })()
        
        metrics = [
            ExactMatch(self.config),
            F1Score(self.config),
            PrecisionScore(self.config),
            RecallScore(self.config),
            ContainsMatch(self.config),
            BLEUScore(self.config)
        ]
        
        out = {}
        for m in metrics:
            score, _ = m.calculate_metric(data)
            out.update(score)
        return out

    def rouge_l(self, predictions: List[str]) -> Dict[str, float]:
        """Compute ROUGE-L score"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            scores = []
            for doc, pred in zip(self.documents, predictions):
                ans = getattr(doc.answers, "answers", getattr(doc, "answers", []))
                refs = ans if isinstance(ans, list) else [str(ans)]
                
                best_score = 0.0
                for ref in refs:
                    score = scorer.score(ref, pred or "")
                    best_score = max(best_score, score['rougeL'].fmeasure)
                scores.append(best_score)
            
            return {"rougeL_f1": sum(scores) / len(scores) if scores else 0.0}
        except ImportError:
            warnings.warn("rouge-score not installed. Install with: pip install rouge-score")
            return {"rougeL_f1": 0.0}

    def bertscore(self, predictions: List[str]) -> Dict[str, float]:
        """Compute BERTScore"""
        try:
            from bert_score import score as bert_score
            
            cands = []
            refs = []
            for doc, p in zip(self.documents, predictions):
                ans = getattr(doc.answers, "answers", getattr(doc, "answers", []))
                refs_row = ans if isinstance(ans, list) else [str(ans)]
                if not refs_row:
                    refs_row = [""]
                
                best_ref = max(refs_row, key=lambda s: len(s))
                cands.append(p or "")
                refs.append(best_ref)

            P, R, F1 = bert_score(cands, refs, lang="en", rescale_with_baseline=True)
            return {
                "bertscore_precision": float(P.mean()),
                "bertscore_recall": float(R.mean()),
                "bertscore_f1": float(F1.mean()),
            }
        except Exception as e:
            warnings.warn(f"BERTScore unavailable: {e}")
            return {"bertscore_f1": 0.0}

    def ragas_generator(
        self,
        predictions: List[str],
        judge: RagasModels,
        use_reordered_contexts: bool = True,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute RAGAS metrics.
        
        Args:
            predictions: Generated answers
            judge: RagasModels instance with LLM/embeddings config
            use_reordered_contexts: Use reranked contexts if available
            metrics: List of metrics to compute. Options:
                - "faithfulness"
                - "response_relevancy"
                - "context_precision"
                - "context_recall"
                If None, computes all available metrics
        """
        try:
            dataset = build_ragas_dataset(
                self.documents,
                predictions,
                use_reordered=use_reordered_contexts
            )
            
            scores = run_ragas_eval(dataset, judge, metrics)
            
            # Add ragas_ prefix to metric names for consistency
            return {f"ragas_{k}": v for k, v in scores.items()}
            
        except Exception as e:
            warnings.warn(f"RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "ragas_faithfulness": 0.0,
                "ragas_response_relevancy": 0.0,
                "ragas_context_precision": 0.0
            }

    def all(
        self,
        predictions: List[str],
        ragas_models: Optional[RagasModels] = None,
        use_reordered_contexts: bool = True,
        ragas_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: Generated answers
            ragas_models: Optional RagasModels for RAGAS evaluation
            use_reordered_contexts: Use reranked contexts if available
            ragas_metrics: Specific RAGAS metrics to compute
        """
        out = {}
        out.update(self.classic(predictions))
        out.update(self.rouge_l(predictions))
        out.update(self.bertscore(predictions))
        
        if ragas_models:
            out.update(self.ragas_generator(
                predictions,
                judge=ragas_models,
                use_reordered_contexts=use_reordered_contexts,
                metrics=ragas_metrics
            ))
        
        return out