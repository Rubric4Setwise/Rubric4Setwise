# rankify/metrics/retriever_metrics.py
from typing import Dict, List, Optional
import warnings

class RetrieverMetrics:
    """
    Retrieval-side metrics:
      - Your Top@K accuracy (bm25 vs reranked)
      - TREC (NDCG/MAP/MRR) via pyserini.trec_eval (kept)
      - RAGAS: Context Precision / Context Recall (+ Entities Recall optional)
    """
    QREL_MAPPING = {  # kept from your Metrics
        'dl19': 'dl19-passage', 'dl20': 'dl20-passage', 'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test', 'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test', 'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test', 'scidocs': 'beir-v1.0.0-scidocs-test', 'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test', 'dbpedia': 'beir-v1.0.0-dbpedia-entity-test', 'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test', 'signal': 'beir-v1.0.0-signal1m-test',
    }

    def __init__(self, documents):
        self.documents = documents

    # ---------- Top-K ----------
    def top_k_accuracy(self, k: int, use_reordered: bool = False) -> float:
        hits, total = 0, 0
        for document in self.documents:
            contexts = document.reorder_contexts if (use_reordered and getattr(document, "reorder_contexts", None)) else document.contexts
            if any(getattr(c, "has_answer", False) for c in (contexts or [])[:k]):
                hits += 1
            total += 1
        return (hits / total) * 100 if total else 0.0

    def topk_many(self, ks=(1, 5, 10, 20, 50, 100), use_reordered: bool = False) -> Dict[str, float]:
        return {f"top_{k}": self.top_k_accuracy(k, use_reordered) for k in ks}

    # ---------- TREC (kept from your Metrics) ----------
    def _generate_trec(self, use_reordered=False) -> str:
        name = "rankify" if use_reordered else "bm25"
        rows = []
        for doc in self.documents:
            contexts = doc.reorder_contexts if (use_reordered and getattr(doc, "reorder_contexts", None)) else doc.contexts
            for rank, ctx in enumerate(contexts or []):
                rows.append(f"{doc.id} Q0 {ctx.id} {rank + 1} {getattr(ctx, 'score', 0.0)} {name}")
        return "\n".join(rows)

    def trec_eval(self, ndcg_cuts=[10], map_cuts=[100], mrr_cuts=[10], qrel='dl19', use_reordered=False) -> Dict[str, float]:
        import tempfile, subprocess, os
        if os.path.exists(qrel):
            qrel_path = qrel
        elif qrel in self.QREL_MAPPING:
            qrel_path = self.QREL_MAPPING[qrel]
        else:
            raise ValueError(f"Invalid qrel: {qrel}")

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as trec_file:
            trec_file.write(self._generate_trec(use_reordered))
            trec_file_path = trec_file.name

        def run(cmd):
            try:
                out = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                lines = out.stdout.strip().splitlines()
                if not lines: return 0.0
                last = lines[-1].split()
                return float(last[2]) if len(last) >= 3 else 0.0
            except Exception as e:
                warnings.warn(f"TREC eval failed: {e}")
                return 0.0

        base = "python -m pyserini.eval.trec_eval"
        results = {}
        for k in ndcg_cuts:
            results[f"ndcg@{k}"] = run(f"{base} -c -m ndcg_cut.{k} {qrel_path} {trec_file_path}")
        for k in map_cuts:
            results[f"map@{k}"] = run(f"{base} -c -m map_cut.{k} {qrel_path} {trec_file_path}")
        for k in mrr_cuts:
            results[f"mrr@{k}"] = run(f"{base} -c -m recip_rank {qrel_path} {trec_file_path}")

        os.remove(trec_file_path)
        return results

    # ---------- RAGAS (retriever) ----------
    def ragas_retriever(self, predictions: List[str], judge_models, use_reordered_contexts: bool = True,
                        with_entities: bool = False) -> Dict[str, float]:
        """
        Context Precision / Recall (optionally Context Entities Recall).
        """
        try:
            from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecallWithReference, ContextEntitiesRecall
            from ragas import evaluate
            from .ragas_bridge import build_ragas_dataset
            ds = build_ragas_dataset(self.documents, predictions, use_reordered=use_reordered_contexts)
            llm, embeddings = judge_models.build()

            metrics = [
                LLMContextPrecisionWithReference(llm=llm),
                LLMContextRecallWithReference(llm=llm),
            ]
            if with_entities:
                metrics.append(ContextEntitiesRecall(embeddings=embeddings))

            res = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
            out = {
                "ragas_context_precision": float(res["context_precision"]),
                "ragas_context_recall": float(res["context_recall"]),
            }
            if with_entities and "context_entities_recall" in res:
                out["ragas_context_entities_recall"] = float(res["context_entities_recall"])
            return out
        except Exception as e:
            warnings.warn(f"RAGAS retriever metrics failed: {e}")
            return {"ragas_context_precision": 0.0, "ragas_context_recall": 0.0}

    # ---------- one-call ----------
    def all(self, predictions: List[str], judge_models=None, use_reordered_contexts: bool = True) -> Dict[str, float]:
        out = {}
        out.update(self.topk_many(use_reordered=use_reordered_contexts))
        # TREC left optional—often needs qrels path
        if judge_models:
            out.update(self.ragas_retriever(predictions, judge_models, use_reordered_contexts))
        return out
