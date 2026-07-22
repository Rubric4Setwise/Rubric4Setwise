# judge_scroing/

LLM-as-judge scoring for the selected doc set of each reranker.

Files
-----
- `scoring.py`          — main entry point (LLM call, resume, Excel summary)
- `scoring_prompt.txt`  — prompt template (uses `{docs_str}` / `{rubrics_str}`)

Input JSONL — required fields per entry
---------------------------------------
- `question`, `answer`, `data_source`
- `docs`:            candidate document set (each item: `original_id`, `doc`, ...)
- `selected_docs`:   `{ranker_name: [original_id, ...]}`
- `hybrid_rubrics`:  `{dim_name: {"rubrics": [...], ...}}` for the 9 dimensions
  (Relevance / Authenticity / Quality — Complementarity / Redundancy / Conflict
  — Completeness / Density / Reachability).

Output
------
- `<output>.jsonl`     — every input entry augmented with
  `rubric_scores_by_ranker = {ranker: {scored_doc_ids, rubric_scores, overall}}`
- `<output>_stats.xlsx` — reranker × dimension summary (scores rescaled to 0–100).

Quick start
-----------
```bash
export LLM_APP_ID=...
export LLM_APP_KEY=...

python scoring.py input.jsonl output.jsonl --limit 100 --concurrency 25
```

Notes
-----
- 0–4 scale, all 9 dimensions in a single LLM call.
- **Relevance gate**: if every doc has Relevance=0, the remaining 8 dimensions
  are marked skipped (`dimension_score = null`) instead of being scored.
- **Conflict** is set to `null` when the doc set has fewer than two comparable
  statements about the same fact (it is not a scoring failure).
- Dimension-level **resume**: rerunning against an existing output only
  re-scores the dimensions that are missing.
