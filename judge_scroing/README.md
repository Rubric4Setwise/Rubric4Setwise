# judge_scroing — Rubric-based LLM-as-judge

Scores the **document set each reranker selected** against the query's rubrics,
instead of scoring a generated answer. One LLM call rates all 9 rubric
dimensions of one doc set on a 0–4 scale; every aggregate (`rubric_avg`,
`dimension_score`, `overall`) is computed in Python, so the judge only ever
supplies atomic scores plus evidence.

```
judge_scroing/
├── scoring.py                    # entry point: LLM calls, resume, Excel summary
└── prompts/
    └── rubric_scoring.yaml       # the judge prompt (key: user_message)
```

## The 9 dimensions

| Level | Dimensions | Scored per |
|---|---|---|
| Doc-Level | Relevance, Authenticity, Quality | each document |
| Set-Level | Complementarity, Redundancy, Conflict | the set as a whole |
| Global-Level | Completeness, Density, Reachability | the set as a whole |

Three rules make the scores comparable across rerankers:

- **Relevance gate** — if *every* document scores Relevance=0, the other 8
  dimensions are marked skipped (`dimension_score: null`) rather than scored.
  A set of irrelevant documents cannot earn credit for being complementary.
- **Conflict is conditional** — `null` when the set contains fewer than two
  comparable statements about the same fact. "Nothing to contradict" is not a
  perfect Conflict score.
- **Dimension-level resume** — re-running against an existing output only
  re-scores the dimensions that are missing, and never retries an explicit
  `null` (a decision, not a failure).

## Input

The judge reads a JSONL where each entry carries the candidate pool, the
per-reranker selections, and the rubrics:

```jsonc
{
  "question": "Which team won the 2019 Cricket World Cup final?",
  "answer": "England",
  "data_source": "hotpotqa",

  // full candidate pool (BM25 top-20); original_id is the join key
  "docs": [
    {"original_id": 1, "doc": "The 2019 Cricket World Cup final was ...", "title": "", "bm25_score": 18.7},
    {"original_id": 2, "doc": "...", "title": "", "bm25_score": 17.9}
  ],

  // what each reranker picked, as original_id lists in selection order
  "selected_docs": {
    "bm25-baseline":  [1, 2, 3, 4, 5],
    "setr":           [1, 4],
    "rubric4setwise": [1, 4, 9]
  },

  // rubrics per dimension
  "hybrid_rubrics": {
    "Relevance":       {"rubrics": ["Does the document discuss the 2019 CWC final?"]},
    "Authenticity":    {"rubrics": ["..."]},
    "Quality":         {"rubrics": ["..."]},
    "Complementarity": {"rubrics": ["..."]},
    "Redundancy":      {"rubrics": ["..."]},
    "Conflict":        {"rubrics": ["..."]},
    "Completeness":    {"rubrics": ["..."]},
    "Density":         {"rubrics": ["..."]},
    "Reachability":    {"rubrics": ["..."]}
  }
}
```

**`all_rankers_summary.jsonl`, produced by the pipeline's summarize phase, is
already in this shape** — its `selected_docs` is exactly a
`{ranker: [original_id, ...]}` mapping and all original input fields (including
`hybrid_rubrics`) are preserved. Feed it straight in:

```bash
python scoring.py <output_dir>/all_rankers_summary.jsonl judged.jsonl
```

The only requirement is that the *input* JSONL of the pipeline already carried
`hybrid_rubrics`; the pipeline copies the field through but never creates it.

Field names are configurable: `--docs-field`, `--selected-field`,
`--rubric-field`. Entries are identified by `(question, answer, data_source)`,
which is also the resume key.

## Output

`<output>.jsonl` — every input entry plus one `rubric_scores_by_ranker` block
per reranker:

```jsonc
{
  "question": "...", "answer": "...", "docs": [...], "selected_docs": {...},

  "rubric_scores_by_ranker": {
    "rubric4setwise": {
      "scored_doc_ids": [1, 4, 9],
      "rubric_scores": {
        "Relevance": {
          "level": "Doc-Level",
          "rubrics": [
            {"rubric": "Does the document discuss the 2019 CWC final?",
             "per_doc": [
               {"doc_id": 1, "score": 4, "evidence": "England won ...", "reason": "states the result"},
               {"doc_id": 4, "score": 3, "evidence": "...", "reason": "..."},
               {"doc_id": 9, "score": 2, "evidence": "...", "reason": "..."}
             ],
             "rubric_avg": 3.0}
          ],
          "dimension_score": 3.0
        },
        "Complementarity": {
          "level": "Set-Level",
          "rubrics": [
            {"rubric": "...",
             "set_level": {"score": 3, "evidence": "...", "reason": "..."},
             "rubric_avg": 3.0}
          ],
          "dimension_score": 3.0
        },
        "Conflict": {"level": "Set-Level", "rubrics": [], "dimension_score": null,
                     "not_applicable": "fewer than two comparable statements about the same fact; Conflict not applicable"}
      },
      "overall": {
        "Doc-Level": 3.2, "Set-Level": 2.5, "Global-Level": 3.0,
        "all_dims_mean": 2.9
      }
    }
  }
}
```

Scores are `0|1|2|3|4` or `null`. A `null` `dimension_score` means *not scored*
(relevance gate, Conflict not applicable, or a failure recorded in
`scoring_error`) — never treat it as 0 when averaging.

`<output>_stats.xlsx` — reranker × dimension summary table, rescaled to 0–100,
grouped the same way as the paper's tables.

## Usage

```bash
export LLM_APP_ID=...
export LLM_APP_KEY=...

python scoring.py all_rankers_summary.jsonl judged.jsonl \
    --limit 100 --concurrency 25
```

Useful flags: `--rankers r1,r2` (subset), `--max-docs N` (cap docs per prompt),
`--offset/--limit` (slice the input), `--model` / `--base-url` (judge model),
`--prompt-file` (a different prompt YAML, or a raw `.txt` template).

## Editing the prompt

`prompts/rubric_scoring.yaml` holds the whole template under `user_message`,
with two placeholders filled by `str.format`:

- `{docs_str}` — the selected documents as `[<original_id>] <text>` blocks
- `{rubrics_str}` — the rubrics, each labeled with its dimension

Literal JSON braces in the output-format examples are escaped as `{{ }}`
because the template goes through `str.format`. If you add examples, escape
them the same way or the prompt will fail to render.
