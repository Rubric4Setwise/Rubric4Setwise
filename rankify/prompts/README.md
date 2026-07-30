# rankify/prompts

Every reranker that talks to an LLM in natural language keeps its prompts here,
one YAML file per ranker, named exactly like the preset id used by
`run_pipeline.py` / `run.sh`. Changing a prompt no longer means touching model
code, and a prompt change shows up as a self-contained diff.

```
rankify/prompts/
├── __init__.py                 # load_reranker_prompts(name) -> dict
├── verify_prompts.py           # regression check against the original code
└── reranker_prompts/
    ├── monot5.yaml             ├── rankvicuna.yaml
    ├── rankt5.yaml             ├── rankzephyr.yaml
    ├── rankllama.yaml          ├── rankgpt.yaml
    ├── rank1-7b.yaml           ├── setr.yaml
    ├── rankr1-7b.yaml          ├── rank4gen.yaml
    ├── setwise-sft-7b.yaml     └── rubric4setwise.yaml
    ├── rearank-7b.yaml
    └── reasonrank-7b.yaml
```

`bm25-baseline` and `bge-reranker-large` have no YAML on purpose: the first does
no model inference at all, the second is a cross-encoder that scores raw
`(query, passage)` pairs without any instruction text.

## Usage

```python
from rankify.prompts import load_reranker_prompts

_P = load_reranker_prompts("setr")          # reads reranker_prompts/setr.yaml
SYSTEM_PROMPT = _P["system_message"]
USER_PROMPT = _P["user_messages"]["selection_only"]
```

A missing file raises `FileNotFoundError` listing the available names. There is
deliberately no silent fallback to a default prompt — that would quietly change
ranking quality.

Set `RANKIFY_PROMPTS_DIR` to load a different prompt directory, e.g. to A/B a
prompt variant without editing the shipped files:

```bash
RANKIFY_PROMPTS_DIR=/path/to/my_prompts python run_pipeline.py rank --reranker setr --gpu 0
```

## File schema

`method` first, then the prompt fields. Keys follow the ranker's own vocabulary,
so a mapping key is always the exact string the code compares against
(`prompt_mode`, language, identifier style, ...).

```yaml
method: "setr"
system_message: |-
  You are RankLLM, ...
default_prompt_mode: "selection_only"       # documentation only
user_messages:
  selection_only: |-
    I will provide you with {num} passages, ...
  selection_IRI: |-
    ...
```

| Key | Meaning |
|---|---|
| `method` | the `method` string in `reranker_presets.py` |
| `system_message` | system turn; `null` means "send no system turn" (rankvicuna) |
| `user_message` / `user_messages` | single user template, or one per mode/language |
| `input_template`, `pointwise_prompt` | the single scoring string used by pointwise rerankers (monot5, rankt5, rank1, rankllama) |
| `passage_template`, `context_item_template` | how one candidate document is rendered |
| `answer_pattern` | regex for parsing the model's answer (not a prompt, but part of the same output contract) |
| `mode_tags`, `post_message`, `prefix_message` | ranker-specific fragments; see each file's header comment |

Placeholders are filled with `str.format`, so literal braces in a template must
be escaped as `{{` / `}}`.

Two things stay outside these files on purpose: the Rank4Gen generator
descriptions (`rankify/models/rank4gen_descriptions/*.json`, sampled at runtime)
and the RankVicuna/RankZephyr few-shot pool (`data/output_v2_aug_filtered.jsonl`,
only used when `num_few_shot_examples > 0`).

## Verifying a prompt was not corrupted

These YAMLs were extracted mechanically from the string literals that used to be
inlined in `rankify/models/*.py`. The checker re-derives those literals from a
git revision and compares them, ignoring placeholder *renames* but requiring the
surrounding text and the number of placeholders to match:

```bash
python -m rankify.prompts.verify_prompts             # against the pinned baseline
python -m rankify.prompts.verify_prompts --rev <sha>  # against another revision
```

The default baseline is `BASELINE_REV` in `verify_prompts.py`, pinned to the last
commit whose reranker sources still carried the prompts inline.

Run this after editing a YAML if your intent was *not* to change the prompt (for
example after reformatting). If you deliberately reword a prompt, the checker
will report it as a mismatch — that is the expected signal, and `BASELINE_REV`
should then move forward.
