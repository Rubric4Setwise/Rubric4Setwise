# dr-agent

A minimal deep-research agent that iteratively **searches → reranks → browses → answers**,
with all tools served through an MCP backend and reranking delegated to the
[Rankify](../short/Rankify_only_ranker) library (including `rubric4setwise`).

```
┌─────────────────┐  tool call   ┌───────────────────────────┐
│  Search Agent   │ ───────────▶ │  MCP backend              │
│  (vLLM / GPT)   │              │  ├─ google_search (serper)│
│                 │ ◀─────────── │  ├─ browse_webpage (jina) │
└─────────────────┘   snippets   │  ├─ snippet_search (S2)   │
                                 │  └─ rankify_reranker ────▶│──▶ Rankify_only_ranker
                                 └───────────────────────────┘        (rubric4setwise, ...)
```

## Layout

```
dr-agent/
├── dr_agent/                          # library code
│   ├── agent_interface.py             # high-level SearchAgent / BrowseAgent
│   ├── workflow.py                    # workflow serving + generate-dataset CLI
│   ├── client.py                      # low-level LLM client wrapper
│   ├── mcp_backend/                   # FastMCP server exposing all tools
│   │   ├── main.py                    #   entry point: `python -m dr_agent.mcp_backend.main`
│   │   └── apis/rankify_reranker_apis.py   # hooks into Rankify_only_ranker
│   ├── tool_interface/                # tool wrappers (search, browse, rerank, chained)
│   ├── shared_prompts/                # system prompts (unified_tool_calling)
│   ├── dataset_utils/                 # loaders for evaluation datasets
│   └── web_api/                       # FastAPI serve endpoints (/chat, /chat/stream)
└── workflows/
    ├── run_agent.py                   # entry point: `serve` / `generate-dataset`
    ├── agent_config.yaml              # main config (local vLLM models + Rankify preset)
    ├── agent_config_oai.yaml          # OpenAI variant for quick debugging
    └── run_eval.sh                    # batch evaluation across multiple ranker presets
```

## Install


```text
conda env create -f rubric4setwise.yml
If there are any issues, you can refer to https://github.com/DataScienceUIBK/rankify
```
or 

```text
conda create -n rubric4setwise python=3.10 -y
cd env
pip install -r rubric4setwise.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
SERPER_API_KEY=xxx   # required for google_search
S2_API_KEY=xxx       # required for snippet_search
JINA_API_KEY=xxx     # required for browse_webpage
```

## Quick start

### 1. Launch the MCP tool backend

```bash
# Uses ../short/Rankify_only_ranker by default; override with RANKIFY_PATH if needed
python -m dr_agent.mcp_backend.main --port 8000
```

The MCP server exposes `google_search`, `browse_webpage`, `snippet_search`,
and `rankify_reranker`. The reranker tool imports Rankify from:

```
RANKIFY_PATH  (env)  or  ../short/Rankify_only_ranker  (default in rankify_reranker_apis.py)
```

### 2. Serve the search-agent LLM with vLLM

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve <your-search-agent> --port 30001
```

Edit `workflows/agent_config.yaml` to point to your model path.
For a GPU-free quick test, use the OpenAI variant:

```bash
export OPENAI_API_KEY=sk-...
python workflows/run_agent.py serve --port 8080 --config workflows/agent_config_oai.yaml
```

### 3. Interactive chat or batch evaluation

Interactive chat (opens a local UI):

```bash
python workflows/run_agent.py serve --port 8080
```

Batch evaluation on a JSONL benchmark — all knobs are on the command line
(input file, ranker preset, top-K, sample size, ports):

```bash
# Default: input = ../SetwiseEvalKit_long.jsonl, ranker = rubric4setwise, top_n = -1
bash workflows/run_eval.sh

# Try another ranker, top-5, first 20 examples
bash workflows/run_eval.sh -r bge-reranker-large -k 5 -n 20

# No reranking baseline
bash workflows/run_eval.sh -r none -n 20

# Custom input / output / ports
bash workflows/run_eval.sh \
    -i /path/to/your.jsonl \
    -o eval_output/my_run.jsonl \
    -r rubric4setwise -k -1 \
    --vllm-port 30001 --mcp-port 8000
```

`bash workflows/run_eval.sh -h` prints the full option list.

Under the hood it calls `run_agent.py generate-dataset <jsonl>`; any JSONL with
a `question` (or `query`/`problem`) field works — `dr_agent/dataset_utils/load_dataset.py::load_custom_json_data`
handles the mapping automatically, so `SetwiseEvalKit_long.jsonl` needs no conversion.

## Using the Rankify ranker

Any Rankify preset can be enabled by setting `ranker_preset` in
`workflows/agent_config.yaml` (or overriding on the CLI). After each search
tool call, the agent chains a `rankify_reranker` MCP call that reranks / selects
documents before feeding them back to the LLM.

| Category            | Presets                                                                                                  | `top_n` |
|---------------------|----------------------------------------------------------------------------------------------------------|---------|
| Cross-encoder       | `bge-reranker-large`                                                                                     | `5`     |
| Pointwise / listwise| `rankllama`, `monot5`, `rankt5`, `rankvicuna`, `rankzephyr`, `rankgpt`                                    | `5`     |
| Reasoning           | `rank1-7b`, `rearank-7b`, `reasonrank-7b`                                                                | `5`     |
| Setwise             | `setwise-sft-7b`                                                                                         | `5`     |
| Set selection       | `setr`, `rank4gen`, **`rubric4setwise`**, **`rubric4setwise-llama8b`**                                   | `-1`    |

Example (in `agent_config.yaml`):

```yaml
ranker_preset: rubric4setwise
ranker_top_n: -1        # set-selection methods pick their own subset
ranker_timeout: 1200
```

Or from the CLI (preferred — this is exactly what `run_eval.sh` does):

```bash
python workflows/run_agent.py generate-dataset ../SetwiseEvalKit_long.jsonl \
    --config workflows/agent_config.yaml \
    --config-overrides "ranker_preset=rubric4setwise,ranker_top_n=-1" \
    --output eval_output/SetwiseEvalKit_long_rubric4setwise.jsonl
```

### How it wires together

* `workflows/run_agent.py` reads `ranker_preset` from the config and, when set,
  wraps every search tool in a `ChainedTool(search → RankifyRerankerTool)`
  (see `dr_agent/tool_interface/`).
* `RankifyRerankerTool` issues an MCP call to `rankify_reranker` on the MCP backend.
* `dr_agent/mcp_backend/apis/rankify_reranker_apis.py` looks up the preset in
  `rankify.config.reranker_presets.RERANKER_PRESETS`, lazily instantiates the
  `Reranking` model (cached in-process), runs it on the search snippets, and
  returns either a ranked list (top-N) or a selected subset (set-selection methods).

To register a new ranker, add its preset to
`Rankify_only_ranker/rankify/config/reranker_presets.py`; no changes are needed
on the dr-agent side other than referencing the new preset name.
