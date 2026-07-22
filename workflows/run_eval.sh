#!/bin/bash
###############################################################################
# Run the agent on a JSONL benchmark with a chosen Rankify ranker.
#
# Prerequisites (in separate shells):
#   CUDA_VISIBLE_DEVICES=0 vllm serve <search-agent-model> --port 30001
#   CUDA_VISIBLE_DEVICES=1 python -m dr_agent.mcp_backend.main --port 8000
# (Or let `run_agent.py` auto-launch them the first time.)
#
# Usage:
#   bash workflows/run_eval.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   -i, --input       Path to input JSONL (default: ../SetwiseEvalKit_long.jsonl)
#   -o, --output      Path to output JSONL
#                     (default: eval_output/<basename>_<ranker>.jsonl)
#   -r, --ranker      Ranker preset (default: rubric4setwise)
#                     Special: "none" disables reranking.
#   -k, --top-n       Top-N to keep after reranking (default: -1 = all;
#                     ignored for set-selection presets)
#   -n, --num         Limit to first N examples (default: all)
#   -c, --config      Base config yaml (default: workflows/agent_config.yaml)
#       --vllm-port   vLLM search-agent port (default: 30001)
#       --mcp-port    MCP backend port      (default: 8000)
#       --batch-size  Concurrent examples   (default: 1)
#
# Example:
#   bash workflows/run_eval.sh -r rubric4setwise -k -1 -n 20
#   bash workflows/run_eval.sh -r bge-reranker-large -k 5
#   bash workflows/run_eval.sh -r none -i /path/to/data.jsonl -o out.jsonl
###############################################################################

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# -------------------- Defaults --------------------
INPUT="$(cd "$SCRIPT_DIR/../.." && pwd)/SetwiseEvalKit_long.jsonl"
OUTPUT=""
RANKER="rubric4setwise"
TOP_N=-1
NUM_EXAMPLES=""
CONFIG_YAML="workflows/agent_config.yaml"
VLLM_PORT=30001
MCP_PORT=8000
BATCH_SIZE=1

# -------------------- Parse args --------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)      INPUT="$2"; shift 2 ;;
        -o|--output)     OUTPUT="$2"; shift 2 ;;
        -r|--ranker)     RANKER="$2"; shift 2 ;;
        -k|--top-n)      TOP_N="$2"; shift 2 ;;
        -n|--num)        NUM_EXAMPLES="$2"; shift 2 ;;
        -c|--config)     CONFIG_YAML="$2"; shift 2 ;;
        --vllm-port)     VLLM_PORT="$2"; shift 2 ;;
        --mcp-port)      MCP_PORT="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)       sed -n '2,32p' "$0"; exit 0 ;;
        *)               echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "[ERROR] Input file not found: $INPUT" >&2
    exit 1
fi

# Default output path derived from input basename + ranker
if [[ -z "$OUTPUT" ]]; then
    BASE="$(basename "$INPUT" .jsonl)"
    mkdir -p eval_output
    OUTPUT="eval_output/${BASE}_${RANKER}.jsonl"
fi
mkdir -p "$(dirname "$OUTPUT")"

# -------------------- Build overrides --------------------
OVERRIDES="search_agent_base_url=http://localhost:${VLLM_PORT}/v1"
OVERRIDES+=",mcp_port=${MCP_PORT}"

if [[ "$RANKER" == "none" || -z "$RANKER" ]]; then
    :
elif [[ "$TOP_N" -gt 0 ]] 2>/dev/null; then
    OVERRIDES+=",ranker_preset=${RANKER},ranker_top_n=${TOP_N}"
else
    OVERRIDES+=",ranker_preset=${RANKER},ranker_top_n=-1"
fi

# -------------------- Info --------------------
echo "###################################################################"
echo "# Input     : $INPUT"
echo "# Output    : $OUTPUT"
echo "# Ranker    : $RANKER   top_n=$TOP_N"
echo "# Config    : $CONFIG_YAML"
echo "# vLLM port : $VLLM_PORT      MCP port: $MCP_PORT"
echo "# Overrides : $OVERRIDES"
echo "###################################################################"

# -------------------- Run --------------------
CMD="python workflows/run_agent.py generate-dataset \"$INPUT\""
CMD+=" --config \"$CONFIG_YAML\""
CMD+=" --config-overrides \"$OVERRIDES\""
CMD+=" --output \"$OUTPUT\""
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --use-cache"
[[ -n "$NUM_EXAMPLES" ]] && CMD+=" --num-examples $NUM_EXAMPLES"

echo "Run: $CMD"
START=$(date +%s)
eval "$CMD"
ELAPSED=$(( $(date +%s) - START ))
echo "[DONE] $((ELAPSED / 60))m $((ELAPSED % 60))s   -> $OUTPUT"
