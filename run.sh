#!/bin/bash
###############################################################################
# Short-Form Reranker Pipeline
#
# Three phases, both GPU phases are data-parallel and run one job at a time:
#   1. Ranking     - per reranker: shard the dataset across N GPUs, merge, score
#   2. Generation  - per reranker x generator: shard across N GPUs, merge, score
#   3. Summarize   - aggregate metrics across all rerankers
#
# Every GPU loads its own full copy of the model and handles 1/N of the data;
# jobs are processed strictly one after another, so a job always has all N GPUs.
#
# Usage:  bash run.sh
#
# Note: after changing NUM_GPUS, delete any leftover ranker_cache_shard_*.jsonl
# and output_shard_*.jsonl first — stale shards were cut for a different
# num_shards and merging them would duplicate or drop entries.
###############################################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_pipeline.py"

# ============================================================
# Config (edit these)
# ============================================================
INPUT_FILE="/path/to/your_data_bm25top20.jsonl"      # JSONL with BM25 top-20 docs
OUTPUT_DIR="./outputs/short_reranker"
OUTPUT_FILENAME="output.jsonl"
NUM_ENTRIES="all"        # "all" or an integer, e.g. "100"

# GPUs
NUM_GPUS=4
GPU_IDS=(0 1 2 3)

# Generator
GENERATORS=("meta-llama/Llama-3.1-8B-Instruct")
GENERATOR_METHOD="basic-rag"
GENERATOR_BACKEND="vllm"
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.4

# Ranking / eval
TOP_K=5
TOP_K_FOR_EVAL="1 3 5 10"
NDCG_CUTS="3 5 10"
EVALUATE_RANKER="true"
EVALUATE_METRICS="true"
SAVE_INDIVIDUAL_SCORES="true"

# vLLM sampling
MAX_TOKENS=20
TEMPERATURE=0
TOP_P=0.9
REPETITION_PENALTY=1.3

# Rerankers covered by this release (matches the paper's Ranker table).
# Note: `rubric4setwise` requires the input JSONL to carry a `hybrid_rubrics`
# (or legacy `rubric`) field per query. Leave it commented out if your data
# does not contain pre-computed rubrics.
RERANKERS=(
    "bm25-baseline"          # Only Retrieval baseline
    "bge-reranker-large"
    "monot5"
    "rankt5"
    "rankllama"
    "rankvicuna"
    "rankzephyr"
    "setwise-sft-7b"
    "rank1-7b"
    "rearank-7b"
    "reasonrank-7b"
    "setr"
    "rank4gen"
    # "rubric4setwise"       # Ours: rubric-guided set selection (needs hybrid_rubrics)
)

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ============================================================
# Small helpers
# ============================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log()   { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()    { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} $*"; }
err()   { echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $*"; }
phase() { echo -e "\n${CYAN}==== $* ====${NC}\n"; }

# ============================================================
# Phase 1: Ranking (data-parallel shards per reranker)
# ============================================================
phase "PHASE 1: RANKING  (${#RERANKERS[@]} rerankers x ${NUM_GPUS} shards)"
RANK_START=$(date +%s)

for reranker in "${RERANKERS[@]}"; do
    CACHE_FILE="${OUTPUT_DIR}/${reranker}/ranker_cache.jsonl"
    if [ -f "$CACHE_FILE" ]; then
        log "${reranker}: cache exists, skip"
        continue
    fi

    log "Ranking with ${reranker}"

    declare -a SHARD_PIDS; SHARD_FAIL=0
    for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ID=${GPU_IDS[$shard_id]}
        SHARD_FILE="${OUTPUT_DIR}/${reranker}/ranker_cache_shard_${shard_id}.jsonl"
        if [ -f "$SHARD_FILE" ]; then SHARD_PIDS[$shard_id]=0; continue; fi

        LOG_FILE="${LOG_DIR}/rank_${reranker}_shard${shard_id}.log"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python "${PYTHON_SCRIPT}" rank \
            --reranker "$reranker" --gpu "$GPU_ID" \
            --input "$INPUT_FILE" --output-dir "$OUTPUT_DIR" \
            --num-entries "$NUM_ENTRIES" --top-k "$TOP_K" \
            --top-k-for-eval $TOP_K_FOR_EVAL --ndcg-cuts $NDCG_CUTS \
            --shard-id "$shard_id" --num-shards "$NUM_GPUS" \
            --no-evaluate-ranker > "$LOG_FILE" 2>&1 &
        SHARD_PIDS[$shard_id]=$!
        log "  shard ${shard_id} -> GPU ${GPU_ID} (pid=${SHARD_PIDS[$shard_id]})"
    done

    for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
        pid=${SHARD_PIDS[$shard_id]}
        if [ "$pid" -ne 0 ]; then
            wait "$pid" 2>/dev/null
            [ $? -ne 0 ] && { err "  shard ${shard_id} failed"; SHARD_FAIL=1; }
        fi
    done
    unset SHARD_PIDS
    [ "$SHARD_FAIL" -eq 1 ] && { err "${reranker} failed, skip merge"; continue; }

    MERGE_CMD=(python "${PYTHON_SCRIPT}" merge-shards
        --reranker "$reranker" --output-dir "$OUTPUT_DIR"
        --num-shards "$NUM_GPUS" --input "$INPUT_FILE"
        --num-entries "$NUM_ENTRIES" --top-k "$TOP_K"
        --top-k-for-eval $TOP_K_FOR_EVAL --ndcg-cuts $NDCG_CUTS)
    [ "$EVALUATE_RANKER" = "true" ] && MERGE_CMD+=(--evaluate-ranker) || MERGE_CMD+=(--no-evaluate-ranker)
    "${MERGE_CMD[@]}" > "${LOG_DIR}/rank_${reranker}_merge.log" 2>&1 \
        && ok "  ${reranker} done" || err "  ${reranker} merge failed"
done

ok "Ranking done in $(( ($(date +%s) - RANK_START) / 60 )) min"

# ============================================================
# Phase 2: Generation (data-parallel shards per reranker x generator)
# ============================================================
phase "PHASE 2: GENERATION  ($((${#RERANKERS[@]} * ${#GENERATORS[@]})) jobs x ${NUM_GPUS} shards)"
GEN_START=$(date +%s)

for reranker in "${RERANKERS[@]}"; do
    for generator in "${GENERATORS[@]}"; do
        GEN_NAME="${generator##*/}"
        GEN_DIR="${generator//\//_}"
        GEN_OUT_DIR="${OUTPUT_DIR}/${reranker}/${GEN_DIR}"
        OUT_FILE="${GEN_OUT_DIR}/${OUTPUT_FILENAME}"
        [ -f "$OUT_FILE" ] && { log "skip ${reranker}+${GEN_NAME}"; continue; }

        CACHE_FILE="${OUTPUT_DIR}/${reranker}/ranker_cache.jsonl"
        [ ! -f "$CACHE_FILE" ] && { warn "no cache for ${reranker}"; continue; }

        log "Generating with ${reranker}+${GEN_NAME}"

        declare -a GEN_PIDS; GEN_FAIL=0
        for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
            GPU_ID=${GPU_IDS[$shard_id]}
            SHARD_FILE="${GEN_OUT_DIR}/output_shard_${shard_id}.jsonl"
            if [ -f "$SHARD_FILE" ]; then GEN_PIDS[$shard_id]=0; continue; fi

            LOG_FILE="${LOG_DIR}/gen_${reranker}_${GEN_NAME}_shard${shard_id}.log"
            CUDA_VISIBLE_DEVICES="$GPU_ID" python "${PYTHON_SCRIPT}" generate \
                --reranker "$reranker" --generator "$generator" --gpu "$GPU_ID" \
                --input "$INPUT_FILE" --output-dir "$OUTPUT_DIR" \
                --num-entries "$NUM_ENTRIES" \
                --generator-method "$GENERATOR_METHOD" --generator-backend "$GENERATOR_BACKEND" \
                --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --top-k "$TOP_K" --max-tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" --top-p "$TOP_P" \
                --repetition-penalty "$REPETITION_PENALTY" \
                --shard-id "$shard_id" --num-shards "$NUM_GPUS" \
                --no-evaluate-metrics > "$LOG_FILE" 2>&1 &
            GEN_PIDS[$shard_id]=$!
            log "  shard ${shard_id} -> GPU ${GPU_ID} (pid=${GEN_PIDS[$shard_id]})"
        done

        for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
            pid=${GEN_PIDS[$shard_id]}
            if [ "$pid" -ne 0 ]; then
                wait "$pid" 2>/dev/null
                [ $? -ne 0 ] && { err "  shard ${shard_id} failed"; GEN_FAIL=1; }
            fi
        done
        unset GEN_PIDS
        [ "$GEN_FAIL" -eq 1 ] && { err "${reranker}+${GEN_NAME} failed, skip merge"; continue; }

        GEN_MERGE_CMD=(python "${PYTHON_SCRIPT}" merge-gen-shards
            --reranker "$reranker" --generator "$generator"
            --output-dir "$OUTPUT_DIR" --num-shards "$NUM_GPUS" --top-k "$TOP_K")
        [ "$EVALUATE_METRICS" = "true" ] && GEN_MERGE_CMD+=(--evaluate-metrics) || GEN_MERGE_CMD+=(--no-evaluate-metrics)
        [ "$SAVE_INDIVIDUAL_SCORES" = "true" ] && GEN_MERGE_CMD+=(--save-individual-scores) || GEN_MERGE_CMD+=(--no-save-individual-scores)
        "${GEN_MERGE_CMD[@]}" > "${LOG_DIR}/gen_${reranker}_${GEN_NAME}_merge.log" 2>&1 \
            && ok "  ${reranker}+${GEN_NAME} done" || err "  ${reranker}+${GEN_NAME} merge failed"
    done
done
ok "Generation done in $(( ($(date +%s) - GEN_START) / 60 )) min"

# ============================================================
# Phase 3: Summarize
# ============================================================
phase "PHASE 3: SUMMARIZE"
python "${PYTHON_SCRIPT}" summarize \
    --input "$INPUT_FILE" --output-dir "$OUTPUT_DIR" \
    --num-entries "$NUM_ENTRIES" \
    --rerankers "${RERANKERS[@]}" --generators "${GENERATORS[@]}"

ok "All done. Outputs at: ${OUTPUT_DIR}"
