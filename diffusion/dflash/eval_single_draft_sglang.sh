#!/usr/bin/env bash
# Standalone SGLang benchmark of a SINGLE draft model (e.g. a PRETRAINED DFLASH draft) on a target.
# Unlike eval_checkpoints_sglang.sh, this needs NO training checkpoints and NO FSDP extraction -- it
# just generates a one-job config and runs it through sglang_run_bench.sh (same server launch /
# speculative args / .md report). draft_model may be an HF id (z-lab/Qwen3-4B-DFlash-b16) or a local
# dir (e.g. an already-extracted global_step_<N>/draft_model).
#
# Env knobs:
#   DRAFT_MODEL          draft to eval (HF id or path)         (REQUIRED)
#   TARGET_MODEL         target model (HF id or path)          [Qwen/Qwen3-4B]
#   DATASETS_CSV         comma list of dataset:N               [gsm8k:128,math500:128]
#   EVAL_LOG_DIR         where the .md report + config go      [<script dir>/sglang-logs]
#   CONCURRENCY          concurrency_num                       [1]
#   ADD_BASE             also run target-only baseline         [false]
#   ENABLE_THINK         tokenizer enable_thinking             [false]
#   ATTENTION_BACKEND    sglang attention backend              [fa3]
#   MAX_NEW_TOKENS       generation cap                        [8192]
#   TP_SIZE              tensor-parallel size                  [1]
#   MEM_FRACTION_STATIC  sglang static mem fraction            [0.75]
#   TEMP                 sampling temperature (0 = greedy)     [0.0]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:?Set DRAFT_MODEL to the draft model to eval (HF id or local path)}"
DATASETS_CSV="${DATASETS_CSV:-gsm8k:128,math500:128}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${SCRIPT_DIR}/sglang-logs}"
CONCURRENCY="${CONCURRENCY:-1}"
ADD_BASE="${ADD_BASE:-false}"
ENABLE_THINK="${ENABLE_THINK:-false}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
TEMP="${TEMP:-0.0}"

mkdir -p "${EVAL_LOG_DIR}"

# DATASETS_CSV "a:1,b:2" -> YAML inline list ["a:1", "b:2"]
IFS=',' read -r -a _DS <<< "${DATASETS_CSV}"
_items=""
for d in "${_DS[@]}"; do _items+="\"${d}\", "; done
DATASETS_YAML="[${_items%, }]"

DRAFT_TAG="$(basename "${DRAFT_MODEL}")"
OUTPUT_MD="${EVAL_LOG_DIR}/draft_${DRAFT_TAG}.md"
CONFIG_YAML="${EVAL_LOG_DIR}/_single_draft_${DRAFT_TAG}.yaml"

cat > "${CONFIG_YAML}" <<EOF
jobs:
  - name: ${DRAFT_TAG}
    gpu: 0
    target_model: ${TARGET_MODEL}
    draft_model: ${DRAFT_MODEL}
    output_md: ${OUTPUT_MD}
    dataset_name: ${DATASETS_YAML}
    concurrency_num: "${CONCURRENCY}"
    add_base: ${ADD_BASE}
    enable_think: ${ENABLE_THINK}
    attention_backends: ${ATTENTION_BACKEND}
    max_new_tokens: ${MAX_NEW_TOKENS}
    tp_size: ${TP_SIZE}
    mem_fraction_static: ${MEM_FRACTION_STATIC}
    temp: ${TEMP}
EOF

echo "[eval-draft] draft:  ${DRAFT_MODEL}"
echo "[eval-draft] target: ${TARGET_MODEL}"
echo "[eval-draft] datasets: ${DATASETS_CSV}"
echo "[eval-draft] config: ${CONFIG_YAML}"
echo "[eval-draft] report: ${OUTPUT_MD}"
bash "${SCRIPT_DIR}/sglang_run_bench.sh" --config-yaml "${CONFIG_YAML}"
echo "[eval-draft] done -> ${OUTPUT_MD}"
