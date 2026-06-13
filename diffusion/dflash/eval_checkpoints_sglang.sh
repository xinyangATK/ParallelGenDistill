#!/usr/bin/env bash
# =============================================================================
# Multi-checkpoint SGLang eval for a Draft-OPD run, built on sglang_run_bench.sh.
#
# verl saves each step as an FSDP-sharded composed student under
#   ${CKPT_DIR}/global_step_<N>/actor/   (model_world_size_*_rank_*.pt + fsdp_config.json)
# SGLang can't load those shards as a speculative draft, so per step we:
#   1. MERGE the draft sub-module -> ${CKPT_DIR}/global_step_<N>/draft_model/ (clean HF
#      dir) with verl/scripts/extract_dflash_draft_from_fsdp.py (CPU, any world size).
#      This is the same `.../global_step_xxxx/draft_model` layout eval_config/two_gpu_example.yaml uses.
#   2. Emit a sglang_run_bench.sh job per checkpoint and run them through that script
#      (same engine/flags as eval_config/draftopd_zlab_b16.yaml).
#
# GPU usage follows sglang_run_bench.sh: one job == TP_SIZE GPU(s); jobs are packed
# into waves of (NUM_GPU / TP_SIZE) and each wave runs in parallel. So a 1-GPU job
# evals checkpoints sequentially; an 8-GPU job evals 8 at a time.
#
# PREREQ: the eval datasets must already be in the HF datasets cache on the blob.
# amlt/amlt_draftopd_eval.yaml runs prepare_eval_data.sh for you before this step; if
# you run this script standalone, run prepare_eval_data.sh once first (otherwise the
# blobfuse statvfs precheck makes benchmark_sglang.py's first build fail).
#
# Usage:
#   CKPT_DIR=/path/to/.../checkpoints bash eval_checkpoints_sglang.sh
#
# Inputs (env; defaults mirror eval_config/draftopd_zlab_b16.yaml):
#   CKPT_DIR            dir holding global_step_<N>/        (REQUIRED)
#   TARGET_MODEL        target model (HF id or path)        [Qwen/Qwen3-4B]
#   DRAFT_REF          reference draft for the extractor    [z-lab/Qwen3-4B-DFlash-b16]
#                       (source of config.json + dflash.py; HF id or local dir)
#   DATASETS_CSV       comma list of dataset:N              [gsm8k:128,math500:128]
#   CONCURRENCY        concurrency_num                      [1]
#   ENABLE_THINK       true/false                           [false]
#   ADD_BASE           also run target-only baseline        [false]
#   ATTENTION_BACKENDS sglang attention backend(s)          [fa3]
#   MAX_NEW_TOKENS     generation cap                       [8192]
#   TP_SIZE            tensor-parallel GPUs per job         [1]
#   MEM_FRACTION       mem_fraction_static                  [0.75]
#   STEPS              optional comma list of step numbers  [all]
#   NUM_GPU            override GPU count (else auto)        [auto]
#   EVAL_LOG_DIR       where per-step .md reports go         [${CKPT_DIR}/../eval_sglang]
#   REUSE_EXTRACTED    skip extraction if draft_model exists [true]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # diffusion/dflash
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXTRACT_PY="${REPO_ROOT}/verl/scripts/extract_dflash_draft_from_fsdp.py"

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the directory containing global_step_<N>/ }"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_REF="${DRAFT_REF:-z-lab/Qwen3-4B-DFlash-b16}"
DATASETS_CSV="${DATASETS_CSV:-gsm8k:128,math500:128}"
CONCURRENCY="${CONCURRENCY:-1}"
ENABLE_THINK="${ENABLE_THINK:-false}"
ADD_BASE="${ADD_BASE:-false}"
ATTENTION_BACKENDS="${ATTENTION_BACKENDS:-fa3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
REUSE_EXTRACTED="${REUSE_EXTRACTED:-true}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_DIR%/}/../eval_sglang}"

[ -f "${EXTRACT_PY}" ] || { echo "[ERR] extractor not found: ${EXTRACT_PY}"; exit 1; }
[ -d "${CKPT_DIR}" ]   || { echo "[ERR] CKPT_DIR does not exist: ${CKPT_DIR}"; exit 1; }
mkdir -p "${EVAL_LOG_DIR}"

# DATASETS_CSV "a:1,b:2" -> YAML inline list ["a:1", "b:2"] for the job config.
IFS=',' read -r -a _DS <<< "${DATASETS_CSV}"
_items=""
for d in "${_DS[@]}"; do d="${d//[[:space:]]/}"; [ -n "${d}" ] && _items+="\"${d}\", "; done
DATASETS_YAML="[${_items%, }]"

# --- GPU count -> jobs per wave (same auto-detect as run_benchmark.sh) ---
if [ -z "${NUM_GPU:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPU=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NUM_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
[ "${NUM_GPU:-0}" -ge 1 ] || NUM_GPU=1
SLOTS=$(( NUM_GPU / TP_SIZE )); [ "${SLOTS}" -ge 1 ] || SLOTS=1
echo "[eval] NUM_GPU=${NUM_GPU} TP_SIZE=${TP_SIZE} -> ${SLOTS} job(s) per wave"

# --- Reference draft (config.json + dflash.py) for the extractor ---
if [ -d "${DRAFT_REF}" ] && [ -f "${DRAFT_REF}/config.json" ]; then
  REF_DRAFT="${DRAFT_REF}"
else
  echo "[eval] resolving reference draft via 'hf download ${DRAFT_REF}' ..."
  REF_DRAFT="$(hf download "${DRAFT_REF}")"
fi
echo "[eval] reference draft dir: ${REF_DRAFT}"

# --- Collect steps (numeric sort, robust to underscores in the path) and extract ---
declare -A WANT=()
if [ -n "${STEPS:-}" ]; then for s in ${STEPS//,/ }; do WANT["$s"]=1; done; fi

mapfile -t STEP_DIRS < <(
  for d in "${CKPT_DIR%/}"/global_step_*/ ; do
    [ -d "$d" ] || continue
    d="${d%/}"; n="$(basename "$d")"; printf '%s\t%s\n' "${n##*_}" "$d"
  done | sort -n -k1,1 | cut -f2-
)
[ "${#STEP_DIRS[@]}" -gt 0 ] || { echo "[ERR] no global_step_* under ${CKPT_DIR}"; exit 1; }

NAMES=(); DRAFTS=()
for step_dir in "${STEP_DIRS[@]}"; do
  name="$(basename "${step_dir}")"; num="${name##*_}"
  [ -z "${STEPS:-}" ] || [ -n "${WANT[$num]:-}" ] || continue
  actor="${step_dir}/actor"
  if [ ! -f "${actor}/fsdp_config.json" ]; then
    echo "[eval] skip ${name}: incomplete checkpoint (no actor/fsdp_config.json)"; continue
  fi
  draft="${step_dir}/draft_model"
  if [ "${REUSE_EXTRACTED}" = "true" ] && [ -f "${draft}/model.safetensors" ]; then
    echo "[eval] reuse draft for ${name}: ${draft}"
  else
    echo "[eval] extract ${name} -> ${draft}"
    python "${EXTRACT_PY}" --actor-dir "${actor}" --reference-draft-dir "${REF_DRAFT}" --target-dir "${draft}"
  fi
  NAMES+=("${name}"); DRAFTS+=("${draft}")
done
[ "${#NAMES[@]}" -gt 0 ] || { echo "[ERR] no checkpoints to eval"; exit 1; }

# --- Run benchmarks in waves of SLOTS jobs (one sglang_run_bench.sh call per wave) ---
total=${#NAMES[@]}; i=0; wave=0
while [ "${i}" -lt "${total}" ]; do
  cfg="${EVAL_LOG_DIR}/_wave_${wave}.yaml"
  echo "jobs:" > "${cfg}"
  slot=0
  while [ "${slot}" -lt "${SLOTS}" ] && [ "${i}" -lt "${total}" ]; do
    start=$(( slot * TP_SIZE ))
    gpus="$(seq -s, "${start}" "$(( start + TP_SIZE - 1 ))")"
    {
      echo "  - name: ${NAMES[$i]}"
      echo "    gpu: \"${gpus}\""
      echo "    target_model: \"${TARGET_MODEL}\""
      echo "    draft_model: \"${DRAFTS[$i]}\""
      echo "    output_md: ${EVAL_LOG_DIR}/${NAMES[$i]}.md"
      echo "    dataset_name: ${DATASETS_YAML}"
      echo "    concurrency_num: \"${CONCURRENCY}\""
      echo "    add_base: ${ADD_BASE}"
      echo "    enable_think: ${ENABLE_THINK}"
      echo "    attention_backends: ${ATTENTION_BACKENDS}"
      echo "    max_new_tokens: ${MAX_NEW_TOKENS}"
      echo "    tp_size: ${TP_SIZE}"
      echo "    mem_fraction_static: ${MEM_FRACTION}"
    } >> "${cfg}"
    i=$(( i + 1 )); slot=$(( slot + 1 ))
  done
  echo "[eval] wave ${wave}: $(grep -c '^  - name:' "${cfg}") job(s) -> ${cfg}"
  bash "${SCRIPT_DIR}/sglang_run_bench.sh" --config-yaml "${cfg}"
  wave=$(( wave + 1 ))
done

echo "[eval] done. per-step reports + .run.log under: ${EVAL_LOG_DIR}"
