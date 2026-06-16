#!/usr/bin/env bash
# Anchored Block-OPD (paradistill) launcher -- on-policy reverse on fresh draft samples.
#
# Free anchors (verl_dflash_response_anchor_mode, default stride_k) cover the response independently of
# SD reject positions, reusing the draftopd scalar two-stream loss (no rollout/engine changes -- only
# the anchor selection and the reverse-stream source). Response tokens use Bernoulli FORWARD KL only;
# the reverse stream is a FRESH on-policy draft sample y_hat re-drawn per (block, offset) slot during
# training and scored against the frozen teacher (reverse KL). Overlapping sampled-mode blocks are each
# kept -- every re-sample is a distinct on-policy draw.
#
# paradistill knobs:    DRAFT_SAMPLE_TEMPERATURE (T_draft, 1.0 = genuine q-samples / unbiased k3 reverse),
#                DRAFT_SAMPLE_SEED (<0 = fresh each step; >=0 = reproducible per forward).
# Anchor knobs:  ANCHOR_MODE (stride_k | sampled), ANCHOR_SAMPLE_RATIO, ANCHOR_SEED.
# Loss / training knobs are passed straight through to run_qwen_gsm8k_forward-ins.sh via the environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANCHOR_MODE=${ANCHOR_MODE:-stride_k}
ANCHOR_SAMPLE_RATIO=${ANCHOR_SAMPLE_RATIO:-1.0}
ANCHOR_SEED=${ANCHOR_SEED:-42}

# Response stream = Bernoulli forward only; the reverse term comes from the fresh on-policy samples.
export FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
export REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
# No rejected-draft position decay -> every reverse token weighted equally.
export REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-False}

DRAFT_SAMPLE_TEMPERATURE=${DRAFT_SAMPLE_TEMPERATURE:-1.0}
DRAFT_SAMPLE_SEED=${DRAFT_SAMPLE_SEED:--1}

TODAY=$(date +"%m-%d")
export EXP_NAME=${EXP_NAME:-"paradistill/${ANCHOR_MODE}-Tdraft${DRAFT_SAMPLE_TEMPERATURE}/student-teacher-${TODAY}"}

exec bash "${SCRIPT_DIR}/run_qwen_gsm8k_forward-ins.sh" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_mode="${ANCHOR_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_sample_ratio="${ANCHOR_SAMPLE_RATIO}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_seed="${ANCHOR_SEED}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_onpolicy_reverse_enabled=True \
    ++actor_rollout_ref.model.override_config.verl_dflash_draft_sample_temperature="${DRAFT_SAMPLE_TEMPERATURE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_draft_sample_seed="${DRAFT_SAMPLE_SEED}" \
    distillation.distillation_loss.onpolicy_reverse_enabled=True \
    "$@"
