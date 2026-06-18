#!/usr/bin/env bash
# paradistill launcher -- on-policy reverse on fresh draft samples.
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
# Optional per-offset decay on the reverse stream: weight each fresh sample by decay^(offset-1), where
# offset (1..K) is the draft head index within its block. Default off (uniform). Set =True (and tune
# REJECTED_DRAFT_POSITION_DECAY, the decay factor) to down-weight far-offset on-policy samples.
export REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-False}

DRAFT_SAMPLE_TEMPERATURE=${DRAFT_SAMPLE_TEMPERATURE:-1.0}
DRAFT_SAMPLE_SEED=${DRAFT_SAMPLE_SEED:--1}

# Resume from the latest checkpoint under default_local_dir if present (disable | auto | resume_path).
# Safe even on a fresh run: auto just starts from scratch when no checkpoint exists. The composed
# student rebuilds the frozen main_model from HF and restores only the trainable draft from the ckpt.
RESUME_MODE=${RESUME_MODE:-disable}

# ONPOLICY_REVERSE=True (default): full paradistill (response forward-KL + fresh on-policy reverse-KL).
# ONPOLICY_REVERSE=False: FORWARD-ONLY ablation -- only the response Bernoulli forward-KL on y_j; no fresh
# sampling AND the rollout-reject reverse stream is zeroed (rejected_draft_stream_weight=0).
ONPOLICY_REVERSE=${ONPOLICY_REVERSE:-True}
case "${ONPOLICY_REVERSE,,}" in
    true | 1 | yes | on)
        REVERSE_OVERRIDES=(
            ++actor_rollout_ref.model.override_config.verl_dflash_onpolicy_reverse_enabled=True
            ++actor_rollout_ref.model.override_config.verl_dflash_draft_sample_temperature="${DRAFT_SAMPLE_TEMPERATURE}"
            ++actor_rollout_ref.model.override_config.verl_dflash_draft_sample_seed="${DRAFT_SAMPLE_SEED}"
            distillation.distillation_loss.onpolicy_reverse_enabled=True
        )
        ;;
    *)
        # FORWARD-ONLY: on-policy reverse off + free (non-reject) anchors -> the model emits NO reverse
        # stream at all (the rollout-reject reverse only runs in reject anchor mode), so no wasted
        # reverse LM-head softmax. ANCHOR_MODE must be stride_k|sampled (paradistill default), not reject.
        REVERSE_OVERRIDES=(
            ++actor_rollout_ref.model.override_config.verl_dflash_onpolicy_reverse_enabled=False
            distillation.distillation_loss.onpolicy_reverse_enabled=False
            distillation.distillation_loss.rejected_draft_stream_weight=0.0
        )
        ;;
esac

TODAY=$(date +"%m-%d")
export EXP_NAME=${EXP_NAME:-"paradistill/${ANCHOR_MODE}-Tdraft${DRAFT_SAMPLE_TEMPERATURE}/student-teacher-${TODAY}"}

exec bash "${SCRIPT_DIR}/run_qwen_gsm8k_forward-ins.sh" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_mode="${ANCHOR_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_sample_ratio="${ANCHOR_SAMPLE_RATIO}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_seed="${ANCHOR_SEED}" \
    "${REVERSE_OVERRIDES[@]}" \
    trainer.resume_mode="${RESUME_MODE}" \
    "$@"
