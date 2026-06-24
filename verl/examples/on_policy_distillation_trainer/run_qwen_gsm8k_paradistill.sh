#!/usr/bin/env bash
# paradistill launcher -- on-policy reverse on fresh draft samples.
#
# paradistill IS the draftopd framework with exactly two genuine differences, both behind switches:
#   (1) ANCHOR SELECTION: free anchors (verl_dflash_response_anchor_mode = stride_k | sampled) cover the
#       response independently of SD reject positions, instead of reject-tied anchors.
#   (2) REVERSE-TOKEN SOURCE: verl_dflash_onpolicy_reverse_enabled=True re-draws a FRESH on-policy draft
#       sample y_hat ~ q per (block, offset) slot in the training forward, instead of reusing the
#       rollout-cached rejected token d. Overlapping sampled-mode blocks are each kept (distinct draws).
# Everything else is the SHARED draftopd loss. The reverse-stream loss is selected by the per-region modes
# REJECT_TOKEN_LOSS_MODE / POST_REJECT_LOSS_MODE (default reverse_kl => k3 on log q(y_hat) vs log p(y_hat),
# unbiased at T_draft=1) -- the SAME path draftopd uses on d. FORWARD_KL_WEIGHT / REVERSE_KL_WEIGHT below
# now govern ONLY the response stream (Bernoulli forward KL on y_j); they no longer touch the reverse stream.
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

# Response stream = Bernoulli forward only (these weights govern ONLY the response stream now; the reverse
# stream's loss is the per-region reject mode below, default reverse_kl => k3 on the fresh on-policy samples).
export FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
export REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
# Reverse-stream loss = k3 reverse KL on the fresh sample (default). paradistill is a single reverse region,
# so REJECT_TOKEN_LOSS_MODE drives it; top-K reject modes are not supported with on-policy reverse.
export REJECT_TOKEN_LOSS_MODE=${REJECT_TOKEN_LOSS_MODE:-reverse_kl}
export POST_REJECT_LOSS_MODE=${POST_REJECT_LOSS_MODE:-reverse_kl}
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
