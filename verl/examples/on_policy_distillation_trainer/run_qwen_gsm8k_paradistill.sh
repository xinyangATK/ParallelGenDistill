#!/usr/bin/env bash
# paradistill launcher -- on-policy reverse on fresh draft samples.
#
# paradistill = the draftopd framework with two genuine differences (anchor selection + a draft-prefix
# on-policy reverse), everything else shared:
#   RESPONSE region -> top-K FORWARD KL vs the teacher at the realized prefix (response_loss_mode=topk_fkl).
#   REVERSE  region -> DRAFT-PREFIX top-K REVERSE KL: the draft samples a fresh block y_hat ~ q from its
#     heads; the frozen teacher runs ONE verify-forward ALONG the fresh block to get p(.| prefix<=anchor,
#     y_hat_<o) at each position; then top-K reverse KL between the draft head dist q_o and that teacher dist
#     (reject_token/post_reject_loss_mode=topk_reverse_kl). Both streams use the same offset decay.
#   (1) ANCHOR SELECTION: free anchors (verl_dflash_response_anchor_mode = stride_k | sampled).
#   (2) REVERSE-TOKEN SOURCE: verl_dflash_onpolicy_reverse_enabled=True (fresh y_hat + teacher verify-forward).
# top-K support is the teacher/student union (TOPK_FKL_*_K both > 0; student_k>0 REQUIRED for topk_reverse_kl).
#
# paradistill knobs:    DRAFT_SAMPLE_TEMPERATURE (T_draft; 0 = GREEDY argmax draftopd-style, >0 = sample), DRAFT_SAMPLE_SEED
#                (<0 = fresh each step; >=0 = reproducible per forward).
# Anchor knobs:  ANCHOR_MODE (stride_k | sampled), ANCHOR_SAMPLE_RATIO, ANCHOR_SEED.
# Loss / training knobs are passed straight through to run_qwen_gsm8k_forward-ins.sh via the environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANCHOR_MODE=${ANCHOR_MODE:-stride_k}
ANCHOR_SAMPLE_RATIO=${ANCHOR_SAMPLE_RATIO:-1.0}
ANCHOR_SEED=${ANCHOR_SEED:-42}

# Per-region loss modes: response = top-K forward KL; reverse = draft-prefix top-K reverse KL. The top-K
# value is precomputed in-model, so FORWARD/REVERSE_KL_WEIGHT do NOT enter either stream -- FORWARD_KL_WEIGHT=1
# is kept only to satisfy the "at least one weight > 0" config check.
export FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
export REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
export RESPONSE_LOSS_MODE=${RESPONSE_LOSS_MODE:-topk_fkl}
export REJECT_ACCEPT_LOSS_MODE=${REJECT_ACCEPT_LOSS_MODE:-topk_fkl}
export REJECT_TOKEN_LOSS_MODE=${REJECT_TOKEN_LOSS_MODE:-topk_reverse_kl}
export POST_REJECT_LOSS_MODE=${POST_REJECT_LOSS_MODE:-topk_reverse_kl}
# Top-K support: union (both > 0). student_k>0 is REQUIRED for topk_reverse_kl (so the draft modes are in S).
export TOPK_FKL_TEACHER_K=${TOPK_FKL_TEACHER_K:-8}
export TOPK_FKL_STUDENT_K=${TOPK_FKL_STUDENT_K:-8}
# Per-offset decay (offset 1..K = draft head index): weight each slot by decay^(offset-1). Applies to BOTH
# the response (forward) and reverse streams with the SAME factor. Default on at factor 0.8.
export REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-True}
export REJECTED_DRAFT_POSITION_DECAY=${REJECTED_DRAFT_POSITION_DECAY:-0.8}

DRAFT_SAMPLE_TEMPERATURE=${DRAFT_SAMPLE_TEMPERATURE:-1.0}
DRAFT_SAMPLE_SEED=${DRAFT_SAMPLE_SEED:--1}

# Resume from the latest checkpoint under default_local_dir if present (disable | auto | resume_path).
# Safe even on a fresh run: auto just starts from scratch when no checkpoint exists. The composed
# student rebuilds the frozen main_model from HF and restores only the trainable draft from the ckpt.
RESUME_MODE=${RESUME_MODE:-disable}

# ONPOLICY_REVERSE=True (default): full paradistill (response forward-KL + fresh on-policy reverse-KL).
# ONPOLICY_REVERSE=False: FORWARD-ONLY ablation -- only the response Bernoulli forward-KL on y_j; no fresh
# sampling AND the rollout-reject reverse stream is zeroed (rejected_draft_stream_weight=0).
# Reverse-stream weight for the on-policy reverse loss. paradistill's reverse value rides in the
# rejected-draft channel, so it is gated by rejected_draft_stream_weight. CRITICAL: force it nonzero here so
# it OVERRIDES forward-ins.sh, which sets rejected_draft_stream_weight=0.0 whenever REJECTED_DRAFT_REVERSE
# is false -- that flag is meant for draftopd's rollout-reject reverse and would otherwise silently ZERO
# paradistill's on-policy reverse loss (while still paying its compute). Decoupling them lets
# REJECTED_DRAFT_REVERSE=False cleanly drop only the inert segment-0 model blocks without killing the loss.
REJECTED_DRAFT_STREAM_WEIGHT=${REJECTED_DRAFT_STREAM_WEIGHT:-1.0}
ONPOLICY_REVERSE=${ONPOLICY_REVERSE:-True}
case "${ONPOLICY_REVERSE,,}" in
    true | 1 | yes | on)
        REVERSE_OVERRIDES=(
            ++actor_rollout_ref.model.override_config.verl_dflash_onpolicy_reverse_enabled=True
            ++actor_rollout_ref.model.override_config.verl_dflash_draft_sample_temperature="${DRAFT_SAMPLE_TEMPERATURE}"
            ++actor_rollout_ref.model.override_config.verl_dflash_draft_sample_seed="${DRAFT_SAMPLE_SEED}"
            distillation.distillation_loss.onpolicy_reverse_enabled=True
            distillation.distillation_loss.rejected_draft_stream_weight="${REJECTED_DRAFT_STREAM_WEIGHT}"
        )
        ;;
    *)
        # FORWARD-ONLY: on-policy reverse off + free (non-reject) anchors -> the model emits NO reverse
        # stream at all (the rollout-reject reverse only runs in reject anchor mode), so no wasted teacher
        # verify-forward. Response stays top-K forward KL. Reject modes are reset to reverse_kl so the
        # "reject-region top-K requires reject anchors / on-policy reverse" guard does not fire.
        export REJECT_TOKEN_LOSS_MODE=reverse_kl
        export POST_REJECT_LOSS_MODE=reverse_kl
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
