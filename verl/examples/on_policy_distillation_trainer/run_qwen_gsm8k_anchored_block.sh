#!/usr/bin/env bash
# Anchored Block-OPD (scalar) launcher.
#
# This is a thin wrapper over run_qwen_gsm8k_forward-ins.sh that only flips the anchor selection to
# FREE anchors (default: non-overlapping stride-K cover of the response) instead of the SD-reject-driven
# anchors. Everything else reuses the existing draftopd scalar two-stream loss, so there are NO changes
# to the rollout, engine, or loss code -- only verl_dflash_response_anchor_mode.
#
# Mapping to the Anchored Block-OPD doc (per-position / scalar form), with CORRECTED_TOKEN_FORWARD_ONLY=True:
#   agree positions          -> symmetric KL   (FORWARD_KL_WEIGHT forward + REVERSE_KL_WEIGHT reverse)
#   reject position, corr. y -> forward KL only (KL(P || q))            [response stream]
#   reject position, draft d -> reverse KL      (KL(q || P_d))          [rejected-draft stream; no position decay]
# Set CORRECTED_TOKEN_FORWARD_ONLY=False to instead apply symmetric KL to every response token.
#
# Anchor knobs:
#   ANCHOR_MODE          stride_k (default) | sampled
#   ANCHOR_SAMPLE_RATIO  keep-probability per candidate when ANCHOR_MODE=sampled
#   ANCHOR_SEED          RNG seed for ANCHOR_MODE=sampled
#
# Loss / training knobs (FORWARD_KL_WEIGHT, REVERSE_KL_WEIGHT, REJECTED_DRAFT_*, LR, world sizes, model
# and data paths, ...) are passed straight through to run_qwen_gsm8k_forward-ins.sh via the environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANCHOR_MODE=${ANCHOR_MODE:-stride_k}
ANCHOR_SAMPLE_RATIO=${ANCHOR_SAMPLE_RATIO:-1.0}
ANCHOR_SEED=${ANCHOR_SEED:-42}
CORRECTED_TOKEN_FORWARD_ONLY=${CORRECTED_TOKEN_FORWARD_ONLY:-True}  # per-position form: corrected y -> forward-only

# Default to the doc's symmetric per-anchor objective (forward + reverse KL) unless overridden.
export FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
export REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-1.0}

# Variant B (for now): no rejected-draft position decay -> every rejected token d weighted equally.
export REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-False}

TODAY=$(date +"%m-%d")
export EXP_NAME=${EXP_NAME:-"anchored-block-scalar/${ANCHOR_MODE}-fwd${FORWARD_KL_WEIGHT}-rev${REVERSE_KL_WEIGHT}/student-teacher-${TODAY}"}

exec bash "${SCRIPT_DIR}/run_qwen_gsm8k_forward-ins.sh" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_mode="${ANCHOR_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_sample_ratio="${ANCHOR_SAMPLE_RATIO}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_anchor_seed="${ANCHOR_SEED}" \
    distillation.distillation_loss.corrected_token_forward_only="${CORRECTED_TOKEN_FORWARD_ONLY}" \
    "$@"
