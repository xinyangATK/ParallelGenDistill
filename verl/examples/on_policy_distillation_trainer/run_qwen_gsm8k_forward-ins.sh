#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k3}
REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
LR=${LR:-3e-4}
TEST_FREQ=${TEST_FREQ:-250}
SAVE_FREQ=${SAVE_FREQ:-500}
SAVE_START_STEP=${SAVE_START_STEP:-500}
STUDENT_WORLD_SIZE=${STUDENT_WORLD_SIZE:-7}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-21}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT=${MAX_PROMPT:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( PPO_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK:-$(( ROLLOUT_SPEED_TEST_WORKER_COUNT * 2 ))}
train_epochs=${train_epochs:-8}
# Resume from the latest checkpoint under default_local_dir if present (disable | auto | resume_path).
# auto is safe on a fresh run (just starts from scratch when no checkpoint exists). The composed student
# rebuilds the frozen main_model from HF and restores only the trainable draft + optim/sched/RNG. This
# overrides run_qwen_gsm8k.sh's hardcoded trainer.resume_mode=disable (passed later in "$@").
RESUME_MODE=${RESUME_MODE:-disable}
stream_weight=${stream_weight:-1.0}
rejected_draft_stream_weight=${rejected_draft_stream_weight:-1.0}
# REJECTED_DRAFT_REVERSE=True (default) = original draftopd (reverse KL on the rejected draft token).
# =False -> FORWARD-ONLY draftopd: the model skips the reverse stream entirely (no reverse LM-head
# softmax / OOM), trains only the response forward KL. Also zero its loss weight to keep metrics clean.
REJECTED_DRAFT_REVERSE=${REJECTED_DRAFT_REVERSE:-True}
case "${REJECTED_DRAFT_REVERSE,,}" in
    false | 0 | no | off) rejected_draft_stream_weight=0.0 ;;
esac
# Independent reject sub-stream weights. rejected_draft_stream_weight = the REVERSE stream (reject-token, the
# first mismatch d at min offset per anchor); POST_REJECT_STREAM_WEIGHT = the POST-REJECT stream (discarded
# suffix / deep heads). POST_REJECT_STREAM_WEIGHT<0 (default) inherits the reverse weight -> one shared stream
# (unchanged). To score each block position ONCE under top-K, drop the reject-token duplicate: keep the reject
# stream COLLECTED (REJECTED_DRAFT_REVERSE=True) but set REVERSE_STREAM_WEIGHT=0 and POST_REJECT_STREAM_WEIGHT=1.
rejected_draft_stream_weight=${REVERSE_STREAM_WEIGHT:-${rejected_draft_stream_weight}}
POST_REJECT_STREAM_WEIGHT=${POST_REJECT_STREAM_WEIGHT:--1.0}
REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-True}
REJECTED_DRAFT_POSITION_DECAY=${REJECTED_DRAFT_POSITION_DECAY:-0.8}
# draftopd per-region loss selection. The response stream splits into response (accepted tokens) and
# reject-accept (the corrected token y at SD reject positions); the reject stream splits into reject-token
# (first mismatch d, min offset per anchor) and post-reject (the discarded suffix). Defaults below reproduce
# original draftopd (response/reject-accept = Bernoulli forward KL; reject-token/post-reject = reverse KL).
#   *_LOSS_MODE forward regions: bernoulli_fkl | topk_fkl | topk_tv ; reject regions: reverse_kl | topk_fkl |
#   topk_tv | topk_reverse_kl (topk_reverse_kl needs TOPK_FKL_STUDENT_K>0 so the rejected token d is in the top-K).
#   Top-K forward KL is the in-model FKL over the teacher/student top-K. TOPK_FKL_TEACHER_K and
#   TOPK_FKL_STUDENT_K: both > 0 -> union; only one > 0 -> that side (teacher-only is the original).
RESPONSE_LOSS_MODE=${RESPONSE_LOSS_MODE:-bernoulli_fkl}
REJECT_ACCEPT_LOSS_MODE=${REJECT_ACCEPT_LOSS_MODE:-bernoulli_fkl}
REJECT_TOKEN_LOSS_MODE=${REJECT_TOKEN_LOSS_MODE:-reverse_kl}
POST_REJECT_LOSS_MODE=${POST_REJECT_LOSS_MODE:-reverse_kl}
TOPK_FKL_TEACHER_K=${TOPK_FKL_TEACHER_K:-64}
TOPK_FKL_STUDENT_K=${TOPK_FKL_STUDENT_K:-0}
DFLASH_LM_HEAD_CHUNK_SIZE=${DFLASH_LM_HEAD_CHUNK_SIZE:-512}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.2}
ENABLE_THINKING=${ENABLE_THINKING:-False}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-""} # your draft model path.
TRAIN_JSONL=${TRAIN_JSONL:-""} # your data path.
TRAIN_JSONL_FILENAME="$(basename "$TRAIN_JSONL")"
TRAIN_JSONL_NAME="${TRAIN_JSONL_FILENAME%.jsonl}"
TRAIN_FILES="['$TRAIN_JSONL']"
TODAY=$(date +"%m-%d")
EXP_NAME=${EXP_NAME:-"ins-lr-${LR}/student-teacher-${TODAY}/${DISTILLATION_LOSS_MODE}/enable-thinking-${ENABLE_THINKING}/train-${TRAIN_JSONL_NAME}-update-accumulation-steps"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/verl-dflash-opd/${EXP_NAME}"}

exec bash "${SCRIPT_DIR}/run_qwen_gsm8k.sh" \
    data.train_files="${TRAIN_FILES}" \
    data.max_prompt_length="${MAX_PROMPT}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_draft_model_path="${DRAFT_MODEL_PATH}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_lm_head_chunk_size="${DFLASH_LM_HEAD_CHUNK_SIZE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_rejected_draft_reverse_enabled="${REJECTED_DRAFT_REVERSE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_response_loss_mode="${RESPONSE_LOSS_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_reject_accept_loss_mode="${REJECT_ACCEPT_LOSS_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_reject_token_loss_mode="${REJECT_TOKEN_LOSS_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_post_reject_loss_mode="${POST_REJECT_LOSS_MODE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_topk_fkl_teacher_k="${TOPK_FKL_TEACHER_K}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_topk_fkl_student_k="${TOPK_FKL_STUDENT_K}" \
    distillation.distillation_loss.response_loss_mode="${RESPONSE_LOSS_MODE}" \
    distillation.distillation_loss.reject_accept_loss_mode="${REJECT_ACCEPT_LOSS_MODE}" \
    distillation.distillation_loss.reject_token_loss_mode="${REJECT_TOKEN_LOSS_MODE}" \
    distillation.distillation_loss.post_reject_loss_mode="${POST_REJECT_LOSS_MODE}" \
    distillation.distillation_loss.topk_fkl_teacher_k="${TOPK_FKL_TEACHER_K}" \
    distillation.distillation_loss.topk_fkl_student_k="${TOPK_FKL_STUDENT_K}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_NUM_TOKENS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_TOKENS}" \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path="${DRAFT_MODEL_PATH}" \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    distillation.n_gpus_per_node="${TEACHER_WORLD_SIZE}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="${MAX_NUM_TOKENS}" \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_TOKENS}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.reverse_kl_weight="${REVERSE_KL_WEIGHT}" \
    distillation.distillation_loss.forward_kl_weight="${FORWARD_KL_WEIGHT}" \
    distillation.distillation_loss.response_stream_weight="${stream_weight}" \
    distillation.distillation_loss.rejected_draft_stream_weight="${rejected_draft_stream_weight}" \
    distillation.distillation_loss.post_reject_stream_weight="${POST_REJECT_STREAM_WEIGHT}" \
    distillation.distillation_loss.rejected_draft_position_decay_enabled="${REJECTED_DRAFT_POSITION_DECAY_ENABLED}" \
    distillation.distillation_loss.rejected_draft_position_decay="${REJECTED_DRAFT_POSITION_DECAY}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_start_step="${SAVE_START_STEP}" \
    trainer.n_gpus_per_node="${STUDENT_WORLD_SIZE}" \
    trainer.rollout_speed_test_worker_count="${ROLLOUT_SPEED_TEST_WORKER_COUNT}" \
    trainer.rollout_speed_test_max_samples_per_benchmark="${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK}" \
    trainer.total_epochs="${train_epochs}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    "$@"
