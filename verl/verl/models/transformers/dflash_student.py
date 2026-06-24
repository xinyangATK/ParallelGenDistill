# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import logging
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Callable, Optional, cast

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DFLASH_ATTENTION_IMPL_IDS = {
    "eager": 0,
    "sdpa": 1,
    "flex_attention": 2,
}

# Loss modes that compute an in-model top-K divergence from the teacher's full-vocab logits (need the teacher
# logits retained). Forward regions: topk_fkl / topk_tv. Reject regions: topk_fkl / topk_reverse_kl.
_TOPK_FORWARD_MODES = ("topk_fkl", "topk_tv")
_TOPK_REJECT_MODES = ("topk_fkl", "topk_reverse_kl")

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    BlockMask = None
    create_block_mask = None
    FLEX_ATTENTION_AVAILABLE = False


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def resolve_target_layer_ids(main_model: PreTrainedModel, draft_model: PreTrainedModel) -> list[int]:
    if hasattr(draft_model, "target_layer_ids") and getattr(draft_model, "target_layer_ids") is not None:
        return [int(layer_id) for layer_id in getattr(draft_model, "target_layer_ids")]

    draft_config = getattr(draft_model, "config", None)
    dflash_config = getattr(draft_config, "dflash_config", None) if draft_config is not None else None
    if isinstance(dflash_config, dict) and dflash_config.get("target_layer_ids") is not None:
        return [int(layer_id) for layer_id in dflash_config["target_layer_ids"]]

    num_target_layers = int(getattr(main_model.config, "num_hidden_layers"))
    num_draft_layers = int(getattr(draft_model.config, "num_hidden_layers"))
    return build_target_layer_ids(num_target_layers=num_target_layers, num_draft_layers=num_draft_layers)


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len

    q_indices = torch.arange(q_len, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(batch_size, 1, num_blocks, 1).repeat_interleave(block_size, dim=2)
    mask_context = (kv_indices < seq_len) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= seq_len
    kv_block_ids = (kv_indices - seq_len) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(batch_size, 1, num_blocks, 1).repeat_interleave(block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> BlockMask:
    if not FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise RuntimeError("flex_attention is not available for DFLASH block mask construction.")

    batch_size, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=num_blocks - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < num_blocks
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    return create_block_mask(
        dflash_mask_mod, B=batch_size, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
    )


def _set_config_attn_implementation(config: Any, attn_impl: str) -> None:
    for attr_name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
        if hasattr(config, attr_name):
            setattr(config, attr_name, attn_impl)


class ComposedDFlashStudentForCausalLM(PreTrainedModel):
    """Train-only composed model: frozen target model + trainable DFLASH draft."""

    config_class = PretrainedConfig
    base_model_prefix = "main_model"
    supports_gradient_checkpointing = True

    @staticmethod
    def _normalize_wrapper_attn_implementation(config: PretrainedConfig) -> PretrainedConfig:
        """Force wrapper-level attention impl to eager for HF init checks.

        The composed student wrapper itself has no native attention blocks, but the
        inherited PreTrainedModel init path still validates attn implementation and
        rejects flash_attention_2 for unknown architectures.
        """
        wrapper_config = copy.deepcopy(config)
        for attr_name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
            if hasattr(wrapper_config, attr_name):
                setattr(wrapper_config, attr_name, "eager")
        return wrapper_config

    def __init__(
        self,
        config: PretrainedConfig,
        main_model: PreTrainedModel,
        draft_model: PreTrainedModel,
    ):
        super().__init__(self._normalize_wrapper_attn_implementation(config))
        self.main_model = main_model
        self.draft_model = draft_model
        self.target_layer_ids = resolve_target_layer_ids(main_model=main_model, draft_model=draft_model)
        self._configure_draft_attention()
        self.freeze_main_model()

    def _configure_draft_attention(self) -> None:
        requested_impl = (
            getattr(self.config, "verl_dflash_attention_impl", None)
            or os.getenv("VERL_DFLASH_ATTENTION_IMPL")
            or "flex_attention"
        )
        requested_impl = str(requested_impl).lower()
        if requested_impl == "auto":
            requested_impl = "flex_attention"

        if requested_impl == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            logger.warning("flex_attention is not available; falling back to sdpa for DFLASH draft attention.")
            requested_impl = "sdpa"

        if requested_impl not in DFLASH_ATTENTION_IMPL_IDS:
            logger.warning(
                "Unsupported DFLASH draft attention implementation %s. "
                "Arbitrary DFLASH block masks are only wired for flex_attention, sdpa, or eager; falling back to %s.",
                requested_impl,
                "flex_attention" if FLEX_ATTENTION_AVAILABLE else "sdpa",
            )
            requested_impl = "flex_attention" if FLEX_ATTENTION_AVAILABLE else "sdpa"

        draft_config = getattr(self.draft_model, "config", None)
        if draft_config is not None:
            _set_config_attn_implementation(draft_config, requested_impl)

    def freeze_main_model(self) -> None:
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.main_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen teacher in eval mode even while training the draft module.
        self.main_model.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for wrapped submodules."""
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        kwargs = gradient_checkpointing_kwargs or {}
        for module in (self.main_model, self.draft_model):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for wrapped submodules."""
        super().gradient_checkpointing_disable()
        for module in (self.main_model, self.draft_model):
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Optional[Callable[..., Any]] = None,
    ):
        """HF hook for toggling gradient checkpointing support."""
        setattr(self, "gradient_checkpointing", enable)
        if hasattr(self, "config"):
            setattr(self.config, "gradient_checkpointing", enable)

        for child in (self.main_model, self.draft_model):
            if hasattr(child, "_set_gradient_checkpointing"):
                try:
                    if gradient_checkpointing_func is None:
                        child._set_gradient_checkpointing(enable=enable)
                    else:
                        child._set_gradient_checkpointing(
                            enable=enable,
                            gradient_checkpointing_func=gradient_checkpointing_func,
                        )
                except TypeError:
                    # Compatibility for older HF-style signatures used by some remote-code models.
                    child._set_gradient_checkpointing(enable)
            else:
                setattr(child, "gradient_checkpointing", enable)
                if hasattr(child, "config"):
                    setattr(child.config, "gradient_checkpointing", enable)

    def get_input_embeddings(self):
        return self.main_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.main_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.main_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.main_model.set_output_embeddings(value)

    def _extract_target_hidden(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected_states = []
        for layer_id in self.target_layer_ids:
            hidden_index = int(layer_id) + 1
            if hidden_index >= len(hidden_states):
                raise ValueError(
                    f"target_layer_id={layer_id} out of range for hidden_states size={len(hidden_states)}"
                )
            selected_states.append(hidden_states[hidden_index])
        return torch.cat(selected_states, dim=-1)

    def _resolve_position_ids(
        self,
        noise_embedding: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        if position_ids is not None:
            return position_ids
        batch_size, seq_len = noise_embedding.shape[:2]
        position_ids_tensor = (
            torch.arange(seq_len, device=noise_embedding.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        return cast(torch.LongTensor, position_ids_tensor)

    def _get_mask_token_id(self) -> int:
        mask_token_id = getattr(self.draft_model, "mask_token_id", None)
        if mask_token_id is None:
            draft_config = getattr(self.draft_model, "config", None)
            dflash_config = getattr(draft_config, "dflash_config", None) if draft_config is not None else None
            if isinstance(dflash_config, dict):
                mask_token_id = dflash_config.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError(
                "DFLASH OPD requires draft_model.mask_token_id or config.dflash_config['mask_token_id']."
            )
        return int(mask_token_id)

    def _get_block_size(self) -> int:
        block_size = getattr(self.draft_model, "block_size", None)
        if block_size is None:
            block_size = getattr(getattr(self.draft_model, "config", None), "block_size", None)
        if block_size is None:
            raise ValueError("DFLASH OPD requires draft_model.block_size or draft_model.config.block_size.")
        return int(block_size)

    def _get_lm_head_chunk_size(self) -> int:
        chunk_size = getattr(self.config, "verl_dflash_lm_head_chunk_size", None)
        if chunk_size is None:
            chunk_size = os.getenv("VERL_DFLASH_LM_HEAD_CHUNK_SIZE", "2048")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"verl_dflash_lm_head_chunk_size must be positive, got {chunk_size}.")
        return chunk_size

    def _get_response_anchor_stride(self) -> int:
        stride = getattr(self.config, "verl_dflash_response_anchor_stride", None)
        if stride is None:
            stride = os.getenv("VERL_DFLASH_RESPONSE_ANCHOR_STRIDE", "1")
        stride = int(stride)
        if stride <= 0:
            raise ValueError(f"verl_dflash_response_anchor_stride must be positive, got {stride}.")
        return stride

    def _get_random_response_anchor_enabled(self) -> bool:
        value = getattr(self.config, "verl_dflash_random_response_anchor_enabled", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RANDOM_RESPONSE_ANCHOR_ENABLED", "0")
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_random_response_anchor_seed(self) -> int:
        value = getattr(self.config, "verl_dflash_random_response_anchor_seed", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RANDOM_RESPONSE_ANCHOR_SEED", "42")
        return int(value)

    def _get_response_anchor_mode(self) -> str:
        """paradistill anchor selection: 'reject' (default, SD-reject-driven) or a free mode
        ('stride_k' | 'sampled') that covers the response independently of reject positions."""
        value = getattr(self.config, "verl_dflash_response_anchor_mode", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RESPONSE_ANCHOR_MODE", "reject")
        return str(value)

    def _get_response_anchor_sample_ratio(self) -> float:
        value = getattr(self.config, "verl_dflash_response_anchor_sample_ratio", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RESPONSE_ANCHOR_SAMPLE_RATIO", "1.0")
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"verl_dflash_response_anchor_sample_ratio must be in (0, 1], got {value}.")
        return value

    def _get_response_anchor_seed(self) -> int:
        value = getattr(self.config, "verl_dflash_response_anchor_seed", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RESPONSE_ANCHOR_SEED", "42")
        return int(value)

    def _get_onpolicy_reverse_enabled(self) -> bool:
        """paradistill: enable the on-policy reverse stream on fresh draft samples instead
        of the rollout-reject driven rejected-draft stream."""
        value = getattr(self.config, "verl_dflash_onpolicy_reverse_enabled", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_ONPOLICY_REVERSE_ENABLED", "0")
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_rejected_draft_reverse_enabled(self) -> bool:
        """draftopd (reject-mode) toggle for the rollout-reject reverse stream. Default True = original
        draftopd (response forward-KL + reverse-KL on the rejected draft token d, decayed by offset).
        Set False for a forward-only draftopd: skip _collect_rejected_draft_log_probs ENTIRELY in the
        model (not just zero its loss weight, which still pays the full reverse LM-head softmax / OOMs).
        No effect with free anchors (those already have no rollout-reject reverse)."""
        value = getattr(self.config, "verl_dflash_rejected_draft_reverse_enabled", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_REJECTED_DRAFT_REVERSE_ENABLED", "1")
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_draft_sample_temperature(self) -> float:
        """T_draft for drawing fresh on-policy samples y_hat_j ~ q. 1.0 gives genuine q-samples (an
        unbiased k3 reverse-KL estimate); other values trade bias for exploration."""
        value = getattr(self.config, "verl_dflash_draft_sample_temperature", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_DRAFT_SAMPLE_TEMPERATURE", "1.0")
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"verl_dflash_draft_sample_temperature must be positive, got {value}.")
        return value

    def _get_draft_sample_seed(self) -> int:
        """Seed for the paradistill draft sampler. A negative value uses the global RNG (genuinely fresh
        samples that advance with training); a non-negative value makes sampling reproducible per call."""
        value = getattr(self.config, "verl_dflash_draft_sample_seed", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_DRAFT_SAMPLE_SEED", "-1")
        return int(value)

    # Per-region loss-mode getters. Forward regions (response, reject-accept): bernoulli_fkl | topk_fkl |
    # topk_tv; reject regions (reject-token, post-reject): reverse_kl | topk_fkl | topk_reverse_kl.
    # Defaults = original draftopd.
    _FORWARD_REGION_MODES = ("bernoulli_fkl", "topk_fkl", "topk_tv")
    _REJECT_REGION_MODES = ("reverse_kl", "topk_fkl", "topk_reverse_kl")

    def _get_region_loss_mode(self, config_key: str, env_key: str, default: str, allowed: tuple) -> str:
        value = getattr(self.config, config_key, None)
        if value is None:
            value = os.getenv(env_key, default)
        value = str(value).lower()
        if value not in allowed:
            raise ValueError(f"{config_key} must be one of {allowed}, got {value!r}.")
        return value

    def _get_response_loss_mode(self) -> str:
        return self._get_region_loss_mode(
            "verl_dflash_response_loss_mode", "VERL_DFLASH_RESPONSE_LOSS_MODE", "bernoulli_fkl",
            self._FORWARD_REGION_MODES,
        )

    def _get_reject_accept_loss_mode(self) -> str:
        return self._get_region_loss_mode(
            "verl_dflash_reject_accept_loss_mode", "VERL_DFLASH_REJECT_ACCEPT_LOSS_MODE", "bernoulli_fkl",
            self._FORWARD_REGION_MODES,
        )

    def _get_reject_token_loss_mode(self) -> str:
        return self._get_region_loss_mode(
            "verl_dflash_reject_token_loss_mode", "VERL_DFLASH_REJECT_TOKEN_LOSS_MODE", "reverse_kl",
            self._REJECT_REGION_MODES,
        )

    def _get_post_reject_loss_mode(self) -> str:
        return self._get_region_loss_mode(
            "verl_dflash_post_reject_loss_mode", "VERL_DFLASH_POST_REJECT_LOSS_MODE", "reverse_kl",
            self._REJECT_REGION_MODES,
        )

    def _get_topk_fkl_k(self, config_key: str, env_key: str, default: str) -> int:
        value = getattr(self.config, config_key, None)
        if value is None:
            value = os.getenv(env_key, default)
        value = int(value)
        if value < 0:
            raise ValueError(f"{config_key} must be >= 0, got {value}.")
        return value

    def _get_topk_fkl_teacher_k(self) -> int:
        return self._get_topk_fkl_k("verl_dflash_topk_fkl_teacher_k", "VERL_DFLASH_TOPK_FKL_TEACHER_K", "64")

    def _get_topk_fkl_student_k(self) -> int:
        return self._get_topk_fkl_k("verl_dflash_topk_fkl_student_k", "VERL_DFLASH_TOPK_FKL_STUDENT_K", "0")

    @staticmethod
    def _resolve_topk_fkl_mode(teacher_k: int, student_k: int) -> str:
        """Top-K index set from the two K: both > 0 -> union, else the single non-zero side."""
        if teacher_k > 0 and student_k > 0:
            return "union"
        if teacher_k > 0:
            return "teacher"
        if student_k > 0:
            return "student"
        raise ValueError("top-K forward KL requires verl_dflash_topk_fkl_teacher_k>0 or _student_k>0.")

    def _get_rejected_draft_max_tokens_per_sample(self) -> Optional[int]:
        value = getattr(self.config, "verl_dflash_rejected_draft_max_tokens_per_sample", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_REJECTED_DRAFT_MAX_TOKENS_PER_SAMPLE")
        if value is None or str(value).lower() in {"", "none", "null"}:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_dflash_rejected_draft_max_tokens_per_sample must be positive, got {value}.")
        return value

    def _is_dflash_profiling_enabled(self) -> bool:
        return os.getenv("VERL_DFLASH_PROFILE", "0").lower() in {"1", "true", "yes", "on"}

    def _maybe_sync_for_profile(self, enabled: bool, device: torch.device) -> None:
        if enabled and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _is_oom_error(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    def _draft_sdpa_context(self):
        if torch.cuda.is_available():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
                enable_cudnn=False,
            )
        return nullcontext()

    def _create_position_ids_for_anchors(self, anchor_positions: torch.Tensor, block_size: int) -> torch.Tensor:
        batch_size, num_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(batch_size, -1)

    def _create_noise_embedding_for_anchors(
        self,
        input_ids: torch.LongTensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        mask_token_id = self._get_mask_token_id()
        num_blocks = anchor_positions.shape[1]

        noise_ids = torch.full(
            (batch_size, num_blocks * block_size),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_starts = torch.arange(num_blocks, device=device) * block_size
        block_starts = block_starts.unsqueeze(0).expand(batch_size, -1)

        safe_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, safe_anchor_positions)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_blocks)
        noise_ids[batch_indices, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(mask_token_id, dtype=torch.long, device=device),
        )
        return self.get_input_embeddings()(noise_ids)

    def _compute_selected_lm_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        token_ids: torch.LongTensor,
        chunk_size: int,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, empty if calculate_entropy else None

        selected_hidden = draft_hidden[batch_indices, draft_indices, :]
        log_prob_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            logits = output_embeddings(selected_hidden[start:end])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            labels = token_ids[start:end].to(device=log_probs.device)
            log_prob_chunks.append(log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
            if calculate_entropy:
                entropy_chunks.append(-(log_probs.exp() * log_probs).sum(dim=-1))

        selected_log_probs = torch.cat(log_prob_chunks, dim=0)
        selected_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return selected_log_probs, selected_entropy

    def _compute_response_and_reverse_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        token_ids: torch.LongTensor,
        teacher_logits: torch.Tensor,
        row_indices: torch.LongTensor,
        chunk_size: int,
        calculate_entropy: bool,
        sample_temperature: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """paradistill fused LM-head pass: from ONE draft->vocab projection per slot,
        return both the response-forward and the on-policy reverse terms.

        At each ``(batch_indices, draft_indices)`` slot it computes the draft log-probs ONCE and returns:
          - ``log q(y_j)`` gathered at the realized response token ``token_ids`` (with gradient) -- the
            response-forward term -- plus optional draft entropy;
          - a fresh draft sample ``y_hat ~ q_phi^(j)`` drawn at ``sample_temperature`` (no gradient) with
            ``log q(y_hat)`` from the SAME log-probs (with gradient) and ``log p(y_hat)`` from the frozen
            teacher logits at ``row_indices`` (no gradient) -- the reverse term -- plus ``y_hat`` ids.
        Gradients are identical to projecting twice: both gathers backprop through the shared log-probs
        into ``draft_hidden`` (the head is the frozen main-model head, so only draft params get grad).
        ``log q`` uses temperature 1 so the k3 reverse-KL is unbiased when ``sample_temperature == 1``.
        """
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            empty_ids = batch_indices.new_empty((0,), dtype=torch.long)
            return empty, (empty if calculate_entropy else None), empty, empty, empty_ids

        selected_hidden = draft_hidden[batch_indices, draft_indices, :]
        resp_log_q_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        rev_log_q_chunks: list[torch.Tensor] = []
        rev_log_p_chunks: list[torch.Tensor] = []
        sample_chunks: list[torch.Tensor] = []
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            draft_logits = output_embeddings(selected_hidden[start:end]).float()
            draft_log_probs = F.log_softmax(draft_logits, dim=-1)  # shared by both terms
            # response-forward term: log q at the realized token y_j
            resp_labels = token_ids[start:end].to(device=draft_log_probs.device)
            resp_log_q_chunks.append(draft_log_probs.gather(dim=-1, index=resp_labels.unsqueeze(-1)).squeeze(-1))
            if calculate_entropy:
                entropy_chunks.append(-(draft_log_probs.exp() * draft_log_probs).sum(dim=-1))
            # on-policy reverse term: sample y_hat ~ q at T_draft, reuse draft_log_probs for log q(y_hat)
            with torch.no_grad():
                sample_probs = F.softmax(draft_logits / sample_temperature, dim=-1)
                sampled = torch.multinomial(sample_probs, num_samples=1, generator=generator).squeeze(-1)
                teacher_chunk_logits = teacher_logits[
                    batch_indices[start:end], row_indices[start:end], :
                ].float()
                teacher_log_probs = F.log_softmax(teacher_chunk_logits, dim=-1)
                rev_log_p_chunks.append(teacher_log_probs.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1))
            rev_log_q_chunks.append(draft_log_probs.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1))
            sample_chunks.append(sampled)

        resp_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return (
            torch.cat(resp_log_q_chunks, dim=0),
            resp_entropy,
            torch.cat(rev_log_q_chunks, dim=0),
            torch.cat(rev_log_p_chunks, dim=0),
            torch.cat(sample_chunks, dim=0),
        )

    @staticmethod
    def _topk_candidate_support(student_logits, teacher_chunk_logits, mode, topk_teacher, topk_student):
        """Top-K support ``S`` as gathered candidate columns: returns ``(cand_idx, keep_bool)``.

        ``S`` = the teacher top-``topk_teacher`` and/or student top-``topk_student`` tokens (``mode``);
        ``union`` dedups student columns already in the teacher top-K (``keep`` zeros them) so each token
        in ``S`` contributes once. Grad-free (detached indices); call under no_grad. Shared by the top-K
        forward-KL / TV / reverse-KL helpers.
        """
        index_parts: list[torch.Tensor] = []
        keep_parts: list[torch.Tensor] = []
        teacher_idx = None
        if mode in ("teacher", "union"):
            teacher_idx = teacher_chunk_logits.topk(topk_teacher, dim=-1).indices  # (n, K_t)
            index_parts.append(teacher_idx)
            keep_parts.append(torch.ones_like(teacher_idx, dtype=torch.bool))
        if mode in ("student", "union"):
            student_idx = student_logits.detach().topk(topk_student, dim=-1).indices  # (n, K_s)
            index_parts.append(student_idx)
            if teacher_idx is None:
                keep_parts.append(torch.ones_like(student_idx, dtype=torch.bool))
            else:
                dup = (student_idx.unsqueeze(-1) == teacher_idx.unsqueeze(-2)).any(dim=-1)  # (n, K_s)
                keep_parts.append(~dup)
        return torch.cat(index_parts, dim=-1), torch.cat(keep_parts, dim=-1)

    def _compute_topk_forward_kl(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        teacher_logits: torch.Tensor,
        row_indices: torch.LongTensor,
        chunk_size: int,
        mode: str,
        topk_teacher: int,
        topk_student: int,
        calculate_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-slot top-K forward KL ``sum_{v in S} p_t(v) (log p_t(v) - log q_s(v))``.

        ``S`` = the teacher top-``topk_teacher`` and/or student top-``topk_student`` token sets (``mode``);
        ``union`` takes both (the two K may differ). Teacher = frozen realized-prefix logits at
        ``row_indices`` (no grad); student = draft head projection at ``(batch_indices, draft_indices)``
        (grad through ``log q_s`` only). The partial-support sum is clamped to >=0 (top-K mass < 1).

        Implemented by GATHERING only the candidate indices (the union of the teacher/student top-K) --
        no full-vocab teacher log-probs / probs / boolean mask / product are materialized (only the
        unavoidable full student log-softmax, kept for the gradient). ``union`` is deduplicated via
        ``keep`` (student columns whose token is already in the teacher top-K are zeroed), so each token
        in S contributes exactly once -- numerically and gradient-equivalent to the masked full-vocab
        form (the sole difference is float reduction order over |S| terms vs the full vocabulary).
        """
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, (empty if calculate_entropy else None)

        selected_hidden = draft_hidden[batch_indices, draft_indices, :]
        fkl_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        vocab_size = int(teacher_logits.shape[-1])
        topk_teacher = min(int(topk_teacher), vocab_size)
        topk_student = min(int(topk_student), vocab_size)
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            student_logits = output_embeddings(selected_hidden[start:end]).float()
            student_log_probs = F.log_softmax(student_logits, dim=-1)  # full vocab: needed for the gather + grad
            with torch.no_grad():
                teacher_chunk_logits = teacher_logits[batch_indices[start:end], row_indices[start:end], :].float()
                teacher_lse = torch.logsumexp(teacher_chunk_logits, dim=-1, keepdim=True)  # (n, 1)
                cand_idx, keep = self._topk_candidate_support(
                    student_logits, teacher_chunk_logits, mode, topk_teacher, topk_student
                )
                keep = keep.to(teacher_chunk_logits.dtype)  # (n, |S_cand|)
                teacher_logp_sel = teacher_chunk_logits.gather(-1, cand_idx) - teacher_lse  # log p_t over S
                teacher_probs_sel = teacher_logp_sel.exp()
            student_logp_sel = student_log_probs.gather(-1, cand_idx)  # log q_s over S (keeps grad)
            fkl = (teacher_probs_sel * (teacher_logp_sel - student_logp_sel) * keep).sum(dim=-1).clamp_min(0.0)
            fkl_chunks.append(fkl)
            if calculate_entropy:
                entropy_chunks.append(-(student_log_probs.exp() * student_log_probs).sum(dim=-1))

        selected_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return torch.cat(fkl_chunks, dim=0), selected_entropy

    def _compute_topk_tv(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        teacher_logits: torch.Tensor,
        row_indices: torch.LongTensor,
        chunk_size: int,
        mode: str,
        topk_teacher: int,
        topk_student: int,
        calculate_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-slot top-K total-variation distance ``0.5 * sum_{v in S} |p_t(v) - q_s(v)|``.

        Same support ``S`` and conventions as ``_compute_topk_forward_kl`` (``mode``/``topk_*``; teacher at
        ``row_indices`` under no_grad, gradient through ``q_s`` only). ``p_t, q_s`` are full-vocab softmax;
        only the candidate columns and the two log-sum-exps are materialized (no full-vocab softmax). The
        partial-support sum of absolute differences is always >= 0, so no clamp is needed.
        """
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, (empty if calculate_entropy else None)

        selected_hidden = draft_hidden[batch_indices, draft_indices, :]
        tv_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        vocab_size = int(teacher_logits.shape[-1])
        topk_teacher = min(int(topk_teacher), vocab_size)
        topk_student = min(int(topk_student), vocab_size)
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            student_logits = output_embeddings(selected_hidden[start:end]).float()
            student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)  # (n, 1); grad to all logits
            with torch.no_grad():
                teacher_chunk_logits = teacher_logits[batch_indices[start:end], row_indices[start:end], :].float()
                teacher_lse = torch.logsumexp(teacher_chunk_logits, dim=-1, keepdim=True)
                cand_idx, keep = self._topk_candidate_support(
                    student_logits, teacher_chunk_logits, mode, topk_teacher, topk_student
                )
                keep = keep.to(teacher_chunk_logits.dtype)
                teacher_probs_sel = (teacher_chunk_logits.gather(-1, cand_idx) - teacher_lse).exp()  # p_t over S
            student_probs_sel = (student_logits.gather(-1, cand_idx) - student_lse).exp()  # q_s over S (keeps grad)
            tv = (0.5 * (teacher_probs_sel - student_probs_sel).abs() * keep).sum(dim=-1)
            tv_chunks.append(tv)
            if calculate_entropy:
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                entropy_chunks.append(-(student_log_probs.exp() * student_log_probs).sum(dim=-1))

        selected_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return torch.cat(tv_chunks, dim=0), selected_entropy

    def _compute_topk_reverse_kl(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        teacher_logits: torch.Tensor,
        row_indices: torch.LongTensor,
        chunk_size: int,
        mode: str,
        topk_teacher: int,
        topk_student: int,
        calculate_entropy: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-slot top-K reverse KL ``sum_{v in S} q_s(v) (log q_s(v) - log p_t(v))``.

        Reverse-direction analog of ``_compute_topk_forward_kl`` over the SAME support ``S`` (``mode``/
        ``topk_*``; teacher at ``row_indices`` under no_grad). Gradient flows through ``q_s`` in BOTH the
        weight ``q_s(v)`` and ``log q_s(v)``. ``q_s, p_t`` are full-vocab softmax via logsumexp+gather (no
        full-vocab softmax materialized). The partial-support sum is clamped to >=0 (top-K mass < 1). At a
        reject position the top-K support contains both the corrected token y (teacher mode) and the rejected
        token d (student mode), so one reverse KL trains both without separating them.
        """
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, (empty if calculate_entropy else None)

        selected_hidden = draft_hidden[batch_indices, draft_indices, :]
        rkl_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        vocab_size = int(teacher_logits.shape[-1])
        topk_teacher = min(int(topk_teacher), vocab_size)
        topk_student = min(int(topk_student), vocab_size)
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            student_logits = output_embeddings(selected_hidden[start:end]).float()
            student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)  # (n, 1); grad to all logits
            with torch.no_grad():
                teacher_chunk_logits = teacher_logits[batch_indices[start:end], row_indices[start:end], :].float()
                teacher_lse = torch.logsumexp(teacher_chunk_logits, dim=-1, keepdim=True)
                cand_idx, keep = self._topk_candidate_support(
                    student_logits, teacher_chunk_logits, mode, topk_teacher, topk_student
                )
                keep = keep.to(teacher_chunk_logits.dtype)
                teacher_logp_sel = teacher_chunk_logits.gather(-1, cand_idx) - teacher_lse  # log p_t over S
            student_logp_sel = student_logits.gather(-1, cand_idx) - student_lse  # log q_s over S (keeps grad)
            student_probs_sel = student_logp_sel.exp()  # q_s over S (grad through the weight too)
            rkl = (student_probs_sel * (student_logp_sel - teacher_logp_sel) * keep).sum(dim=-1).clamp_min(0.0)
            rkl_chunks.append(rkl)
            if calculate_entropy:
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                entropy_chunks.append(-(student_log_probs.exp() * student_log_probs).sum(dim=-1))

        selected_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return torch.cat(rkl_chunks, dim=0), selected_entropy

    def _build_random_response_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        batch_idx: int,
        prompt_len: int,
        response_len: int,
        segment_lens: list[int],
        seed: int,
    ) -> tuple[list[int], list[int]]:
        if response_len <= 0 or not segment_lens:
            return [], []
        segment_lens = [int(segment_len) for segment_len in segment_lens if int(segment_len) > 0]
        if not segment_lens:
            return [], []
        token_count = sum(segment_lens)
        if token_count > response_len:
            raise ValueError(
                "Random DFLASH response anchors cannot preserve token count: "
                f"segment token count {token_count} exceeds response_len={response_len}."
            )
        valid_len = max(0, min(int(prompt_len + response_len), int(input_ids.shape[1])))
        if valid_len > 0:
            sample_ids = input_ids[batch_idx, :valid_len].to(dtype=torch.long)
            weights = torch.arange(1, valid_len + 1, dtype=torch.long, device=input_ids.device)
            token_hash = int((sample_ids * weights).sum().item())
        else:
            token_hash = 0
        rng = random.Random(int(seed) + batch_idx * 1000003 + token_hash)
        segment_lens = list(segment_lens)
        rng.shuffle(segment_lens)
        gaps = [0 for _ in range(len(segment_lens) + 1)]
        for _ in range(response_len - token_count):
            gaps[rng.randrange(len(gaps))] += 1

        anchors_resp: list[int] = []
        cursor = gaps[0]
        for segment_idx, segment_len in enumerate(segment_lens):
            anchors_resp.append(cursor - 1)
            cursor += segment_len + gaps[segment_idx + 1]
        return anchors_resp, segment_lens

    @staticmethod
    def _per_sample_slot_columns(batch_indices: torch.LongTensor, batch_size: int) -> tuple[torch.LongTensor, int]:
        """Map flat per-slot ``batch_indices`` to a per-sample (batch, width) layout.

        Returns ``(col, width)`` where ``col[k]`` is the column of slot ``k`` within its sample's row
        (its rank among that sample's slots) and ``width`` is the max slots over samples. Used to pack the
        flat paradistill reverse stream into per-sample rows so the engine's dynamic-batch restore (which
        reorders model outputs by sample) and the loss both see the standard (batch, width) rejected layout.
        """
        counts = torch.bincount(batch_indices, minlength=batch_size)
        width = int(counts.max().item()) if batch_indices.numel() > 0 else 0
        order = torch.argsort(batch_indices, stable=True)
        group_start = torch.cumsum(counts, dim=0) - counts
        col = torch.empty_like(batch_indices)
        col[order] = torch.arange(batch_indices.numel(), device=batch_indices.device) - group_start[batch_indices[order]]
        return col, width

    @staticmethod
    def _build_free_response_anchors(
        *,
        response_len: int,
        num_predict: int,
        mode: str,
        sample_ratio: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        """Enumerate free (reject-independent) ``(anchor_resp, segment_len)`` for paradistill.

        Anchors are in response coordinates (``-1`` is the last prompt token). Anchor ``a`` distills
        the next ``segment_len`` response tokens (positions ``a+1 .. a+segment_len``); ``segment_len``
        is capped at ``num_predict`` (= draft block heads) and clamped to the response end. Modes:
          - ``stride_k``: non-overlapping blocks covering the response (anchors at -1, -1+K, ...).
          - ``sampled``: pick EXACTLY ``round(sample_ratio * response_len)`` anchors total (N = real
            response length, padding-free). The prompt-end anchor ``-1`` is always one of them and the
            remaining are drawn without replacement from response positions ``0 .. response_len-2`` (the
            last response token is never an anchor). At least the ``-1`` anchor is always kept.
        """
        if response_len <= 0 or num_predict <= 0:
            return [], []
        last = response_len - 1  # last response position usable as a prediction target
        if mode == "sampled":
            rng = random.Random(int(seed))
            pool = list(range(0, last))  # response anchor positions, excluding -1 and the last token
            num_anchors = max(1, min(round(sample_ratio * response_len), len(pool) + 1))
            candidates = [-1] + sorted(rng.sample(pool, num_anchors - 1))
        elif mode == "stride_k":
            candidates = list(range(-1, last, num_predict))
        else:
            raise ValueError(f"Unknown response anchor mode {mode!r}; expected reject, stride_k, or sampled.")
        anchors: list[int] = []
        segments: list[int] = []
        for anchor_resp in candidates:
            segment_len = min(num_predict, last - anchor_resp)
            if segment_len <= 0:
                continue
            anchors.append(anchor_resp)
            segments.append(segment_len)
        return anchors, segments

    def _build_opd_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        reject_token_indices: torch.LongTensor,
        draft_block_size: int,
        response_anchor_stride: int = 1,
        random_response_anchor_enabled: bool = False,
        random_response_anchor_seed: int = 42,
        anchor_mode: str = "reject",
        anchor_sample_ratio: float = 1.0,
        anchor_seed: int = 42,
        rejected_draft_anchor_indices: Optional[torch.LongTensor] = None,
        rejected_draft_offsets: Optional[torch.LongTensor] = None,
        rejected_draft_mask: Optional[torch.Tensor] = None,
        include_rejected_draft_anchors: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        all_anchors: list[list[int]] = []
        all_segment_lens: list[list[int]] = []
        all_row_starts: list[list[int]] = []
        max_blocks = 0
        empty_reject_sample_count = 0
        skipped_sample_count = 0
        total_reject_count = 0
        response_anchor_count = 0
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            sample_anchors: list[int] = []
            sample_segment_lens: list[int] = []
            sample_row_starts: list[int] = []
            response_anchors_resp: list[int] = []
            response_segment_lens: list[int] = []

            if anchor_mode != "reject":
                # Free anchors cover the response independently of SD reject positions; the rest of the
                # plan (sequence-coord conversion + optional rejected-draft anchors) is shared.
                response_anchors_resp, response_segment_lens = self._build_free_response_anchors(
                    response_len=response_len,
                    num_predict=draft_block_size - 1,
                    mode=anchor_mode,
                    sample_ratio=anchor_sample_ratio,
                    seed=anchor_seed + batch_idx,
                )
            else:
                if response_len <= 0:
                    anchors_resp: list[int] = []
                    boundaries_resp: list[int] = []
                else:
                    raw_rejects = reject_token_indices[batch_idx]
                    rejects = sorted(
                        {
                            int(idx.item())
                            for idx in raw_rejects
                            if int(idx.item()) >= 0 and int(idx.item()) < response_len
                        }
                    )
                    total_reject_count += len(rejects)
                    if len(rejects) == 0:
                        empty_reject_sample_count += 1
                        anchors_resp = []
                        boundaries_resp = []
                    elif rejects[-1] < response_len - 1:
                        rejects.append(response_len - 1)
                        anchors_resp = [-1] + rejects
                        boundaries_resp = rejects
                    else:
                        anchors_resp = [-1] + rejects
                        boundaries_resp = rejects

                if response_anchor_stride > 1 and anchors_resp:
                    anchor_boundary_pairs = list(zip(anchors_resp, boundaries_resp, strict=False))
                    anchor_boundary_pairs = [
                        pair
                        for pair_idx, pair in enumerate(anchor_boundary_pairs)
                        if pair_idx % response_anchor_stride == 0 or pair_idx == len(anchor_boundary_pairs) - 1
                    ]
                    anchors_resp = [pair[0] for pair in anchor_boundary_pairs]
                    boundaries_resp = [pair[1] for pair in anchor_boundary_pairs]
                for anchor_resp, boundary_resp in zip(anchors_resp, boundaries_resp, strict=False):
                    full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                    segment_len = boundary_resp - anchor_resp
                    if segment_len <= 0:
                        continue
                    segment_len = min(segment_len, draft_block_size - 1)
                    if full_anchor < 0 or full_anchor >= seq_len - 1:
                        continue
                    response_anchors_resp.append(anchor_resp)
                    response_segment_lens.append(segment_len)

                if random_response_anchor_enabled and response_len > 0 and response_segment_lens:
                    response_anchors_resp, response_segment_lens = self._build_random_response_anchor_plan(
                        input_ids=input_ids,
                        batch_idx=batch_idx,
                        prompt_len=prompt_len,
                        response_len=response_len,
                        segment_lens=response_segment_lens,
                        seed=random_response_anchor_seed,
                    )

            for anchor_resp, segment_len in zip(response_anchors_resp, response_segment_lens, strict=False):
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= seq_len - 1:
                    continue
                sample_anchors.append(full_anchor)
                sample_segment_lens.append(segment_len)
                sample_row_starts.append(full_anchor)

            if response_len > 0 and len(sample_anchors) == 0:
                skipped_sample_count += 1
            response_anchor_count += len(sample_anchors)

            if (
                include_rejected_draft_anchors
                and rejected_draft_anchor_indices is not None
                and rejected_draft_offsets is not None
                and rejected_draft_mask is not None
            ):
                valid_len = prompt_len + response_len
                rejected_count = rejected_draft_anchor_indices.shape[1]
                existing_anchors = set(sample_anchors)
                for item_idx in range(rejected_count):
                    if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                        continue
                    offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                    if offset <= 0 or offset >= draft_block_size:
                        continue
                    anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                    full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                    if full_anchor < 0 or full_anchor >= min(valid_len, seq_len):
                        continue
                    if full_anchor in existing_anchors:
                        continue
                    sample_anchors.append(full_anchor)
                    sample_segment_lens.append(0)
                    sample_row_starts.append(full_anchor)
                    existing_anchors.add(full_anchor)

            all_anchors.append(sample_anchors)
            all_segment_lens.append(sample_segment_lens)
            all_row_starts.append(sample_row_starts)
            max_blocks = max(max_blocks, len(sample_anchors))

        max_blocks = max(max_blocks, 1)

        anchor_positions = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        segment_lens = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        row_starts = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        block_keep_mask = torch.zeros((batch_size, max_blocks), dtype=torch.bool, device=device)

        for batch_idx, sample_anchors in enumerate(all_anchors):
            n_blocks = len(sample_anchors)
            if n_blocks == 0:
                continue
            anchor_positions[batch_idx, :n_blocks] = torch.tensor(sample_anchors, dtype=torch.long, device=device)
            segment_lens[batch_idx, :n_blocks] = torch.tensor(
                all_segment_lens[batch_idx], dtype=torch.long, device=device
            )
            row_starts[batch_idx, :n_blocks] = torch.tensor(all_row_starts[batch_idx], dtype=torch.long, device=device)
            block_keep_mask[batch_idx, :n_blocks] = True

        if attention_mask is not None:
            valid_seq_lens = attention_mask.long().sum(dim=1)
        else:
            valid_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        opd_metrics = {
            "valid_anchor_count": response_anchor_count,
            "skipped_sample_count": skipped_sample_count,
            "empty_reject_sample_count": empty_reject_sample_count,
            "total_reject_count": total_reject_count,
            "sample_count": batch_size,
        }
        return anchor_positions, segment_lens, row_starts, block_keep_mask, valid_seq_lens, opd_metrics

    def _build_rejected_draft_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        draft_block_size: int,
        max_tokens_per_sample: Optional[int],
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if (
            rejected_draft_anchor_indices is None
            or rejected_draft_offsets is None
            or rejected_draft_token_ids is None
            or rejected_draft_mask is None
            or not bool(rejected_draft_mask.any())
        ):
            anchor_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            block_keep_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            return anchor_positions, block_keep_mask

        all_anchors: list[list[int]] = []
        max_blocks = 0
        rejected_width = int(rejected_draft_anchor_indices.shape[1])
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            valid_len = min(prompt_len + response_len, seq_len)
            sample_anchors: list[int] = []
            existing_anchors: set[int] = set()
            selected_count = 0
            for item_idx in range(rejected_width):
                if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                    continue
                if max_tokens_per_sample is not None and selected_count >= max_tokens_per_sample:
                    continue
                offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                token_id = int(rejected_draft_token_ids[batch_idx, item_idx].item())
                if offset <= 0 or offset >= draft_block_size or token_id < 0:
                    continue
                anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= valid_len:
                    continue
                if full_anchor not in existing_anchors:
                    sample_anchors.append(full_anchor)
                    existing_anchors.add(full_anchor)
                selected_count += 1

            all_anchors.append(sample_anchors)
            max_blocks = max(max_blocks, len(sample_anchors))

        max_blocks = max(max_blocks, 1)
        anchor_positions = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        block_keep_mask = torch.zeros((batch_size, max_blocks), dtype=torch.bool, device=device)
        for batch_idx, sample_anchors in enumerate(all_anchors):
            n_blocks = len(sample_anchors)
            if n_blocks == 0:
                continue
            anchor_positions[batch_idx, :n_blocks] = torch.tensor(sample_anchors, dtype=torch.long, device=device)
            block_keep_mask[batch_idx, :n_blocks] = True
        return anchor_positions, block_keep_mask

    def _collect_rejected_draft_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        anchor_positions: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        draft_block_size: int,
        lm_head_chunk_size: int,
        max_tokens_per_sample: Optional[int],
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_teacher_logprobs: Optional[torch.Tensor],
        rejected_draft_mask: Optional[torch.Tensor],
        reject_token_mode: str = "reverse_kl",
        post_reject_mode: str = "reverse_kl",
        teacher_logits: Optional[torch.Tensor] = None,
        topk_fkl_mode: str = "teacher",
        topk_fkl_teacher_k: int = 0,
        topk_fkl_student_k: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per rollout-rejected-draft slot, return (student, teacher, mask, is_reject_token), each
        (batch, rejected_width). The min offset per anchor is the reject token (first mismatch d); larger
        offsets are the post-reject suffix. Each region's loss mode picks per slot:
          - reverse_kl: student = log q(d), teacher = cached scalar log p(d) (the loss takes reverse-KL);
          - topk_fkl: student = top-K forward KL vs the teacher's REALIZED-position top-K (teacher forcing,
            teacher_logits at full_anchor+offset-1), teacher channel 0; slots whose realized target is past
            the response end are dropped (the reject token itself is never OOB).
        """
        batch_size = int(prompt_lengths.shape[0])
        rejected_width = 1
        if rejected_draft_anchor_indices is not None and rejected_draft_anchor_indices.dim() >= 2:
            rejected_width = max(1, int(rejected_draft_anchor_indices.shape[1]))

        student_tensor = draft_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        teacher_tensor = draft_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        mask_tensor = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=draft_hidden.device)
        is_reject_token_tensor = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=draft_hidden.device)
        any_topk = reject_token_mode in _TOPK_REJECT_MODES or post_reject_mode in _TOPK_REJECT_MODES
        any_reverse = reject_token_mode == "reverse_kl" or post_reject_mode == "reverse_kl"
        if (
            rejected_draft_anchor_indices is None
            or rejected_draft_offsets is None
            or rejected_draft_token_ids is None
            or rejected_draft_mask is None
            or not bool(rejected_draft_mask.any())
            or (any_topk and teacher_logits is None)
            or (any_reverse and rejected_draft_teacher_logprobs is None)
        ):
            return student_tensor, teacher_tensor, mask_tensor, is_reject_token_tensor

        # Materialize the small (batch x width) metadata once to drive the per-slot routing on the host --
        # avoids one GPU->CPU sync per slot (.item()/nonzero) in the loops below.
        prompt_lens = prompt_lengths.tolist()
        response_lens = response_lengths.tolist()
        mask_l = rejected_draft_mask.tolist()
        offset_l = rejected_draft_offsets.tolist()
        token_l = rejected_draft_token_ids.tolist()
        anchor_l = rejected_draft_anchor_indices.tolist()
        anchor_pos_l = anchor_positions.tolist()
        keep_l = block_keep_mask.tolist()
        teacher_lp_l = rejected_draft_teacher_logprobs.tolist() if any_reverse else None
        # Per-sample {full_anchor: block_idx} over the kept response blocks (first match wins).
        anchor_to_block = [
            {pos: blk for blk, (pos, kept) in reversed(list(enumerate(zip(anchor_pos_l[b], keep_l[b])))) if kept}
            for b in range(batch_size)
        ]

        # Pass 1: collect candidate slots (basic filters + per-sample cap) and the min offset per anchor.
        candidates: list[tuple] = []  # (batch_idx, item_idx, block_idx, offset, token_id, full_anchor, valid_len)
        min_offset_by_anchor: list[dict[int, int]] = [dict() for _ in range(batch_size)]
        selected_counts = [0 for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lens[batch_idx])
            valid_len = prompt_len + int(response_lens[batch_idx])
            mapping = anchor_to_block[batch_idx]
            for item_idx in range(rejected_width):
                if not mask_l[batch_idx][item_idx]:
                    continue
                if max_tokens_per_sample is not None and selected_counts[batch_idx] >= max_tokens_per_sample:
                    continue
                offset = int(offset_l[batch_idx][item_idx])
                token_id = int(token_l[batch_idx][item_idx])
                if offset <= 0 or offset >= draft_block_size or token_id < 0:
                    continue
                anchor_resp = int(anchor_l[batch_idx][item_idx])
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= valid_len:
                    continue
                block_idx = mapping.get(full_anchor)
                if block_idx is None:
                    continue
                candidates.append((batch_idx, item_idx, block_idx, offset, token_id, full_anchor, valid_len))
                selected_counts[batch_idx] += 1
                prev = min_offset_by_anchor[batch_idx].get(full_anchor)
                if prev is None or offset < prev:
                    min_offset_by_anchor[batch_idx][full_anchor] = offset

        # Pass 2: route each slot by its region's mode -> reverse-KL data (log q(d) + cached log p(d)), or
        # an in-model top-K forward KL / top-K reverse KL at the realized teacher position. Both top-K modes
        # ride in the student channel (teacher channel 0) and drop slots whose realized target is OOB.
        rev_batch: list[int] = []
        rev_draft: list[int] = []
        rev_tokens: list[int] = []
        rev_items: list[tuple[int, int]] = []
        rev_teacher: list[float] = []
        # one (batch, draft, rows, items) bucket per top-K mode keyed by the in-model helper.
        topk_fns = {"topk_fkl": self._compute_topk_forward_kl, "topk_reverse_kl": self._compute_topk_reverse_kl}
        topk_buckets = {m: {"batch": [], "draft": [], "rows": [], "items": []} for m in topk_fns}
        for batch_idx, item_idx, block_idx, offset, token_id, full_anchor, valid_len in candidates:
            is_reject_token = offset == min_offset_by_anchor[batch_idx][full_anchor]
            mode = reject_token_mode if is_reject_token else post_reject_mode
            draft_index = block_idx * draft_block_size + offset
            if mode in topk_buckets:
                if full_anchor + offset >= valid_len:
                    continue  # realized teacher target past the response end
                bucket = topk_buckets[mode]
                bucket["batch"].append(batch_idx)
                bucket["draft"].append(draft_index)
                bucket["rows"].append(full_anchor + offset - 1)
                bucket["items"].append((batch_idx, item_idx))
            else:  # reverse_kl
                rev_batch.append(batch_idx)
                rev_draft.append(draft_index)
                rev_tokens.append(token_id)
                rev_items.append((batch_idx, item_idx))
                rev_teacher.append(float(teacher_lp_l[batch_idx][item_idx]))
            mask_tensor[batch_idx, item_idx] = True
            is_reject_token_tensor[batch_idx, item_idx] = is_reject_token

        def _long(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=draft_hidden.device)

        if rev_batch:
            rev_values, _ = self._compute_selected_lm_log_probs(
                draft_hidden=draft_hidden,
                output_embeddings=output_embeddings,
                batch_indices=_long(rev_batch),
                draft_indices=_long(rev_draft),
                token_ids=_long(rev_tokens),
                chunk_size=lm_head_chunk_size,
                calculate_entropy=False,
            )
            rev_rows = _long([b for b, _ in rev_items])
            rev_cols = _long([c for _, c in rev_items])
            student_tensor[rev_rows, rev_cols] = rev_values
            teacher_tensor[rev_rows, rev_cols] = torch.tensor(
                rev_teacher, dtype=torch.float32, device=draft_hidden.device
            )
        for mode, bucket in topk_buckets.items():
            if not bucket["batch"]:
                continue
            values, _ = topk_fns[mode](
                draft_hidden=draft_hidden,
                output_embeddings=output_embeddings,
                batch_indices=_long(bucket["batch"]),
                draft_indices=_long(bucket["draft"]),
                teacher_logits=teacher_logits,
                row_indices=_long(bucket["rows"]),
                chunk_size=lm_head_chunk_size,
                mode=topk_fkl_mode,
                topk_teacher=topk_fkl_teacher_k,
                topk_student=topk_fkl_student_k,
            )
            student_tensor[_long([b for b, _ in bucket["items"]]), _long([c for _, c in bucket["items"]])] = values
        return student_tensor, teacher_tensor, mask_tensor, is_reject_token_tensor

    def _run_dflash_draft_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        target_hidden: torch.Tensor,
        anchor_positions: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        draft_block_size: int,
        profile_enabled: bool,
        checkpoint_forward: bool = False,
    ) -> tuple[torch.Tensor, str, float]:
        batch_size, seq_len = input_ids.shape
        noise_embedding = self._create_noise_embedding_for_anchors(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            block_size=draft_block_size,
        )
        context_position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        draft_position_ids = self._create_position_ids_for_anchors(anchor_positions, block_size=draft_block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        draft_config = getattr(self.draft_model, "config", None)
        attn_impl = str(getattr(draft_config, "_attn_implementation", "eager"))
        if attn_impl == "flex_attention":
            try:
                draft_attention_mask = create_dflash_block_mask(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    seq_len=seq_len,
                    block_size=draft_block_size,
                    device=input_ids.device,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create flex_attention BlockMask for DFLASH OPD (%s); falling back to sdpa.",
                    exc,
                )
                attn_impl = "sdpa"
                if draft_config is not None:
                    _set_config_attn_implementation(draft_config, attn_impl)
                draft_attention_mask = create_dflash_sdpa_mask(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    seq_len=seq_len,
                    block_size=draft_block_size,
                    device=input_ids.device,
                )
        else:
            draft_attention_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=seq_len,
                block_size=draft_block_size,
                device=input_ids.device,
            )

        def _draft_forward(noise_embedding_arg: torch.Tensor, target_hidden_arg: torch.Tensor) -> torch.Tensor:
            draft_outputs = self.draft_model(
                position_ids=full_position_ids,
                attention_mask=draft_attention_mask,
                noise_embedding=noise_embedding_arg,
                target_hidden=target_hidden_arg,
                use_cache=False,
            )
            draft_hidden = draft_outputs[0] if isinstance(draft_outputs, tuple) else draft_outputs
            if hasattr(draft_hidden, "last_hidden_state"):
                draft_hidden = draft_hidden.last_hidden_state
            return draft_hidden

        def _maybe_checkpoint_draft_forward() -> torch.Tensor:
            if checkpoint_forward and torch.is_grad_enabled():
                return torch_checkpoint.checkpoint(
                    _draft_forward,
                    noise_embedding,
                    target_hidden,
                    use_reentrant=False,
                )
            return _draft_forward(noise_embedding, target_hidden)

        draft_start_time = time.perf_counter()
        try:
            with self._draft_sdpa_context():
                draft_hidden = _maybe_checkpoint_draft_forward()
        except RuntimeError as exc:
            if attn_impl != "flex_attention" or self._is_oom_error(exc):
                raise
            logger.warning(
                "DFLASH draft flex_attention forward failed (%s); retrying this micro-batch with sdpa.",
                exc,
            )
            attn_impl = "sdpa"
            if draft_config is not None:
                _set_config_attn_implementation(draft_config, attn_impl)
            draft_attention_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=seq_len,
                block_size=draft_block_size,
                device=input_ids.device,
            )
            with self._draft_sdpa_context():
                draft_hidden = _maybe_checkpoint_draft_forward()
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        draft_forward_ms = (time.perf_counter() - draft_start_time) * 1000.0
        return draft_hidden, attn_impl, draft_forward_ms

    def _forward_opd(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        reject_token_indices: torch.LongTensor,
        rejected_draft_anchor_indices: Optional[torch.LongTensor] = None,
        rejected_draft_offsets: Optional[torch.LongTensor] = None,
        rejected_draft_token_ids: Optional[torch.LongTensor] = None,
        rejected_draft_teacher_logprobs: Optional[torch.Tensor] = None,
        rejected_draft_mask: Optional[torch.Tensor] = None,
        calculate_entropy: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"DFLASH OPD requires padded 2D input_ids, got shape={tuple(input_ids.shape)}.")

        profile_enabled = self._is_dflash_profiling_enabled()
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        total_start_time = time.perf_counter()
        teacher_forward_ms = 0.0
        draft_forward_ms = 0.0
        lm_head_ms = 0.0

        # paradistill: replace the rollout-reject reverse stream with a fresh on-policy
        # draft sample drawn at each response-prediction position. Read the flag early so the teacher
        # forward can retain its full-vocab logits (needed to score log p(y_hat_j)).
        onpolicy_reverse_enabled = self._get_onpolicy_reverse_enabled()
        # Per-region loss selection (see _get_*_loss_mode). Any region on top-K needs the teacher's
        # full-vocab logits at the realized positions, so retain them (as paradistill also does).
        response_loss_mode = self._get_response_loss_mode()
        reject_accept_loss_mode = self._get_reject_accept_loss_mode()
        reject_token_loss_mode = self._get_reject_token_loss_mode()
        post_reject_loss_mode = self._get_post_reject_loss_mode()
        reject_token_topk = reject_token_loss_mode in _TOPK_REJECT_MODES
        post_reject_topk = post_reject_loss_mode in _TOPK_REJECT_MODES
        # A region "uses top-K" (needs the teacher's full-vocab logits + an in-model top-K helper) when its
        # mode is a top-K divergence: forward regions topk_fkl / topk_tv, reject regions topk_fkl /
        # topk_reverse_kl.
        topk_enabled = (
            response_loss_mode in _TOPK_FORWARD_MODES
            or reject_accept_loss_mode in _TOPK_FORWARD_MODES
            or reject_token_topk
            or post_reject_topk
        )
        retain_teacher_logits = onpolicy_reverse_enabled or topk_enabled

        target_kwargs = dict(kwargs)
        target_kwargs.pop("dflash_prompt_lengths", None)
        target_kwargs.pop("dflash_response_lengths", None)
        target_kwargs.pop("dflash_reject_token_indices", None)
        target_kwargs.pop("dflash_rejected_draft_anchor_indices", None)
        target_kwargs.pop("dflash_rejected_draft_offsets", None)
        target_kwargs.pop("dflash_rejected_draft_token_ids", None)
        target_kwargs.pop("dflash_rejected_draft_teacher_logprobs", None)
        target_kwargs.pop("dflash_rejected_draft_mask", None)
        target_kwargs.pop("dflash_calculate_entropy", None)

        teacher_start_time = time.perf_counter()
        with torch.no_grad():
            teacher_outputs = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                **target_kwargs,
            )
            if teacher_outputs.hidden_states is None:
                raise RuntimeError("Teacher model did not return hidden states required by DFLASH draft.")
            target_hidden = self._extract_target_hidden(teacher_outputs.hidden_states)
            # paradistill scores the fresh draft sample against the frozen teacher's realized-prefix
            # distribution p(.|s_t, y_<j); top-K forward KL scores the draft head against the same teacher
            # realized-position distribution. Either way these logits are already computed by main_model.
            teacher_logits = teacher_outputs.logits if retain_teacher_logits else None
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        teacher_forward_ms = (time.perf_counter() - teacher_start_time) * 1000.0
        if retain_teacher_logits and teacher_logits is None:
            raise RuntimeError(
                "DFLASH on-policy reverse / top-K forward KL require full-vocab teacher logits from the frozen "
                "main model, but main_model did not return logits."
            )

        # SGLang DFLASH uses block_size as the total block length: one anchor
        # token plus block_size - 1 future-token predictions.
        draft_block_size = self._get_block_size()
        lm_head_chunk_size = self._get_lm_head_chunk_size()
        response_anchor_stride = self._get_response_anchor_stride()
        random_response_anchor_enabled = self._get_random_response_anchor_enabled()
        random_response_anchor_seed = self._get_random_response_anchor_seed()
        response_anchor_mode = self._get_response_anchor_mode()
        response_anchor_sample_ratio = self._get_response_anchor_sample_ratio()
        response_anchor_seed = self._get_response_anchor_seed()
        rejected_draft_max_tokens_per_sample = self._get_rejected_draft_max_tokens_per_sample()
        draft_sample_temperature = self._get_draft_sample_temperature()
        draft_sample_seed = self._get_draft_sample_seed()
        if onpolicy_reverse_enabled and random_response_anchor_enabled:
            raise NotImplementedError(
                "paradistill on-policy reverse (verl_dflash_onpolicy_reverse_enabled) is not supported together "
                "with verl_dflash_random_response_anchor_enabled."
            )
        if topk_enabled and (onpolicy_reverse_enabled or random_response_anchor_enabled):
            raise NotImplementedError(
                "draftopd top-K losses (topk_fkl / topk_tv) are not supported together with on-policy reverse "
                "or random response anchors."
            )
        if (reject_token_topk or post_reject_topk) and response_anchor_mode != "reject":
            raise NotImplementedError(
                "draftopd reject-region top-K losses (reject_token/post_reject loss_mode=topk_fkl|topk_reverse_kl) "
                "require reject-mode anchors."
            )
        topk_fkl_teacher_k = self._get_topk_fkl_teacher_k() if topk_enabled else 0
        topk_fkl_student_k = self._get_topk_fkl_student_k() if topk_enabled else 0
        topk_fkl_mode = self._resolve_topk_fkl_mode(topk_fkl_teacher_k, topk_fkl_student_k) if topk_enabled else "teacher"
        if (
            reject_token_loss_mode == "topk_reverse_kl" or post_reject_loss_mode == "topk_reverse_kl"
        ) and topk_fkl_student_k <= 0:
            # The rejected token d is a student-top-K token generally outside the teacher top-K, so teacher-only
            # support would train only the corrected token y. Require student support so S covers both y and d.
            raise ValueError(
                "topk_reverse_kl requires verl_dflash_topk_fkl_student_k > 0 (union support); with student_k=0 "
                "the top-K support is teacher-only and excludes the rejected token d."
            )
        split_random_rejected_pass = random_response_anchor_enabled
        # The rollout-reject stream exists only for reject-mode anchors (draftopd); free anchors (paradistill)
        # have none. verl_dflash_rejected_draft_reverse_enabled=False skips the whole reject-stream compute
        # (no LM-head softmax over the slots), not just zeros its loss.
        reject_stream_active = (
            response_anchor_mode == "reject" and self._get_rejected_draft_reverse_enabled()
        )
        need_rollout_reject_anchors = reject_stream_active
        anchor_positions, segment_lens, row_starts, block_keep_mask, valid_seq_lens, opd_metrics = (
            self._build_opd_anchor_plan(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                reject_token_indices=reject_token_indices,
                draft_block_size=draft_block_size,
                response_anchor_stride=response_anchor_stride,
                random_response_anchor_enabled=random_response_anchor_enabled,
                random_response_anchor_seed=random_response_anchor_seed,
                anchor_mode=response_anchor_mode,
                anchor_sample_ratio=response_anchor_sample_ratio,
                anchor_seed=response_anchor_seed,
                rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                rejected_draft_offsets=rejected_draft_offsets,
                rejected_draft_mask=rejected_draft_mask,
                include_rejected_draft_anchors=not split_random_rejected_pass and need_rollout_reject_anchors,
            )
        )

        batch_size, seq_len = input_ids.shape

        def _plan_block_stats(
            positions: torch.Tensor,
            keep_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, int, torch.Tensor]:
            actual = keep_mask.sum()
            if bool(keep_mask.any()):
                padded = batch_size * int(positions.shape[1])
                max_per_sample = keep_mask.sum(dim=1).max().to(dtype=torch.float32)
            else:
                padded = 0
                max_per_sample = actual.to(dtype=torch.float32)
            return actual, padded, max_per_sample

        rejected_anchor_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        rejected_block_keep_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device)
        if split_random_rejected_pass:
            rejected_anchor_positions, rejected_block_keep_mask = self._build_rejected_draft_anchor_plan(
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                draft_block_size=draft_block_size,
                max_tokens_per_sample=rejected_draft_max_tokens_per_sample,
                rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                rejected_draft_offsets=rejected_draft_offsets,
                rejected_draft_token_ids=rejected_draft_token_ids,
                rejected_draft_mask=rejected_draft_mask,
            )

        response_actual_block_count, response_padded_block_count, response_max_blocks = _plan_block_stats(
            anchor_positions, block_keep_mask
        )
        rejected_actual_block_count, rejected_padded_block_count, rejected_max_blocks = _plan_block_stats(
            rejected_anchor_positions, rejected_block_keep_mask
        )
        actual_block_count = response_actual_block_count + rejected_actual_block_count
        padded_block_count = response_padded_block_count + rejected_padded_block_count
        max_blocks_per_sample = torch.maximum(response_max_blocks, rejected_max_blocks)
        if split_random_rejected_pass:
            max_blocks_per_sample = (block_keep_mask.sum(dim=1) + rejected_block_keep_mask.sum(dim=1)).max().to(
                dtype=torch.float32
            )
        draft_q_token_count = padded_block_count * draft_block_size

        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Main model output embeddings are required for composed DFLASH student logits.")

        lm_head_start_time = time.perf_counter()
        log_probs_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        loss_mask_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        entropy_by_seq = (
            target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32) if calculate_entropy else None
        )

        rejected_width = 1
        if rejected_draft_anchor_indices is not None and rejected_draft_anchor_indices.dim() >= 2:
            rejected_width = max(1, int(rejected_draft_anchor_indices.shape[1]))
        rejected_student_log_probs = target_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        rejected_teacher_log_probs = target_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        rejected_loss_mask = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=input_ids.device)
        # Per-slot reject-token vs post-reject tag (see _collect_rejected_draft_log_probs); used by the loss
        # only for mixed reject regions. All-False for paradistill / no-reject.
        rejected_is_reject_token = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=input_ids.device)
        # paradistill reverse stream is kept FLAT (one entry per re-sampled (block, offset) slot), NOT scattered
        # by sequence position. Overlapping blocks (sampled-mode anchors) each re-sample their own fresh
        # draft token from a different head/context, so every slot is a distinct on-policy sample and must
        # contribute its own reverse loss -- no per-position dedup. N_rejected = total slots (>= N_response).
        onpolicy_flat_log_q = onpolicy_flat_log_p = onpolicy_flat_offsets = None
        response_lm_token_count = 0
        attn_impl = str(getattr(getattr(self.draft_model, "config", None), "_attn_implementation", "eager"))
        ran_draft_forward = False

        if bool(block_keep_mask.any()):
            draft_hidden, attn_impl, response_draft_forward_ms = self._run_dflash_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                draft_block_size=draft_block_size,
                profile_enabled=profile_enabled,
                checkpoint_forward=split_random_rejected_pass,
            )
            draft_forward_ms += response_draft_forward_ms
            ran_draft_forward = True

            response_batch_indices: list[torch.Tensor] = []
            response_draft_indices: list[torch.Tensor] = []
            response_row_indices: list[torch.Tensor] = []
            response_labels: list[torch.Tensor] = []
            for block_offset in range(1, draft_block_size):
                active_blocks = block_keep_mask & (segment_lens >= block_offset)
                if not bool(active_blocks.any()):
                    continue
                block_indices = torch.nonzero(active_blocks, as_tuple=False)
                batch_indices = block_indices[:, 0]
                anchor_block_indices = block_indices[:, 1]
                row_indices = row_starts[batch_indices, anchor_block_indices] + (block_offset - 1)
                label_indices = row_indices + 1
                in_bounds = label_indices < valid_seq_lens[batch_indices]
                if not bool(in_bounds.any()):
                    continue
                batch_indices = batch_indices[in_bounds]
                anchor_block_indices = anchor_block_indices[in_bounds]
                row_indices = row_indices[in_bounds]
                label_indices = label_indices[in_bounds]
                flat_draft_indices = anchor_block_indices * draft_block_size + block_offset
                response_batch_indices.append(batch_indices)
                response_draft_indices.append(flat_draft_indices)
                response_row_indices.append(row_indices)
                response_labels.append(input_ids[batch_indices, label_indices])

            if response_batch_indices:
                response_batch_tensor = torch.cat(response_batch_indices, dim=0)
                response_draft_tensor = torch.cat(response_draft_indices, dim=0)
                response_row_tensor = torch.cat(response_row_indices, dim=0)
                response_label_tensor = torch.cat(response_labels, dim=0)
                if onpolicy_reverse_enabled:
                    # paradistill: ONE draft->vocab projection per slot yields both the response-forward
                    # log q(y_j) AND a fresh on-policy reverse sample y_hat with log q(y_hat) [grad] /
                    # log p(y_hat) [no grad]. Slots stay flat (overlap duplicates of sampled-mode anchors
                    # retained), so the reverse loss is computed on every re-sampled block.
                    sample_generator = None
                    if draft_sample_seed >= 0:
                        sample_generator = torch.Generator(device=draft_hidden.device).manual_seed(
                            int(draft_sample_seed)
                        )
                    selected_log_probs, selected_entropy, onpolicy_flat_log_q, onpolicy_flat_log_p, _ = (
                        self._compute_response_and_reverse_log_probs(
                            draft_hidden=draft_hidden,
                            output_embeddings=output_embeddings,
                            batch_indices=response_batch_tensor,
                            draft_indices=response_draft_tensor,
                            token_ids=response_label_tensor,
                            teacher_logits=teacher_logits,
                            row_indices=response_row_tensor,
                            chunk_size=lm_head_chunk_size,
                            calculate_entropy=calculate_entropy,
                            sample_temperature=draft_sample_temperature,
                            generator=sample_generator,
                        )
                    )
                    # Each slot's draft index packs anchor_block * block_size + offset, so the offset (the
                    # draft head index, 1..K) recovers from a modulo -- used for optional per-offset decay.
                    onpolicy_flat_offsets = response_draft_tensor % draft_block_size
                    log_probs_by_seq[response_batch_tensor, response_row_tensor] = selected_log_probs
                    loss_mask_by_seq[response_batch_tensor, response_row_tensor] = 1.0
                    if entropy_by_seq is not None and selected_entropy is not None:
                        entropy_by_seq[response_batch_tensor, response_row_tensor] = selected_entropy
                else:
                    # Forward response stream (SEMANTIC OVERLOAD): each slot's log_probs channel holds log q(y_j)
                    # (bernoulli_fkl) or a precomputed top-K divergence (topk_fkl / topk_tv). The response
                    # (accepted) region uses response_loss_mode, the reject-accept region (corrected y) uses
                    # reject_accept_loss_mode.
                    def _forward_region_values(mode, sel_batch, sel_draft, sel_row, sel_labels):
                        if mode in ("topk_fkl", "topk_tv"):
                            topk_fn = self._compute_topk_forward_kl if mode == "topk_fkl" else self._compute_topk_tv
                            return topk_fn(
                                draft_hidden=draft_hidden,
                                output_embeddings=output_embeddings,
                                batch_indices=sel_batch,
                                draft_indices=sel_draft,
                                teacher_logits=teacher_logits,
                                row_indices=sel_row,
                                chunk_size=lm_head_chunk_size,
                                mode=topk_fkl_mode,
                                topk_teacher=topk_fkl_teacher_k,
                                topk_student=topk_fkl_student_k,
                                calculate_entropy=calculate_entropy,
                            )
                        return self._compute_selected_lm_log_probs(  # bernoulli_fkl -> log q(y_j)
                            draft_hidden=draft_hidden,
                            output_embeddings=output_embeddings,
                            batch_indices=sel_batch,
                            draft_indices=sel_draft,
                            token_ids=sel_labels,
                            chunk_size=lm_head_chunk_size,
                            calculate_entropy=calculate_entropy,
                        )

                    response_split = (
                        response_anchor_mode == "reject" and response_loss_mode != reject_accept_loss_mode
                    )
                    if not response_split:
                        # Uniform: one helper call on the full response tensors (pre-refactor hot path -- no
                        # mask/nonzero/gather).
                        selected_log_probs, selected_entropy = _forward_region_values(
                            response_loss_mode, response_batch_tensor, response_draft_tensor,
                            response_row_tensor, response_label_tensor,
                        )
                        log_probs_by_seq[response_batch_tensor, response_row_tensor] = selected_log_probs
                        loss_mask_by_seq[response_batch_tensor, response_row_tensor] = 1.0
                        if entropy_by_seq is not None and selected_entropy is not None:
                            entropy_by_seq[response_batch_tensor, response_row_tensor] = selected_entropy
                    else:
                        # Mixed forward modes: split by region (corrected = the predicted label token sits at an
                        # SD reject position) and dispatch each subset to its mode's helper.
                        corrected_seq_mask = torch.zeros(
                            (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
                        )
                        if reject_token_indices is not None and reject_token_indices.numel() > 0:
                            rj = reject_token_indices.to(torch.long)
                            rj_valid = (rj >= 0) & (rj < response_lengths.unsqueeze(1))
                            seq_pos = (prompt_lengths.unsqueeze(1) + rj).clamp_(0, seq_len - 1)
                            rows = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand_as(rj)
                            corrected_seq_mask[rows[rj_valid], seq_pos[rj_valid]] = True
                        slot_is_corrected = corrected_seq_mask[response_batch_tensor, response_row_tensor + 1]
                        for mode in dict.fromkeys((response_loss_mode, reject_accept_loss_mode)):
                            sel = torch.zeros(response_batch_tensor.numel(), dtype=torch.bool, device=input_ids.device)
                            if response_loss_mode == mode:
                                sel = sel | slot_is_corrected.logical_not()
                            if reject_accept_loss_mode == mode:
                                sel = sel | slot_is_corrected
                            sub = torch.nonzero(sel, as_tuple=False).squeeze(-1)
                            sub_batch = response_batch_tensor[sub]
                            sub_row = response_row_tensor[sub]
                            sub_vals, sub_entropy = _forward_region_values(
                                mode, sub_batch, response_draft_tensor[sub], sub_row, response_label_tensor[sub],
                            )
                            log_probs_by_seq[sub_batch, sub_row] = sub_vals
                            loss_mask_by_seq[sub_batch, sub_row] = 1.0
                            if entropy_by_seq is not None and sub_entropy is not None:
                                entropy_by_seq[sub_batch, sub_row] = sub_entropy
                response_lm_token_count = response_batch_tensor.numel()

            if onpolicy_reverse_enabled:
                # paradistill feeds the fresh-sample reverse stream through the rejected-draft channel.
                # Pack the flat per-slot tensors into per-SAMPLE rows (batch, width) -- overlap duplicates
                # kept as separate columns, per-slot offsets (1..K) carried for optional decay -- so the
                # engine's per-sample dynamic-batch restore and the loss both see the standard layout. When
                # there are no slots the zero-initialized (batch, rejected_width) buffers already give an
                # all-False mask (the loss skips the reverse stream), so only the populated case is handled.
                if onpolicy_flat_log_q is not None and onpolicy_flat_log_q.numel() > 0:
                    col, width = self._per_sample_slot_columns(response_batch_tensor, batch_size)
                    rejected_student_log_probs = target_hidden.new_zeros((batch_size, width), dtype=torch.float32)
                    rejected_teacher_log_probs = target_hidden.new_zeros((batch_size, width), dtype=torch.float32)
                    rejected_loss_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=input_ids.device)
                    rejected_offsets = torch.zeros((batch_size, width), dtype=torch.long, device=input_ids.device)
                    rejected_student_log_probs[response_batch_tensor, col] = onpolicy_flat_log_q
                    rejected_teacher_log_probs[response_batch_tensor, col] = onpolicy_flat_log_p
                    rejected_loss_mask[response_batch_tensor, col] = True
                    rejected_offsets[response_batch_tensor, col] = onpolicy_flat_offsets.to(torch.long)
                    rejected_draft_offsets = rejected_offsets
            elif reject_stream_active and not split_random_rejected_pass:
                # draftopd reject stream: reject-token + post-reject regions, each reverse_kl or top-K FKL
                # (SEMANTIC OVERLOAD: both ride in the student/teacher channels; offsets stay raw for decay).
                (
                    rejected_student_log_probs,
                    rejected_teacher_log_probs,
                    rejected_loss_mask,
                    rejected_is_reject_token,
                ) = self._collect_rejected_draft_log_probs(
                    draft_hidden=draft_hidden,
                    output_embeddings=output_embeddings,
                    prompt_lengths=prompt_lengths,
                    response_lengths=response_lengths,
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    draft_block_size=draft_block_size,
                    lm_head_chunk_size=lm_head_chunk_size,
                    max_tokens_per_sample=rejected_draft_max_tokens_per_sample,
                    rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                    rejected_draft_offsets=rejected_draft_offsets,
                    rejected_draft_token_ids=rejected_draft_token_ids,
                    rejected_draft_teacher_logprobs=rejected_draft_teacher_logprobs,
                    rejected_draft_mask=rejected_draft_mask,
                    reject_token_mode=reject_token_loss_mode,
                    post_reject_mode=post_reject_loss_mode,
                    teacher_logits=teacher_logits,
                    topk_fkl_mode=topk_fkl_mode,
                    topk_fkl_teacher_k=topk_fkl_teacher_k,
                    topk_fkl_student_k=topk_fkl_student_k,
                )
            del draft_hidden

        if split_random_rejected_pass and bool(rejected_block_keep_mask.any()):
            rejected_draft_hidden, rejected_attn_impl, rejected_draft_forward_ms = self._run_dflash_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=rejected_anchor_positions,
                block_keep_mask=rejected_block_keep_mask,
                draft_block_size=draft_block_size,
                profile_enabled=profile_enabled,
                checkpoint_forward=True,
            )
            draft_forward_ms += rejected_draft_forward_ms
            attn_impl = rejected_attn_impl
            ran_draft_forward = True
            (
                rejected_student_log_probs,
                rejected_teacher_log_probs,
                rejected_loss_mask,
                rejected_is_reject_token,
            ) = self._collect_rejected_draft_log_probs(
                draft_hidden=rejected_draft_hidden,
                output_embeddings=output_embeddings,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                anchor_positions=rejected_anchor_positions,
                block_keep_mask=rejected_block_keep_mask,
                draft_block_size=draft_block_size,
                lm_head_chunk_size=lm_head_chunk_size,
                max_tokens_per_sample=rejected_draft_max_tokens_per_sample,
                rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                rejected_draft_offsets=rejected_draft_offsets,
                rejected_draft_token_ids=rejected_draft_token_ids,
                rejected_draft_teacher_logprobs=rejected_draft_teacher_logprobs,
                rejected_draft_mask=rejected_draft_mask,
                reject_token_mode=reject_token_loss_mode,
                post_reject_mode=post_reject_loss_mode,
                teacher_logits=teacher_logits,
                topk_fkl_mode=topk_fkl_mode,
                topk_fkl_teacher_k=topk_fkl_teacher_k,
                topk_fkl_student_k=topk_fkl_student_k,
            )
            del rejected_draft_hidden

        if not ran_draft_forward:
            trainable_param = next((param for param in self.draft_model.parameters() if param.requires_grad), None)
            if trainable_param is None:
                raise RuntimeError("DFLASH OPD requires at least one trainable draft parameter.")
            zero_with_grad = trainable_param.flatten()[0].float() * 0.0
            log_probs_by_seq = log_probs_by_seq + zero_with_grad
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        lm_head_ms = (time.perf_counter() - lm_head_start_time) * 1000.0
        rejected_lm_token_count = rejected_loss_mask.sum()
        selected_lm_token_count = loss_mask_by_seq.sum() + rejected_lm_token_count

        # Return a plain container so FSDP can discover tensors and attach
        # pre-backward hooks. A custom object can leave FSDP in IDLE at backward.
        output = {
            "dflash_log_probs": log_probs_by_seq,
            "dflash_loss_mask": loss_mask_by_seq,
            "dflash_opd_valid_anchor_count": log_probs_by_seq.new_tensor(opd_metrics["valid_anchor_count"]),
            "dflash_opd_skipped_sample_count": log_probs_by_seq.new_tensor(opd_metrics["skipped_sample_count"]),
            "dflash_opd_empty_reject_sample_count": log_probs_by_seq.new_tensor(
                opd_metrics["empty_reject_sample_count"]
            ),
            "dflash_opd_total_reject_count": log_probs_by_seq.new_tensor(opd_metrics["total_reject_count"]),
            "dflash_opd_sample_count": log_probs_by_seq.new_tensor(opd_metrics["sample_count"]),
            "dflash_opd_target_token_count": loss_mask_by_seq.sum(),
            "dflash_rejected_draft_student_log_probs": rejected_student_log_probs,
            "dflash_rejected_draft_teacher_log_probs": rejected_teacher_log_probs,
            "dflash_rejected_draft_loss_mask": rejected_loss_mask,
            "dflash_rejected_draft_offsets": rejected_draft_offsets
            if rejected_draft_offsets is not None
            else torch.zeros_like(rejected_loss_mask, dtype=torch.long),
            "dflash_rejected_draft_is_reject_token": rejected_is_reject_token,
            "dflash_opd_rejected_draft_token_count": rejected_lm_token_count,
            "dflash_opd_actual_block_count": actual_block_count.to(dtype=torch.float32),
            "dflash_opd_padded_block_count": log_probs_by_seq.new_tensor(padded_block_count),
            "dflash_opd_max_blocks_per_sample": max_blocks_per_sample,
            "dflash_opd_draft_q_token_count": log_probs_by_seq.new_tensor(draft_q_token_count),
            "dflash_opd_response_lm_token_count": log_probs_by_seq.new_tensor(response_lm_token_count),
            "dflash_opd_rejected_lm_token_count": rejected_lm_token_count,
            "dflash_opd_selected_lm_token_count": selected_lm_token_count,
            "dflash_opd_lm_head_chunk_size": log_probs_by_seq.new_tensor(lm_head_chunk_size),
            "dflash_opd_response_anchor_stride": log_probs_by_seq.new_tensor(response_anchor_stride),
            "dflash_opd_rejected_draft_max_tokens_per_sample": log_probs_by_seq.new_tensor(
                rejected_draft_max_tokens_per_sample or 0
            ),
            "dflash_opd_attention_impl_id": log_probs_by_seq.new_tensor(
                DFLASH_ATTENTION_IMPL_IDS.get(attn_impl, -1)
            ),
        }
        if profile_enabled:
            output.update(
                {
                    "dflash_opd_profile_teacher_forward_ms": log_probs_by_seq.new_tensor(teacher_forward_ms),
                    "dflash_opd_profile_draft_forward_ms": log_probs_by_seq.new_tensor(draft_forward_ms),
                    "dflash_opd_profile_lm_head_ms": log_probs_by_seq.new_tensor(lm_head_ms),
                    "dflash_opd_profile_total_forward_ms": log_probs_by_seq.new_tensor(
                        (time.perf_counter() - total_start_time) * 1000.0
                    ),
                }
            )
        if entropy_by_seq is not None:
            output["dflash_entropy"] = entropy_by_seq
        # Top-K FKL reuses existing channels (no new output keys); see the SEMANTIC OVERLOAD notes above.
        return output

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        dflash_prompt_lengths: Optional[torch.LongTensor] = None,
        dflash_response_lengths: Optional[torch.LongTensor] = None,
        dflash_reject_token_indices: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_anchor_indices: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_offsets: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_token_ids: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_teacher_logprobs: Optional[torch.Tensor] = None,
        dflash_rejected_draft_mask: Optional[torch.Tensor] = None,
        dflash_calculate_entropy: bool = False,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("ComposedDFlashStudentForCausalLM requires either input_ids or inputs_embeds.")

        if dflash_reject_token_indices is not None:
            if dflash_prompt_lengths is None or dflash_response_lengths is None:
                raise ValueError("DFLASH OPD requires prompt and response lengths with reject-token indices.")
            if input_ids is None:
                raise ValueError("DFLASH OPD path requires input_ids.")
            return self._forward_opd(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_lengths=dflash_prompt_lengths,
                response_lengths=dflash_response_lengths,
                reject_token_indices=dflash_reject_token_indices,
                rejected_draft_anchor_indices=dflash_rejected_draft_anchor_indices,
                rejected_draft_offsets=dflash_rejected_draft_offsets,
                rejected_draft_token_ids=dflash_rejected_draft_token_ids,
                rejected_draft_teacher_logprobs=dflash_rejected_draft_teacher_logprobs,
                rejected_draft_mask=dflash_rejected_draft_mask,
                calculate_entropy=bool(dflash_calculate_entropy),
                **kwargs,
            )

        # Forward the teacher under no_grad to produce DFLASH context features.
        with torch.no_grad():
            teacher_outputs = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            if teacher_outputs.hidden_states is None:
                raise RuntimeError("Teacher model did not return hidden states required by DFLASH draft.")

            if inputs_embeds is None:
                noise_embedding = self.get_input_embeddings()(input_ids)
            else:
                noise_embedding = inputs_embeds
            target_hidden = self._extract_target_hidden(teacher_outputs.hidden_states)

        draft_position_ids = self._resolve_position_ids(noise_embedding=noise_embedding, position_ids=position_ids)
        draft_outputs = self.draft_model(
            position_ids=draft_position_ids,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=bool(use_cache),
        )

        if isinstance(draft_outputs, tuple):
            draft_hidden = draft_outputs[0]
        elif hasattr(draft_outputs, "last_hidden_state"):
            draft_hidden = draft_outputs.last_hidden_state
        else:
            draft_hidden = draft_outputs

        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Main model output embeddings are required for composed DFLASH student logits.")
        logits = output_embeddings(draft_hidden)
        model_hidden_states = (draft_hidden,) if output_hidden_states else None

        if not return_dict:
            output = (logits,)
            if model_hidden_states is not None:
                output = output + (model_hidden_states,)
            return output

        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=model_hidden_states,
        )


def build_composed_dflash_student(
    *,
    main_model_path: str,
    draft_model_path: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    config: PretrainedConfig,
) -> ComposedDFlashStudentForCausalLM:
    main_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=main_model_path,
        torch_dtype=torch_dtype,
        config=config,
        trust_remote_code=trust_remote_code,
    )

    try:
        draft_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=draft_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except OSError as exc:
        logger.warning(
            "Failed to load draft model weights from %s (%s). Falling back to config-only initialization.",
            draft_model_path,
            exc,
        )
        draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
        draft_model = AutoModel.from_config(draft_config, trust_remote_code=True)

    model = ComposedDFlashStudentForCausalLM(config=config, main_model=main_model, draft_model=draft_model)
    return model
