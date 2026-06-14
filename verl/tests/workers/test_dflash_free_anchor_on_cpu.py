"""CPU tests for Anchored Block-OPD free anchors (verl_dflash_response_anchor_mode).

Free (stride_k / sampled) anchors reuse the existing DFLASH OPD anchor plan + forward + scalar
two-stream loss; only the response-anchor selection changes. These tests cover the free-anchor
enumeration and that `_build_opd_anchor_plan` produces the right plan in free mode while leaving the
default reject mode unchanged.
"""

from types import SimpleNamespace

import torch
from tensordict import NonTensorData, TensorDict

from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.core_algos import kl_penalty


def _single_sequence_logprobs(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return torch.nested.as_nested_tensor([values], layout=torch.jagged)


def _bernoulli_forward_kl(student, teacher):
    student = torch.as_tensor(student, dtype=torch.float32)
    teacher = torch.as_tensor(teacher, dtype=torch.float32)
    sp, tp = student.exp(), teacher.exp()
    return tp * (teacher - student) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))


def test_free_response_anchors_stride_k_covers_response_once():
    # response_len=10, num_predict=3 (block_size=4): non-overlapping stride-3 cover.
    anchors, segments = ComposedDFlashStudentForCausalLM._build_free_response_anchors(
        response_len=10, num_predict=3, mode="stride_k", sample_ratio=1.0, seed=0
    )
    assert anchors == [-1, 2, 5, 8]  # response coords; -1 is the last prompt token
    # anchor a predicts response positions a+1 .. a+seg; the union must cover [0..9] exactly once.
    covered = []
    for a, seg in zip(anchors, segments, strict=True):
        covered.extend(a + j for j in range(1, seg + 1))
    assert covered == list(range(0, 10))
    assert sum(segments) == 10  # == response_len


def test_free_response_anchors_clamp_and_sampled():
    # last block truncates so heads never predict past the response end.
    anchors, segments = ComposedDFlashStudentForCausalLM._build_free_response_anchors(
        response_len=5, num_predict=4, mode="stride_k", sample_ratio=1.0, seed=0
    )
    assert list(zip(anchors, segments, strict=True)) == [(-1, 4), (3, 1)]

    sampled_a = ComposedDFlashStudentForCausalLM._build_free_response_anchors(
        response_len=20, num_predict=3, mode="sampled", sample_ratio=0.5, seed=7
    )
    sampled_b = ComposedDFlashStudentForCausalLM._build_free_response_anchors(
        response_len=20, num_predict=3, mode="sampled", sample_ratio=0.5, seed=7
    )
    assert sampled_a == sampled_b  # reproducible given the seed
    assert sampled_a[0][0] == -1  # first anchor always kept


def test_build_opd_anchor_plan_stride_k_mode_uses_free_anchors():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(12, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    anchor_positions, segment_lens, row_starts, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([10]),
        reject_token_indices=torch.tensor([[-1]]),  # ignored in free mode
        draft_block_size=4,
        anchor_mode="stride_k",
    )
    assert anchor_positions[block_keep_mask].tolist() == [1, 4, 7, 10]  # prompt_len-1 + free anchors
    assert torch.equal(row_starts, anchor_positions)
    assert int(segment_lens[block_keep_mask].sum().item()) == 10  # covers the whole response
    assert metrics["valid_anchor_count"] == 4
    assert metrics["total_reject_count"] == 0  # reject path not taken


def test_build_opd_anchor_plan_default_mode_is_unchanged():
    # Default anchor_mode="reject" must reproduce the existing reject-driven behavior.
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(25, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    _, segment_lens, _, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([5]),
        response_lengths=torch.tensor([20]),
        reject_token_indices=torch.tensor([[2]]),
        draft_block_size=16,
    )
    assert metrics["valid_anchor_count"] == 2
    assert int(segment_lens[block_keep_mask].max().item()) == 15


def _make_response_stream_case(reject_indices):
    # 1 prompt token + 3 response tokens; reject at the given response positions.
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, -0.6, 0.0]),
        },
        batch_size=[1],
    )
    data.set("dflash_reject_token_indices", NonTensorData([reject_indices]))
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, -0.3, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
    }
    return data, model_output


def _loss_cfg(corrected_token_forward_only):
    return SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.5,
        forward_kl_weight=1.0,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
        corrected_token_forward_only=corrected_token_forward_only,
    )


def test_corrected_token_forward_only_zeros_reverse_at_reject_positions():
    # reject at response position 1 -> token y_1 should be forward-KL only; positions 0,2 stay symmetric.
    data, model_output = _make_response_stream_case(reject_indices=[1])
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})

    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=_loss_cfg(True)), model_output, data)

    student = torch.tensor([-0.4, -0.9, -0.3])
    teacher = torch.tensor([-0.5, -0.7, -0.6])
    k3 = kl_penalty(student, teacher, "k3")
    bern = _bernoulli_forward_kl(student, teacher)
    rev_mask = torch.tensor([1.0, 0.0, 1.0])  # reverse off at the corrected (reject) position
    per_token = 0.5 * k3 * rev_mask + 1.0 * bern
    assert torch.allclose(loss, per_token.mean())


def test_corrected_token_forward_only_false_keeps_symmetric_everywhere():
    data, model_output = _make_response_stream_case(reject_indices=[1])
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})

    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=_loss_cfg(False)), model_output, data)

    student = torch.tensor([-0.4, -0.9, -0.3])
    teacher = torch.tensor([-0.5, -0.7, -0.6])
    per_token = 0.5 * kl_penalty(student, teacher, "k3") + 1.0 * _bernoulli_forward_kl(student, teacher)
    assert torch.allclose(loss, per_token.mean())
