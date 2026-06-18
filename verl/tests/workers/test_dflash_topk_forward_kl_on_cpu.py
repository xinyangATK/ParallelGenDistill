"""CPU tests for draftopd top-K forward KL (verl_dflash_topk_fkl_{response,reject}_enabled).

The top-K forward KL is computed in-model from the frozen teacher's full logits:
``sum_{v in S} p_t(v) (log p_t(v) - log q_s(v))`` where ``S`` is the teacher / student / union top-K set.
Request 1 swaps the response forward term for this; request 2 swaps the reject stream's reverse KL for a
top-K forward KL on the draft's full-block-depth predictions (still using the rejected-draft channel,
weight, and optional offset decay). These tests cover the new helper and both loss-consumption paths.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM
from verl.trainer.distillation.losses import (
    compute_distillation_loss_reverse_kl_estimator,
    distillation_loss,
)
from verl.workers.utils.padding import no_padding_2_padding


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    batch, draft_len, seq, hidden, vocab = 2, 5, 7, 4, 6
    draft_hidden = torch.randn(batch, draft_len, hidden, dtype=torch.float32)
    teacher_logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    output_embeddings = torch.nn.Linear(hidden, vocab, bias=False)
    batch_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    draft_indices = torch.tensor([1, 3, 2], dtype=torch.long)
    row_indices = torch.tensor([2, 4, 3], dtype=torch.long)
    token_ids = torch.tensor([0, 2, 1], dtype=torch.long)
    return draft_hidden, teacher_logits, output_embeddings, batch_indices, draft_indices, row_indices, token_ids


def _manual_topk_fkl(draft_hidden, teacher_logits, out, b, d, r, mode, k):
    s_logits = out(draft_hidden[b, d, :]).float()
    s_lp = F.log_softmax(s_logits, dim=-1)
    t_logits = teacher_logits[b, r, :].float()
    t_lp = F.log_softmax(t_logits, dim=-1)
    t_p = t_lp.exp()
    sel = torch.zeros_like(t_lp, dtype=torch.bool)
    if mode in ("teacher", "union"):
        sel.scatter_(-1, t_logits.topk(k, dim=-1).indices, True)
    if mode in ("student", "union"):
        sel.scatter_(-1, s_logits.topk(k, dim=-1).indices, True)
    return (t_p * (t_lp - s_lp) * sel).sum(dim=-1).clamp_min(0.0)


def _call(student, draft_hidden, teacher_logits, out, b, d, r, *, mode, k=2, chunk_size=64,
          calculate_entropy=False):
    return student._compute_topk_forward_kl(
        draft_hidden=draft_hidden,
        output_embeddings=out,
        batch_indices=b,
        draft_indices=d,
        teacher_logits=teacher_logits,
        row_indices=r,
        chunk_size=chunk_size,
        mode=mode,
        topk=k,
        calculate_entropy=calculate_entropy,
    )


def test_topk_forward_kl_matches_manual_each_mode():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs()
    for mode in ("teacher", "student", "union"):
        fkl, _ = _call(student, draft_hidden, teacher_logits, out, b, d, r, mode=mode, k=2, chunk_size=2)
        expected = _manual_topk_fkl(draft_hidden, teacher_logits, out, b, d, r, mode, k=2)
        assert torch.allclose(fkl, expected, atol=1e-6), mode
        assert fkl.shape == (b.numel(),)
        assert (fkl >= -1e-6).all()  # clamped to >=0 (top-K mass < 1)


def test_topk_forward_kl_union_dedups_overlapping_indices():
    # The boolean union mask must not double-count tokens shared by the teacher and student top-K.
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs(seed=3)
    union, _ = _call(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=3)
    manual = _manual_topk_fkl(draft_hidden, teacher_logits, out, b, d, r, "union", k=3)
    assert torch.allclose(union, manual, atol=1e-6)


def test_topk_forward_kl_grad_flows_through_student_only():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs()
    draft_hidden.requires_grad_(True)
    teacher_logits.requires_grad_(True)
    fkl, _ = _call(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=2, chunk_size=2)
    assert fkl.requires_grad
    fkl.sum().backward()
    assert teacher_logits.grad is None  # teacher term is detached (no_grad)
    touched = draft_hidden.grad.abs().sum(dim=-1) > 0
    assert touched[0, 1] and touched[0, 3] and touched[1, 2]


def test_topk_forward_kl_handles_empty_selection():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, *_ = _make_inputs()
    empty = torch.empty((0,), dtype=torch.long)
    fkl, ent = _call(student, draft_hidden, teacher_logits, out, empty, empty, empty, mode="teacher")
    assert fkl.numel() == 0 and ent is None


def test_collect_rejected_draft_topk_fkl_uses_realized_teacher_and_drops_oob():
    # Request 2 aligns to draftopd's reject geometry: the SAME rollout-reject slots, only the loss differs.
    # Here block_size K=3 (offsets 1,2), prompt_len=2, response_len=4 (valid_len=6), two block anchors at
    # full positions 2 and 5. Slot A (anchor 2, offset 1) -> draft slot 1, realized target pos 3 (teacher
    # row 2) -> kept. Slot B (anchor 5, offset 2) -> realized target pos 7 >= valid_len -> DROPPED.
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    torch.manual_seed(0)
    hidden, vocab, K = 4, 6, 3
    draft_hidden = torch.randn(1, 2 * K, hidden)  # 2 blocks x K slots
    teacher_logits = torch.randn(1, 6, vocab)
    out = torch.nn.Linear(hidden, vocab, bias=False)
    long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    student_t, teacher_t, mask_t = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=out,
        prompt_lengths=long([2]),
        response_lengths=long([4]),
        anchor_positions=long([[2, 5]]),
        block_keep_mask=torch.tensor([[True, True]]),
        draft_block_size=K,
        lm_head_chunk_size=64,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=long([[0, 3]]),   # anchor_resp -> full 2 and 5
        rejected_draft_offsets=long([[1, 2]]),
        rejected_draft_token_ids=long([[3, 2]]),
        rejected_draft_teacher_logprobs=None,           # unused for the FKL path
        rejected_draft_mask=torch.tensor([[True, True]]),
        compute_topk_fkl=True,
        teacher_logits=teacher_logits,
        topk_fkl_mode="teacher",
        topk_fkl_k=2,
    )
    assert mask_t.tolist() == [[True, False]]           # slot B dropped (realized target out of bounds)
    assert teacher_t.abs().sum() == 0                   # teacher channel unused for the FKL path
    expected = _manual_topk_fkl(
        draft_hidden, teacher_logits, out, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), "teacher", k=2
    )
    assert torch.allclose(student_t[0, 0], expected[0], atol=1e-6)  # FKL at draft slot 1 vs teacher row 2
    assert student_t[0, 1] == 0.0


def test_collect_rejected_draft_baseline_path_unchanged():
    # compute_topk_fkl=False keeps the baseline reverse-stream behavior: student = log q(d) at the slot,
    # teacher = the cached scalar log p(d).
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    torch.manual_seed(1)
    hidden, vocab, K = 4, 6, 3
    draft_hidden = torch.randn(1, K, hidden)
    out = torch.nn.Linear(hidden, vocab, bias=False)
    long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    student_t, teacher_t, mask_t = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=out,
        prompt_lengths=long([2]),
        response_lengths=long([4]),
        anchor_positions=long([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=K,
        lm_head_chunk_size=64,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=long([[0]]),
        rejected_draft_offsets=long([[1]]),
        rejected_draft_token_ids=long([[3]]),
        rejected_draft_teacher_logprobs=torch.tensor([[-0.7]]),
        rejected_draft_mask=torch.tensor([[True]]),
    )
    assert mask_t.tolist() == [[True]]
    expected_logq = F.log_softmax(out(draft_hidden[0, 1]).float(), -1)[3]  # log q at token 3, draft slot 1
    assert torch.allclose(student_t[0, 0], expected_logq, atol=1e-6)
    assert torch.allclose(teacher_t[0, 0], torch.tensor(-0.7), atol=1e-6)


def _data_one_sample():
    return TensorDict(
        {
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1, 1]]),
            "teacher_logprobs": torch.nested.as_nested_tensor(
                [torch.tensor([-0.5, -0.7, -0.6, 0.0]).reshape(-1, 1)], layout=torch.jagged
            ),
        },
        batch_size=[1],
    )


def test_response_loss_fn_consumes_model_topk_fkl():
    # Request 1: the k3 estimator returns the per-position top-K FKL directly (not the scalar Bernoulli-
    # forward / reverse combine) when topk_fkl_response_enabled. SEMANTIC OVERLOAD: the FKL rides in the
    # "log_probs" channel (no new key), so model_output["log_probs"] holds the FKL, not log q(y_j).
    data = _data_one_sample()
    fkl_per_pos = torch.tensor([0.11, 0.22, 0.33, 0.0])
    model_output = {
        "log_probs": fkl_per_pos,
        "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
    }
    cfg = SimpleNamespace(
        loss_mode="k3", loss_max_clamp=None, log_prob_min_clamp=None, forward_kl_weight=1.0,
        reverse_kl_weight=0.0, topk_fkl_response_enabled=True,
    )
    losses, metrics = compute_distillation_loss_reverse_kl_estimator(
        SimpleNamespace(), SimpleNamespace(distillation_loss=cfg), model_output, data
    )
    assert torch.allclose(losses, no_padding_2_padding(fkl_per_pos, data), atol=1e-6)
    assert "distillation/response_topk_fkl_loss" in metrics


def test_reject_loss_uses_topk_fkl_with_offset_decay_and_slot_count():
    # Request 2: the reject stream loss is the model-emitted top-K FKL (not reverse KL), decayed by
    # offset and normalized by the actual local slot count + decay-weight sum (engine counts ignored).
    data = _data_one_sample()
    logq_y = torch.tensor([-0.4, -0.9, -0.3, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    reject_fkl = torch.tensor([[0.3, 0.5, 0.2]])
    offsets = torch.tensor([[1, 2, 3]])
    model_output = {
        "log_probs": logq_y,
        "opd_loss_mask": mask,
        # SEMANTIC OVERLOAD: the rejected-draft student channel carries the per-slot FKL (teacher stays 0).
        "opd_rejected_draft_student_log_probs": reject_fkl,
        "opd_rejected_draft_teacher_log_probs": torch.zeros_like(reject_fkl),
        "opd_rejected_draft_loss_mask": torch.ones_like(reject_fkl, dtype=torch.bool),
        "opd_rejected_draft_offsets": offsets,
    }
    decay = 0.5
    cfg = SimpleNamespace(
        loss_mode="k3", loss_max_clamp=None, log_prob_min_clamp=None, use_policy_gradient=False,
        reverse_kl_weight=0.0, forward_kl_weight=1.0, rejected_draft_use_reverse_kl=True,
        rejected_draft_position_decay_enabled=True, rejected_draft_position_decay=decay,
        response_stream_weight=1.0, rejected_draft_stream_weight=1.0,
        onpolicy_reverse_enabled=False, topk_fkl_reject_enabled=True, topk_fkl_response_enabled=False,
    )
    config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={
            "batch_num_tokens": 3.0,
            "opd_rejected_draft_batch_num_tokens": 99.0,  # rollout pollution, must be ignored
            "opd_rejected_draft_batch_effective_num_tokens": 77.0,
            "dp_size": 1,
        },
    )
    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)

    s = torch.tensor([-0.4, -0.9, -0.3])
    t = torch.tensor([-0.5, -0.7, -0.6])
    sp, tp = s.exp(), t.exp()
    fwd = tp * (t - s) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))  # response Bernoulli forward
    w = torch.tensor([decay**0, decay**1, decay**2])  # decay^(offset-1)
    # reject loss = the model FKL values directly (no kl_penalty), decay-weighted in num + denom.
    expected = (fwd.sum() + (w * reject_fkl.squeeze(0)).sum()) / (1.0 * 3.0 + 1.0 * w.sum())
    assert torch.allclose(loss, expected, atol=1e-6)
