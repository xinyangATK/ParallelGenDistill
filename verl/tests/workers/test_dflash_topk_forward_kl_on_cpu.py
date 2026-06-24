"""CPU tests for draftopd top-K forward KL within the per-region loss framework.

The top-K forward KL is computed in-model from the frozen teacher's full logits:
``sum_{v in S} p_t(v) (log p_t(v) - log q_s(v))`` where ``S`` is the teacher / student / union top-K set.
The response stream's reject-accept region and the reject stream's reject-token / post-reject regions can
each select it (vs Bernoulli forward / reverse KL). These tests cover the helper, the per-slot reject split
(reject-token = min offset per anchor; mixed reverse/top-K), and the loss-consumption paths.
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
from verl.trainer.ppo.core_algos import kl_penalty
from verl.workers.config.distillation import DistillationLossConfig
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


def _manual_topk_fkl(draft_hidden, teacher_logits, out, b, d, r, mode, k, k_student=None):
    k_student = k if k_student is None else k_student
    s_logits = out(draft_hidden[b, d, :]).float()
    s_lp = F.log_softmax(s_logits, dim=-1)
    t_logits = teacher_logits[b, r, :].float()
    t_lp = F.log_softmax(t_logits, dim=-1)
    t_p = t_lp.exp()
    sel = torch.zeros_like(t_lp, dtype=torch.bool)
    if mode in ("teacher", "union"):
        sel.scatter_(-1, t_logits.topk(k, dim=-1).indices, True)
    if mode in ("student", "union"):
        sel.scatter_(-1, s_logits.topk(k_student, dim=-1).indices, True)
    return (t_p * (t_lp - s_lp) * sel).sum(dim=-1).clamp_min(0.0)


def _call(student, draft_hidden, teacher_logits, out, b, d, r, *, mode, k=2, k_student=None,
          chunk_size=64, calculate_entropy=False):
    return student._compute_topk_forward_kl(
        draft_hidden=draft_hidden,
        output_embeddings=out,
        batch_indices=b,
        draft_indices=d,
        teacher_logits=teacher_logits,
        row_indices=r,
        chunk_size=chunk_size,
        mode=mode,
        topk_teacher=k,
        topk_student=k if k_student is None else k_student,
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


def test_topk_forward_kl_asymmetric_union_k():
    # union with DIFFERENT teacher and student top-K (verl_dflash_topk_fkl_student_k): the support set is
    # teacher-top-k_teacher U student-top-k_student; the partial-support sum must match the manual one.
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs(seed=5)
    fkl, _ = _call(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=4, k_student=2, chunk_size=2)
    expected = _manual_topk_fkl(draft_hidden, teacher_logits, out, b, d, r, "union", k=4, k_student=2)
    assert torch.allclose(fkl, expected, atol=1e-6)


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


def _manual_topk_tv(draft_hidden, teacher_logits, out, b, d, r, mode, k, k_student=None):
    k_student = k if k_student is None else k_student
    s_logits = out(draft_hidden[b, d, :]).float()
    s_p = F.softmax(s_logits, dim=-1)
    t_logits = teacher_logits[b, r, :].float()
    t_p = F.softmax(t_logits, dim=-1)
    sel = torch.zeros_like(t_p, dtype=torch.bool)
    if mode in ("teacher", "union"):
        sel.scatter_(-1, t_logits.topk(k, dim=-1).indices, True)
    if mode in ("student", "union"):
        sel.scatter_(-1, s_logits.topk(k_student, dim=-1).indices, True)
    return (0.5 * (t_p - s_p).abs() * sel).sum(dim=-1)


def _call_tv(student, draft_hidden, teacher_logits, out, b, d, r, *, mode, k=2, k_student=None, chunk_size=64):
    return student._compute_topk_tv(
        draft_hidden=draft_hidden, output_embeddings=out, batch_indices=b, draft_indices=d,
        teacher_logits=teacher_logits, row_indices=r, chunk_size=chunk_size, mode=mode,
        topk_teacher=k, topk_student=k if k_student is None else k_student,
    )


def test_topk_tv_matches_manual_each_mode():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs()
    for mode in ("teacher", "student", "union"):
        tv, _ = _call_tv(student, draft_hidden, teacher_logits, out, b, d, r, mode=mode, k=2, chunk_size=2)
        expected = _manual_topk_tv(draft_hidden, teacher_logits, out, b, d, r, mode, k=2)
        assert torch.allclose(tv, expected, atol=1e-6), mode
        assert tv.shape == (b.numel(),)
        assert (tv >= 0).all()  # TV is a sum of absolute differences -> always >= 0


def test_topk_tv_asymmetric_union_k():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs(seed=5)
    tv, _ = _call_tv(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=4, k_student=2, chunk_size=2)
    expected = _manual_topk_tv(draft_hidden, teacher_logits, out, b, d, r, "union", k=4, k_student=2)
    assert torch.allclose(tv, expected, atol=1e-6)


def test_topk_tv_grad_flows_through_student_only():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs()
    draft_hidden.requires_grad_(True)
    teacher_logits.requires_grad_(True)
    tv, _ = _call_tv(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=2, chunk_size=2)
    assert tv.requires_grad
    tv.sum().backward()
    assert teacher_logits.grad is None  # teacher term is detached (no_grad)
    touched = draft_hidden.grad.abs().sum(dim=-1) > 0
    assert touched[0, 1] and touched[0, 3] and touched[1, 2]


def _manual_topk_reverse_kl(draft_hidden, teacher_logits, out, b, d, r, mode, k, k_student=None):
    k_student = k if k_student is None else k_student
    s_logits = out(draft_hidden[b, d, :]).float()
    s_lp = F.log_softmax(s_logits, dim=-1)
    s_p = s_lp.exp()
    t_logits = teacher_logits[b, r, :].float()
    t_lp = F.log_softmax(t_logits, dim=-1)
    sel = torch.zeros_like(s_lp, dtype=torch.bool)
    if mode in ("teacher", "union"):
        sel.scatter_(-1, t_logits.topk(k, dim=-1).indices, True)
    if mode in ("student", "union"):
        sel.scatter_(-1, s_logits.topk(k_student, dim=-1).indices, True)
    return (s_p * (s_lp - t_lp) * sel).sum(dim=-1).clamp_min(0.0)


def _call_rkl(student, draft_hidden, teacher_logits, out, b, d, r, *, mode, k=2, k_student=None, chunk_size=64):
    return student._compute_topk_reverse_kl(
        draft_hidden=draft_hidden, output_embeddings=out, batch_indices=b, draft_indices=d,
        teacher_logits=teacher_logits, row_indices=r, chunk_size=chunk_size, mode=mode,
        topk_teacher=k, topk_student=k if k_student is None else k_student,
    )


def test_topk_reverse_kl_matches_manual_each_mode():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs()
    for mode in ("teacher", "student", "union"):
        rkl, _ = _call_rkl(student, draft_hidden, teacher_logits, out, b, d, r, mode=mode, k=2, chunk_size=2)
        expected = _manual_topk_reverse_kl(draft_hidden, teacher_logits, out, b, d, r, mode, k=2)
        assert torch.allclose(rkl, expected, atol=1e-6), mode
        assert rkl.shape == (b.numel(),)
        assert (rkl >= -1e-6).all()  # clamped to >=0 (partial top-K support)


def test_topk_reverse_kl_asymmetric_union_k():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs(seed=5)
    rkl, _ = _call_rkl(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=4, k_student=2, chunk_size=2)
    expected = _manual_topk_reverse_kl(draft_hidden, teacher_logits, out, b, d, r, "union", k=4, k_student=2)
    assert torch.allclose(rkl, expected, atol=1e-6)


def test_topk_reverse_kl_grad_flows_through_student_only():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, out, b, d, r, _ = _make_inputs()
    draft_hidden.requires_grad_(True)
    teacher_logits.requires_grad_(True)
    rkl, _ = _call_rkl(student, draft_hidden, teacher_logits, out, b, d, r, mode="union", k=2, chunk_size=2)
    assert rkl.requires_grad
    rkl.sum().backward()
    assert teacher_logits.grad is None  # teacher term detached (no_grad); reverse KL grad flows through q only
    # grad reaches the student (per-slot grad may be 0 where clamp_min(0) zeroes a negative partial-support sum).
    assert draft_hidden.grad is not None and draft_hidden.grad.abs().sum() > 0


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
    student_t, teacher_t, mask_t, is_rt = student._collect_rejected_draft_log_probs(
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
        reject_token_mode="topk_fkl",
        post_reject_mode="topk_fkl",
        teacher_logits=teacher_logits,
        topk_fkl_mode="teacher",
        topk_fkl_teacher_k=2,
    )
    assert mask_t.tolist() == [[True, False]]           # slot B dropped (realized target out of bounds)
    assert teacher_t.abs().sum() == 0                   # teacher channel unused for the FKL path
    expected = _manual_topk_fkl(
        draft_hidden, teacher_logits, out, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), "teacher", k=2
    )
    assert torch.allclose(student_t[0, 0], expected[0], atol=1e-6)  # FKL at draft slot 1 vs teacher row 2
    assert student_t[0, 1] == 0.0


def test_collect_rejected_draft_baseline_path_unchanged():
    # Default reverse_kl regions keep the baseline reverse-stream behavior: student = log q(d) at the slot,
    # teacher = the cached scalar log p(d); the single slot is the reject token (min offset for its anchor).
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    torch.manual_seed(1)
    hidden, vocab, K = 4, 6, 3
    draft_hidden = torch.randn(1, K, hidden)
    out = torch.nn.Linear(hidden, vocab, bias=False)
    long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    student_t, teacher_t, mask_t, is_rt = student._collect_rejected_draft_log_probs(
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
    assert is_rt.tolist() == [[True]]  # min offset per anchor = reject token
    expected_logq = F.log_softmax(out(draft_hidden[0, 1]).float(), -1)[3]  # log q at token 3, draft slot 1
    assert torch.allclose(student_t[0, 0], expected_logq, atol=1e-6)
    assert torch.allclose(teacher_t[0, 0], torch.tensor(-0.7), atol=1e-6)


def test_collect_rejected_draft_is_reject_token_min_offset_and_mixed_regions():
    # One anchor (full pos 2), block K=4, offsets 1,2,3 -> reject token = offset 1 (min), post-reject = 2,3.
    # Mixed regions: reject_token=reverse_kl, post_reject=topk_fkl. valid_len=2+5=7 so all realized targets
    # (3,4,5) are in bounds. The reject-token slot carries log q(d); the post-reject slots carry the FKL.
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    torch.manual_seed(2)
    hidden, vocab, K = 4, 6, 4
    draft_hidden = torch.randn(1, K, hidden)
    teacher_logits = torch.randn(1, 7, vocab)
    out = torch.nn.Linear(hidden, vocab, bias=False)
    long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    student_t, teacher_t, mask_t, is_rt = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=out,
        prompt_lengths=long([2]),
        response_lengths=long([5]),
        anchor_positions=long([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=K,
        lm_head_chunk_size=64,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=long([[0, 0, 0]]),  # all anchor_resp 0 -> full 2
        rejected_draft_offsets=long([[1, 2, 3]]),
        rejected_draft_token_ids=long([[3, 4, 5]]),
        rejected_draft_teacher_logprobs=torch.tensor([[-0.7, -0.8, -0.9]]),
        rejected_draft_mask=torch.tensor([[True, True, True]]),
        reject_token_mode="reverse_kl",
        post_reject_mode="topk_fkl",
        teacher_logits=teacher_logits,
        topk_fkl_mode="teacher",
        topk_fkl_teacher_k=2,
    )
    assert is_rt.tolist() == [[True, False, False]]  # offset 1 is the reject token
    assert mask_t.tolist() == [[True, True, True]]
    # reject-token slot (offset 1): reverse data -> student = log q(d=3) at draft slot 1, teacher = -0.7.
    assert torch.allclose(student_t[0, 0], F.log_softmax(out(draft_hidden[0, 1]).float(), -1)[3], atol=1e-6)
    assert torch.allclose(teacher_t[0, 0], torch.tensor(-0.7), atol=1e-6)
    # post-reject slots (offsets 2,3): top-K FKL vs teacher rows full_anchor+offset-1 = 3,4; teacher chan 0.
    for col, (draft_slot, trow, d) in enumerate([(2, 3, 4), (3, 4, 5)], start=1):
        exp = _manual_topk_fkl(
            draft_hidden, teacher_logits, out, torch.tensor([0]), torch.tensor([draft_slot]), torch.tensor([trow]),
            "teacher", k=2,
        )
        assert torch.allclose(student_t[0, col], exp[0], atol=1e-6)
        assert teacher_t[0, col] == 0.0


def test_collect_rejected_draft_reverse_token_topk_reverse_kl_postreject():
    # Mixed 3-way routing: reject_token=reverse_kl, post_reject=topk_reverse_kl -> reverse-data slot + two
    # top-K reverse-KL slots (student channel = the in-model reverse KL vs the realized teacher; teacher 0).
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    torch.manual_seed(2)
    hidden, vocab, K = 4, 6, 4
    draft_hidden = torch.randn(1, K, hidden)
    teacher_logits = torch.randn(1, 7, vocab)
    out = torch.nn.Linear(hidden, vocab, bias=False)
    long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    student_t, teacher_t, mask_t, is_rt = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden, output_embeddings=out, prompt_lengths=long([2]), response_lengths=long([5]),
        anchor_positions=long([[2]]), block_keep_mask=torch.tensor([[True]]), draft_block_size=K,
        lm_head_chunk_size=64, max_tokens_per_sample=None,
        rejected_draft_anchor_indices=long([[0, 0, 0]]), rejected_draft_offsets=long([[1, 2, 3]]),
        rejected_draft_token_ids=long([[3, 4, 5]]), rejected_draft_teacher_logprobs=torch.tensor([[-0.7, -0.8, -0.9]]),
        rejected_draft_mask=torch.tensor([[True, True, True]]),
        reject_token_mode="reverse_kl", post_reject_mode="topk_reverse_kl",
        teacher_logits=teacher_logits, topk_fkl_mode="teacher", topk_fkl_teacher_k=2,
    )
    assert is_rt.tolist() == [[True, False, False]] and mask_t.tolist() == [[True, True, True]]
    # reject-token (offset 1): reverse data.
    assert torch.allclose(student_t[0, 0], F.log_softmax(out(draft_hidden[0, 1]).float(), -1)[3], atol=1e-6)
    assert torch.allclose(teacher_t[0, 0], torch.tensor(-0.7), atol=1e-6)
    # post-reject (offsets 2,3): top-K REVERSE KL vs teacher rows 3,4; teacher channel 0.
    for col, (draft_slot, trow) in enumerate([(2, 3), (3, 4)], start=1):
        exp = _manual_topk_reverse_kl(
            draft_hidden, teacher_logits, out, torch.tensor([0]), torch.tensor([draft_slot]), torch.tensor([trow]),
            "teacher", k=2,
        )
        assert torch.allclose(student_t[0, col], exp[0], atol=1e-6)
        assert teacher_t[0, col] == 0.0


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


def _response_cfg(**kw):
    base = dict(
        loss_mode="k3", loss_max_clamp=None, log_prob_min_clamp=None, forward_kl_weight=1.0,
        reverse_kl_weight=0.0, response_loss_mode="bernoulli_fkl", reject_accept_loss_mode="bernoulli_fkl",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_response_loss_fn_consumes_model_topk_fkl():
    # response_loss_mode=topk_fkl (and reject_accept_loss_mode=topk_fkl) -> the estimator returns the
    # per-position top-K FKL directly. SEMANTIC OVERLOAD: the FKL rides in the "log_probs" channel.
    data = _data_one_sample()
    fkl_per_pos = torch.tensor([0.11, 0.22, 0.33, 0.0])
    model_output = {"log_probs": fkl_per_pos, "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0])}
    cfg = _response_cfg(response_loss_mode="topk_fkl", reject_accept_loss_mode="topk_fkl")
    losses, metrics = compute_distillation_loss_reverse_kl_estimator(
        SimpleNamespace(), SimpleNamespace(distillation_loss=cfg), model_output, data
    )
    assert torch.allclose(losses, no_padding_2_padding(fkl_per_pos, data), atol=1e-6)
    assert "distillation/response_forward_loss" in metrics


def test_response_loss_fn_consumes_model_topk_tv():
    # topk_tv rides in the same log_probs channel as topk_fkl; the loss consumes it directly.
    data = _data_one_sample()
    tv_per_pos = torch.tensor([0.05, 0.12, 0.09, 0.0])
    model_output = {"log_probs": tv_per_pos, "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0])}
    cfg = _response_cfg(response_loss_mode="topk_tv", reject_accept_loss_mode="topk_tv")
    losses, metrics = compute_distillation_loss_reverse_kl_estimator(
        SimpleNamespace(), SimpleNamespace(distillation_loss=cfg), model_output, data
    )
    assert torch.allclose(losses, no_padding_2_padding(tv_per_pos, data), atol=1e-6)
    assert "distillation/response_forward_loss" in metrics


def _reject_cfg(reject_token_mode, post_reject_mode, decay):
    return SimpleNamespace(
        loss_mode="k3", loss_max_clamp=None, log_prob_min_clamp=None, use_policy_gradient=False,
        reverse_kl_weight=0.0, forward_kl_weight=1.0,
        response_loss_mode="bernoulli_fkl", reject_accept_loss_mode="bernoulli_fkl",
        reject_token_loss_mode=reject_token_mode, post_reject_loss_mode=post_reject_mode,
        rejected_draft_position_decay_enabled=True, rejected_draft_position_decay=decay,
        response_stream_weight=1.0, rejected_draft_stream_weight=1.0, onpolicy_reverse_enabled=False,
    )


def _reject_config():
    return SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={
            "batch_num_tokens": 3.0,
            "opd_rejected_draft_batch_num_tokens": 99.0,  # rollout pollution, must be ignored
            "opd_rejected_draft_batch_effective_num_tokens": 77.0,
            "dp_size": 1,
        },
    )


def _response_bernoulli(s, t):
    sp, tp = s.exp(), t.exp()
    return tp * (t - s) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))


def test_reject_loss_uniform_topk_with_offset_decay_and_slot_count():
    # reject_token=post_reject=topk_fkl: the reject stream loss is the model-emitted FKL (student channel),
    # decayed by offset and normalized by the actual local slot count + decay-weight sum (engine ignored).
    data = _data_one_sample()
    reject_fkl = torch.tensor([[0.3, 0.5, 0.2]])
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, -0.3, 0.0]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "opd_rejected_draft_student_log_probs": reject_fkl,
        "opd_rejected_draft_teacher_log_probs": torch.zeros_like(reject_fkl),
        "opd_rejected_draft_loss_mask": torch.ones_like(reject_fkl, dtype=torch.bool),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2, 3]]),
        "opd_rejected_draft_is_reject_token": torch.tensor([[True, False, False]]),
    }
    decay = 0.5
    cfg = _reject_cfg("topk_fkl", "topk_fkl", decay)
    loss, _ = distillation_loss(_reject_config(), SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)
    fwd = _response_bernoulli(torch.tensor([-0.4, -0.9, -0.3]), torch.tensor([-0.5, -0.7, -0.6]))
    w = torch.tensor([decay**0, decay**1, decay**2])  # decay^(offset-1)
    expected = (fwd.sum() + (w * reject_fkl.squeeze(0)).sum()) / (1.0 * 3.0 + 1.0 * w.sum())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_reject_loss_uniform_topk_reverse_kl_consumed_directly():
    # reject_token=post_reject=topk_reverse_kl: the reject loss is the model-emitted reverse-KL value in the
    # student channel (consumed directly like topk_fkl), decayed by offset and normalized by the slot count.
    data = _data_one_sample()
    reject_rkl = torch.tensor([[0.4, 0.6, 0.1]])
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, -0.3, 0.0]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "opd_rejected_draft_student_log_probs": reject_rkl,
        "opd_rejected_draft_teacher_log_probs": torch.zeros_like(reject_rkl),
        "opd_rejected_draft_loss_mask": torch.ones_like(reject_rkl, dtype=torch.bool),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2, 3]]),
        "opd_rejected_draft_is_reject_token": torch.tensor([[True, False, False]]),
    }
    decay = 0.5
    cfg = _reject_cfg("topk_reverse_kl", "topk_reverse_kl", decay)
    loss, _ = distillation_loss(_reject_config(), SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)
    fwd = _response_bernoulli(torch.tensor([-0.4, -0.9, -0.3]), torch.tensor([-0.5, -0.7, -0.6]))
    w = torch.tensor([decay**0, decay**1, decay**2])
    expected = (fwd.sum() + (w * reject_rkl.squeeze(0)).sum()) / (1.0 * 3.0 + 1.0 * w.sum())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_reject_loss_mixed_reverse_token_topk_postreject():
    # reject_token=reverse_kl, post_reject=topk_fkl: per slot the loss is kl_penalty(log q(d), log p(d)) on
    # the reject-token slot (is_reject_token=True) and the FKL on the post-reject slots, by is_reject_token.
    data = _data_one_sample()
    student_chan = torch.tensor([[-0.4, 0.5, 0.2]])  # slot0 = log q(d); slots 1,2 = FKL values
    teacher_chan = torch.tensor([[-0.9, 0.0, 0.0]])  # slot0 = cached log p(d); FKL slots = 0
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, -0.3, 0.0]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "opd_rejected_draft_student_log_probs": student_chan,
        "opd_rejected_draft_teacher_log_probs": teacher_chan,
        "opd_rejected_draft_loss_mask": torch.ones_like(student_chan, dtype=torch.bool),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2, 3]]),
        "opd_rejected_draft_is_reject_token": torch.tensor([[True, False, False]]),
    }
    decay = 0.5
    cfg = _reject_cfg("reverse_kl", "topk_fkl", decay)
    loss, _ = distillation_loss(_reject_config(), SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)
    rev0 = kl_penalty(logprob=student_chan[:, 0], ref_logprob=teacher_chan[:, 0], kl_penalty="k3")[0]
    reject_losses = torch.tensor([rev0.item(), 0.5, 0.2])  # slot0 reverse KL, slots 1,2 raw FKL
    fwd = _response_bernoulli(torch.tensor([-0.4, -0.9, -0.3]), torch.tensor([-0.5, -0.7, -0.6]))
    w = torch.tensor([decay**0, decay**1, decay**2])
    expected = (fwd.sum() + (w * reject_losses).sum()) / (1.0 * 3.0 + 1.0 * w.sum())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_topk_reverse_kl_requires_student_k_for_d_coverage():
    # topk_reverse_kl needs student_k>0 so the rejected token d (outside the teacher top-K) is in the support;
    # student_k=0 (teacher-only) would silently train only y, so the config must raise.
    raised = False
    try:
        DistillationLossConfig(reject_token_loss_mode="topk_reverse_kl", topk_fkl_teacher_k=64, topk_fkl_student_k=0)
    except ValueError:
        raised = True
    assert raised, "topk_reverse_kl with student_k=0 must raise"
    # allowed with student_k>0 (union support covers both y and d); also valid on post_reject.
    DistillationLossConfig(reject_token_loss_mode="topk_reverse_kl", topk_fkl_teacher_k=8, topk_fkl_student_k=8)
    DistillationLossConfig(post_reject_loss_mode="topk_reverse_kl", topk_fkl_teacher_k=8, topk_fkl_student_k=8)
    # default (no topk_reverse_kl) is unaffected even with student_k=0.
    DistillationLossConfig(topk_fkl_student_k=0)
