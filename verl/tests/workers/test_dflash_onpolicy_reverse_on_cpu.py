"""CPU tests for paradistill (verl_dflash_onpolicy_reverse_enabled).

paradistill replaces the rollout-reject reverse stream with a fresh on-policy draft sample drawn at each
response-prediction position. The genuinely new code is the fused sampler/scorer
``_compute_response_and_reverse_log_probs``: from ONE draft->vocab projection per slot it returns the
response-forward log q(y_j) and the reverse pair log q(y_hat) [grad] / log p(y_hat) [no grad] for a
fresh y_hat ~ q drawn at T_draft, feeding the otherwise-unchanged rejected-draft k3 reverse loss.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.core_algos import kl_penalty


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    batch, draft_len, seq, hidden, vocab = 2, 5, 7, 4, 6
    draft_hidden = torch.randn(batch, draft_len, hidden, dtype=torch.float32)
    teacher_logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    output_embeddings = torch.nn.Linear(hidden, vocab, bias=False)
    batch_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    draft_indices = torch.tensor([1, 3, 2], dtype=torch.long)
    row_indices = torch.tensor([2, 4, 3], dtype=torch.long)
    token_ids = torch.tensor([0, 2, 1], dtype=torch.long)  # realized y_j per slot (response-forward term)
    return draft_hidden, teacher_logits, output_embeddings, batch_indices, draft_indices, row_indices, token_ids


def _fused(student, draft_hidden, teacher_logits, output_embeddings, b_idx, d_idx, r_idx, t_idx, *,
           chunk_size=64, sample_temperature=1.0, calculate_entropy=False, generator=None):
    return student._compute_response_and_reverse_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        batch_indices=b_idx,
        draft_indices=d_idx,
        token_ids=t_idx,
        teacher_logits=teacher_logits,
        row_indices=r_idx,
        chunk_size=chunk_size,
        calculate_entropy=calculate_entropy,
        sample_temperature=sample_temperature,
        generator=generator,
    )


def test_fused_response_and_reverse_logprobs_match_gathered_distributions():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, output_embeddings, b_idx, d_idx, r_idx, t_idx = _make_inputs()

    resp_log_q, resp_entropy, log_q, log_p, sampled = _fused(
        student, draft_hidden, teacher_logits, output_embeddings, b_idx, d_idx, r_idx, t_idx,
        generator=torch.Generator().manual_seed(123),
    )
    draft_log_probs = F.log_softmax(output_embeddings(draft_hidden[b_idx, d_idx, :]).float(), dim=-1)

    # response-forward term: log q(y_j) at the realized token, from the shared projection
    assert torch.allclose(resp_log_q, draft_log_probs.gather(-1, t_idx.unsqueeze(-1)).squeeze(-1), atol=1e-6)
    assert resp_entropy is None  # calculate_entropy=False
    # reverse term: log q(y_hat) from the SAME draft log-probs at the drawn token
    assert torch.allclose(log_q, draft_log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1), atol=1e-6)
    # log p(y_hat): frozen teacher's realized-prefix distribution at row_indices, gathered at y_hat
    teacher_lp = F.log_softmax(teacher_logits[b_idx, r_idx, :].float(), dim=-1)
    assert torch.allclose(log_p, teacher_lp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1), atol=1e-6)

    assert resp_log_q.shape == log_q.shape == log_p.shape == sampled.shape == (b_idx.numel(),)
    assert sampled.min() >= 0 and sampled.max() < draft_log_probs.shape[-1]


def test_fused_reverse_sampling_is_reproducible_with_seed():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    inputs = _make_inputs()

    def _draw():
        return _fused(student, *inputs, sample_temperature=0.7, generator=torch.Generator().manual_seed(7))[4]

    assert torch.equal(_draw(), _draw())


def test_fused_grad_flows_through_both_student_terms_not_teacher():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, output_embeddings, b_idx, d_idx, r_idx, t_idx = _make_inputs()
    draft_hidden.requires_grad_(True)

    resp_log_q, _, log_q, log_p, _ = _fused(
        student, draft_hidden, teacher_logits, output_embeddings, b_idx, d_idx, r_idx, t_idx,
        chunk_size=2,  # exercise multi-chunk path
        generator=torch.Generator().manual_seed(5),
    )
    assert resp_log_q.requires_grad and log_q.requires_grad  # both student terms differentiable
    assert not log_p.requires_grad  # teacher term is detached (no_grad)
    # both forward (y_j) and reverse (y_hat) backprop into draft_hidden -- gradient-identical to two
    # separate projections (they sum at the shared draft log-probs).
    (resp_log_q.sum() + log_q.sum()).backward()
    touched = draft_hidden.grad.abs().sum(dim=-1) > 0
    assert touched[0, 1] and touched[0, 3] and touched[1, 2]


def test_fused_reverse_keeps_overlapping_slots_without_dedup():
    # Two slots target the SAME sequence row (row 2) but come from different draft blocks (cols 1 and 3).
    # They re-sample independently, so the helper returns one entry per slot (no dedup).
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, output_embeddings, *_ = _make_inputs()
    b_idx = torch.tensor([0, 0], dtype=torch.long)
    d_idx = torch.tensor([1, 3], dtype=torch.long)
    r_idx = torch.tensor([2, 2], dtype=torch.long)
    t_idx = torch.tensor([0, 1], dtype=torch.long)

    _, _, log_q, log_p, sampled = _fused(
        student, draft_hidden, teacher_logits, output_embeddings, b_idx, d_idx, r_idx, t_idx,
        generator=torch.Generator().manual_seed(0),
    )
    assert log_q.shape == log_p.shape == sampled.shape == (2,)  # both overlapping slots kept
    for k, col in enumerate(d_idx.tolist()):
        draft_lp = F.log_softmax(output_embeddings(draft_hidden[0, col]).float(), -1)
        assert torch.allclose(log_q[k], draft_lp[sampled[k]], atol=1e-6)


def _nested_logprobs(vals):
    v = torch.as_tensor(vals, dtype=torch.float32).reshape(-1, 1)
    return torch.nested.as_nested_tensor([v], layout=torch.jagged)


def test_paradistill_loss_uses_flat_slot_count_with_overlap_and_ignores_polluted_count():
    # paradistill keeps ALL re-sampled (block, offset) slots (overlapping sampled-mode blocks are NOT deduped),
    # so the reverse stream has more entries than response positions and the loss must normalize it by the
    # actual flat slot count -- not batch_num_tokens, and not the engine's rollout-reject pollution.
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1, 1]]),
            "teacher_logprobs": _nested_logprobs([-0.5, -0.7, -0.6, 0.0]),  # log p(y_j)
        },
        batch_size=[1],
    )
    logq_y = torch.tensor([-0.4, -0.9, -0.3, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])  # 3 response-forward positions
    # 5 re-sampled reverse slots (> 3 response positions: overlapping blocks), flat (1, 5).
    logq_yhat = torch.tensor([[-1.1, -0.2, -0.9, -0.8, -0.3]])
    logp_yhat = torch.tensor([[-0.6, -0.5, -0.4, -1.0, -0.7]])
    model_output = {
        "log_probs": logq_y,
        "opd_loss_mask": mask,
        "opd_rejected_draft_student_log_probs": logq_yhat,
        "opd_rejected_draft_teacher_log_probs": logp_yhat,
        "opd_rejected_draft_loss_mask": torch.ones_like(logq_yhat, dtype=torch.bool),
    }
    cfg = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.0,
        forward_kl_weight=1.0,
        rejected_draft_use_reverse_kl=True,
        rejected_draft_position_decay_enabled=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
        onpolicy_reverse_enabled=True,
    )
    # opd_rejected_draft_batch_num_tokens=99 is the rollout-reject pollution that must be ignored.
    config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={
            "batch_num_tokens": 3.0,
            "opd_rejected_draft_batch_num_tokens": 99.0,
            "dp_size": 1,
        },
    )
    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)

    s = torch.tensor([-0.4, -0.9, -0.3])
    t = torch.tensor([-0.5, -0.7, -0.6])
    sp, tp = s.exp(), t.exp()
    fwd = tp * (t - s) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))  # Bernoulli forward on 3 y's
    rev = kl_penalty(logq_yhat.squeeze(0), logp_yhat.squeeze(0), "k3")  # k3 on all 5 re-sampled yhat slots
    # denom = w_resp * batch_num_tokens(3) + w_rej * flat_slot_count(5)
    expected = (fwd.sum() + rev.sum()) / (1.0 * 3.0 + 1.0 * 5.0)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_paradistill_loss_applies_offset_decay_and_normalizes_by_decay_weight():
    # paradistill with position decay ON: each reverse slot is weighted decay^(offset-1) in BOTH the
    # numerator and the denominator (sum of weights). The engine's (rollout-based) counts are ignored.
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1, 1]]),
            "teacher_logprobs": _nested_logprobs([-0.5, -0.7, -0.6, 0.0]),
        },
        batch_size=[1],
    )
    logq_y = torch.tensor([-0.4, -0.9, -0.3, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    logq_yhat = torch.tensor([[-1.1, -0.2, -0.9]])
    logp_yhat = torch.tensor([[-0.6, -0.5, -0.4]])
    offsets = torch.tensor([[1, 2, 3]])  # per-slot draft offset (1..K)
    model_output = {
        "log_probs": logq_y,
        "opd_loss_mask": mask,
        "opd_rejected_draft_student_log_probs": logq_yhat,
        "opd_rejected_draft_teacher_log_probs": logp_yhat,
        "opd_rejected_draft_loss_mask": torch.ones_like(logq_yhat, dtype=torch.bool),
        "opd_rejected_draft_offsets": offsets,
    }
    decay = 0.5
    cfg = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.0,
        forward_kl_weight=1.0,
        rejected_draft_use_reverse_kl=True,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=decay,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
        onpolicy_reverse_enabled=True,
    )
    # Pollute BOTH engine counts (99 raw, 77 effective) to prove the paradistill override ignores them.
    config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={
            "batch_num_tokens": 3.0,
            "opd_rejected_draft_batch_num_tokens": 99.0,
            "opd_rejected_draft_batch_effective_num_tokens": 77.0,
            "dp_size": 1,
        },
    )
    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)

    s = torch.tensor([-0.4, -0.9, -0.3])
    t = torch.tensor([-0.5, -0.7, -0.6])
    sp, tp = s.exp(), t.exp()
    fwd = tp * (t - s) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))
    w = torch.tensor([decay**0, decay**1, decay**2])  # decay^(offset-1) = [1, 0.5, 0.25]
    rev = kl_penalty(logq_yhat.squeeze(0), logp_yhat.squeeze(0), "k3")
    # numerator reverse = sum_j w_j * rev_j ; denom reverse = sum_j w_j (= 1.75)
    expected = (fwd.sum() + (w * rev).sum()) / (1.0 * 3.0 + 1.0 * w.sum())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_fused_handles_empty_selection():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden, teacher_logits, output_embeddings, *_ = _make_inputs()
    empty = torch.empty((0,), dtype=torch.long)
    resp_log_q, resp_entropy, log_q, log_p, sampled = _fused(
        student, draft_hidden, teacher_logits, output_embeddings, empty, empty, empty, empty, chunk_size=8
    )
    assert resp_log_q.numel() == log_q.numel() == log_p.numel() == sampled.numel() == 0
    assert resp_entropy is None
