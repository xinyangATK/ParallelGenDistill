"""CPU tests for paradistill DRAFT-PREFIX on-policy reverse (verl_dflash_onpolicy_reverse_enabled).

paradistill now: the draft samples a fresh block y_hat ~ q from its heads; the frozen teacher runs ONE
verify-forward ALONG the fresh block (see test_dflash_teacher_verify_on_cpu) to get p(.| prefix<=anchor,
y_hat_<o); then a top-K REVERSE KL between the draft head dist q_o and that teacher dist is precomputed
in-model and rides in the rejected-draft student channel (reject_token/post_reject_loss_mode=topk_reverse_kl).
These tests cover the new sampler, the teacher-hidden generalization of the reverse-KL helper, and the
loss-side reading of the precomputed reverse value with per-offset decay + flat-slot normalization.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM
from verl.trainer.distillation.losses import distillation_loss


def _student() -> ComposedDFlashStudentForCausalLM:
    return object.__new__(ComposedDFlashStudentForCausalLM)


def test_sample_draft_tokens_deterministic_and_in_vocab():
    torch.manual_seed(0)
    student = _student()
    batch, draft_len, hidden, vocab = 2, 5, 4, 9
    draft_hidden = torch.randn(batch, draft_len, hidden)
    emb = torch.nn.Linear(hidden, vocab, bias=False)
    b_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    d_idx = torch.tensor([1, 3, 2], dtype=torch.long)

    def draw():
        return student._sample_draft_tokens(
            draft_hidden=draft_hidden, output_embeddings=emb, batch_indices=b_idx, draft_indices=d_idx,
            sample_temperature=1.0, chunk_size=2, generator=torch.Generator().manual_seed(7),
        )

    s = draw()
    assert s.shape == (3,) and s.dtype == torch.long
    assert int(s.min()) >= 0 and int(s.max()) < vocab
    assert torch.equal(s, draw())  # reproducible with seeded generator
    # empty selection -> empty
    empty = torch.empty((0,), dtype=torch.long)
    assert student._sample_draft_tokens(
        draft_hidden=draft_hidden, output_embeddings=emb, batch_indices=empty, draft_indices=empty,
        sample_temperature=1.0, chunk_size=2,
    ).numel() == 0


def test_topk_reverse_kl_teacher_hidden_matches_pre_projected_logits():
    # The teacher-hidden path (project per chunk via teacher_output_embeddings) must equal feeding
    # pre-projected teacher logits -- this is what lets the verify-forward hidden feed the reverse KL.
    torch.manual_seed(1)
    student = _student()
    batch, qlen, seq, hidden, vocab = 1, 6, 6, 5, 11
    draft_hidden = torch.randn(batch, qlen, hidden, requires_grad=True)
    teacher_hidden = torch.randn(batch, seq, hidden)
    out_emb = torch.nn.Linear(hidden, vocab, bias=False)
    b_idx = torch.tensor([0, 0, 0], dtype=torch.long)
    d_idx = torch.tensor([1, 3, 5], dtype=torch.long)
    r_idx = torch.tensor([0, 2, 4], dtype=torch.long)

    # reference: pre-project teacher hidden to logits, use the logits path
    teacher_logits = out_emb(teacher_hidden).detach()
    ref, _ = student._compute_topk_reverse_kl(
        draft_hidden=draft_hidden, output_embeddings=out_emb, batch_indices=b_idx, draft_indices=d_idx,
        teacher_logits=teacher_logits, row_indices=r_idx, chunk_size=2, mode="union", topk_teacher=4, topk_student=4,
    )
    got, _ = student._compute_topk_reverse_kl(
        draft_hidden=draft_hidden, output_embeddings=out_emb, batch_indices=b_idx, draft_indices=d_idx,
        teacher_logits=teacher_hidden, teacher_output_embeddings=out_emb, row_indices=r_idx, chunk_size=2,
        mode="union", topk_teacher=4, topk_student=4,
    )
    assert torch.allclose(ref, got, atol=1e-6), f"teacher-hidden != logits path: {(ref-got).abs().max()}"
    # gradient flows through the draft head only (teacher detached)
    got.sum().backward()
    assert draft_hidden.grad is not None and draft_hidden.grad.abs().sum() > 0


def _nested_logprobs(vals):
    v = torch.as_tensor(vals, dtype=torch.float32).reshape(-1, 1)
    return torch.nested.as_nested_tensor([v], layout=torch.jagged)


def _paradistill_loss(*, reverse_values, offsets=None, response_offsets=None, decay_enabled=False, decay=0.5,
                      batch_num_tokens=3.0, polluted_reject=99.0):
    # response forward stream: top-K forward KL values precomputed in the log_probs channel (semantic
    # overload); reverse stream: top-K reverse KL values precomputed in the rejected-draft student channel
    # (reject_token/post_reject_loss_mode=topk_reverse_kl -> the loss reads them directly).
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "response_mask": torch.tensor([[1, 1, 1]]),
            "teacher_logprobs": _nested_logprobs([0.0, 0.0, 0.0, 0.0]),
        },
        batch_size=[1],
    )
    resp_fwd = torch.tensor([0.10, 0.20, 0.30, 0.0])  # precomputed top-K forward KL per response position
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    rev = torch.tensor([reverse_values])  # (1, M) precomputed top-K reverse KL
    model_output = {
        "log_probs": resp_fwd,
        "opd_loss_mask": mask,
        "opd_rejected_draft_student_log_probs": rev,
        "opd_rejected_draft_teacher_log_probs": torch.zeros_like(rev),  # unused for top-K modes
        "opd_rejected_draft_loss_mask": torch.ones_like(rev, dtype=torch.bool),
    }
    if offsets is not None:
        model_output["opd_rejected_draft_offsets"] = torch.tensor([offsets], dtype=torch.long)
    if response_offsets is not None:
        model_output["opd_response_offsets"] = torch.tensor(response_offsets, dtype=torch.long)
    cfg = SimpleNamespace(
        loss_mode="k3", loss_max_clamp=None, log_prob_min_clamp=None, use_policy_gradient=False,
        reverse_kl_weight=0.0, forward_kl_weight=1.0,
        response_loss_mode="topk_fkl", reject_accept_loss_mode="topk_fkl",
        reject_token_loss_mode="topk_reverse_kl", post_reject_loss_mode="topk_reverse_kl",
        topk_fkl_student_k=8, topk_fkl_teacher_k=8,
        rejected_draft_position_decay_enabled=decay_enabled, rejected_draft_position_decay=decay,
        response_stream_weight=1.0, rejected_draft_stream_weight=1.0, onpolicy_reverse_enabled=True,
    )
    config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={
            "batch_num_tokens": batch_num_tokens,
            "opd_rejected_draft_batch_num_tokens": polluted_reject,  # must be ignored
            "dp_size": 1,
        },
    )
    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=cfg), model_output, data, dp_group=None)
    return loss, resp_fwd[:3]


def test_paradistill_reads_precomputed_topk_reverse_and_uses_flat_slot_count():
    # reverse stream has 5 precomputed top-K reverse-KL slots (> 3 response positions: overlap), loss must
    # use the actual flat slot count (5), NOT batch_num_tokens and NOT the polluted engine reject count (99).
    rev = [0.11, 0.22, 0.33, 0.44, 0.55]
    loss, fwd = _paradistill_loss(reverse_values=rev)
    expected = (fwd.sum() + torch.tensor(rev).sum()) / (1.0 * 3.0 + 1.0 * 5.0)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_paradistill_reverse_offset_decay_applied_in_num_and_denom():
    # Reverse decay only (no response offsets supplied) -> response stream stays uniform.
    rev = [0.11, 0.22, 0.33]
    offsets = [1, 2, 3]
    decay = 0.5
    loss, fwd = _paradistill_loss(reverse_values=rev, offsets=offsets, decay_enabled=True, decay=decay)
    w = torch.tensor([decay**0, decay**1, decay**2])  # decay^(offset-1)
    expected = (fwd.sum() + (w * torch.tensor(rev)).sum()) / (1.0 * 3.0 + 1.0 * w.sum())
    assert torch.allclose(loss, expected, atol=1e-6)


def test_paradistill_both_streams_decayed_with_shared_factor():
    # BOTH streams decayed by decay^(offset-1): response weighted by its per-position offsets, reverse by its
    # per-slot offsets; each stream's denominator is its own decay-weight sum (the engine token count is ignored).
    rev = [0.11, 0.22, 0.33]
    offsets = [1, 2, 3]
    response_offsets = [1, 2, 3, 0]  # offset per seq position; last is padding (masked out)
    decay = 0.5
    loss, fwd = _paradistill_loss(
        reverse_values=rev, offsets=offsets, response_offsets=response_offsets, decay_enabled=True, decay=decay,
    )
    w = torch.tensor([decay**0, decay**1, decay**2])  # decay^(offset-1), shared by both streams
    num = (w * fwd).sum() + (w * torch.tensor(rev)).sum()
    denom = 1.0 * w.sum() + 1.0 * w.sum()  # response denom = reverse denom = sum of decay weights
    expected = num / denom
    assert torch.allclose(loss, expected, atol=1e-6)
