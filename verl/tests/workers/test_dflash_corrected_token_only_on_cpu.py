"""CPU tests for draftopd ``corrected_token_only`` (distillation.distillation_loss.corrected_token_only).

When enabled, draftopd's loss keeps ONLY the response Bernoulli forward-KL at SD reject positions (the
corrected token y): the accepted-position response loss is masked out and the rejected-draft reverse stream
is dropped entirely. The remaining loss is the mean Bernoulli forward-KL over corrected tokens, normalized by
the GLOBAL corrected-token count (so it stays a true mean, comparable to baseline draftopd).

The draftopd conda env has no pytest, so these use plain asserts and run directly via __main__.
"""

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu


def _bernoulli_forward_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    sp, tp = student.exp(), teacher.exp()
    return tp * (teacher - student) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))


def _data_one_sample(reject_indices):
    data = TensorDict(
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
    # response-coordinate SD reject positions (the corrected tokens to train on).
    tu.assign_non_tensor_data(data, "dflash_reject_token_indices", [list(reject_indices)])
    return data


def _cfg(**overrides):
    base = dict(
        loss_mode="k3",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.0,
        forward_kl_weight=1.0,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=0.0,
        rejected_draft_use_reverse_kl=True,
        rejected_draft_position_decay_enabled=False,
        rejected_draft_position_decay=0.9,
        onpolicy_reverse_enabled=False,
        topk_fkl_response_enabled=False,
        topk_fkl_reject_enabled=False,
        corrected_token_only=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _config():
    return SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={"batch_num_tokens": 3.0, "dp_size": 1})


# student log q(y) at each response position; the teacher_logprobs in _data_one_sample are [-0.5,-0.7,-0.6].
_LOGQ_Y = torch.tensor([-0.4, -0.9, -0.3, 0.0])
_TEACHER = torch.tensor([-0.5, -0.7, -0.6])
_OPD_LOSS_MASK = torch.tensor([1.0, 1.0, 1.0, 0.0])


def test_corrected_token_only_keeps_only_reject_position_forward_kl():
    data = _data_one_sample(reject_indices=[1])
    model_output = {"log_probs": _LOGQ_Y, "opd_loss_mask": _OPD_LOSS_MASK}
    loss, metrics = distillation_loss(
        _config(), SimpleNamespace(distillation_loss=_cfg()), model_output, data, dp_group=None
    )
    # Only response position 1 is a reject; loss = mean Bernoulli forward-KL over that single corrected token.
    expected = _bernoulli_forward_kl(_LOGQ_Y[1], _TEACHER[1])
    assert torch.allclose(loss, expected, atol=1e-6), (loss, expected)
    assert abs(float(metrics["distillation/corrected_token_count"].aggregate()) - 1.0) < 1e-6


def test_corrected_token_only_averages_over_multiple_reject_positions():
    data = _data_one_sample(reject_indices=[0, 2])
    model_output = {"log_probs": _LOGQ_Y, "opd_loss_mask": _OPD_LOSS_MASK}
    loss, _ = distillation_loss(_config(), SimpleNamespace(distillation_loss=_cfg()), model_output, data, dp_group=None)
    fwd = _bernoulli_forward_kl(_LOGQ_Y[:3], _TEACHER)
    expected = (fwd[0] + fwd[2]) / 2.0  # mean over the two corrected tokens
    assert torch.allclose(loss, expected, atol=1e-6), (loss, expected)


def test_corrected_token_only_drops_rejected_draft_stream():
    # A non-empty rejected-draft reverse stream must be IGNORED when corrected_token_only is on, so the loss
    # equals the corrected-only forward-KL with no reverse contribution.
    data = _data_one_sample(reject_indices=[1])
    reject = torch.tensor([[0.0, 0.0, 0.0]])
    model_output = {
        "log_probs": _LOGQ_Y,
        "opd_loss_mask": _OPD_LOSS_MASK,
        "opd_rejected_draft_student_log_probs": torch.tensor([[-0.4, -0.5, -0.6]]),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.9, -0.8, -0.7]]),
        "opd_rejected_draft_loss_mask": torch.ones_like(reject, dtype=torch.bool),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2, 3]]),
    }
    loss, metrics = distillation_loss(
        _config(), SimpleNamespace(distillation_loss=_cfg()), model_output, data, dp_group=None
    )
    expected = _bernoulli_forward_kl(_LOGQ_Y[1], _TEACHER[1])
    assert torch.allclose(loss, expected, atol=1e-6), (loss, expected)
    assert "distillation/rejected_draft_loss" not in metrics  # reject stream never entered


def test_corrected_token_only_empty_rejects_gives_grad_safe_zero():
    data = _data_one_sample(reject_indices=[])
    logq = _LOGQ_Y.clone().requires_grad_(True)
    model_output = {"log_probs": logq, "opd_loss_mask": _OPD_LOSS_MASK}
    loss, _ = distillation_loss(_config(), SimpleNamespace(distillation_loss=_cfg()), model_output, data, dp_group=None)
    assert abs(loss.item()) < 1e-9, loss
    loss.backward()  # must stay connected to the student log-probs (grad path, all-zero grads)
    assert logq.grad is not None


def test_corrected_token_only_rejects_policy_gradient():
    data = _data_one_sample(reject_indices=[1])
    model_output = {"log_probs": _LOGQ_Y, "opd_loss_mask": _OPD_LOSS_MASK}
    raised = False
    try:
        distillation_loss(
            _config(),
            SimpleNamespace(distillation_loss=_cfg(use_policy_gradient=True)),
            model_output,
            data,
            dp_group=None,
        )
    except NotImplementedError:
        raised = True
    assert raised, "corrected_token_only must reject use_policy_gradient=True"


if __name__ == "__main__":
    test_corrected_token_only_keeps_only_reject_position_forward_kl()
    test_corrected_token_only_averages_over_multiple_reject_positions()
    test_corrected_token_only_drops_rejected_draft_stream()
    test_corrected_token_only_empty_rejects_gives_grad_safe_zero()
    test_corrected_token_only_rejects_policy_gradient()
    print("all corrected_token_only tests passed")
