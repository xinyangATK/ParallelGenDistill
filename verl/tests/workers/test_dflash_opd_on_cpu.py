from types import SimpleNamespace

import numpy as np
import torch
from tensordict import NonTensorData, TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM
from verl.protocol import DataProto
from verl.trainer.distillation.losses import (
    distillation_loss,
    get_effective_distillation_response_mask,
    get_rejected_draft_distillation_stream,
)
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.rollout.sglang_rollout.utils import align_dflash_reject_token_mask


def _single_sequence_logprobs(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return torch.nested.as_nested_tensor([values], layout=torch.jagged)


def _object_array(*values):
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array


def _local_forward_kl(student_log_probs, teacher_log_probs):
    student_log_probs = torch.as_tensor(student_log_probs, dtype=torch.float32)
    teacher_log_probs = torch.as_tensor(teacher_log_probs, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    return teacher_probs * (teacher_log_probs - student_log_probs) + (1.0 - teacher_probs) * (
        torch.log1p(-teacher_probs) - torch.log1p(-student_probs)
    )


class _CaseLogTokenizer:
    _pieces = {
        1: "A",
        2: "B",
        3: "C",
        4: "D",
        5: "E",
        99: "<prompt>",
        101: "x",
        102: "<eos>",
    }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._pieces[int(token_id)] for token_id in ids)


def test_align_dflash_reject_token_mask_adds_prefill_token():
    assert align_dflash_reject_token_mask([False, True, False], response_len=4) == [
        False,
        False,
        True,
        False,
    ]
    assert align_dflash_reject_token_mask([False, True, False, True], response_len=4) == [
        False,
        True,
        False,
        True,
    ]
    assert align_dflash_reject_token_mask([True], response_len=4) == [False, True, False, False]


def test_case_log_reconstructs_draft_blocks_with_accepted_prefixes():
    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[99]], dtype=torch.long),
                "responses": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "dflash_reject_token_indices": _object_array([2, 4]),
            "dflash_rejected_draft_anchor_indices": _object_array([0, 0]),
            "dflash_rejected_draft_offsets": _object_array([2, 3]),
            "dflash_rejected_draft_token_ids": _object_array([101, 102]),
        },
    )
    worker = object.__new__(AgentLoopWorker)
    worker.tokenizer = _CaseLogTokenizer()

    text = worker._build_draft_token_case_text(
        batch=batch,
        sample_idx=0,
        response_ids=[1, 2, 3, 4, 5],
    )
    record = worker._build_case_log_record(batch=batch, sample_idx=0)

    assert text == "AB[{B}x<eos>]CD[{D}]E"
    assert record == {"response_text_with_draft_token": "AB[{B}x<eos>]CD[{D}]E"}


def test_case_log_only_writes_on_update_accumulation_boundaries():
    assert AgentLoopManager._should_write_case_log_step(0, 25)
    assert not AgentLoopManager._should_write_case_log_step(24, 25)
    assert AgentLoopManager._should_write_case_log_step(25, 25)
    assert AgentLoopManager._should_write_case_log_step(torch.tensor(50), "25")
    assert not AgentLoopManager._should_write_case_log_step(None, 25)
    assert AgentLoopManager._should_write_case_log_step(3, 1)


def test_prepare_composed_dflash_inputs_preserves_true_lengths_after_no_padding():
    prompts = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 24, 25]], dtype=torch.long)
    responses = torch.tensor([[31, 32, 0, 0], [41, 42, 43, 0]], dtype=torch.long)
    padded_input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long
    )
    padded_position_ids = torch.arange(padded_input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand_as(
        padded_input_ids
    )
    response_mask = attention_mask[:, prompts.shape[1] :]
    input_ids = torch.nested.as_nested_tensor(
        [padded_input_ids[i, attention_mask[i].bool()] for i in range(2)], layout=torch.jagged
    )
    position_ids = torch.nested.as_nested_tensor(
        [padded_position_ids[i, attention_mask[i].bool()] for i in range(2)], layout=torch.jagged
    )

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(batch, dflash_reject_token_indices=[[1], [2]])
    tu.assign_non_tensor(
        batch,
        dflash_rejected_draft_anchor_indices=[[0, 1], []],
        dflash_rejected_draft_offsets=[[2, 3], []],
        dflash_rejected_draft_token_ids=[[101, 102], []],
        dflash_rejected_draft_teacher_logprobs=[[-0.1, -0.2], []],
    )

    engine = object.__new__(FSDPEngineWithLMHead)
    model_inputs, output_args = engine._prepare_composed_dflash_inputs(batch)

    assert output_args == {"dflash_opd": True}
    assert model_inputs["dflash_prompt_lengths"].tolist() == [3, 5]
    assert model_inputs["dflash_response_lengths"].tolist() == [2, 3]
    assert model_inputs["input_ids"].tolist() == [
        [11, 12, 13, 31, 32, 0, 0, 0],
        [21, 22, 23, 24, 25, 41, 42, 43],
    ]
    assert model_inputs["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    assert model_inputs["dflash_rejected_draft_anchor_indices"].tolist() == [[0, 1], [-2, -2]]
    assert model_inputs["dflash_rejected_draft_offsets"].tolist() == [[2, 3], [-1, -1]]
    assert model_inputs["dflash_rejected_draft_token_ids"].tolist() == [[101, 102], [-1, -1]]
    assert torch.allclose(
        model_inputs["dflash_rejected_draft_teacher_logprobs"],
        torch.tensor([[-0.1, -0.2], [0.0, 0.0]], dtype=torch.float32),
    )
    assert model_inputs["dflash_rejected_draft_mask"].tolist() == [[True, True], [False, False]]


def test_dflash_anchor_plan_skips_empty_rejects_and_uses_block_size_as_total_length():
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

    _, _, _, empty_keep_mask, _, empty_metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([5]),
        response_lengths=torch.tensor([20]),
        reject_token_indices=torch.tensor([[-1]]),
        draft_block_size=16,
    )
    assert not bool(empty_keep_mask.any())
    assert empty_metrics["empty_reject_sample_count"] == 1
    assert empty_metrics["skipped_sample_count"] == 1


def test_dflash_anchor_plan_can_stride_response_anchors():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(12, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    _, _, _, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([10]),
        reject_token_indices=torch.tensor([[1, 2, 3, 4]]),
        draft_block_size=4,
        response_anchor_stride=2,
    )

    assert int(block_keep_mask.sum().item()) == 3
    assert metrics["valid_anchor_count"] == 3


def test_rejected_draft_logits_are_selected_from_anchor_and_offset():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden = torch.zeros(1, 4, 10, dtype=torch.float32)
    draft_hidden[0, 2, 7] = 3.0
    output_embeddings = torch.nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        output_embeddings.weight.copy_(torch.eye(10))

    student_log_probs, teacher_log_probs, mask, _ = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        lm_head_chunk_size=2,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[0]]),
        rejected_draft_offsets=torch.tensor([[2]]),
        rejected_draft_token_ids=torch.tensor([[7]]),
        rejected_draft_teacher_logprobs=torch.tensor([[-1.25]]),
        rejected_draft_mask=torch.tensor([[True]]),
    )

    expected = torch.log_softmax(draft_hidden[0, 2], dim=-1)[7]
    assert torch.allclose(student_log_probs, expected.view(1, 1))
    assert torch.allclose(teacher_log_probs, torch.tensor([[-1.25]]))
    assert mask.tolist() == [[True]]


def test_empty_rejected_draft_metadata_returns_empty_stream():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    output_embeddings = torch.nn.Linear(10, 10, bias=False)

    student_log_probs, teacher_log_probs, mask, _ = student._collect_rejected_draft_log_probs(
        draft_hidden=torch.zeros(1, 4, 10, dtype=torch.float32),
        output_embeddings=output_embeddings,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        lm_head_chunk_size=2,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[-2]]),
        rejected_draft_offsets=torch.tensor([[-1]]),
        rejected_draft_token_ids=torch.tensor([[-1]]),
        rejected_draft_teacher_logprobs=torch.tensor([[0.0]]),
        rejected_draft_mask=torch.tensor([[False]]),
    )

    assert student_log_probs.shape == (1, 1)
    assert teacher_log_probs.shape == (1, 1)
    assert mask.tolist() == [[False]]


def test_rejected_draft_max_tokens_per_sample_limits_loss_tokens():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden = torch.zeros(1, 4, 10, dtype=torch.float32)
    draft_hidden[0, 1, 3] = 2.0
    draft_hidden[0, 2, 4] = 2.0
    output_embeddings = torch.nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        output_embeddings.weight.copy_(torch.eye(10))

    student_log_probs, _, mask, _ = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        lm_head_chunk_size=2,
        max_tokens_per_sample=1,
        rejected_draft_anchor_indices=torch.tensor([[0, 0]]),
        rejected_draft_offsets=torch.tensor([[1, 2]]),
        rejected_draft_token_ids=torch.tensor([[3, 4]]),
        rejected_draft_teacher_logprobs=torch.tensor([[-1.0, -2.0]]),
        rejected_draft_mask=torch.tensor([[True, True]]),
    )

    assert mask.tolist() == [[True, False]]
    assert student_log_probs[0, 0].item() != 0
    assert student_log_probs[0, 1].item() == 0


def test_selected_lm_log_probs_match_full_logits_path():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden = torch.randn(2, 5, 8, dtype=torch.float32)
    output_embeddings = torch.nn.Linear(8, 11, bias=False)
    batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    draft_indices = torch.tensor([1, 3, 0, 4], dtype=torch.long)
    token_ids = torch.tensor([2, 5, 7, 1], dtype=torch.long)

    selected, entropy = student._compute_selected_lm_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        batch_indices=batch_indices,
        draft_indices=draft_indices,
        token_ids=token_ids,
        chunk_size=2,
        calculate_entropy=True,
    )

    full_logits = output_embeddings(draft_hidden)
    full_log_probs = torch.log_softmax(full_logits.float(), dim=-1)
    expected = full_log_probs[batch_indices, draft_indices, token_ids]
    selected_full_log_probs = full_log_probs[batch_indices, draft_indices]
    expected_entropy = -(selected_full_log_probs.exp() * selected_full_log_probs).sum(dim=-1)
    assert torch.allclose(selected, expected)
    assert entropy is not None
    assert torch.allclose(entropy, expected_entropy)


def test_rejected_draft_stream_accepts_nested_postprocess_output():
    student = torch.nested.as_nested_tensor(
        [torch.tensor([-1.1, -0.8]), torch.tensor([0.0])], layout=torch.jagged
    )
    teacher = torch.nested.as_nested_tensor(
        [torch.tensor([-0.6, -1.2]), torch.tensor([0.0])], layout=torch.jagged
    )
    mask = torch.nested.as_nested_tensor(
        [torch.tensor([True, True]), torch.tensor([False])], layout=torch.jagged
    )

    stream = get_rejected_draft_distillation_stream(
        {
            "opd_rejected_draft_student_log_probs": student,
            "opd_rejected_draft_teacher_log_probs": teacher,
            "opd_rejected_draft_loss_mask": mask,
        }
    )

    assert stream is not None
    student_values, teacher_values, mask_values = stream
    assert torch.allclose(student_values, torch.tensor([-1.1, -0.8, 0.0]))
    assert torch.allclose(teacher_values, torch.tensor([-0.6, -1.2, 0.0]))
    assert mask_values.tolist() == [True, True, False]


def test_effective_distillation_response_mask_applies_opd_loss_mask():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[4, 5, 6, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        },
        batch_size=[1],
    )
    model_output = {
        "opd_loss_mask": torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    }

    effective_mask = get_effective_distillation_response_mask(data=data, model_output=model_output)
    assert effective_mask.tolist() == [[True, True, False, False]]


def test_combined_k3_loss_includes_rejected_draft_tokens():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": torch.tensor([[-1.1, -0.8]], dtype=torch.float32),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.6, -1.2]], dtype=torch.float32),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    response_losses = kl_penalty(
        torch.tensor([-0.4, -0.9]),
        torch.tensor([-0.5, -0.7]),
        "k3",
    )
    rejected_losses = kl_penalty(
        torch.tensor([-1.1, -0.8]),
        torch.tensor([-0.6, -1.2]),
        "k3",
    )
    assert torch.allclose(loss, torch.cat([response_losses, rejected_losses]).mean())
    assert metrics["distillation/rejected_draft_token_count"].values == [2.0]


def test_rejected_draft_can_use_reverse_kl_while_response_uses_forward_kl():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": torch.tensor([[-1.1, -0.8]], dtype=torch.float32),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.6, -1.2]], dtype=torch.float32),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.0,
        forward_kl_weight=1.0,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=2.0,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    response_student = torch.tensor([-0.4, -0.9])
    response_teacher = torch.tensor([-0.5, -0.7])
    rejected_student = torch.tensor([-1.1, -0.8])
    rejected_teacher = torch.tensor([-0.6, -1.2])
    response_losses = _local_forward_kl(response_student, response_teacher)
    rejected_losses = kl_penalty(rejected_student, rejected_teacher, "k3")
    rejected_weights = torch.tensor([1.0, 0.5])
    expected = (response_losses.sum() + 2.0 * (rejected_losses * rejected_weights).sum()) / (
        2.0 + 2.0 * rejected_weights.sum()
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/forward_kl_loss"].values[0]),
        response_losses.mean(),
    )
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/rejected_draft_loss"].values[0]),
        rejected_losses.mean(),
    )
    assert "distillation/rejected_draft_forward_kl_loss" not in metrics


def test_rejected_draft_position_decay_weights_loss_by_draft_offset():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, 0.0]),
        },
        batch_size=[1],
    )
    rejected_student = torch.tensor([-1.1, -0.8, -1.4], dtype=torch.float32)
    rejected_teacher = torch.tensor([-0.6, -1.2, -1.0], dtype=torch.float32)
    model_output = {
        "log_probs": torch.tensor([-0.4, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True, True]]),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2, 3]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=0.0,
        rejected_draft_stream_weight=1.0,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, _ = distillation_loss(config, distillation_config, model_output, data)

    rejected_losses = kl_penalty(rejected_student, rejected_teacher, "k3")
    weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    assert torch.allclose(loss, (rejected_losses * weights).sum() / weights.sum())


def _single_sample_position_decay_combined_loss(
    response_student,
    response_teacher,
    rejected_student,
    rejected_teacher,
    offsets,
    global_info,
):
    response_student = torch.tensor(response_student, dtype=torch.float32)
    response_teacher = torch.tensor(response_teacher, dtype=torch.float32)
    rejected_student = torch.tensor(rejected_student, dtype=torch.float32)
    rejected_teacher = torch.tensor(rejected_teacher, dtype=torch.float32)
    offsets = torch.tensor(offsets, dtype=torch.long)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs(torch.cat([response_teacher, torch.tensor([0.0])])),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.cat([response_student, torch.tensor([0.0])]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.ones((1, rejected_student.numel()), dtype=torch.bool),
        "opd_rejected_draft_offsets": offsets.unsqueeze(0),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info=global_info)
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=2.0,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    loss, _ = distillation_loss(config, distillation_config, model_output, data)
    response_numerator = kl_penalty(response_student, response_teacher, "k3").sum()
    weights = torch.pow(torch.tensor(0.5), (offsets.to(torch.float32) - 1.0).clamp_min(0.0))
    rejected_numerator = (kl_penalty(rejected_student, rejected_teacher, "k3") * weights).sum()
    return loss, response_numerator, rejected_numerator


def test_position_decay_combined_k3_loss_is_invariant_to_micro_batch_splits():
    offsets = [1, 2, 3]
    weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    global_info = {
        "dp_size": 1,
        "batch_num_tokens": 4,
        "opd_rejected_draft_batch_num_tokens": 6,
        "opd_rejected_draft_batch_effective_num_tokens": float((weights.sum() * 2).item()),
    }
    loss_a, response_a, rejected_a = _single_sample_position_decay_combined_loss(
        [-0.4, -0.9],
        [-0.5, -0.7],
        [-1.1, -0.8, -1.4],
        [-0.6, -1.2, -1.0],
        offsets,
        global_info,
    )
    loss_b, response_b, rejected_b = _single_sample_position_decay_combined_loss(
        [-0.3, -1.2],
        [-0.4, -0.9],
        [-0.7, -1.4, -1.0],
        [-0.8, -1.0, -0.5],
        offsets,
        global_info,
    )

    expected = (response_a + response_b + 2.0 * (rejected_a + rejected_b)) / (4.0 + 2.0 * weights.sum() * 2.0)
    assert torch.allclose(loss_a + loss_b, expected)


def _single_sample_combined_loss(response_student, response_teacher, rejected_student, rejected_teacher, global_info):
    response_student = torch.tensor(response_student, dtype=torch.float32)
    response_teacher = torch.tensor(response_teacher, dtype=torch.float32)
    rejected_student = torch.tensor(rejected_student, dtype=torch.float32)
    rejected_teacher = torch.tensor(rejected_teacher, dtype=torch.float32)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs(torch.cat([response_teacher, torch.tensor([0.0])])),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.cat([response_student, torch.tensor([0.0])]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info=global_info)
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    loss, _ = distillation_loss(config, distillation_config, model_output, data)
    numerator = torch.cat(
        [
            kl_penalty(response_student, response_teacher, "k3"),
            kl_penalty(rejected_student, rejected_teacher, "k3"),
        ]
    ).sum()
    return loss, numerator


def test_combined_k3_loss_is_invariant_to_micro_batch_splits():
    global_info = {
        "dp_size": 1,
        "batch_num_tokens": 4,
        "opd_rejected_draft_batch_num_tokens": 4,
    }
    loss_a, numerator_a = _single_sample_combined_loss(
        [-0.4, -0.9], [-0.5, -0.7], [-1.1, -0.8], [-0.6, -1.2], global_info
    )
    loss_b, numerator_b = _single_sample_combined_loss(
        [-0.3, -1.2], [-0.4, -0.9], [-0.7, -1.4], [-0.8, -1.0], global_info
    )

    expected = (numerator_a + numerator_b) / 8
    assert torch.allclose(loss_a + loss_b, expected)


def test_combined_k3_loss_dp_scaled_average_matches_global_mean():
    global_info = {
        "dp_size": 2,
        "batch_num_tokens": 4,
        "opd_rejected_draft_batch_num_tokens": 4,
    }
    rank0_loss, numerator_a = _single_sample_combined_loss(
        [-0.4, -0.9], [-0.5, -0.7], [-1.1, -0.8], [-0.6, -1.2], global_info
    )
    rank1_loss, numerator_b = _single_sample_combined_loss(
        [-0.3, -1.2], [-0.4, -0.9], [-0.7, -1.4], [-0.8, -1.0], global_info
    )

    expected = (numerator_a + numerator_b) / 8
    assert torch.allclose((rank0_loss + rank1_loss) / 2, expected)


def test_count_dflash_rejected_draft_tokens_respects_empty_values_and_limit():
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            [[1, 2], [], [3, -1]], batch_size=3, max_tokens_per_sample=None
        )
        == 3
    )
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            [[1, 2], [], [3, -1]], batch_size=3, max_tokens_per_sample=1
        )
        == 2
    )
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            None, batch_size=3, max_tokens_per_sample=None
        )
        == 0
    )


def _bernoulli_forward_kl(student, teacher):
    student = torch.as_tensor(student, dtype=torch.float32)
    teacher = torch.as_tensor(teacher, dtype=torch.float32)
    sp, tp = student.exp(), teacher.exp()
    return tp * (teacher - student) + (1.0 - tp) * (torch.log1p(-tp) - torch.log1p(-sp))


def _response_stream_case(reject_indices):
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


def _symmetric_k3_cfg():
    return SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.5,
        forward_kl_weight=1.0,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )


def test_corrected_token_is_forward_only_by_default():
    # By default (no flag), the corrected token at a reject position drops its reverse-KL term;
    # agree positions stay symmetric. reject at response position 1.
    data, model_output = _response_stream_case(reject_indices=[1])
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})

    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=_symmetric_k3_cfg()), model_output, data)

    student = torch.tensor([-0.4, -0.9, -0.3])
    teacher = torch.tensor([-0.5, -0.7, -0.6])
    k3 = kl_penalty(student, teacher, "k3")
    bern = _bernoulli_forward_kl(student, teacher)
    reverse_on = torch.tensor([1.0, 0.0, 1.0])  # reverse dropped at the corrected (reject) position
    per_token = 0.5 * k3 * reverse_on + 1.0 * bern
    assert torch.allclose(loss, per_token.mean())


def test_no_reject_means_pure_symmetric_kl():
    # No reject positions -> every response token keeps the full symmetric KL.
    data, model_output = _response_stream_case(reject_indices=[])
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})

    loss, _ = distillation_loss(config, SimpleNamespace(distillation_loss=_symmetric_k3_cfg()), model_output, data)

    student = torch.tensor([-0.4, -0.9, -0.3])
    teacher = torch.tensor([-0.5, -0.7, -0.6])
    per_token = 0.5 * kl_penalty(student, teacher, "k3") + 1.0 * _bernoulli_forward_kl(student, teacher)
    assert torch.allclose(loss, per_token.mean())
