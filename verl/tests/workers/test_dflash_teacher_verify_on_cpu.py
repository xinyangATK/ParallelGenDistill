"""CPU oracle test for the paradistill draft-prefix teacher VERIFY forward.

Proves ``_run_teacher_block_verify_forward`` (ONE packed forward over [context | appended fresh-block
slots] with the block-diagonal causal mask + position_ids = anchor+offset) is MATHEMATICALLY EQUIVALENT
to the naive per-block plain-causal teacher forward on ``[prefix<=anchor, y_hat_1..y_hat_{o-1}]`` -- i.e.
to "feed prefix + fresh tokens to the teacher and read each position". The draft head predicting
``anchor+o`` is matched to the teacher distribution at block slot ``o-1`` = ``p(.| prefix<=anchor, y_hat_<o)``.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from verl.models.transformers.dflash_student import (
    ComposedDFlashStudentForCausalLM,
    create_dflash_verify_sdpa_mask,
)


def _tiny_main_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


def _student_with(main_model) -> ComposedDFlashStudentForCausalLM:
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    torch.nn.Module.__init__(student)  # set up _modules so submodule assignment works
    student.main_model = main_model
    return student


def test_verify_forward_matches_per_block_plain_causal():
    main_model = _tiny_main_model()
    student = _student_with(main_model)
    vocab = main_model.config.vocab_size

    seq_len, block_size = 6, 3  # slot 0 = anchor, slots 1,2 = y_hat_1,y_hat_2 ; heads o=1,2
    input_ids = torch.randint(0, vocab, (1, seq_len))
    anchors = [2, 3]
    fresh = {2: [11, 22], 3: [33, 5]}  # [y_hat_1, y_hat_2] per anchor

    anchor_positions = torch.tensor([anchors], dtype=torch.long)
    block_keep_mask = torch.ones((1, len(anchors)), dtype=torch.bool)
    num_blocks = len(anchors)

    # pack slot tokens: slot 0 = anchor token, slots>=1 = fresh y_hat
    slot_ids = torch.zeros(1, num_blocks * block_size, dtype=torch.long)
    for bi, a in enumerate(anchors):
        base = bi * block_size
        slot_ids[0, base] = input_ids[0, a]
        for s in range(1, block_size):
            slot_ids[0, base + s] = fresh[a][s - 1]

    slot_hidden = student._run_teacher_block_verify_forward(
        input_ids=input_ids,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        slot_token_ids=slot_ids,
        block_size=block_size,
    )
    slot_logits = main_model.get_output_embeddings()(slot_hidden).float()  # (1, Q, V)

    def ref(prefix_ids):
        return main_model(input_ids=prefix_ids).logits.float()[0, -1]

    max_abs = 0.0
    for bi, a in enumerate(anchors):
        base = bi * block_size
        for o in range(1, block_size):  # heads o = 1..K ; teacher dist at slot o-1
            prefix = input_ids[:, : a + 1].clone()
            for j in range(1, o):
                prefix = torch.cat([prefix, torch.tensor([[fresh[a][j - 1]]])], dim=1)
            expected = ref(prefix)
            got = slot_logits[0, base + (o - 1)]
            max_abs = max(max_abs, (expected - got).abs().max().item())
    assert max_abs < 1e-4, f"verify-forward not equivalent to per-block causal forward: max|diff|={max_abs}"


def test_verify_mask_block_isolation_and_causality():
    # block B's slots must NOT attend to block A's slots; within a block, slot s sees only slots 0..s.
    seq_len, block_size = 5, 3
    anchor_positions = torch.tensor([[1, 2]], dtype=torch.long)
    keep = torch.ones((1, 2), dtype=torch.bool)
    mask = create_dflash_verify_sdpa_mask(anchor_positions, keep, seq_len, block_size, torch.device("cpu"))
    allow = mask[0, 0] == 0.0  # True where attended
    L = seq_len + 2 * block_size

    # block A = slots [5,6,7] (anchor 1), block B = slots [8,9,10] (anchor 2)
    # cross-block: block B slot 8 must not see block A slots 5,6,7
    assert not allow[8, 5] and not allow[8, 6] and not allow[8, 7]
    # within block A: slot 7 (offset 2) sees 5,6,7 ; slot 5 (offset 0) sees only 5
    assert allow[7, 5] and allow[7, 6] and allow[7, 7]
    assert allow[5, 5] and not allow[5, 6] and not allow[5, 7]
    # block A (anchor 1) context: sees kv < 1 -> only position 0
    assert allow[5, 0] and not allow[5, 1] and not allow[5, 2]
    # block B (anchor 2) context: sees kv < 2 -> positions 0,1
    assert allow[8, 0] and allow[8, 1] and not allow[8, 2]
    # context rows are plain causal and never attend to slots
    assert allow[3, 0] and allow[3, 3] and not allow[3, 4]
    assert not allow[3, seq_len]


def test_verify_mask_respects_block_keep():
    seq_len, block_size = 4, 2
    anchor_positions = torch.tensor([[1, 2]], dtype=torch.long)
    keep = torch.tensor([[True, False]], dtype=torch.bool)  # drop block B
    mask = create_dflash_verify_sdpa_mask(anchor_positions, keep, seq_len, block_size, torch.device("cpu"))
    allow = mask[0, 0] == 0.0
    # block B slots (6,7) are dropped -> attend to nothing
    assert not allow[6].any() and not allow[7].any()
    # block A slots still active
    assert allow[seq_len].any()
