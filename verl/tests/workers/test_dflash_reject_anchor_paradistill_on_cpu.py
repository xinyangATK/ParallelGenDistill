"""CPU tests: the draftopd REJECT-anchor scheme composes with the paradistill DRAFT-PREFIX flow.

paradistill's draft-prefix reverse (sample y_hat -> teacher verify-forward -> top-K reverse KL) is
anchor-mode-agnostic: it consumes (anchor_positions, block_keep_mask, response slots) from
``_build_opd_anchor_plan`` and feeds them to ``_run_teacher_block_verify_forward`` (covered by
test_dflash_teacher_verify_on_cpu). These tests pin the ONLY anchor-mode-specific piece for the combo
``anchor_mode=reject`` + ``onpolicy_reverse``: that the reject-mode plan is well-formed for the reverse
block (anchors AT reject positions, segments capped to K, no-reject samples skipped, and the optional
rollout-reject merge adds only inert segment-0 blocks).
"""

import torch

from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM


def _student():
    return object.__new__(ComposedDFlashStudentForCausalLM)


def _plan(reject_token_indices, *, prompt_len=2, resp_len=8, block_size=4,
          include_rejected=False, rej_anchor=None, rej_offset=None, rej_mask=None):
    s = _student()
    seq_len = prompt_len + resp_len
    input_ids = torch.arange(seq_len).view(1, -1)
    kwargs = dict(
        input_ids=input_ids, attention_mask=torch.ones(1, seq_len),
        prompt_lengths=torch.tensor([prompt_len]), response_lengths=torch.tensor([resp_len]),
        reject_token_indices=reject_token_indices, draft_block_size=block_size,
        anchor_mode="reject", include_rejected_draft_anchors=include_rejected,
    )
    if include_rejected:
        kwargs.update(rejected_draft_anchor_indices=rej_anchor, rejected_draft_offsets=rej_offset,
                      rejected_draft_mask=rej_mask)
    return s._build_opd_anchor_plan(**kwargs)


def test_reject_anchor_plan_places_anchors_at_reject_positions():
    # rejects at response positions 1,4 ; prompt_len=2 -> seq anchors -1->1, 1->3, 4->6
    ap, seg, rs, keep, vsl, _ = _plan(torch.tensor([[1, 4]]))
    active = keep[0].tolist()
    anchors = [a for a, k in zip(ap[0].tolist(), active) if k]
    segs = [g for g, k in zip(seg[0].tolist(), active) if k]
    assert anchors == [1, 3, 6], anchors                     # prompt-end (-1->1) + each reject (->3,->6)
    assert segs == [2, 3, 3], segs                           # to next reject, capped at block_size-1=3
    assert rs[0].tolist()[: len(anchors)] == [1, 3, 6]       # row_starts == full anchors
    assert int(vsl[0]) == prompt_len_resp_len()


def prompt_len_resp_len():
    return 2 + 8


def test_reject_anchor_segment_capped_at_block_size_minus_1():
    # single reject at 1, then a long gap to the end -> the gap segment is capped at K=3 (tail uncovered)
    ap, seg, rs, keep, vsl, _ = _plan(torch.tensor([[1]]))
    active = keep[0].tolist()
    segs = [g for g, k in zip(seg[0].tolist(), active) if k]
    assert max(segs) <= 3, segs                              # no segment exceeds block_size-1
    # anchor at the reject (seq pos 3) exists with a capped segment
    anchors = [a for a, k in zip(ap[0].tolist(), active) if k]
    assert 3 in anchors


def test_reject_anchor_no_reject_sample_is_skipped():
    ap, seg, rs, keep, vsl, m = _plan(torch.tensor([[-1, -1]]))  # no valid rejects
    assert not bool(keep.any())                               # no active block -> not trained
    assert m.get("skipped_sample_count") == 1


def test_rollout_reject_merge_adds_only_inert_segment0_blocks():
    # include_rejected_draft_anchors merges rollout-reject anchors as segment_len=0 blocks (never enumerated
    # as response slots: the response loop needs segment_len >= offset>=1). So they cannot be fresh-sampled.
    base = _plan(torch.tensor([[1, 4]]), include_rejected=False)
    merged = _plan(
        torch.tensor([[1, 4]]), include_rejected=True,
        rej_anchor=torch.tensor([[5]]), rej_offset=torch.tensor([[1]]), rej_mask=torch.tensor([[True]]),
    )
    base_active = [(a, g) for a, g, k in zip(base[0][0].tolist(), base[1][0].tolist(), base[3][0].tolist()) if k]
    merged_active = [(a, g) for a, g, k in zip(merged[0][0].tolist(), merged[1][0].tolist(), merged[3][0].tolist()) if k]
    # the original reject-position anchors (segment>0) are unchanged
    assert [ag for ag in merged_active if ag[1] > 0] == base_active
    # the merge added at least one extra block, and every extra block has segment_len == 0 (inert)
    extra = [ag for ag in merged_active if ag not in base_active]
    assert len(extra) >= 1 and all(g == 0 for _, g in extra), merged_active
