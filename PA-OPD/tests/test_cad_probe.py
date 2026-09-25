"""CAD GT-prefix probe regressions, including legal early termination."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from verl.trainer.ppo.pa_opd_direct import (
    build_direct_probe_plan, build_direct_jsd_token_metadata,
    compute_constrained_teacher_reliability,
)


def _plan(answer):
    tokens = {'A': 0, 'B': 1, 'C': 2, 'D': 3, ',A': 4, ',B': 5, ',C': 6, ',D': 7}
    with patch('verl.trainer.ppo.pa_opd_direct._one_token_suffix',
               side_effect=lambda tokenizer, prefix, text: tokens[text]):
        return build_direct_probe_plan(SimpleNamespace(eos_token_id=8), 'Q', 'ABCD', answer)


def _batch(plans):
    steps = max(len(p.response_token_ids) for p in plans)
    width = max(len(a) for p in plans for a in p.allowed_token_ids)
    target = torch.zeros(len(plans), steps, dtype=torch.long)
    mask = torch.zeros_like(target)
    allowed = torch.full((len(plans), steps, width), -1, dtype=torch.long)
    scores = torch.full((len(plans), steps, 9), -30.)
    for row, plan in enumerate(plans):
        for step, (token, candidates) in enumerate(zip(plan.response_token_ids, plan.allowed_token_ids)):
            target[row, step] = token
            mask[row, step] = 1
            allowed[row, step, :len(candidates)] = torch.tensor(candidates)
            scores[row, step, token] = 0.
    return scores, target, mask, allowed


def test_probe_single_multi_all_options_and_padding():
    plans = [_plan('A'), _plan('D'), _plan('A,C'), _plan('D,C,B,A')]
    for plan in plans:
        assert 8 not in plan.allowed_token_ids[0]  # no empty answer
        assert all(a[0] == 8 for a in plan.allowed_token_ids[1:])
    assert plans[-1].canonical_answer == 'A,B,C,D'
    assert plans[-1].allowed_token_ids[-1] == (8,)
    args = _batch(plans)
    assert compute_constrained_teacher_reliability(*args)[0].tolist() == [True] * 4


@pytest.mark.parametrize('step,competitor', [(0, 1), (1, 8), (1, 6), (2, 8), (3, 7)])
def test_probe_rejects_wrong_first_skipped_label_early_eos_and_extra_label(step, competitor):
    args = _batch([_plan('A,B,C')])
    args[0][0, step, competitor] = 1.
    assert not compute_constrained_teacher_reliability(*args)[0].item()


def test_eos_wins_a_tie_after_first_label():
    args = _batch([_plan('A,B')])
    args[0][0, 1, 8] = 0.
    assert not compute_constrained_teacher_reliability(*args)[0].item()


def test_jsd_metadata_carries_early_eos_and_masks_after_divergence():
    plan = _plan('A,B,C')
    responses = torch.tensor([plan.response_token_ids, [0, 8, 0, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    answer_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    metadata = build_direct_jsd_token_metadata(responses, response_mask, answer_mask, [plan, plan])
    assert metadata.allowed_token_ids[:, 1, 0].tolist() == [8, 8]
    assert metadata.target_token_ids[:, 1].tolist() == [5, 5]
    assert metadata.prefix_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
