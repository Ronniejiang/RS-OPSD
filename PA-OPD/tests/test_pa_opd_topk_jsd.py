"""Standalone regression tests for GT-safe direct PA-OPDVR top-k JSD."""

from types import SimpleNamespace

import torch

from verl.trainer.ppo.core_algos import compute_pa_opd_safe_topk_jsd_loss
from verl.trainer.ppo.pa_opd_direct import DirectProbePlan, build_direct_jsd_token_metadata
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import SelfDistillationConfig


def test_direct_jsd_masks_gt_prefix_and_stops_after_first_divergence() -> None:
    plan = DirectProbePlan((10, 99), ((10, 12), (99, 13)), "A")
    response_ids = torch.tensor([[44, 12, 99, 0], [44, 10, 99, 0]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    answer_mask = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]])
    metadata = build_direct_jsd_token_metadata(response_ids, response_mask, answer_mask, [plan, plan])
    assert metadata.prefix_mask.tolist() == [[0, 1, 0, 0], [0, 1, 1, 0]]
    assert metadata.target_token_ids.tolist() == [[-1, 10, -1, -1], [-1, 10, 99, -1]]
    assert metadata.allowed_token_ids[0, 1, :2].tolist() == [10, 12]
    assert metadata.allowed_token_ids[1, 2, :2].tolist() == [99, 13]


def test_shared_support_contains_topk_and_legal_answer_tokens_once() -> None:
    topk = torch.tensor([[[9, 10, 11]]])
    mandatory = torch.tensor([[[10, 12, -1]]])
    support, valid = DataParallelPPOActor._merge_topk_support(topk, mandatory)
    assert support[0, 0, valid[0, 0]].tolist() == [9, 10, 11, 12]


def test_safe_topk_jsd_removes_non_gt_teacher_option_mass_and_backpropagates() -> None:
    student = torch.log(torch.tensor([[[0.20, 0.20, 0.20]]], requires_grad=True))
    student.retain_grad()
    teacher = torch.log(torch.tensor([[[0.10, 0.80, 0.05]]]))
    token_ids = torch.tensor([[[10, 12, 50]]])
    valid = torch.ones_like(token_ids, dtype=torch.bool)
    target = torch.tensor([[10]])
    allowed = torch.tensor([[[10, 12]]])
    prefix = torch.ones((1, 1), dtype=torch.long)
    loss, metrics = compute_pa_opd_safe_topk_jsd_loss(
        student, teacher, token_ids, valid, target, allowed, prefix, torch.tensor([True]),
        SimpleNamespace(is_clip=None), loss_agg_mode="token-mean",
    )
    loss.backward()
    assert loss.item() > 0.0
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert metrics["topk_jsd/teacher_removed_non_gt_mass"] > 0.79


def test_topk_jsd_config_requires_direct_answer_and_tail_bucket() -> None:
    config = SelfDistillationConfig(
        pa_opd_enabled=True,
        pa_opd_direct_answer=True,
        pa_opd_topk_jsd_enabled=True,
        teacher_input_mode="global_plus_crop",
        full_logit_distillation=True,
        distillation_topk=64,
        distillation_add_tail=True,
        alpha=0.5,
    )
    assert config.pa_opd_topk_jsd_enabled
    try:
        SelfDistillationConfig(
            pa_opd_enabled=True,
            pa_opd_direct_answer=True,
            pa_opd_topk_jsd_enabled=True,
            teacher_input_mode="global_plus_crop",
            full_logit_distillation=True,
            distillation_topk=64,
            distillation_add_tail=False,
            alpha=0.5,
        )
    except ValueError as exc:
        assert "tail" in str(exc)
    else:
        raise AssertionError("top-k JSD unexpectedly accepted no tail bucket")
