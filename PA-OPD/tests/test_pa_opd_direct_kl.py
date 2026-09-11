"""Regression coverage for the portable 2K direct PA-OPDVR + KL recipe."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.trainer.ppo.utils import need_reference_policy
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import SelfDistillationConfig


def test_direct_2k_kl_config_uses_frozen_reference_and_ema_teacher() -> None:
    config = (Path(__file__).resolve().parents[1] / "verl/trainer/config/pa_opdvr_direct_2k_kl.yaml").read_text()
    assert "use_kl_loss: true" in config
    assert "kl_loss_coef: 0.001" in config
    assert "kl_loss_type: low_var_kl" in config
    assert "teacher_model_source: fixed" in config
    assert "fixed_teacher_ema: true" in config
    assert "pa_opd_direct_answer: true" in config
    assert "pa_opd_reward_free: true" in config
    assert "enable_thinking: false" in config
    assert "n_gpus_per_node: ${oc.env:PA_OPD_NUM_GPUS,4}" in config


def test_actor_kl_flag_activates_frozen_reference_policy() -> None:
    config = OmegaConf.create(
        {"algorithm": {"use_kl_in_reward": False}, "actor_rollout_ref": {"actor": {"use_kl_loss": True}}}
    )
    assert need_reference_policy(config)


def test_low_variance_kl_is_differentiable_and_response_masked() -> None:
    current_log_prob = torch.tensor([[-2.0, -0.4]], requires_grad=True)
    reference_log_prob = torch.tensor([[-1.0, -1.0]])
    response_mask = torch.tensor([[1, 0]])
    loss = agg_loss(kl_penalty(current_log_prob, reference_log_prob, "low_var_kl"), response_mask, "token-mean")
    (loss * 0.001).backward()
    assert loss.item() > 0
    assert current_log_prob.grad is not None
    assert current_log_prob.grad[0, 0].abs().item() > 0
    assert current_log_prob.grad[0, 1].item() == 0


def _minimal_task_runner_config(teacher_model_source: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "trainer": {"use_legacy_worker_impl": "auto"},
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "model": {},
                "actor": {
                    "strategy": "fsdp",
                    "policy_loss": {"loss_mode": "vopd"},
                    "use_kl_loss": True,
                    "self_distillation": {"teacher_model_source": teacher_model_source},
                },
            },
        }
    )


def test_fixed_teacher_source_passes_native_kl_compatibility_guard() -> None:
    TaskRunner().add_actor_rollout_worker(_minimal_task_runner_config("fixed"))
    try:
        TaskRunner().add_actor_rollout_worker(_minimal_task_runner_config("legacy"))
    except ValueError as exc:
        assert "teacher_model_source=current or fixed" in str(exc)
    else:
        raise AssertionError("legacy teacher unexpectedly passed the KL compatibility guard")


def test_fixed_teacher_ema_updates_a_distinct_teacher_module() -> None:
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = OmegaConf.create(
        {
            "self_distillation": {
                "teacher_model_source": "fixed",
                "fixed_teacher_ema": True,
                "teacher_regularization": "ema",
                "teacher_update_rate": 0.25,
            },
            "policy_loss": {"loss_mode": "vopd"},
        }
    )
    actor.actor_module = nn.Linear(1, 1, bias=False)
    actor.teacher_module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.actor_module.weight.fill_(8.0)
        actor.teacher_module.weight.fill_(0.0)
    actor._update_teacher()
    assert actor.teacher_module.weight.item() == 2.0


def test_fixed_teacher_ema_configuration_is_explicit_and_validated() -> None:
    config = SelfDistillationConfig(
        teacher_model_source="fixed", teacher_model_path="/tmp/base-model", fixed_teacher_ema=True
    )
    assert config.fixed_teacher_ema
