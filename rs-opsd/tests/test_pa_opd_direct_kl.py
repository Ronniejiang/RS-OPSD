"""Regression coverage for the portable 2K direct RS-OPSD + KL recipe."""

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


def test_resource_pool_honors_whole_gpu_reservation(monkeypatch):
    from verl.trainer.ppo import ray_trainer
    created = []

    def fake_pool(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(ray_trainer, "RayResourcePool", fake_pool)
    monkeypatch.setattr(ray_trainer.ResourcePoolManager, "_check_resource_available", lambda self: None)
    manager = ray_trainer.ResourcePoolManager({"global_pool": [16]}, {}, max_colocate_count=1)
    manager.create_resource_pool()
    assert created[0]["max_colocate_count"] == 1
    assert created[0]["process_on_nodes"] == [16]


def test_resource_pool_rejects_zero_colocation():
    import pytest
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager
    with pytest.raises(ValueError, match="positive integer"):
        ResourcePoolManager({"global_pool": [16]}, {}, max_colocate_count=0).create_resource_pool()


def test_task_runner_passes_explicit_colocation_to_pool_manager():
    config = OmegaConf.create({
        "trainer": {"n_gpus_per_node": 16, "nnodes": 1, "ray_max_colocate_count": 1},
        "reward_model": {"enable_resource_pool": False},
    })
    manager = TaskRunner().init_resource_pool_mgr(config)
    assert manager.max_colocate_count == 1
    assert manager.resource_pool_spec == {"global_pool": [16]}


def test_direct_2k_kl_config_uses_frozen_reference_and_ema_teacher(monkeypatch) -> None:
    config = (Path(__file__).resolve().parents[1] / "verl/trainer/config/rs_opsd_direct_2k_kl.yaml").read_text()
    assert "use_kl_loss: true" in config
    assert "kl_loss_coef: 0.001" in config
    assert "kl_loss_type: low_var_kl" in config
    assert "teacher_model_source: fixed" in config
    assert "fixed_teacher_ema: true" in config
    assert "pa_opd_direct_answer: true" in config
    assert "pa_opd_reward_free: true" in config
    assert "enable_thinking: false" in config
    # Per-node device count must not be confused with the global world size.
    monkeypatch.setenv("PA_OPD_NUM_GPUS", "16")
    monkeypatch.setenv("PA_OPD_GPUS_PER_NODE", "8")
    monkeypatch.setenv("PA_OPD_NNODES", "2")
    parsed = OmegaConf.create(config)
    assert parsed.trainer.n_gpus_per_node == 8
    assert parsed.trainer.nnodes == 2


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


def test_frozen_heterogeneous_teacher_never_receives_student_updates():
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = OmegaConf.create({
        'self_distillation': {'teacher_model_source': 'fixed', 'fixed_teacher_ema': False,
                             'teacher_regularization': 'ema', 'teacher_update_rate': 0.05},
        'policy_loss': {'loss_mode': 'vopd'},
    })
    # Different hidden dimensions/parameter layouts, same output vocabulary.
    actor.actor_module = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 7))
    actor.teacher_module = nn.Sequential(nn.Linear(3, 8), nn.Linear(8, 7)).requires_grad_(False).eval()
    before = {k: v.clone() for k, v in actor.teacher_module.state_dict().items()}
    optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.1)
    x = torch.ones(1, 3)
    for _ in range(3):
        optimizer.zero_grad()
        with torch.no_grad():
            target = actor.teacher_module(x).softmax(-1)
        loss = -(target * actor.actor_module(x).log_softmax(-1)).sum()
        loss.backward()
        optimizer.step()
        actor._update_teacher()
    for key, value in actor.teacher_module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in actor.teacher_module.parameters())


def test_fixed_non_ema_teacher_is_checkpointed_separately():
    # Execute the production checkpoint-selection block without allocating
    # distributed models. This covers legacy EMA and frozen heterogeneous cases.
    import ast
    from types import SimpleNamespace
    source = (Path(__file__).resolve().parents[1] / 'verl/workers/fsdp_workers.py').read_text()
    tree = ast.parse(source)
    predicate = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'uses_checkpointed_teacher' for t in n.targets))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == 'uses_checkpointed_teacher')
    block = compile(ast.Module(body=[predicate, branch], type_ignores=[]), '<teacher-checkpoint>', 'exec')
    for source_name, ema, regularization, expected in [
        ('fixed', False, 'ema', 'fixed'), ('fixed', True, 'ema', 'fixed'),
        ('legacy', False, 'ema', 'ref'), ('current', False, 'ema', None),
    ]:
        worker = SimpleNamespace(config=OmegaConf.create({'actor': {'policy_loss': {'loss_mode': 'vopd'}}}),
                                 teacher_module_fsdp=object(), ref_module_fsdp=object(), rank=1,
                                 ema_teacher_checkpoint_manager=None)
        context = {'self': worker, 'teacher_model_source': source_name,
                   'self_distillation_cfg': {'fixed_teacher_ema': ema, 'teacher_regularization': regularization},
                   'FSDPCheckpointManager': lambda **kwargs: kwargs}
        exec(block, context)
        manager = worker.ema_teacher_checkpoint_manager
        if expected is None:
            assert manager is None
        else:
            assert manager['model'] is (worker.teacher_module_fsdp if expected == 'fixed' else worker.ref_module_fsdp)
            assert manager['optimizer'] is None
            assert manager['checkpoint_config'] == {'load_contents': ['model'], 'save_contents': ['model']}
