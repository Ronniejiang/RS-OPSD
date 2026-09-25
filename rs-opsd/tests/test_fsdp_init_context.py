import torch
from torch import nn
from verl.utils.fsdp_utils import get_init_weight_context_manager


def test_no_sync_loads_real_weights_even_on_nonzero_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    context = get_init_weight_context_manager(use_meta_tensor=True, sync_module_states=False)
    with context():
        module = nn.Linear(2, 2)
    assert module.weight.device.type == "cpu"
    assert not module.weight.is_meta


def test_sync_keeps_meta_optimization_on_nonzero_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    context = get_init_weight_context_manager(use_meta_tensor=True, sync_module_states=True)
    with context():
        module = nn.Linear(2, 2)
    assert module.weight.is_meta


def test_rank_zero_has_real_weights_for_broadcast(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    context = get_init_weight_context_manager(use_meta_tensor=True, sync_module_states=True)
    with context():
        module = nn.Linear(2, 2)
    assert module.weight.device.type == "cpu"
