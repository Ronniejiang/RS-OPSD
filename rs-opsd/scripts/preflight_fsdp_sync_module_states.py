#!/usr/bin/env python3
"""Exercise PPU NCCL plus the FSDP constructor with a chosen sync mode."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"FSDP_SYNC_MODULE_STATES must be boolean, got {value!r}")


def main() -> int:
    sync_module_states = _as_bool(os.environ.get("FSDP_SYNC_MODULE_STATES", "false"))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        # Confirm that the common NCCL process group itself can communicate.
        probe = torch.tensor([float(rank)], device="cuda")
        dist.all_reduce(probe)
        expected = world_size * (world_size - 1) / 2
        if probe.item() != expected:
            raise RuntimeError(f"all_reduce mismatch: {probe.item()} != {expected}")

        # Each rank uses the same deterministic initialized parameters. This
        # matches loading a shared immutable Hugging Face checkpoint per rank.
        torch.manual_seed(20260913)
        module = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 64)).to("cuda")
        wrapped = FSDP(module, device_id=local_rank, sync_module_states=sync_module_states)
        parameter_sum = sum(parameter.detach().float().sum() for parameter in wrapped.parameters())
        gathered = [torch.zeros_like(parameter_sum) for _ in range(world_size)]
        dist.all_gather(gathered, parameter_sum)
        if rank == 0:
            print(
                "FSDP_PPU2_PREFLIGHT_OK "
                f"world_size={world_size} sync_module_states={sync_module_states} "
                f"all_reduce_sum={probe.item():.1f} parameter_sums={[item.item() for item in gathered]}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
