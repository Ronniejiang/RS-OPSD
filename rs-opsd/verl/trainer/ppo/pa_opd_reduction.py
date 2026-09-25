"""Optimizer-step token denominators for PA-OPD with averaged FSDP gradients."""

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GlobalTokenMean:
    # Counts are global over one optimizer mini-batch, not one micro-batch.
    counts: torch.Tensor
    dp_size: int

    def kwargs(self, term: str) -> dict:
        index = {"semantic": 0, "jsd_prefix": 1, "response": 2}[term]
        # An all-empty term contributes differentiable zero on every rank.
        return {"batch_num_tokens": self.counts[index].clamp_min(1), "dp_size": self.dp_size}


@torch.no_grad()
def collect_global_token_mean(
    batch: Mapping[str, torch.Tensor], *, device: torch.device | str | int, group=None,
) -> GlobalTokenMean:
    """Collect all three counts once, before splitting an optimizer mini-batch.

    Reliability and signed-advantage gates stay in the numerator. The presence
    mask belongs in the semantic/JSD denominators, but not the reference KL.
    Caller must use loss_scale_factor=1: dp_size below compensates FSDP AVG;
    no additional micro-batch sample-count scaling may be applied.
    """
    response = batch["response_mask"].detach().float()
    semantic = batch["pa_opd_semantic_token_mask"].detach().float()
    prefix = batch.get("pa_opd_jsd_prefix_mask")
    prefix = torch.zeros_like(response) if prefix is None else prefix.detach().bool().float()
    present = batch.get("self_distillation_mask")
    if present is not None:
        present = present.detach().float().unsqueeze(1)
        semantic = semantic * present
        prefix = prefix * present
    counts = torch.stack((semantic.sum(), prefix.sum(), response.sum())).to(device)
    dp_size = 1
    if dist.is_available() and dist.is_initialized():
        dp_size = dist.get_world_size(group)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    return GlobalTokenMean(counts=counts, dp_size=dp_size)
