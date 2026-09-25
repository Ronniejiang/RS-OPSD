"""Shared probability helpers for CAD Teacher reliability gating."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


def normalize_option_label(answer: object, option_labels: Sequence[str]) -> int:
    """Return the configured option index for a single-label legacy probe."""

    text = str(answer).strip().upper()
    match = re.fullmatch(r"(?:<ANSWER>\s*)?\(?\s*([A-Z])\s*\)?(?:\s*</ANSWER>)?", text)
    if match is None:
        raise ValueError(f"PA-OPD expected an option label, got {answer!r}")
    label = match.group(1)
    normalized_labels = [str(item).strip().upper() for item in option_labels]
    if label not in normalized_labels:
        raise ValueError(f"PA-OPD answer {label!r} is not in {normalized_labels!r}")
    return normalized_labels.index(label)


def option_probabilities(logits: torch.Tensor, candidate_token_ids: torch.Tensor) -> torch.Tensor:
    """Normalize next-token logits over the multiple-choice candidates only."""

    if logits.ndim != 2:
        raise ValueError(f"Expected probe logits shaped [batch, vocab], got {tuple(logits.shape)}")
    if candidate_token_ids.ndim != 2 or candidate_token_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            "candidate_token_ids must be shaped [batch, num_options] and match the probe batch, "
            f"got {tuple(candidate_token_ids.shape)} for {tuple(logits.shape)}"
        )
    candidate_logits = logits.gather(dim=1, index=candidate_token_ids.to(device=logits.device))
    return torch.softmax(candidate_logits, dim=-1)


def compute_teacher_reliability(
    teacher_option_probs: torch.Tensor,
    gt_option_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the hard privileged reliability gate and Teacher GT probability."""

    if teacher_option_probs.ndim != 2:
        raise ValueError("teacher_option_probs must be shaped [batch, num_options].")
    if gt_option_indices.shape != teacher_option_probs.shape[:1]:
        raise ValueError("gt_option_indices must be shaped [batch].")

    gt_indices = gt_option_indices.to(device=teacher_option_probs.device, dtype=torch.long)
    teacher_gt = teacher_option_probs.gather(1, gt_indices.unsqueeze(1)).squeeze(1)
    teacher_reliable = teacher_option_probs.argmax(dim=-1).eq(gt_indices)
    return teacher_reliable.detach(), teacher_gt.detach()


__all__ = [
    "compute_teacher_reliability",
    "normalize_option_label",
    "option_probabilities",
]
