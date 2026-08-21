"""Pure helpers shared by PA-OPD's trainer and actor workers.

The helpers deliberately keep answer-probe scoring separate from the rollout
trajectory: a single per-sample privileged weight is computed before rollout
and is later applied to every valid reasoning token.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


def normalize_option_label(answer: object, option_labels: Sequence[str]) -> int:
    """Return the configured option index for a dataset answer.

    Vision-OPD labels are normally ``A``--``D``. Accept common wrappers such
    as ``(A)`` and ``<answer>A</answer>`` but reject ambiguous answers early.
    """

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


def compute_privileged_advantage(
    teacher_option_probs: torch.Tensor,
    student_option_probs: torch.Tensor,
    gt_option_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PA-OPD's detached sample weights and teacher-correct flags."""

    if teacher_option_probs.shape != student_option_probs.shape:
        raise ValueError("Teacher and student option probabilities must have identical shapes.")
    if gt_option_indices.shape != teacher_option_probs.shape[:1]:
        raise ValueError("gt_option_indices must be shaped [batch].")

    gt_indices = gt_option_indices.to(device=teacher_option_probs.device, dtype=torch.long)
    teacher_correct = teacher_option_probs.argmax(dim=-1).eq(gt_indices)
    teacher_gt = teacher_option_probs.gather(1, gt_indices.unsqueeze(1)).squeeze(1)
    student_gt = student_option_probs.gather(1, gt_indices.unsqueeze(1)).squeeze(1)
    weights = (teacher_gt - student_gt).clamp_min(0.0) * teacher_correct.to(teacher_gt.dtype)
    return weights.detach(), teacher_correct.detach()


def _find_subsequences(token_ids: list[int], pattern: Sequence[int]) -> list[int]:
    if not pattern:
        raise ValueError("PA-OPD tag tokenization produced an empty token sequence.")
    width = len(pattern)
    pattern_list = list(pattern)
    return [idx for idx in range(len(token_ids) - width + 1) if token_ids[idx : idx + width] == pattern_list]


def build_strict_reasoning_mask(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    close_think_token_ids: Sequence[int],
    open_answer_token_ids: Sequence[int],
    close_answer_token_ids: Sequence[int],
    open_think_token_ids: Sequence[int],
    response_texts: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return content tokens from one well-formed prefilled-``<think>`` response.

    The template supplies the sole opening ``<think>`` before generation. For
    Qwen3, ordinary XML-like tags such as ``<answer>`` can merge with their
    following token under BPE. Format validity is therefore checked on decoded
    response text when available, while the dedicated closing-think token maps
    the textual decision back to the generated reasoning-token span.
    """

    if response_ids.shape != response_mask.shape:
        raise ValueError("response_ids and response_mask must have the same shape.")
    if response_texts is not None and len(response_texts) != response_ids.shape[0]:
        raise ValueError("response_texts must have one item per response row.")

    reasoning_mask = torch.zeros_like(response_mask)
    valid_format = torch.zeros(response_ids.shape[0], dtype=torch.bool, device=response_ids.device)
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    for row_idx in range(response_ids.shape[0]):
        length = int(response_mask[row_idx].sum().item())
        token_ids = response_ids[row_idx, :length].detach().cpu().tolist()
        close_think = _find_subsequences(token_ids, close_think_token_ids)
        if response_texts is not None:
            assistant_output = "<think>" + str(response_texts[row_idx])
            positions = [assistant_output.find(tag) for tag in tags]
            is_valid = all(assistant_output.count(tag) == 1 for tag in tags) and positions == sorted(positions)
        else:
            open_think = _find_subsequences(token_ids, open_think_token_ids)
            open_answer = _find_subsequences(token_ids, open_answer_token_ids)
            close_answer = _find_subsequences(token_ids, close_answer_token_ids)
            is_valid = (
                len(open_think) == 0
                and len(close_think) == 1
                and len(open_answer) == 1
                and len(close_answer) == 1
                and close_think[0] < open_answer[0] < close_answer[0]
            )
        if is_valid and len(close_think) == 1:
            reasoning_mask[row_idx, : close_think[0]] = 1
            valid_format[row_idx] = True
    return reasoning_mask.to(dtype=response_mask.dtype), valid_format

