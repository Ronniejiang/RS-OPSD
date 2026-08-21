"""Shared output protocol and rule rewards for PA-OPD.

Qwen3's chat template may prefill ``<think>`` outside the response token
tensor.  ``canonicalize_response`` restores that prefix before applying the
same strict parser in training, rollout diagnostics, and evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_RESERVED_TAGS = ("<think>", "</think>", "<answer>", "</answer>")
_FORMAT_RE = re.compile(
    r"\A<think>"
    r"(?P<reasoning>(?:(?!<think>|</think>|<answer>|</answer>)[\s\S])*)"
    r"</think>\s*<answer>\s*(?P<answer>[ABCD])\s*</answer>\s*\Z"
)
_GT_RE = re.compile(r"(?:\A|[^A-Za-z])([ABCD])(?:[^A-Za-z]|\Z)", re.IGNORECASE)


@dataclass(frozen=True)
class PAOPDProtocolResult:
    format_valid: bool
    answer: str | None
    canonical_response: str


def canonicalize_response(solution_str: Any) -> str:
    """Return the full assistant response expected by the strict parser."""

    text = "" if solution_str is None else str(solution_str)
    text = text.strip()
    if not text.startswith("<think>"):
        text = "<think>" + text
    return text


def parse_pa_opd_response(solution_str: Any) -> PAOPDProtocolResult:
    """Strictly parse one reasoning block followed by one A-D answer block."""

    canonical = canonicalize_response(solution_str)
    match = _FORMAT_RE.fullmatch(canonical)
    if match is None:
        return PAOPDProtocolResult(False, None, canonical)
    return PAOPDProtocolResult(True, match.group("answer").upper(), canonical)


def normalize_ground_truth(ground_truth: Any) -> str | None:
    """Extract an A-D label from the dataset's common scalar/dict formats."""

    if isinstance(ground_truth, dict):
        for key in ("answer", "label", "ground_truth", "target"):
            if key in ground_truth:
                return normalize_ground_truth(ground_truth[key])
        return None

    text = "" if ground_truth is None else str(ground_truth).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = _GT_RE.search(text)
    return match.group(1).upper() if match is not None else None


def score_pa_opd_response(
    solution_str: Any,
    ground_truth: Any,
    *,
    format_coef: float = 0.1,
    accuracy_coef: float = 1.0,
) -> dict[str, float | str | None]:
    """Compute decomposed format/accuracy rewards and their weighted sum."""

    parsed = parse_pa_opd_response(solution_str)
    target = normalize_ground_truth(ground_truth)
    format_reward = float(parsed.format_valid)
    accuracy_reward = float(parsed.format_valid and target is not None and parsed.answer == target)
    return {
        "score": format_coef * format_reward + accuracy_coef * accuracy_reward,
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "parsed_answer": parsed.answer,
        "ground_truth_answer": target,
    }


__all__ = [
    "PAOPDProtocolResult",
    "canonicalize_response",
    "normalize_ground_truth",
    "parse_pa_opd_response",
    "score_pa_opd_response",
]
