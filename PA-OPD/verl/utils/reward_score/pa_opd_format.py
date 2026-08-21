"""Format-only reward used by PA-OPD's GRPO branch."""

from __future__ import annotations


def is_valid_pa_opd_format(response: str) -> bool:
    """Check the four tags on the complete assistant message.

    PA-OPD's chat template pre-fills the opening ``<think>`` before rollout,
    so reward-manager responses are reconstructed with that fixed prefix.
    """

    assistant_output = "<think>" + response
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(assistant_output.count(tag) != 1 for tag in tags):
        return False
    positions = [assistant_output.index(tag) for tag in tags]
    return positions == sorted(positions)


def compute_score(solution_str: str, **_: object) -> dict[str, float]:
    """Return a binary format reward without consulting answer correctness."""

    valid = is_valid_pa_opd_format(solution_str)
    return {"score": float(valid), "format_valid": float(valid)}
