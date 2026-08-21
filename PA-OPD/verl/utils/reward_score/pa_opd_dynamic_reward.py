"""Custom rule reward for the two PA-OPD runtime modes."""

from __future__ import annotations

import os
from typing import Any

from verl.trainer.ppo.pa_opd_runtime import ADAPTIVE_FORMAT_MODE, OPD_RLVR_MODE
from verl.utils.reward_score.pa_opd_protocol import score_pa_opd_response


def compute_score(
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **_: Any,
) -> float:
    """Return format-only reward or 0.1-format + 1.0-accuracy reward.

    The mode is read per call so Ray workers do not retain stale state when a
    new run is launched in the same environment.
    """

    del extra_info
    mode = os.environ.get("PA_OPD_MODE", ADAPTIVE_FORMAT_MODE)
    format_coef = float(os.environ.get("PA_OPD_FORMAT_REWARD_COEF", "0.1"))
    accuracy_coef = float(os.environ.get("PA_OPD_ACCURACY_REWARD_COEF", "1.0"))
    parts = score_pa_opd_response(
        solution_str,
        ground_truth,
        format_coef=format_coef,
        accuracy_coef=accuracy_coef,
    )
    if mode == ADAPTIVE_FORMAT_MODE:
        return float(parts["format_reward"])
    if mode == OPD_RLVR_MODE:
        return float(parts["score"])
    raise ValueError(f"Unsupported PA_OPD_MODE={mode!r}")
