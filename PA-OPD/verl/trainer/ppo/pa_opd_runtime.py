"""Runtime controller for PA-OPD rollout/reward modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ADAPTIVE_FORMAT_MODE = "adaptive_format"
OPD_RLVR_MODE = "opd_rlvr"
SUPPORTED_MODES = {ADAPTIVE_FORMAT_MODE, OPD_RLVR_MODE}
STATE_VERSION = 1


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)


@dataclass
class PAOPDRuntimeState:
    phase: str = "format_recovery"
    consecutive_good_steps: int = 0
    last_format_valid_rate: float | None = None
    effective_rollout_n: int = 8


class PAOPDRuntimeController:
    """Choose rollout multiplicity and RLVR activation for each train step.

    State transitions apply to the *next* step.  In adaptive mode, three
    consecutive recovery steps at or above the threshold enter n=1 OPD-only
    mode.  Any sub-threshold OPD-only step returns to recovery immediately.
    """

    def __init__(self, config: Any):
        self.mode = str(_cfg_get(config, "mode", ADAPTIVE_FORMAT_MODE))
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported PA-OPD mode {self.mode!r}; expected one of {sorted(SUPPORTED_MODES)}")

        adaptive = _cfg_get(config, "adaptive_format", None)
        self.recovery_rollout_n = int(_cfg_get(adaptive, "recovery_rollout_n", 8))
        self.opd_only_rollout_n = int(_cfg_get(adaptive, "opd_only_rollout_n", 1))
        self.valid_threshold = float(_cfg_get(adaptive, "valid_threshold", 0.90))
        self.stable_steps = int(_cfg_get(adaptive, "stable_steps", 3))
        self.rlvr_loss_coef = float(_cfg_get(config, "rlvr_loss_coef", 0.1))

        if self.recovery_rollout_n < 2:
            raise ValueError("recovery_rollout_n must be >= 2 so grouped RLVR has within-group samples")
        if self.opd_only_rollout_n != 1:
            raise ValueError("opd_only_rollout_n must be 1 for single-trajectory PA-OPD")
        if not 0.0 <= self.valid_threshold <= 1.0:
            raise ValueError("valid_threshold must be in [0, 1]")
        if self.stable_steps < 1:
            raise ValueError("stable_steps must be >= 1")

        initial_n = self.recovery_rollout_n
        self.state = PAOPDRuntimeState(effective_rollout_n=initial_n)
        if self.mode == OPD_RLVR_MODE:
            self.state.phase = "opd_rlvr"

    @property
    def effective_rollout_n(self) -> int:
        if self.mode == OPD_RLVR_MODE:
            return self.recovery_rollout_n
        return self.state.effective_rollout_n

    @property
    def rlvr_active(self) -> bool:
        return self.mode == OPD_RLVR_MODE or self.state.phase == "format_recovery"

    def observe_format_valid_rate(self, rate: float) -> str | None:
        """Update state from the current rollout and return a transition label."""

        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"format valid rate must be in [0, 1], got {rate}")
        self.state.last_format_valid_rate = rate

        if self.mode == OPD_RLVR_MODE:
            self.state.phase = "opd_rlvr"
            self.state.effective_rollout_n = self.recovery_rollout_n
            return None

        if self.state.phase == "format_recovery":
            if rate >= self.valid_threshold:
                self.state.consecutive_good_steps += 1
            else:
                self.state.consecutive_good_steps = 0
            if self.state.consecutive_good_steps >= self.stable_steps:
                self.state.phase = "opd_only"
                self.state.effective_rollout_n = self.opd_only_rollout_n
                self.state.consecutive_good_steps = 0
                return "format_recovery_to_opd_only"
            return None

        if self.state.phase != "opd_only":
            raise RuntimeError(f"Invalid adaptive PA-OPD phase: {self.state.phase!r}")
        if rate < self.valid_threshold:
            self.state.phase = "format_recovery"
            self.state.effective_rollout_n = self.recovery_rollout_n
            self.state.consecutive_good_steps = 0
            return "opd_only_to_format_recovery"
        self.state.effective_rollout_n = self.opd_only_rollout_n
        return None

    def metrics(self, transition: str | None = None) -> dict[str, float]:
        transition_code = {
            None: 0.0,
            "format_recovery_to_opd_only": 1.0,
            "opd_only_to_format_recovery": -1.0,
        }[transition]
        return {
            "pa_opd/effective_rollout_n": float(self.effective_rollout_n),
            "pa_opd/format_recovery_active": float(self.state.phase == "format_recovery"),
            "pa_opd/format_stable_steps": float(self.state.consecutive_good_steps),
            "pa_opd/format_state_transition": transition_code,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "mode": self.mode,
            "recovery_rollout_n": self.recovery_rollout_n,
            "opd_only_rollout_n": self.opd_only_rollout_n,
            "valid_threshold": self.valid_threshold,
            "stable_steps": self.stable_steps,
            "rlvr_loss_coef": self.rlvr_loss_coef,
            "state": asdict(self.state),
        }

    def load_state_dict(self, saved: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for key in (
            "version",
            "mode",
            "recovery_rollout_n",
            "opd_only_rollout_n",
            "valid_threshold",
            "stable_steps",
            "rlvr_loss_coef",
        ):
            if saved.get(key) != expected[key]:
                raise RuntimeError(
                    f"Strict PA-OPD resume rejected: checkpoint {key}={saved.get(key)!r}, "
                    f"current config has {expected[key]!r}"
                )
        state = saved.get("state")
        if not isinstance(state, Mapping):
            raise RuntimeError("Strict PA-OPD resume rejected: missing runtime state")
        self.state = PAOPDRuntimeState(**dict(state))
        if self.state.effective_rollout_n != self.effective_rollout_n:
            raise RuntimeError("Strict PA-OPD resume rejected: inconsistent effective_rollout_n")


__all__ = [
    "ADAPTIVE_FORMAT_MODE",
    "OPD_RLVR_MODE",
    "PAOPDRuntimeController",
    "PAOPDRuntimeState",
]
