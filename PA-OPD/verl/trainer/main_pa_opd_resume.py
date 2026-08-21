"""PA-OPD entrypoint with deterministic legacy-checkpoint migration.

The normal PA-OPD loader remains strict.  This entrypoint permits exactly one
legacy case: all model/training state (including the EMA Teacher) loaded
successfully, but ``pa_opd_runtime.json`` is absent.  It reconstructs adaptive-format state from complete rollout JSONL files
ending at the resumed global step, or initializes the deterministic fixed-n
OPD-RLVR state without inferring rollout history.  Any earlier checkpoint-loading failure is propagated unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from verl.trainer import main_pa_opd as _main_pa_opd  # installs the standard strict hooks
from verl.trainer.ppo.pa_opd_runtime import ADAPTIVE_FORMAT_MODE, OPD_RLVR_MODE, PAOPDRuntimeState
from verl.utils.reward_score.pa_opd_protocol import parse_pa_opd_response


def _checkpoint_folder(trainer: Any) -> Path:
    resume_mode = str(trainer.config.trainer.resume_mode)
    if resume_mode == "resume_path" and trainer.config.trainer.resume_from_path:
        return Path(str(trainer.config.trainer.resume_from_path)).expanduser().resolve()
    return Path(str(trainer.config.trainer.default_local_dir)).expanduser().resolve() / (
        f"global_step_{trainer.global_steps}"
    )


def _load_rollout_format_rate(path: Path, expected_rows: int) -> float:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if "output" not in row:
                raise RuntimeError(f"Legacy rollout {path}:{line_number} has no output field")
            rows.append(row)

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Strict legacy migration rejected incomplete rollout {path}: "
            f"expected {expected_rows} rows, found {len(rows)}"
        )
    valid = sum(parse_pa_opd_response(row["output"]).format_valid for row in rows)
    return valid / len(rows)


def _reconstruct_runtime_state(trainer: Any) -> None:
    controller = trainer.pa_opd_runtime_controller
    if controller.mode == OPD_RLVR_MODE:
        # This fixed-n mode has no adaptive history to infer. Make the initial
        # state explicit so a following PA-OPD checkpoint writes a strict
        # runtime sidecar, while preserving its intended n=8 RLVR behavior.
        controller.state = PAOPDRuntimeState(
            phase=OPD_RLVR_MODE,
            consecutive_good_steps=0,
            last_format_valid_rate=None,
            effective_rollout_n=controller.recovery_rollout_n,
        )
        print(
            "[PA-OPD] initialized fixed OPD-RLVR runtime from legacy checkpoint: "
            f"phase={controller.state.phase}, next_n={controller.effective_rollout_n}"
        )
        return
    if controller.mode != ADAPTIVE_FORMAT_MODE:
        raise RuntimeError(f"Unsupported PA-OPD legacy migration mode: {controller.mode!r}")

    checkpoint_folder = _checkpoint_folder(trainer)
    rollout_dir = checkpoint_folder.parent.parent / "rollouts"
    expected_rows = int(trainer.config.data.train_batch_size) * controller.recovery_rollout_n
    latest_step = int(trainer.global_steps)
    consecutive_good = 0
    inspected: list[tuple[int, float]] = []

    for step in range(latest_step, max(0, latest_step - controller.stable_steps), -1):
        rollout_path = rollout_dir / f"{step}.jsonl"
        if not rollout_path.is_file():
            raise FileNotFoundError(
                "Strict legacy migration needs uninterrupted rollout history; "
                f"missing {rollout_path}"
            )
        rate = _load_rollout_format_rate(rollout_path, expected_rows)
        inspected.append((step, rate))
        if rate < controller.valid_threshold:
            break
        consecutive_good += 1

    if consecutive_good >= controller.stable_steps:
        controller.state = PAOPDRuntimeState(
            phase="opd_only",
            consecutive_good_steps=0,
            last_format_valid_rate=inspected[0][1],
            effective_rollout_n=controller.opd_only_rollout_n,
        )
    else:
        controller.state = PAOPDRuntimeState(
            phase="format_recovery",
            consecutive_good_steps=consecutive_good,
            last_format_valid_rate=inspected[0][1],
            effective_rollout_n=controller.recovery_rollout_n,
        )

    rates = ", ".join(f"step {step}={rate:.6f}" for step, rate in inspected)
    print(
        "[PA-OPD] deterministically reconstructed adaptive runtime from legacy rollouts: "
        f"{rates}; phase={controller.state.phase}, "
        f"consecutive_good_steps={controller.state.consecutive_good_steps}, "
        f"next_n={controller.effective_rollout_n}"
    )


def _install_legacy_migration_hook() -> None:
    from verl.trainer.ppo import ray_trainer

    trainer_cls = ray_trainer.RayPPOTrainer
    if getattr(trainer_cls, "_pa_opd_legacy_migration_installed", False):
        return
    strict_load_checkpoint = trainer_cls._load_checkpoint

    def load_checkpoint_with_legacy_migration(self, *args, **kwargs):
        try:
            return strict_load_checkpoint(self, *args, **kwargs)
        except FileNotFoundError as exc:
            # Do not suppress missing EMA Teacher shards, actor shards, data.pt,
            # or any other checkpoint error.  The standard loader only emits
            # this exact marker after all those components loaded successfully.
            if "pa_opd_runtime.json" not in str(exc):
                raise
            legacy_flag = self.config.get("pa_opd_legacy_rollout_resume", os.environ.get("PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME", "0"))
            if str(legacy_flag).strip().lower() not in {"1", "true", "yes", "on"}:
                raise RuntimeError(
                    "Legacy rollout migration is disabled. Set "
                    "PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME=1 for the one-time migration."
                ) from exc
            _reconstruct_runtime_state(self)
            return None

    trainer_cls._load_checkpoint = load_checkpoint_with_legacy_migration
    trainer_cls._pa_opd_legacy_migration_installed = True


_install_legacy_migration_hook()


if __name__ == "__main__":
    from verl.trainer.main_ppo import main

    _main_pa_opd._require_ray_managed_cuda_visibility()
    main()
