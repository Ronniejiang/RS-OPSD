"""PA-OPD entrypoint with adaptive format supervision.

This module keeps the vendored VERL trainer intact and installs narrowly
scoped lifecycle hooks before invoking its Hydra entrypoint.  The hooks own
the per-step rollout multiplicity, diagnostics, and strict runtime resume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from verl.trainer.ppo.pa_opd_runtime import PAOPDRuntimeController
from verl.utils.reward_score.pa_opd_protocol import parse_pa_opd_response, score_pa_opd_response


_RUNTIME_STATE_FILENAME = "pa_opd_runtime.json"


def _require_ray_managed_cuda_visibility() -> None:
    """Prevent a cluster-wide Ray NOSET flag from collapsing all FSDP ranks.

    PA-OPD uses one Ray actor per GPU. ``RAY_EXPERIMENTAL_NOSET_*`` is only
    appropriate when callers manually assign devices (for example large TP
    rollout servers); with this FSDP resource pool it causes every actor to
    retain the TaskRunner's visible device. The launcher and worker runtime
    env repeat this guard; doing it here also protects direct module launches.
    """

    if os.environ.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None) is not None:
        print("[PA-OPD] cleared RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES; Ray will bind one GPU per FSDP rank")


class _RuntimeAwareDataLoader:
    """Delegate a StatefulDataLoader while selecting n before every step."""

    def __init__(self, wrapped: Any, trainer: Any):
        self._wrapped = wrapped
        self._trainer = trainer

    def __iter__(self):
        for item in self._wrapped:
            controller = self._trainer.pa_opd_runtime_controller
            rollout_n = controller.effective_rollout_n
            self._trainer.config.actor_rollout_ref.rollout.n = rollout_n
            self._trainer._pa_opd_step_rollout_n = rollout_n
            self._trainer._pa_opd_step_rlvr_active = controller.rlvr_active
            yield item

    def __len__(self):
        return len(self._wrapped)

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


def _checkpoint_folder(trainer: Any, *, for_save: bool = False) -> Path:
    """Return the exact checkpoint directory for a load or a save.

    ``resume_from_path`` must only influence loading. Reusing it while saving
    silently writes PA-OPD runtime state into the old checkpoint, whereas
    VERL writes actor/data state into
    ``default_local_dir/global_step_<current>``.
    """

    resume_mode = str(trainer.config.trainer.resume_mode)
    if not for_save and resume_mode == "resume_path" and trainer.config.trainer.resume_from_path:
        return Path(str(trainer.config.trainer.resume_from_path)).expanduser().resolve()
    return Path(str(trainer.config.trainer.default_local_dir)).expanduser().resolve() / (
        f"global_step_{trainer.global_steps}"
    )


def _extract_ground_truths(batch: Any) -> list[Any]:
    return [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]


def _as_python_list(value: Any) -> list[Any]:
    """Convert VERL's object arrays/lists to one value per rollout row."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first_image_path(value: Any) -> str | None:
    """Extract a serializable path from PA-OPD's image metadata container."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _first_image_path(item)
            if path:
                return path
        return None
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path:
            return path
        image = value.get("image")
        if isinstance(image, str) and image:
            return image
        return None
    return value if isinstance(value, str) and value else None


def _install_runtime_hooks() -> None:
    from verl.trainer.ppo import ray_trainer

    trainer_cls = ray_trainer.RayPPOTrainer
    if getattr(trainer_cls, "_pa_opd_runtime_hooks_installed", False):
        return

    original_init = trainer_cls.__init__
    original_fit = trainer_cls.fit
    original_save_checkpoint = trainer_cls._save_checkpoint
    original_load_checkpoint = trainer_cls._load_checkpoint
    original_log_rollout_data = trainer_cls._log_rollout_data

    def init_with_runtime(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        runtime_cfg = self.config.get("pa_opd_runtime", None)
        if runtime_cfg is None:
            raise RuntimeError("PA-OPD entrypoint requires a top-level pa_opd_runtime config")
        self.pa_opd_runtime_controller = PAOPDRuntimeController(runtime_cfg)
        self._pa_opd_step_rollout_n = self.pa_opd_runtime_controller.effective_rollout_n
        self._pa_opd_step_rlvr_active = self.pa_opd_runtime_controller.rlvr_active
        self.train_dataloader = _RuntimeAwareDataLoader(self.train_dataloader, self)

        reward_cfg = runtime_cfg.get("reward", {})
        os.environ["PA_OPD_MODE"] = self.pa_opd_runtime_controller.mode
        os.environ["PA_OPD_FORMAT_REWARD_COEF"] = str(reward_cfg.get("format_coef", 0.1))
        os.environ["PA_OPD_ACCURACY_REWARD_COEF"] = str(reward_cfg.get("accuracy_coef", 1.0))

    def save_checkpoint_with_runtime(self, *args, **kwargs):
        result = original_save_checkpoint(self, *args, **kwargs)
        folder = _checkpoint_folder(self, for_save=True)
        folder.mkdir(parents=True, exist_ok=True)
        state_path = folder / _RUNTIME_STATE_FILENAME
        tmp_path = folder / f".{_RUNTIME_STATE_FILENAME}.tmp"
        payload = self.pa_opd_runtime_controller.state_dict()
        payload["global_step"] = int(self.global_steps)
        with tmp_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, state_path)
        print(f"[PA-OPD] saved strict runtime state to {state_path}")
        return result

    def load_checkpoint_with_runtime(self, *args, **kwargs):
        result = original_load_checkpoint(self, *args, **kwargs)
        if int(getattr(self, "global_steps", 0)) == 0:
            return result
        state_path = _checkpoint_folder(self) / _RUNTIME_STATE_FILENAME
        if not state_path.is_file():
            raise FileNotFoundError(
                "Strict PA-OPD resume requires pa_opd_runtime.json. "
                f"The checkpoint {state_path.parent} predates adaptive runtime checkpointing; "
                "start a new run or resume a checkpoint produced by this entrypoint."
            )
        with state_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if int(payload.get("global_step", -1)) != int(self.global_steps):
            raise RuntimeError(
                "Strict PA-OPD resume rejected: runtime-state global_step does not match checkpoint folder"
            )
        self.pa_opd_runtime_controller.load_state_dict(payload)
        print(
            "[PA-OPD] restored runtime state: "
            f"mode={self.pa_opd_runtime_controller.mode}, "
            f"phase={self.pa_opd_runtime_controller.state.phase}, "
            f"next_n={self.pa_opd_runtime_controller.effective_rollout_n}"
        )
        return result

    def log_rollout_data_with_runtime(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        extras = dict(reward_extra_infos_dict)
        for key in (
            "pa_opd_format_valid",
            "pa_opd_accuracy",
            "pa_opd_parsed_answer",
            "pa_opd_rollout_n",
            "pa_opd_source_line",
        ):
            if key in batch.non_tensor_batch:
                extras.setdefault(key, _as_python_list(batch.non_tensor_batch[key]))
        for batch_key, logged_key in (
            ("images", "pa_opd_original_image_path"),
            ("bbox_images", "pa_opd_crop_image_path"),
        ):
            if batch_key in batch.non_tensor_batch:
                extras.setdefault(
                    logged_key,
                    [_first_image_path(value) for value in _as_python_list(batch.non_tensor_batch[batch_key])],
                )
        return original_log_rollout_data(self, batch, extras, timing_raw, rollout_data_dir)

    def fit_with_runtime(self, *args, **kwargs):
        if not hasattr(self, "actor_rollout_wg"):
            raise RuntimeError("PA-OPD runtime hooks require init_workers() before fit()")

        worker_group = self.actor_rollout_wg
        original_update_actor = worker_group.update_actor

        def update_actor_with_runtime(batch):
            current_n = int(self._pa_opd_step_rollout_n)
            rlvr_active = bool(self._pa_opd_step_rlvr_active)
            responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            ground_truths = _extract_ground_truths(batch)
            parsed = [parse_pa_opd_response(text) for text in responses]
            reward_cfg = self.config.pa_opd_runtime.get("reward", {})
            decomposed = [
                score_pa_opd_response(
                    text,
                    gt,
                    format_coef=float(reward_cfg.get("format_coef", 0.1)),
                    accuracy_coef=float(reward_cfg.get("accuracy_coef", 1.0)),
                )
                for text, gt in zip(responses, ground_truths, strict=True)
            ]
            format_flags = np.asarray([float(item.format_valid) for item in parsed], dtype=np.float32)
            accuracy_flags = np.asarray(
                [float(parts["accuracy_reward"]) for parts in decomposed], dtype=np.float32
            )
            format_rate = float(format_flags.mean()) if len(format_flags) else 0.0
            accuracy_rate = float(accuracy_flags.mean()) if len(accuracy_flags) else 0.0

            batch.non_tensor_batch["pa_opd_format_valid"] = format_flags
            batch.non_tensor_batch["pa_opd_accuracy"] = accuracy_flags
            batch.non_tensor_batch["pa_opd_parsed_answer"] = np.asarray(
                [item.answer for item in parsed], dtype=object
            )
            batch.non_tensor_batch["pa_opd_rollout_n"] = np.full(len(format_flags), current_n, dtype=np.int32)
            batch.meta_info["pa_opd_effective_rollout_n"] = current_n
            batch.meta_info["pa_opd_rlvr_active"] = rlvr_active
            batch.meta_info["pa_opd_rlvr_loss_coef"] = float(
                self.pa_opd_runtime_controller.rlvr_loss_coef
            )

            actor_output = original_update_actor(batch)
            transition = self.pa_opd_runtime_controller.observe_format_valid_rate(format_rate)
            controller_metrics = self.pa_opd_runtime_controller.metrics(transition)
            controller_metrics["pa_opd/effective_rollout_n"] = float(current_n)
            controller_metrics["pa_opd/next_rollout_n"] = float(
                self.pa_opd_runtime_controller.effective_rollout_n
            )
            controller_metrics["pa_opd/rlvr_active"] = float(rlvr_active)
            controller_metrics["pa_opd/format_valid_rate"] = format_rate
            controller_metrics["pa_opd/answer_accuracy"] = accuracy_rate

            metrics = actor_output.meta_info.setdefault("metrics", {})
            for key, value in controller_metrics.items():
                existing = metrics.get(key)
                if existing is None:
                    metrics[key] = [value]
                elif isinstance(existing, list):
                    existing.append(value)
                else:
                    metrics[key] = [existing, value]
            if transition is not None:
                print(
                    f"[PA-OPD] {transition} at step {self.global_steps}: "
                    f"format_valid_rate={format_rate:.6f}, "
                    f"next_n={self.pa_opd_runtime_controller.effective_rollout_n}"
                )
            return actor_output

        worker_group.update_actor = update_actor_with_runtime
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            worker_group.update_actor = original_update_actor

    trainer_cls.__init__ = init_with_runtime
    trainer_cls.fit = fit_with_runtime
    trainer_cls._save_checkpoint = save_checkpoint_with_runtime
    trainer_cls._load_checkpoint = load_checkpoint_with_runtime
    trainer_cls._log_rollout_data = log_rollout_data_with_runtime
    trainer_cls._pa_opd_runtime_hooks_installed = True


_install_runtime_hooks()


if __name__ == "__main__":
    from verl.trainer.main_ppo import main

    _require_ray_managed_cuda_visibility()
    main()
