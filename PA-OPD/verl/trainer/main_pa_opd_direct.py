"""No-thinking, reward-free direct-answer PA-OPDVR entrypoint for RS_OPD."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.pa_opd_direct import (
    DirectProbePlan,
    build_direct_jsd_token_metadata,
    build_direct_option_token_mask,
)
from verl.utils.reward_score.pa_opd_direct_protocol import canonicalize_option_set


def _require_ray_managed_cuda_visibility() -> None:
    """Ensure Ray assigns a distinct visible GPU to every FSDP rank."""

    if os.environ.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None) is not None:
        print("[PA-OPD direct] cleared RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")


def _as_python_list(value: Any) -> list[Any]:
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
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            path = _first_image_path(item)
            if path:
                return path
        return None
    if isinstance(value, dict):
        path = value.get("path") or value.get("image")
        return path if isinstance(path, str) and path else None
    return value if isinstance(value, str) and value else None


def _install_direct_hooks() -> None:
    from verl.trainer.ppo import ray_trainer

    trainer_cls = ray_trainer.RayPPOTrainer
    if getattr(trainer_cls, "_pa_opd_direct_hooks_installed", False):
        return

    original_fit = trainer_cls.fit
    original_log_rollout_data = trainer_cls._log_rollout_data

    def log_rollout_data_with_direct_metadata(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        extras = dict(reward_extra_infos_dict)
        for key in (
            "pa_opd_direct_answer_valid",
            "pa_opd_direct_answer_correct",
            "pa_opd_direct_parsed_answer",
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

    def fit_with_direct_protocol(self, *args, **kwargs):
        if not hasattr(self, "actor_rollout_wg"):
            raise RuntimeError("Direct PA-OPDVR hooks require init_workers() before fit()")

        worker_group = self.actor_rollout_wg
        original_update_actor = worker_group.update_actor

        def update_actor_with_direct_protocol(batch):
            response_ids = batch.batch["responses"]
            response_mask = batch.batch["response_mask"]
            decoded = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            ground_truths = _as_python_list(batch.non_tensor_batch.get("reward_model", []))
            extras = _as_python_list(batch.non_tensor_batch.get("extra_info", []))
            if len(ground_truths) != len(batch) or len(extras) != len(batch):
                raise RuntimeError("Direct PA-OPDVR batch metadata does not align with rollout rows")

            option_labels_per_row = []
            canonical_targets = []
            for row_idx, (ground_truth, extra) in enumerate(zip(ground_truths, extras, strict=True)):
                if not isinstance(extra, dict) or not extra.get("option_labels"):
                    raise ValueError(f"Direct PA-OPDVR missing option_labels for rollout row {row_idx}")
                option_labels = extra["option_labels"]
                target = ground_truth.get("ground_truth") if isinstance(ground_truth, dict) else None
                if target is None:
                    target = extra.get("answer")
                option_labels_per_row.append(option_labels)
                canonical_targets.append(canonicalize_option_set(target, option_labels))

            answer_mask, valid_rows, parsed = build_direct_option_token_mask(
                response_ids,
                response_mask,
                tokenizer=self.tokenizer,
                option_labels_per_row=option_labels_per_row,
            )
            answer_correct = np.asarray(
                [
                    float(item.valid and item.canonical_answer == target)
                    for item, target in zip(parsed, canonical_targets, strict=True)
                ],
                dtype=np.float32,
            )
            topk_jsd_enabled = bool(
                self.config.actor_rollout_ref.actor.self_distillation.get("pa_opd_topk_jsd_enabled", False)
            )
            if topk_jsd_enabled:
                probe_ids = batch.batch["pa_opd_probe_responses"]
                probe_mask = batch.batch["pa_opd_probe_response_mask"]
                probe_allowed = batch.batch["pa_opd_probe_allowed_token_ids"]
                plans = []
                for row_idx, canonical_target in enumerate(canonical_targets):
                    target_length = int(probe_mask[row_idx].sum().item())
                    response_token_ids = tuple(int(token) for token in probe_ids[row_idx, :target_length].tolist())
                    allowed_token_ids = tuple(
                        tuple(int(token) for token in row.tolist() if token >= 0)
                        for row in probe_allowed[row_idx, :target_length]
                    )
                    plans.append(DirectProbePlan(response_token_ids, allowed_token_ids, canonical_target))
                jsd_metadata = build_direct_jsd_token_metadata(
                    response_ids, response_mask, answer_mask, plans
                )
                batch.batch["pa_opd_jsd_target_token_ids"] = jsd_metadata.target_token_ids
                batch.batch["pa_opd_jsd_allowed_token_ids"] = jsd_metadata.allowed_token_ids
                batch.batch["pa_opd_jsd_prefix_mask"] = jsd_metadata.prefix_mask
            batch.batch["pa_opd_semantic_token_mask"] = answer_mask
            batch.batch["pa_opd_answer_token_mask"] = answer_mask
            batch.batch["pa_opd_trajectory_correct"] = torch.as_tensor(
                answer_correct, dtype=torch.bool, device=response_ids.device
            )
            batch.non_tensor_batch["pa_opd_direct_answer_valid"] = valid_rows.detach().cpu().numpy().astype(np.float32)
            batch.non_tensor_batch["pa_opd_direct_answer_correct"] = answer_correct
            batch.non_tensor_batch["pa_opd_direct_parsed_answer"] = np.asarray(
                [item.canonical_answer for item in parsed], dtype=object
            )

            actor_output = original_update_actor(batch)
            metrics = actor_output.meta_info.setdefault("metrics", {})
            direct_metrics = {
                "pa_opd/direct_answer_valid_fraction": float(valid_rows.float().mean().item()),
                "pa_opd/direct_answer_accuracy": float(answer_correct.mean()) if len(answer_correct) else 0.0,
                "pa_opd/direct_answer_token_count": float(answer_mask.sum().item()),
            }
            for key, value in direct_metrics.items():
                existing = metrics.get(key)
                if existing is None:
                    metrics[key] = [value]
                elif isinstance(existing, list):
                    existing.append(value)
                else:
                    metrics[key] = [existing, value]
            return actor_output

        worker_group.update_actor = update_actor_with_direct_protocol
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            worker_group.update_actor = original_update_actor

    trainer_cls.fit = fit_with_direct_protocol
    trainer_cls._log_rollout_data = log_rollout_data_with_direct_metadata
    trainer_cls._pa_opd_direct_hooks_installed = True


_install_direct_hooks()


if __name__ == "__main__":
    from verl.trainer.main_ppo import main

    _require_ray_managed_cuda_visibility()
    main()
