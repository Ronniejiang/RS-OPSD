"""Regression coverage for the full/derived/tight direct RS-OPSD recipe."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from verl.utils.dataset.pa_opd_dataset import load_pa_opd_direct_three_image_jsonl
from verl.workers.config.actor import SelfDistillationConfig


_REMOVE_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _record(problem: str, answer: str = "B") -> dict:
    return {
        "images": ["images/full.png"],
        "teacher_images": ["teacher_images/tight.png"],
        "problem": problem,
        "answer": answer,
    }


def _write_paired_dataset(root: Path, *, derived_problem: str | None = None) -> None:
    _write_image(root / "images/full.png", (1, 2, 3))
    _write_image(root / "teacher_images/tight.png", (4, 5, 6))
    _write_image(root / "derived/teacher_images/derived.png", (7, 8, 9))
    problem = (
        "<image> Which option is correct?\n\nA. one\nB. two\nC. three\nD. four\n\n"
        + _REMOVE_HINT
    )
    base = _record(problem)
    derived = {
        **_record(derived_problem if derived_problem is not None else problem),
        "images": ["../images/full.png"],
        "teacher_images": ["teacher_images/derived.png"],
    }
    (root / "train.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
    (root / "derived/train.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "derived/train.jsonl").write_text(json.dumps(derived) + "\n", encoding="utf-8")


def test_three_image_loader_keeps_student_single_image_and_teacher_order() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _write_paired_dataset(root)
        item = load_pa_opd_direct_three_image_jsonl(root / "train.jsonl")[0]

        assert item["data_source"] == "pa_opd_direct_rs_opd_three_image"
        assert item["images"] == [{"path": str(root / "images/full.png")}]
        assert item["teacher_images"] == [
            {"path": str(root / "images/full.png")},
            {"path": str(root / "derived/teacher_images/derived.png")},
            {"path": str(root / "teacher_images/tight.png")},
        ]
        assert item["extra_info"]["teacher_image_order"] == [
            "full_image",
            "derived_teacher_image",
            "teacher_image",
        ]
        teacher_text = item["teacher_prompt"][0]["content"]
        assert teacher_text.count("<image>") == 3
        assert "Image 2 is an expanded high-resolution crop" in teacher_text
        assert "Image 3 is a tight high-resolution crop" in teacher_text
        assert _REMOVE_HINT not in teacher_text
        assert "<think>" not in item["prompt"][0]["content"]


def test_three_image_loader_rejects_misaligned_derived_annotation() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _write_paired_dataset(root, derived_problem="<image> a different question\nA. one\nB. two")
        try:
            load_pa_opd_direct_three_image_jsonl(root / "train.jsonl")
        except ValueError as exc:
            assert "do not describe the same sample" in str(exc)
        else:
            raise AssertionError("misaligned derived annotation unexpectedly loaded")


def test_three_image_config_preserves_direct_2k_kl_protocol_with_safe_student_and_teacher_contexts() -> None:
    root = Path(__file__).resolve().parents[1]
    config = (root / "verl/trainer/config/rs_opsd_direct_2k_three_image_kl.yaml").read_text()
    assert "rs_opsd_direct_2k_kl" in config
    assert "pa_opd_direct_three_image_jsonl: true" in config
    assert "teacher_image_key: teacher_images" in config
    assert "teacher_input_mode: global_plus_derived_plus_crop" in config
    assert "max_model_len: 25088" in config
    assert "ppo_max_token_len_per_gpu: 25088" in config
    assert "max_reprompt_len: 65536" in config
    assert "max_prompt_length: 24576" in config
    assert "n_gpus_per_node: ${oc.decode:${oc.env:PA_OPD_GPUS_PER_NODE,4}}" in config


def test_three_image_teacher_mode_is_valid_for_direct_pa_opd() -> None:
    config = SelfDistillationConfig(
        teacher_model_source="fixed",
        teacher_model_path="/tmp/base-model",
        teacher_input_mode="global_plus_derived_plus_crop",
        pa_opd_enabled=True,
        pa_opd_direct_answer=True,
        pa_opd_reward_free=True,
        full_logit_distillation=False,
        alpha=1.0,
    )
    assert config.teacher_input_mode == "global_plus_derived_plus_crop"
