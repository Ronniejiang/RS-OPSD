import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from verl.trainer.main_ppo import _install_pa_opd_runtime_hooks_in_task_runner
from verl.trainer.main_pa_opd import _checkpoint_folder, _first_image_path, _require_ray_managed_cuda_visibility
from verl.trainer.ppo.pa_opd_runtime import PAOPDRuntimeController
from verl.utils.dataset.pa_opd_dataset import _build_record
from verl.utils.reward_score.pa_opd_protocol import parse_pa_opd_response, score_pa_opd_response
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.ray_utils import ray_noset_visible_devices


def _adaptive_config():
    return {
        "mode": "adaptive_format",
        "rlvr_loss_coef": 0.1,
        "adaptive_format": {
            "recovery_rollout_n": 8,
            "opd_only_rollout_n": 1,
            "valid_threshold": 0.90,
            "stable_steps": 3,
        },
    }


def test_protocol_and_decomposed_rewards():
    valid = "<think>reasoning A and B</think><answer> C </answer>"
    parsed = parse_pa_opd_response(valid)
    assert parsed.format_valid
    assert parsed.answer == "C"
    assert score_pa_opd_response(valid, "C")["score"] == 1.1
    assert score_pa_opd_response(valid, "D")["score"] == 0.1

    for invalid in (
        "<think>x</think><answer>C",
        "<think>x</think><answer>C</answer><answer>C</answer>",
        "<think><answer>A</answer></think><answer>A</answer>",
        "<think>x</think><answer>E</answer>",
    ):
        assert not parse_pa_opd_response(invalid).format_valid


def test_qwen_prefilled_think_is_restored():
    parsed = parse_pa_opd_response("reasoning</think><answer>B</answer>")
    assert parsed.format_valid
    assert parsed.answer == "B"


def test_rollout_image_metadata_has_exact_source_paths():
    assert _first_image_path([{"path": "/tmp/original.png"}]) == "/tmp/original.png"
    assert _first_image_path(np.asarray([[{"path": "/tmp/crop.png"}]], dtype=object)) == "/tmp/crop.png"
    assert _first_image_path([{"image": ""}, None]) is None

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        (root / "images").mkdir()
        (root / "teacher_images").mkdir()
        Image.new("RGB", (2, 2), color="red").save(root / "images" / "original.png")
        Image.new("RGB", (2, 2), color="blue").save(root / "teacher_images" / "crop.png")
        record = _build_record(
            {
                "images": ["images/original.png"],
                "teacher_images": ["teacher_images/crop.png"],
                "problem": "<image>\nWhich option?",
                "answer": "A",
            },
            root,
            source_line=37,
        )
    assert record["pa_opd_source_line"] == 37
    assert record["images"] == [{"path": str(root / "images" / "original.png")}]
    assert record["bbox_images"] == [{"path": str(root / "teacher_images" / "crop.png")}]


def test_adaptive_state_transitions_and_strict_resume():
    controller = PAOPDRuntimeController(_adaptive_config())
    assert controller.effective_rollout_n == 8
    assert controller.rlvr_active
    assert controller.observe_format_valid_rate(0.90) is None
    assert controller.observe_format_valid_rate(0.99) is None
    assert controller.observe_format_valid_rate(0.98) == "format_recovery_to_opd_only"
    assert controller.effective_rollout_n == 1
    assert not controller.rlvr_active

    saved = controller.state_dict()
    restored = PAOPDRuntimeController(_adaptive_config())
    restored.load_state_dict(saved)
    assert restored.effective_rollout_n == 1
    assert restored.observe_format_valid_rate(0.89) == "opd_only_to_format_recovery"
    assert restored.effective_rollout_n == 8
    assert restored.rlvr_active


def test_fixed_opd_rlvr_never_switches():
    config = _adaptive_config()
    config["mode"] = "opd_rlvr"
    controller = PAOPDRuntimeController(config)
    for rate in (1.0, 0.0, 1.0, 0.5):
        assert controller.observe_format_valid_rate(rate) is None
        assert controller.effective_rollout_n == 8
        assert controller.rlvr_active


def test_checkpoint_folder_separates_load_and_save_paths():
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path="/tmp/pa-opd-source/global_step_50",
                default_local_dir="/tmp/pa-opd-output/checkpoints",
            )
        ),
        global_steps=51,
    )
    assert _checkpoint_folder(trainer) == Path("/tmp/pa-opd-source/global_step_50").resolve()
    assert _checkpoint_folder(trainer, for_save=True) == (
        Path("/tmp/pa-opd-output/checkpoints").resolve() / "global_step_51"
    )


def test_pa_opd_clears_ray_noset_cuda_visibility():
    key = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        _require_ray_managed_cuda_visibility()
        assert key not in os.environ
        assert not ray_noset_visible_devices(os.environ)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_task_runner_installs_pa_opd_runtime_hooks():
    assert not _install_pa_opd_runtime_hooks_in_task_runner({})
    assert _install_pa_opd_runtime_hooks_in_task_runner({"pa_opd_runtime": _adaptive_config()})
    assert _install_pa_opd_runtime_hooks_in_task_runner(
        {"pa_opd_runtime": _adaptive_config(), "pa_opd_legacy_rollout_resume": True}
    )
    assert getattr(RayPPOTrainer, "_pa_opd_legacy_migration_installed", False)


def test_pa_opd_teacher_integrates_crop_without_source_leakage():
    trainer = object.__new__(RayPPOTrainer)
    original = Image.new("RGB", (4, 4), color="red")
    crop = Image.new("RGB", (2, 2), color="blue")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Question"}, {"type": "image", "image": original}]}
    ]
    result = trainer._prepare_global_plus_crop_teacher_messages(messages, [crop])
    content = result[0]["content"]
    texts = "\n".join(item["text"] for item in content if item.get("type") == "text")
    images = [item for item in content if item.get("type") == "image"]
    assert "Original full image" in texts
    assert "additional high-resolution detail" in texts
    assert "Use all visual evidence from both images" in texts
    assert "Do not mention, compare, or name the image sources" in texts
    assert "private verification" not in texts
    assert "Use both the global context" not in texts
    assert len(images) == 2
