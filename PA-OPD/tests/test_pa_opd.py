import importlib.util
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pa_opd = _load_module("pa_opd_helpers", ROOT / "verl/trainer/ppo/pa_opd.py")
format_reward = _load_module("pa_opd_format", ROOT / "verl/utils/reward_score/pa_opd_format.py")
build_strict_reasoning_mask = pa_opd.build_strict_reasoning_mask
compute_privileged_advantage = pa_opd.compute_privileged_advantage
option_probabilities = pa_opd.option_probabilities
is_valid_pa_opd_format = format_reward.is_valid_pa_opd_format


def test_probe_probabilities_and_privileged_gate():
    teacher_probs = option_probabilities(
        torch.tensor([[0.0, 3.0, 1.0, 0.0], [4.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    )
    student_probs = option_probabilities(
        torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 3.0, 0.0, 0.0]]),
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
    )
    weights, teacher_correct = compute_privileged_advantage(
        teacher_probs, student_probs, torch.tensor([1, 1])
    )
    assert teacher_correct.tolist() == [True, False]
    assert weights[0].item() > 0
    assert weights[1].item() == 0


def test_reasoning_mask_only_keeps_valid_think_content():
    # Generated tokens: reason, </think>, <answer>, A, </answer>, pad.
    response_ids = torch.tensor([[7, 8, 20, 21, 30, 31, 4, 40, 41, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    mask, valid = build_strict_reasoning_mask(
        response_ids,
        response_mask,
        open_think_token_ids=[10, 11],
        close_think_token_ids=[20, 21],
        open_answer_token_ids=[30, 31],
        close_answer_token_ids=[40, 41],
    )
    assert valid.tolist() == [True]
    assert mask.tolist() == [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]


def test_invalid_think_format_has_no_distillation_mask():
    response_ids = torch.tensor([[7, 20, 21, 30, 31, 4, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    mask, valid = build_strict_reasoning_mask(
        response_ids,
        response_mask,
        open_think_token_ids=[10],
        close_think_token_ids=[20, 21],
        open_answer_token_ids=[30, 31],
        close_answer_token_ids=[40, 41],
    )
    assert valid.tolist() == [False]
    assert mask.sum().item() == 0


def test_format_reward_reconstructs_prefilled_opening_tag():
    assert is_valid_pa_opd_format("reason</think><answer>A</answer>")
    assert not is_valid_pa_opd_format("reason</think><answer>A</answer></think>")


def test_local_data_record_uses_cached_original_and_source_crop(tmp_path):
    prepare_data = _load_module("pa_opd_prepare_data", ROOT / "scripts/prepare_data.py")
    source = tmp_path / "source"
    image_cache = tmp_path / "cache" / "images"
    (source / "teacher_images").mkdir(parents=True)
    image_cache.mkdir(parents=True)
    (image_cache / "original.png").write_bytes(b"original")
    crop = source / "teacher_images" / "crop.png"
    crop.write_bytes(b"crop")

    record = prepare_data.build_record(
        {
            "images": ["images/original.png"],
            "teacher_images": ["teacher_images/crop.png"],
            "problem": "<image>\nQuestion?\n\nOnly focus on the objects inside the red bounding box in the image to answer this question.",
            "answer": "c",
        },
        image_cache,
        source,
    )

    assert record["images"] == [{"path": str(image_cache / "original.png")}]
    assert record["bbox_images"] == [{"path": str(crop)}]
    assert record["reward_model"]["ground_truth"] == "C"
    assert "red bounding box" not in record["prompt"][0]["content"]
    assert "</think><answer>" in record["prompt"][0]["content"]


def test_local_archives_require_explicit_extraction(tmp_path):
    prepare_data = _load_module("pa_opd_prepare_archive", ROOT / "scripts/prepare_data.py")
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "images.tar.gz00").write_bytes(b"archive-part")

    try:
        prepare_data.materialize_source_images(source, tmp_path / "output", extract=False)
    except RuntimeError as exc:
        assert "--extract-source-images" in str(exc)
    else:
        raise AssertionError("expected source archive materialization to require an explicit flag")


def test_local_split_archive_materializes_into_output_cache(tmp_path):
    prepare_data = _load_module("pa_opd_prepare_extract", ROOT / "scripts/prepare_data.py")
    source = tmp_path / "source"
    archive_dir = source / "images"
    archive_dir.mkdir(parents=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "original.png").write_bytes(b"original")
    subprocess.run(
        ["tar", "-czf", str(archive_dir / "images.tar.gz00"), "-C", str(staging), "original.png"],
        check=True,
    )

    output_images = prepare_data.materialize_source_images(source, tmp_path / "output", extract=True)
    assert output_images == tmp_path / "output" / "images"
    assert (output_images / "original.png").read_bytes() == b"original"
