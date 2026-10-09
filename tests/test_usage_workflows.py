"""CPU regression tests for public-data preparation and evaluation recovery."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from PIL import Image
import pytest

from eval.run import completed_ids, load_records, run_dataset
from eval.rescore_lrs_tolerant import load_lrs_records


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "rs-opsd/scripts"
spec = importlib.util.spec_from_file_location("prepare_release", SCRIPTS / "prepare_geoevidence_release.py")
prepare_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_release)


def release_fixture(root):
    rows = []
    for index in range(2):
        row = {
            "sample_id": f"ge_{index}", "question": "Which objects are visible?",
            "choices": [{"label": "A", "text": "Trees"}, {"label": "B", "text": "Roads"}],
            "question_type": "single_choice" if index == 0 else "multiple_choice",
            "answer_labels": ["B"] if index == 0 else ["B", "A"],
            "answer_instruction": "Answer with letters separated by commas.",
            "visual_instruction": "Only focus on the objects inside the red bounding box in the image to answer this question.",
        }
        for view in prepare_release.VIEW_DIRS:
            # Distinct names and extensions ensure aliases are paired by metadata.
            extension = ".png" if view in ("global", "contextual") else ".jpg"
            relative = f"{view}/source/{view}_{index}{extension}"
            row[f"{view}_file_name"] = relative
            path = root / "train" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (index * 20, 10, 30)).save(path)
        rows.append(row)
    (root / "train/metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows


def test_release_conversion(tmp_path):
    release = tmp_path / "release"
    rows = release_fixture(release)
    output = tmp_path / "training"
    assert prepare_release.prepare(release, output) == 2
    base = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    derived = [json.loads(line) for line in (output / "derived/train.jsonl").read_text().splitlines()]
    assert [row["extra_info"]["sample_id"] for row in base] == ["ge_0", "ge_1"]
    assert [row["answer"] for row in base] == ["B", "A,B"]
    assert all(a["problem"] == b["problem"] and a["answer"] == b["answer"] for a, b in zip(base, derived))
    for raw, converted in zip(rows, base):
        image = Path(converted["images"][0])
        crop = Path(converted["teacher_images"][0])
        aliases = {"plain": image, "global": Path("bbox_images") / image.name,
                   "evidence": crop, "contextual": Path("derived") / crop}
        for view, alias in aliases.items():
            assert (output / alias).is_symlink()
            assert (output / alias).resolve() == (release / "train" / raw[f"{view}_file_name"]).resolve()
    # The strict default still rejects external links; the converter opts in explicitly.
    result = subprocess.run([sys.executable, str(SCRIPTS / "rebuild_geoevidence_train_jsonl.py"),
                             "--data-root", str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "escapes data root" in result.stderr
    with pytest.raises(FileExistsError):
        prepare_release.prepare(release, output)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "invalid_answer", "traversal"])
def test_invalid_release_does_not_create_output(tmp_path, failure):
    release = tmp_path / "release"
    rows = release_fixture(release)
    if failure == "missing":
        (release / "train" / rows[1]["contextual_file_name"]).unlink()
    elif failure == "duplicate":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif failure == "invalid_answer":
        rows[1]["answer_labels"] = ["Z"]
    else:
        rows[1]["plain_file_name"] = "plain/../../outside.jpg"
    (release / "train/metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="metadata.jsonl:2"):
        prepare_release.prepare(release, output)
    assert not output.exists()


def test_converted_release_passes_training_view_validation(tmp_path):
    # This integration check is enabled when the training loader's dependencies
    # are installed; the preparation/recovery tests also run in a light CPU env.
    for package in ("datasets", "ray", "tensordict", "transformers"):
        pytest.importorskip(package)
    release = tmp_path / "release"
    release_fixture(release)
    output = tmp_path / "training"
    prepare_release.prepare(release, output)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_geoevidence_views.py"),
         "--data-root", str(output), "--recipe", "direct-2k-three-image-kl",
         "--student-view", "bbox_images", "--teacher-full-view", "bbox_images"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "rs-opsd")},
        capture_output=True, text=True, check=True,
    )
    validation = json.loads(result.stdout)
    assert validation["validated_rows"] == 2
    assert validation["teacher_images"] == 3


def test_evaluation_retries_errors_and_counts_latest_attempt(tmp_path):
    image = tmp_path / "image/sample.png"
    image.parent.mkdir()
    Image.new("RGB", (16, 16)).save(image)
    annotation = {"image": "sample.png", "text": "What shape?", "ground_truth": "square"}
    (tmp_path / "LRS_VQA_merged.jsonl").write_text("\n".join(
        json.dumps({**annotation, "question_id": name}) for name in ("success", "failed")))
    path = tmp_path / "results.jsonl"

    class Generator:
        def generate(self, image, prompt):
            return "square"

    with patch.object(Generator, "generate", side_effect=["square", RuntimeError("temporary failure")]):
        first = run_dataset("lrs-vqa", tmp_path, Generator(), path, False, None, 1, None, 0.85)
    assert first["samples"] == 2 and first["errors"] == 1
    assert completed_ids(path) == {"success"}
    second = run_dataset("lrs-vqa", tmp_path, Generator(), path, True, None, 1, None, 0.85)
    assert second["samples"] == 2 and second["errors"] == 0 and second["accuracy"] == 1
    assert second["new_samples"] == 1 and second["resumed_samples"] == 1
    assert len(path.read_text().splitlines()) == 3  # The attempt log remains durable.
    assert len(load_records(path)) == 2
    assert len(load_lrs_records(path)) == 2
    assert completed_ids(path) == {"success", "failed"}
    with pytest.raises(FileExistsError):
        run_dataset("lrs-vqa", tmp_path, Generator(), path, False, None, 1, None, 0.85)


def test_single_image_cli(tmp_path, capsys):
    from eval import infer

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image)
    with patch.object(sys, "argv", ["infer", "--model-path", str(model), "--image", str(image),
                                    "--question", "What shape?"]), patch.object(infer, "TransformersGenerator") as factory:
        factory.return_value.generate.return_value = "square"
        infer.main()
        assert factory.call_args.kwargs["enable_thinking"] is False
        assert factory.return_value.generate.call_args.args[1] == "What shape?"
    assert capsys.readouterr().out.strip() == "square"
