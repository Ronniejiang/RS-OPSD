import json
from pathlib import Path

import pytest

from verl.utils.dataset.pa_opd_dataset import (
    _REMOVE_HINT, load_pa_opd_direct_jsonl, load_pa_opd_direct_three_image_jsonl,
)


@pytest.fixture
def source(tmp_path):
    for folder in ("images", "bbox_images", "teacher_images", "derived/teacher_images"):
        path = tmp_path / folder / "sample.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    row = {"images": ["images/sample.png"], "teacher_images": ["teacher_images/sample.png"],
           "problem": f"<image> Which? {_REMOVE_HINT}\nA. yes\nB. no", "answer": "A"}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "derived/train.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path


@pytest.mark.parametrize("student", ["images", "bbox_images"])
@pytest.mark.parametrize("three", [False, True])
def test_student_view_does_not_change_teacher_full(source, student, three):
    loader = load_pa_opd_direct_three_image_jsonl if three else load_pa_opd_direct_jsonl
    row = loader(source / "train.jsonl", student_image_mode=student,
                 teacher_full_image_mode="bbox_images", separate_teacher_views=True)[0]
    assert row["images"] == [{"path": str(source / student / "sample.png")}]
    teachers = [item["path"] for item in row["teacher_images"]]
    expected = [str(source / "bbox_images/sample.png")]
    if three:
        expected.append(str(source / "derived/teacher_images/sample.png"))
    expected.append(str(source / "teacher_images/sample.png"))
    assert teachers == expected
    assert row["teacher_prompt"][0]["content"].count("<image>") == len(expected)
    assert (_REMOVE_HINT in row["prompt"][0]["content"]) == (student == "bbox_images")
    assert row["extra_info"]["student_image_path"] == row["images"][0]["path"]
    assert row["extra_info"]["full_image_path"] == teachers[0]
    if not three:
        assert _REMOVE_HINT in row["teacher_prompt"][0]["content"]


def test_bbox_missing_is_not_silently_replaced_with_plain(source):
    (source / "bbox_images/sample.png").unlink()
    with pytest.raises(RuntimeError, match="Failed to load"):
        load_pa_opd_direct_jsonl(source / "train.jsonl", student_image_mode="bbox_images")


def test_plain_and_bbox_students_get_identical_teacher_inputs(source):
    rows = [load_pa_opd_direct_jsonl(source / "train.jsonl", student_image_mode=mode,
            teacher_full_image_mode="bbox_images", separate_teacher_views=True)[0]
            for mode in ("images", "bbox_images")]
    assert rows[0]["teacher_images"] == rows[1]["teacher_images"]
    assert rows[0]["teacher_prompt"] == rows[1]["teacher_prompt"]


def test_global_question_does_not_invent_a_box_hint(source):
    path = source / "train.jsonl"
    path.write_text(path.read_text().replace(_REMOVE_HINT, ""))
    row = load_pa_opd_direct_jsonl(path, student_image_mode="bbox_images")[0]
    assert _REMOVE_HINT not in row["prompt"][0]["content"]
