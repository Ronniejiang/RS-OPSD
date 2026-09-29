#!/usr/bin/env python3
"""Read-only validation of per-sample Student/Teacher view and prompt routing."""
import argparse
import json
from pathlib import Path

from verl.utils.dataset.pa_opd_dataset import (
    _REMOVE_HINT, load_pa_opd_direct_jsonl, load_pa_opd_direct_three_image_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recipe", required=True, choices=["direct-2k-kl", "direct-2k-three-image-kl"])
    parser.add_argument("--student-view", choices=["images", "bbox_images"], required=True)
    parser.add_argument("--teacher-full-view", choices=["images", "bbox_images"], required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    three = args.recipe == "direct-2k-three-image-kl"
    loader = load_pa_opd_direct_three_image_jsonl if three else load_pa_opd_direct_jsonl
    dataset = loader(root / "train.jsonl", student_image_mode=args.student_view,
                     teacher_full_image_mode=args.teacher_full_view, separate_teacher_views=True)
    source = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines() if line.strip()]
    assert len(dataset) == len(source)
    hint_count = 0
    for row, raw in zip(dataset, source, strict=True):
        tail = Path(raw["images"][0]).relative_to("images")
        assert Path(row["images"][0]["path"]) == root / args.student_view / tail
        teachers = row["teacher_images"]
        assert len(teachers) == (3 if three else 2)
        assert Path(teachers[0]["path"]) == root / args.teacher_full_view / tail
        assert Path(teachers[-1]["path"]) == root / raw["teacher_images"][0]
        if three:
            assert Path(teachers[1]["path"]) == root / "derived" / raw["teacher_images"][0]
        hint = _REMOVE_HINT in row["prompt"][0]["content"]
        assert hint == (args.student_view == "bbox_images" and _REMOVE_HINT in raw["problem"])
        hint_count += hint
    print(json.dumps({"validated_rows": len(dataset), "student_view": args.student_view,
                      "teacher_full_view": args.teacher_full_view, "teacher_images": 3 if three else 2,
                      "student_box_hint_rows": hint_count, "first_student": dataset[0]["images"],
                      "first_teacher": dataset[0]["teacher_images"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
