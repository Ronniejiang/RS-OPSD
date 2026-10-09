#!/usr/bin/env python3
"""Convert the extracted GeoEvidence-6K HF release to the training layout.

No image bytes are copied or re-encoded. Keep the extracted release in place:
the training layout refers to it with absolute symbolic links.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


VIEW_DIRS = {
    "plain": "images",
    "global": "bbox_images",
    "evidence": "teacher_images",
    "contextual": "derived/teacher_images",
}


def required_text(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def convert_record(row: dict, release_train: Path) -> tuple[dict, dict[str, Path]]:
    sample_id = required_text(row, "sample_id")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError(f"Unsafe sample_id: {sample_id!r}")
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list")
    labels = [required_text(choice, "label") for choice in choices]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[A-Z]", label) for label in labels):
        raise ValueError("choices must have distinct uppercase letter labels")
    answers = row.get("answer_labels")
    if (not isinstance(answers, list) or not answers
            or any(not isinstance(label, str) or label not in labels for label in answers)
            or len(set(answers)) != len(answers)):
        raise ValueError("answer_labels must contain distinct labels from choices")
    question_type = row.get("question_type")
    if question_type not in ("single_choice", "multiple_choice"):
        raise ValueError("Unsupported question_type")
    if question_type == "single_choice" and len(answers) != 1:
        raise ValueError("single_choice requires exactly one answer")
    question = required_text(row, "question")
    hint = row.get("visual_instruction", "")
    if not isinstance(hint, str):
        raise ValueError("visual_instruction must be a string")
    if hint.strip():
        question += "\n\n" + hint
    options = "\n".join(f"{label}. {required_text(choice, 'text')}" for label, choice in zip(labels, choices))
    problem = question + "\n\n" + options + "\n\n" + required_text(row, "answer_instruction")
    links: dict[str, Path] = {}
    for view, directory in VIEW_DIRS.items():
        relative = Path(required_text(row, f"{view}_file_name"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != view:
            raise ValueError(f"Invalid {view} path: {relative}")
        source = (release_train / relative).resolve()
        if not source.is_relative_to(release_train.resolve()):
            raise ValueError(f"{view} path escapes the release: {relative}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing {view} image: {source}")
        links[f"{directory}/{sample_id}{relative.suffix}"] = source
    image_path = next(name for name in links if name.startswith("images/"))
    global_path = next(name for name in links if name.startswith("bbox_images/"))
    links[image_path.replace("images/", "bbox_images/", 1)] = links.pop(global_path)
    crop_path = next(name for name in links if name.startswith("teacher_images/"))
    # The derived loader/validator expects contextual and evidence aliases to
    # share a path, even if their release filenames or extensions differ.
    contextual_path = next(name for name in links if name.startswith("derived/"))
    links[f"derived/{crop_path}"] = links.pop(contextual_path)
    return {
        "problem": problem,
        "answer": ",".join(label for label in labels if label in answers),
        "images": [image_path],
        "teacher_images": [crop_path],
        "extra_info": {"sample_id": sample_id, "question_type": question_type,
                       "source": row.get("source"), "source_image_id": row.get("source_image_id")},
    }, links


def prepare(release_root: Path, output_root: Path) -> int:
    release_root = release_root.resolve()
    output_root = output_root.resolve()
    if output_root == release_root or output_root.is_relative_to(release_root):
        raise ValueError("output-root must be outside the downloaded release")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_root}")
    records = []
    links: dict[str, Path] = {}
    seen = set()
    with (release_root / "train/metadata.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record, row_links = convert_record(json.loads(line), release_root / "train")
                sample_id = record["extra_info"]["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")
                seen.add(sample_id)
            except (ValueError, TypeError, AttributeError, FileNotFoundError) as error:
                raise ValueError(f"metadata.jsonl:{line_number}: {error}") from error
            records.append(record)
            links.update(row_links)
    if not records:
        raise ValueError("metadata.jsonl contains no samples")
    # Validate the whole release before creating any output.
    output_root.mkdir(parents=True, exist_ok=True)
    for relative, source in links.items():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
    (output_root / "final_annotations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for script in ("rebuild_geoevidence_train_jsonl.py", "rebuild_geoevidence_derived_train_jsonl.py"):
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)),
                        "--data-root", str(output_root), "--allow-external-image-links"], check=True)
    print(f"Prepared {len(records)} samples at {output_root}; keep {release_root} in place.")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True,
                        help="Downloaded dataset root; extract all four view ZIPs here first.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New or empty training directory outside the downloaded release.")
    args = parser.parse_args()
    prepare(args.release_root.expanduser(), args.output_root.expanduser())


if __name__ == "__main__":
    main()
