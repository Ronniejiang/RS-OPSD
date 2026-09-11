"""Direct Vision-OPD-6K JSONL loader used by the PA-OPD recipe.

The source dataset stays immutable: records keep absolute paths to its original
image and evidence-crop files, and no copied image cache or Parquet manifest is
created in the code workspace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import datasets

from verl.utils.reward_score.pa_opd_direct_protocol import canonicalize_option_set, normalize_option_labels


_REMOVE_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."
_FORMAT_INSTRUCTION = (
    "\n\nContinue the already-open <think> block with reasoning, then emit "
    "</think><answer>the option label only</answer>."
)

_DIRECT_OPTION_INSTRUCTION = (
    "\n\nOutput only the selected option letter or comma-separated option letters. "
    "For multiple selections, use commas and no explanation."
)
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-Za-z])\.\s+")


# Keep the privileged three-view contract aligned with
# eval/rs_opd_original_teacher_crops_three_images.py. These instructions are
# Teacher-only: the Student continues to receive only the released full image.
_TRIPLE_LOCAL_HINT = (
    "You are given three views of the same aerial scene. Image 1 is the complete scene for global context. "
    "Image 2 is an expanded high-resolution crop centred on the target; its red bounding box marks the exact target region. "
    "Image 3 is a tight high-resolution crop of that target without a red bounding box. "
    "Use image 3 for the finest target detail, image 2 to identify the exact boxed target and nearby context, "
    "and image 1 only when global context is needed. Return only the letter of the correct option."
)
_TRIPLE_GLOBAL_HINT = (
    "You are given three complementary views of the same complete aerial scene. Image 1 is the released full scene, "
    "image 2 is a supplemental teacher view, and image 3 is a high-resolution teacher view. "
    "Use the views together to answer the question. Return only the letter of the correct option."
)
_THREE_IMAGE_ORDER = ("full_image", "derived_teacher_image", "teacher_image")


def _clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{_REMOVE_HINT}", "")
    return text.replace(_REMOVE_HINT, "").strip()

def _extract_option_labels(question: str) -> tuple[str, ...]:
    """Extract ordered multiple-choice labels from answer-option lines only."""

    return normalize_option_labels(match.group(1) for match in _OPTION_LINE_RE.finditer(question))


def _build_direct_record(
    item: dict[str, Any], source_dir: Path, source_line: int
) -> dict[str, Any]:
    """Create the no-thinking RS_OPD record without structural tags."""

    image_path = source_dir / item["images"][0]
    crop_path = source_dir / item["teacher_images"][0]
    if not image_path.is_file():
        raise FileNotFoundError(f"Original image referenced by record is missing: {image_path}")
    if not crop_path.is_file():
        raise FileNotFoundError(f"Evidence crop referenced by record is missing: {crop_path}")

    question = _clean_question(item.get("problem", ""))
    option_labels = _extract_option_labels(question)
    answer = canonicalize_option_set(item.get("answer", ""), option_labels)
    student_images = [{"path": str(image_path)}]
    image_prefix = "<image>\n"
    return {
        "data_source": "pa_opd_direct_rs_opd",
        "pa_opd_source_line": int(source_line),
        "prompt": [{"role": "user", "content": f"{image_prefix}{question}{_DIRECT_OPTION_INSTRUCTION}"}],
        "images": student_images,
        "bbox_images": [{"path": str(crop_path)}],
        "ability": "visual_question_answering",
        "reward_model": {"style": "none", "ground_truth": answer},
        "extra_info": {
            "answer": answer,
            "question": question,
            "option_labels": list(option_labels),
            "source_extra_info": item.get("extra_info", {}),
        },
    }


def load_pa_opd_direct_jsonl(jsonl_path: str | Path) -> datasets.Dataset:
    """Load RS_OPD as direct option-set PA-OPDVR records in memory."""

    path = Path(jsonl_path).resolve()
    if path.name != "train.jsonl":
        raise ValueError(f"Direct PA-OPDVR expects an RS_OPD train.jsonl, got: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(_build_direct_record(json.loads(line), path.parent, source_line=line_number))
            except Exception as exc:
                raise RuntimeError(f"Failed to load direct PA-OPDVR record at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"Direct PA-OPDVR source JSONL is empty: {path}")
    return datasets.Dataset.from_list(records)



def _read_strict_jsonl_records(path: Path, *, description: str) -> list[tuple[int, dict[str, Any]]]:
    """Read non-empty JSON objects while retaining source-line diagnostics."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} annotations were not found: {path}")
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {description} at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object in {description} at {path}:{line_number}")
            records.append((line_number, item))
    if not records:
        raise ValueError(f"{description} annotations are empty: {path}")
    return records


def _strict_single_image_path(
    item: dict[str, Any], source_dir: Path, *, field: str, description: str, line_number: int
) -> Path:
    values = item.get(field)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise ValueError(
            f"{description} at line {line_number} requires exactly one non-empty '{field}' path"
        )
    image_path = (source_dir / values[0]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"{description} at line {line_number} references a missing image: {image_path}")
    return image_path


def _build_three_image_teacher_prompt(problem: str) -> str:
    """Render the evaluator's ordered three-view prompt for a Teacher template."""

    hint = _TRIPLE_LOCAL_HINT if _REMOVE_HINT in problem else _TRIPLE_GLOBAL_HINT
    return f"{hint}\n\n{_clean_question(problem)}{_DIRECT_OPTION_INSTRUCTION}"


def load_pa_opd_direct_three_image_jsonl(jsonl_path: str | Path) -> datasets.Dataset:
    """Load direct RS_OPD records with full/derived/tight Teacher privilege.

    The base dataset owns the released full image and original tight crop. Its
    ``derived`` sibling owns the matching expanded red-box crop. Pairing is
    positional and strict so a Teacher can never receive another sample's
    evidence.
    """

    path = Path(jsonl_path).resolve()
    if path.name != "train.jsonl":
        raise ValueError(f"Three-image direct PA-OPDVR expects an RS_OPD train.jsonl, got: {path}")
    derived_path = path.parent / "derived" / "train.jsonl"
    base_records = _read_strict_jsonl_records(path, description="base RS_OPD")
    derived_records = _read_strict_jsonl_records(derived_path, description="derived RS_OPD")
    if len(base_records) != len(derived_records):
        raise ValueError(
            "Base and derived RS_OPD record counts differ: "
            f"{len(base_records)} != {len(derived_records)}"
        )

    records: list[dict[str, Any]] = []
    for sample_index, ((base_line, base), (derived_line, derived)) in enumerate(
        zip(base_records, derived_records, strict=True), start=1
    ):
        if base.get("problem") != derived.get("problem") or base.get("answer") != derived.get("answer"):
            raise ValueError(
                "Base and derived RS_OPD annotations do not describe the same sample at "
                f"position {sample_index} (base line {base_line}, derived line {derived_line})"
            )

        try:
            record = _build_direct_record(base, path.parent, source_line=base_line)
            derived_crop_path = _strict_single_image_path(
                derived,
                derived_path.parent,
                field="teacher_images",
                description="derived RS_OPD",
                line_number=derived_line,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load three-image direct PA-OPDVR record at {path}:{base_line}"
            ) from exc

        full_image = record["images"][0]
        tight_crop = record["bbox_images"][0]
        record["teacher_images"] = [full_image, {"path": str(derived_crop_path)}, tight_crop]
        record["teacher_prompt"] = [
            {
                "role": "user",
                "content": "<image>\n<image>\n<image>\n" + _build_three_image_teacher_prompt(base["problem"]),
            }
        ]
        record["data_source"] = "pa_opd_direct_rs_opd_three_image"
        record["extra_info"].update(
            {
                "teacher_image_order": list(_THREE_IMAGE_ORDER),
                "derived_teacher_image": str(derived_crop_path),
            }
        )
        records.append(record)
    return datasets.Dataset.from_list(records)
