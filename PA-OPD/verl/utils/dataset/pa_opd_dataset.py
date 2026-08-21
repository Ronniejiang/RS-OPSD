"""Direct Vision-OPD-6K JSONL loader used by the PA-OPD recipe.

The source dataset stays immutable: records keep absolute paths to its original
image and evidence-crop files, and no copied image cache or Parquet manifest is
created in the code workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets


_REMOVE_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."
_FORMAT_INSTRUCTION = (
    "\n\nContinue the already-open <think> block with reasoning, then emit "
    "</think><answer>the option label only</answer>."
)


def _clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{_REMOVE_HINT}", "")
    return text.replace(_REMOVE_HINT, "").strip()


def _build_record(item: dict[str, Any], source_dir: Path, source_line: int) -> dict[str, Any]:
    image_path = source_dir / item["images"][0]
    crop_path = source_dir / item["teacher_images"][0]
    if not image_path.is_file():
        raise FileNotFoundError(f"Original image referenced by record is missing: {image_path}")
    if not crop_path.is_file():
        raise FileNotFoundError(f"Evidence crop referenced by record is missing: {crop_path}")

    question = _clean_question(item.get("problem", ""))
    answer = str(item.get("answer", "")).strip().upper()
    return {
        "data_source": "pa_opd_mcq",
        "pa_opd_source_line": int(source_line),
        "prompt": [{"role": "user", "content": f"<image>\n{question}{_FORMAT_INSTRUCTION}"}],
        "images": [{"path": str(image_path)}],
        "bbox_images": [{"path": str(crop_path)}],
        "ability": "visual_question_answering",
        "reward_model": {"style": "none", "ground_truth": answer},
        "extra_info": {
            "answer": answer,
            "question": question,
            "source_extra_info": item.get("extra_info", {}),
        },
    }


def load_pa_opd_jsonl(jsonl_path: str | Path) -> datasets.Dataset:
    """Create an in-memory training dataset from Vision-OPD-6K ``train.jsonl``."""
    path = Path(jsonl_path).resolve()
    if path.name != "train.jsonl":
        raise ValueError(f"PA-OPD expects Vision-OPD-6K train.jsonl, got: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(_build_record(json.loads(line), path.parent, source_line=line_number))
            except Exception as exc:
                raise RuntimeError(f"Failed to load PA-OPD record at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"PA-OPD source JSONL is empty: {path}")
    return datasets.Dataset.from_list(records)
