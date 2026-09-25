#!/usr/bin/env python3
"""Atomically rebuild GeoEvidence ``train.jsonl`` from final_annotations.json.

This deliberately creates no images.  ``final_annotations.json`` already
references the released 2K Student full views and longest-edge-2K Teacher
crops.  Every retained row is validated before the existing manifest is
replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing final_annotations.json, images/, and teacher_images/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to DATA_ROOT/train.jsonl. The replacement is atomic.",
    )
    return parser.parse_args()


def one_existing_image(row: dict[str, Any], field: str, root: Path, row_number: int) -> None:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str) or not value[0]:
        raise ValueError(f"row {row_number}: {field} must be a one-element non-empty string list")
    path = (root / value[0]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} escapes data root: {value[0]!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"row {row_number}: missing {field}: {path}")


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    annotations_path = root / "final_annotations.json"
    output_path = (args.output or root / "train.jsonl").resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(f"missing annotations: {annotations_path}")
    try:
        output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside data-root so relative image paths remain valid") from exc

    with annotations_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected a non-empty JSON object list: {annotations_path}")

    for row_number, row in enumerate(rows, start=1):
        one_existing_image(row, "images", root, row_number)
        one_existing_image(row, "teacher_images", root, row_number)
        if not isinstance(row.get("problem"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"row {row_number}: expected string problem and answer")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    try:
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Rebuilt {output_path} from {annotations_path}: {len(rows)} validated rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
