#!/usr/bin/env python3
"""Create the strict derived-view manifest for the three-image PA-OPD recipe.

The base ``train.jsonl`` owns the released full image and original tight crop.
The ``derived/teacher_images`` tree contains a same-relative-path expanded crop
for each base crop.  This script creates ``derived/train.jsonl`` atomically,
retaining the base row order so the three-image loader can pair records without
ever mixing evidence between samples.
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
        help="Directory containing train.jsonl and derived/teacher_images/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to DATA_ROOT/derived/train.jsonl. The replacement is atomic.",
    )
    return parser.parse_args()


def _one_relative_image(item: dict[str, Any], field: str, *, row_number: int) -> str:
    values = item.get(field)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"row {row_number}: {field} must be a one-element non-empty string list")
    candidate = Path(values[0])
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"row {row_number}: {field} must remain a safe relative path: {values[0]!r}")
    return values[0]


def _checked_file(root: Path, relative_path: str, *, field: str, row_number: int) -> None:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} escapes its data root: {relative_path!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"row {row_number}: missing {field}: {path}")


def _derived_record(base: dict[str, Any], *, row_number: int, root: Path) -> dict[str, Any]:
    full_image = _one_relative_image(base, "images", row_number=row_number)
    tight_crop = _one_relative_image(base, "teacher_images", row_number=row_number)
    _checked_file(root, full_image, field="images", row_number=row_number)
    _checked_file(root, tight_crop, field="teacher_images", row_number=row_number)
    _checked_file(root / "derived", tight_crop, field="derived teacher_images", row_number=row_number)
    if not isinstance(base.get("problem"), str) or not isinstance(base.get("answer"), str):
        raise ValueError(f"row {row_number}: expected string problem and answer")

    # Keep semantic fields byte-for-byte compatible with the base manifest. The
    # derived loader only consumes this row's teacher crop; paths below remain
    # valid if the JSONL is inspected independently from ``derived/``.
    derived = dict(base)
    derived["images"] = [f"../{full_image}"]
    derived["teacher_images"] = [tight_crop]
    if "original_images" in derived:
        original = _one_relative_image(base, "original_images", row_number=row_number)
        derived["original_images"] = [f"../{original}"]
    return derived


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    source = root / "train.jsonl"
    output = (args.output or root / "derived" / "train.jsonl").resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing base manifest: {source}")
    try:
        output.relative_to(root / "derived")
    except ValueError as exc:
        raise ValueError("output must be inside DATA_ROOT/derived") from exc

    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                base = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source}:{row_number}") from exc
            if not isinstance(base, dict):
                raise ValueError(f"expected an object at {source}:{row_number}")
            records.append(_derived_record(base, row_number=row_number, root=root))
    if not records:
        raise ValueError(f"base manifest is empty: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Rebuilt {output} from {source}: {len(records)} strict derived-view rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
