"""Recompute LRS-VQA tolerant results without rerunning model inference.

The input JSONL is never modified. The command writes a new annotated JSONL and
a separate summary so historical benchmark runs can be compared safely.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .lrs_semantic import (
    BGEEmbedder,
    LRSSemanticConfig,
    PairSimilarityScorer,
    annotate_lrs_semantic_records,
    write_jsonl_atomic,
)
from .metrics import summarize_records


def _path(value: str) -> Path:
    return Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=_path, required=True, help="Existing LRS-VQA result JSONL.")
    parser.add_argument("--output", type=_path, required=True, help="New annotated JSONL path.")
    parser.add_argument("--summary-output", type=_path, required=True, help="New JSON summary path.")
    parser.add_argument("--semantic-model", type=_path, required=True, help="Local BGE HF model directory.")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input to preserve the original result file")
    if not args.input.is_file():
        parser.error(f"input JSONL not found: {args.input}")
    try:
        args.semantic_config = LRSSemanticConfig(args.semantic_model, args.threshold, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    return args


def load_lrs_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict) or record.get("dataset") != "lrs-vqa":
                raise ValueError(f"Expected an lrs-vqa record at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No LRS-VQA records found in {path}")
    return records


def rescore_records(
    records: list[dict[str, Any]], scorer: PairSimilarityScorer, threshold: float
) -> dict[str, Any]:
    """Annotate records in place and return their aggregate metrics."""
    annotate_lrs_semantic_records(records, scorer, threshold)
    return summarize_records(records)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    records = load_lrs_records(args.input)
    summary = rescore_records(records, BGEEmbedder(args.semantic_config), args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output, records)
    write_summary(
        args.summary_output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input),
            "output": str(args.output),
            "lrs_semantic": args.semantic_config.as_dict(),
            "result": summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
