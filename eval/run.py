"""Evaluate an OPD-V served model on XLRS-Bench, MME-RealWorld RS, and LRS-VQA."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

from .adapters import EvalSample, inspect_dataset, iter_dataset
from .metrics import build_prompt, score_prediction, summarize_records
from .model import OPDVOpenAIGenerator


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("lrs-vqa", "mme-realworld-rs", "xlrs-bench")


def _path(value: str) -> Path:
    return Path(value).expanduser()


def selected_datasets(values: list[str] | None) -> tuple[str, ...]:
    """Normalize repeated and comma-separated ``--dataset`` values."""
    if not values:
        return DATASETS
    names = [name.strip() for value in values for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--dataset must name at least one dataset")
    unknown = [name for name in names if name not in (*DATASETS, "all")]
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {', '.join(unknown)}")
    if "all" in names:
        if len(names) != 1:
            raise ValueError("--dataset all cannot be combined with specific datasets")
        return DATASETS
    return tuple(dict.fromkeys(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="DATASET",
        help="Repeat or comma-separate: lrs-vqa, mme-realworld-rs, xlrs-bench, all.",
    )
    parser.add_argument("--lrs-root", type=_path, default=None)
    parser.add_argument("--mme-root", type=_path, default=None)
    parser.add_argument("--xlrs-root", type=_path, default=None)
    parser.add_argument("--api-base", "--api_base", default=None, help="OpenAI-compatible server base URL.")
    parser.add_argument("--api-key", "--api_key", default="EMPTY")
    parser.add_argument("--model-id", "--model_id", default=None, help="Model ID exposed by the server.")
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=4096)
    parser.add_argument("--max-retries", "--max_retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--parallel-workers", "--parallel_workers", type=int, default=32)
    parser.add_argument(
        "--enable-thinking",
        choices=("True", "False"),
        default=None,
        help="Pass chat_template_kwargs.enable_thinking to OPD-V's OpenAI-compatible server.",
    )
    parser.add_argument("--max-pixels", type=int, default=16_777_216)
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--output-root", type=_path, default=REPO_ROOT / "eval" / "results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip sample IDs already written to JSONL output.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new samples per selected dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset layouts without starting inference.")
    args = parser.parse_args()
    try:
        args.datasets = selected_datasets(args.dataset)
    except ValueError as error:
        parser.error(str(error))
    for name in ("max_tokens", "max_retries", "parallel_workers", "max_pixels"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.dry_run:
        if not args.api_base:
            parser.error("--api-base is required unless --dry-run is used")
        if not args.model_id:
            parser.error("--model-id is required unless --dry-run is used")
    args.enable_thinking = None if args.enable_thinking is None else args.enable_thinking == "True"
    return args


def roots_from_args(args: argparse.Namespace) -> dict[str, Path | None]:
    return {"lrs-vqa": args.lrs_root, "mme-realworld-rs": args.mme_root, "xlrs-bench": args.xlrs_root}


def require_selected_roots(datasets: tuple[str, ...], roots: dict[str, Path | None]) -> None:
    missing = [dataset for dataset in datasets if roots[dataset] is None]
    if missing:
        raise ValueError("required dataset roots are missing: " + ", ".join(missing))


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return args.model_id.replace("/", "_").replace(" ", "_")


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                ids.add(str(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid result JSONL at {path}:{line_number}") from error
    return ids


def record_for_error(sample: EvalSample, error: Exception, elapsed_sec: float = 0.0) -> dict[str, Any]:
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "image": sample.image_display_path,
        "question": sample.question,
        "choices": sample.choices,
        "ground_truth": sample.ground_truth,
        "prediction": "",
        "parsed_answer": "",
        "correct": False,
        "metrics": {},
        "elapsed_sec": round(elapsed_sec, 3),
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "metadata": sample.metadata,
    }


def _generate_record(sample: EvalSample, generator: OPDVOpenAIGenerator) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        prediction = generator.generate(sample.load_image(), build_prompt(sample))
        scored = score_prediction(sample, prediction)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "image": sample.image_display_path,
            "question": sample.question,
            "choices": sample.choices,
            "ground_truth": sample.ground_truth,
            "prediction": prediction,
            "parsed_answer": scored["parsed_answer"],
            "correct": scored["correct"],
            "metrics": scored["metrics"],
            "elapsed_sec": round(elapsed, 3),
            "status": "ok",
            "metadata": sample.metadata,
        }
    except Exception as error:  # Keep a durable failure record and continue the long benchmark run.
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return record_for_error(sample, error, elapsed)


def _next_sample(iterator, completed: set[str], limit: int | None, submitted: int):
    """Return the next unfinished sample and the updated skipped count."""
    skipped = 0
    if limit is not None and submitted >= limit:
        return None, skipped
    for sample in iterator:
        if sample.sample_id in completed:
            skipped += 1
            continue
        return sample, skipped
    return None, skipped


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dataset(
    dataset: str,
    root: Path,
    generator: OPDVOpenAIGenerator,
    output_path: Path,
    resume: bool,
    limit: int | None,
    parallel_workers: int,
) -> dict[str, Any]:
    if output_path.exists() and not resume:
        raise FileExistsError(f"Result file already exists: {output_path}. Use --resume or choose --run-name.")
    completed = completed_ids(output_path) if resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter_dataset(dataset, root)
    submitted = 0
    skipped = 0
    pending: dict[Future[dict[str, Any]], EvalSample] = {}
    print(f"[run] {dataset}: writing {output_path}", flush=True)

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor, output_path.open("a", encoding="utf-8") as handle:
        def fill_pending() -> None:
            nonlocal submitted, skipped
            while len(pending) < parallel_workers:
                sample, skipped_now = _next_sample(iterator, completed, limit, submitted)
                skipped += skipped_now
                if sample is None:
                    return
                pending[executor.submit(_generate_record, sample, generator)] = sample
                submitted += 1

        fill_pending()
        while pending:
            future = next(as_completed(pending))
            sample = pending.pop(future)
            try:
                record = future.result()
            except Exception as error:  # Defensive: _generate_record normally handles every exception.
                record = record_for_error(sample, error)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            outcome = "correct" if record["correct"] else ("error" if record["status"] == "error" else "incorrect")
            print(f"[run] {dataset} {sample.sample_id}: {outcome} ({record['elapsed_sec']:.1f}s)", flush=True)
            fill_pending()

    records = load_records(output_path)
    summary = summarize_records(records)
    summary.update({"dataset": dataset, "new_samples": submitted, "resumed_samples": skipped})
    return summary


def save_summary(path: Path, summaries: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"api_base": args.api_base, "model_id": args.model_id, "enable_thinking": args.enable_thinking},
        "inference": {
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
            "parallel_workers": args.parallel_workers,
            "request_timeout": args.request_timeout,
            "max_pixels": args.max_pixels,
            "image_format": args.image_format,
            "jpeg_quality": args.jpeg_quality,
        },
        "datasets": summaries,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    roots = roots_from_args(args)
    try:
        require_selected_roots(args.datasets, roots)
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    inspections = {dataset: inspect_dataset(dataset, roots[dataset]).as_dict() for dataset in args.datasets}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "datasets": inspections}, ensure_ascii=False, indent=2))
        return

    run_dir = args.output_root / resolve_run_name(args)
    generator = OPDVOpenAIGenerator(
        api_base=args.api_base,
        api_key=args.api_key,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        enable_thinking=args.enable_thinking,
        max_pixels=args.max_pixels,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )
    summaries: dict[str, Any] = {"inspections": inspections}
    try:
        for dataset in args.datasets:
            summaries[dataset] = run_dataset(
                dataset=dataset,
                root=roots[dataset],
                generator=generator,
                output_path=run_dir / f"{dataset}.jsonl",
                resume=args.resume,
                limit=args.limit,
                parallel_workers=args.parallel_workers,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            save_summary(run_dir / "summary.json", summaries, args)
    except Exception:
        traceback.print_exc()
        run_dir.mkdir(parents=True, exist_ok=True)
        save_summary(run_dir / "summary.json", summaries, args)
        raise


if __name__ == "__main__":
    main()
