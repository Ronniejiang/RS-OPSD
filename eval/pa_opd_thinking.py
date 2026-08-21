"""Evaluate an OPD-V model with PA-OPD's prefixed ``<think>`` protocol.

This is intentionally separate from :mod:`eval.run`: normal evaluation keeps
its existing prompt and answer-only JSONL output, while this entry point uses
the PA-OPD rollout prompt and writes the complete reasoning completion.

Example:
  python -m eval.pa_opd_thinking \
    --dataset mme-realworld-rs,xlrs-bench \
    --mme-root /path/to/MME-RealWorld --xlrs-root /path/to/XLRS-Bench \
    --api-base http://127.0.0.1:8000/v1 --model-id pa-opd-qwen3-vl-8b-step195
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import threading
import time
import traceback
from typing import Any

from PIL import Image

from .adapters import EvalSample, inspect_dataset, iter_dataset
from .metrics import score_prediction, summarize_records


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("lrs-vqa", "mme-realworld-rs", "xlrs-bench")

# Kept byte-for-byte compatible with PA-OPD's dataset preparation prompt.
PA_OPD_FORMAT_INSTRUCTION = (
    "\n\nContinue the already-open <think> block with reasoning, then emit "
    "</think><answer>the option label only</answer>."
)
OPEN_THINK = "<think>"
CLOSE_THINK = "</think>"
OPEN_ANSWER = "<answer>"
CLOSE_ANSWER = "</answer>"
PA_OPD_TAGS = (OPEN_THINK, CLOSE_THINK, OPEN_ANSWER, CLOSE_ANSWER)


@dataclass(frozen=True)
class GenerationTrace:
    """The unmodified text returned by the OpenAI-compatible API and metadata."""

    content: str
    reasoning_content: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None


@dataclass(frozen=True)
class PAOPDTrace:
    """One completion represented in exactly the same prefix convention as rollout JSONL."""

    input: str
    output: str
    canonical_response: str
    format_valid: bool
    answer: str | None
    reasoning: str | None


class PAOPDOpenAIGenerator:
    """OpenAI-compatible visual generation that preserves the full response text."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_id: str,
        max_tokens: int,
        max_retries: int,
        request_timeout: float,
        max_pixels: int,
        image_format: str,
        jpeg_quality: int,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.max_pixels = max_pixels
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self._thread_local = threading.local()

    def _client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "PA-OPD thinking inference requires the 'openai' package. "
                    "Install it in the prebuilt evaluation environment."
                ) from error
            client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.request_timeout)
            self._thread_local.client = client
        return client

    def image_to_data_uri(self, image: Image.Image) -> str:
        """Convert an image to a bounded data URI without retaining it in JSONL."""
        image = image.convert("RGB")
        width, height = image.size
        pixels = width * height
        if pixels > self.max_pixels:
            scale = (self.max_pixels / pixels) ** 0.5
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )

        encoded = BytesIO()
        if self.image_format == "jpeg":
            image.save(encoded, format="JPEG", quality=self.jpeg_quality, optimize=True)
            mime_type = "image/jpeg"
        else:
            image.save(encoded, format="PNG", optimize=True)
            mime_type = "image/png"
        payload = base64.b64encode(encoded.getvalue()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump(mode="json")
        if isinstance(usage, dict):
            return usage
        return {"value": str(usage)}

    @staticmethod
    def _message_reasoning_content(message: Any) -> str | None:
        """Read provider-specific reasoning fields without depending on one vLLM version."""
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = getattr(message, key, None)
            if isinstance(value, str) and value:
                return value
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("reasoning_content", "reasoning", "thinking"):
                value = extra.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    def generate(self, image: Image.Image, prompt: str) -> GenerationTrace:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_to_data_uri(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        client = self._client()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # PA-OPD's merged template uses this generation turn to prefill
                # ``<think>``.  The explicit flag also handles compatible models
                # whose template still consults enable_thinking.
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                choice = response.choices[0]
                content = choice.message.content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The model response did not contain text content.")
                return GenerationTrace(
                    content=content,
                    reasoning_content=self._message_reasoning_content(choice.message),
                    finish_reason=getattr(choice, "finish_reason", None),
                    usage=self._usage_to_dict(getattr(response, "usage", None)),
                )
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(1.0)
        raise RuntimeError(f"API request failed after {self.max_retries} attempts: {last_error}")


def build_pa_opd_prompt(sample: EvalSample) -> str:
    """Build the PA-OPD user text, excluding the image placeholder added by vLLM."""
    question = sample.question.strip()
    if not sample.choices:
        # LRS-VQA is free-form; retain the same think/answer envelope while
        # asking for its textual reference answer instead of an option label.
        return (
            f"{question}\n\nAnswer the question concisely."
            "\n\nContinue the already-open <think> block with reasoning, then emit "
            "</think><answer>the concise answer only</answer>."
        )

    choices = "\n".join(sample.choices)
    return f"{question}\n\n{choices}\n\nAnswer with the option's letter from the given choices.{PA_OPD_FORMAT_INSTRUCTION}"


def build_rollout_input(prompt: str) -> str:
    """Serialize a user turn exactly like PA-OPD's persisted rollout input."""
    return f"user\n\n{prompt}\nassistant\n<think>\n"


def completion_after_prefilled_think(api_content: str) -> str:
    """Remove only a duplicated opening tag if a server returns the full turn.

    PA-OPD rollouts store the generation *after* the template-provided
    ``<think>``.  vLLM normally returns that suffix directly, but this makes
    traces portable across servers that include the opening tag in ``content``.
    The original server text is always retained separately in ``api_content``.
    """
    prefix_trimmed = api_content.lstrip()
    if not prefix_trimmed.startswith(OPEN_THINK):
        return api_content
    suffix = prefix_trimmed[len(OPEN_THINK):]
    if suffix.startswith("\r\n"):
        return suffix[2:]
    if suffix.startswith("\n"):
        return suffix[1:]
    return suffix


def is_valid_pa_opd_format(completion: str) -> bool:
    """Match PA-OPD's format reward on a suffix after a prefilled ``<think>``."""
    assistant_output = OPEN_THINK + completion
    if any(assistant_output.count(tag) != 1 for tag in PA_OPD_TAGS):
        return False
    positions = [assistant_output.index(tag) for tag in PA_OPD_TAGS]
    return positions == sorted(positions)


def build_pa_opd_trace(prompt: str, api_content: str) -> PAOPDTrace:
    """Convert an API response into PA-OPD rollout fields and parsed reasoning."""
    output = completion_after_prefilled_think(api_content)
    canonical = OPEN_THINK + output
    format_valid = is_valid_pa_opd_format(output)
    answer: str | None = None
    reasoning: str | None = None
    if format_valid:
        think_end = canonical.index(CLOSE_THINK)
        answer_start = canonical.index(OPEN_ANSWER, think_end) + len(OPEN_ANSWER)
        answer_end = canonical.index(CLOSE_ANSWER, answer_start)
        reasoning = canonical[len(OPEN_THINK):think_end]
        answer = canonical[answer_start:answer_end].strip()
    return PAOPDTrace(
        input=build_rollout_input(prompt),
        output=output,
        canonical_response=canonical,
        format_valid=format_valid,
        answer=answer,
        reasoning=reasoning,
    )


def answer_candidate(trace: PAOPDTrace) -> str:
    """Return a best-effort answer for benchmark scoring, even if tags are malformed."""
    if trace.answer is not None:
        return trace.answer
    if OPEN_ANSWER in trace.canonical_response and CLOSE_ANSWER in trace.canonical_response:
        return trace.canonical_response.rsplit(OPEN_ANSWER, 1)[1].split(CLOSE_ANSWER, 1)[0].strip()
    after_think = trace.canonical_response.rsplit(CLOSE_THINK, 1)
    return after_think[-1].strip()


def _path(value: str) -> Path:
    return Path(value).expanduser()


def selected_datasets(values: list[str] | None) -> tuple[str, ...]:
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
    parser.add_argument("--max-pixels", type=int, default=16_777_216)
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--output-root", type=_path, default=REPO_ROOT / "eval" / "thinking_results")
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
    return args.model_id.replace("/", "_").replace(" ", "_") + "-pa-opd-thinking"


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
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
        "gts": sample.ground_truth,
        "input": None,
        "output": None,
        "prediction": "",
        "parsed_answer": "",
        "correct": False,
        "metrics": {},
        "score": 0.0,
        "pa_opd_format_valid": 0.0,
        "pa_opd_accuracy": 0.0,
        "pa_opd_parsed_answer": None,
        "pa_opd_rollout_n": 1,
        "elapsed_sec": round(elapsed_sec, 3),
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "metadata": sample.metadata,
    }


def generate_record(sample: EvalSample, generator: PAOPDOpenAIGenerator) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        prompt = build_pa_opd_prompt(sample)
        api_response = generator.generate(sample.load_image(), prompt)
        trace = build_pa_opd_trace(prompt, api_response.content)
        prediction = answer_candidate(trace)
        scored = score_prediction(sample, prediction)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        pa_accuracy = float(trace.format_valid and scored["correct"])
        return {
            # The following PA-OPD fields are compatible with rollout JSONL.
            "input": trace.input,
            "output": trace.output,
            "gts": sample.ground_truth,
            "score": float(trace.format_valid),
            "pa_opd_format_valid": float(trace.format_valid),
            "pa_opd_accuracy": pa_accuracy,
            "pa_opd_parsed_answer": trace.answer,
            "pa_opd_rollout_n": 1,
            # Benchmark fields and complete inference trace.
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "image": sample.image_display_path,
            "question": sample.question,
            "choices": sample.choices,
            "ground_truth": sample.ground_truth,
            "prompt": prompt,
            "prediction": prediction,
            "parsed_answer": scored["parsed_answer"],
            "correct": scored["correct"],
            "metrics": scored["metrics"],
            "pa_opd_canonical_response": trace.canonical_response,
            "pa_opd_reasoning": trace.reasoning,
            "api_content": api_response.content,
            "api_reasoning_content": api_response.reasoning_content,
            "api_finish_reason": api_response.finish_reason,
            "api_usage": api_response.usage,
            "elapsed_sec": round(elapsed, 3),
            "status": "ok",
            "metadata": sample.metadata,
        }
    except Exception as error:  # Persist failures and let a multi-hour run continue.
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return record_for_error(sample, error, elapsed)


def _next_sample(iterator, completed: set[str], limit: int | None, submitted: int):
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


def thinking_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_records(records)
    successful = [record for record in records if record.get("status") == "ok"]
    summary.update(
        {
            "pa_opd_format_valid": sum(float(record.get("pa_opd_format_valid", 0.0)) for record in records)
            / len(records)
            if records
            else 0.0,
            "pa_opd_accuracy": sum(float(record.get("pa_opd_accuracy", 0.0)) for record in records) / len(records)
            if records
            else 0.0,
            "truncated": sum(record.get("api_finish_reason") == "length" for record in successful),
        }
    )
    return summary


def run_dataset(
    dataset: str,
    root: Path,
    generator: PAOPDOpenAIGenerator,
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
    print(f"[thinking] {dataset}: writing {output_path}", flush=True)

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor, output_path.open("a", encoding="utf-8") as handle:
        def fill_pending() -> None:
            nonlocal submitted, skipped
            while len(pending) < parallel_workers:
                sample, skipped_now = _next_sample(iterator, completed, limit, submitted)
                skipped += skipped_now
                if sample is None:
                    return
                pending[executor.submit(generate_record, sample, generator)] = sample
                submitted += 1

        fill_pending()
        while pending:
            future = next(as_completed(pending))
            sample = pending.pop(future)
            try:
                record = future.result()
            except Exception as error:  # Defensive: generate_record persists normal failures itself.
                record = record_for_error(sample, error)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            outcome = "correct" if record["correct"] else ("error" if record["status"] == "error" else "incorrect")
            print(
                f"[thinking] {dataset} {sample.sample_id}: {outcome}; "
                f"format={record['pa_opd_format_valid']:.0f} ({record['elapsed_sec']:.1f}s)",
                flush=True,
            )
            fill_pending()

    records = load_records(output_path)
    summary = thinking_summary(records)
    summary.update({"dataset": dataset, "new_samples": submitted, "resumed_samples": skipped})
    return summary


def save_summary(path: Path, summaries: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "pa-opd-thinking",
        "rollout_protocol": {
            "format_instruction": PA_OPD_FORMAT_INSTRUCTION.strip(),
            "template_prefill": "<think>\\n",
            "result_jsonl": "Each row includes training-compatible input/output/gts and full API content.",
        },
        "model": {"api_base": args.api_base, "model_id": args.model_id, "enable_thinking": True},
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
        print(json.dumps({"dry_run": True, "mode": "pa-opd-thinking", "datasets": inspections}, ensure_ascii=False, indent=2))
        return

    run_dir = args.output_root / resolve_run_name(args)
    generator = PAOPDOpenAIGenerator(
        api_base=args.api_base,
        api_key=args.api_key,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
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
