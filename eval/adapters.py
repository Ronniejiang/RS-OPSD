"""Dataset readers exposing a common representation for the three RS benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


# LRS-VQA can contain images larger than Pillow's decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class EvalSample:
    """A benchmark sample independent of its original on-disk layout."""

    dataset: str
    sample_id: str
    image_ref: Path | Image.Image
    image_display_path: str
    question: str
    ground_truth: str
    choices: list[str] | None
    metadata: dict[str, Any]

    def load_image(self) -> Image.Image:
        if isinstance(self.image_ref, Path):
            with Image.open(self.image_ref) as image:
                return image.convert("RGB").copy()
        return self.image_ref.convert("RGB")


@dataclass(frozen=True)
class DatasetInspection:
    dataset: str
    annotations: int
    eligible: int
    missing_images: int | None
    root: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "annotations": self.annotations,
            "eligible": self.eligible,
            "missing_images": self.missing_images,
            "root": self.root,
        }


def resolve_lrs_image(root: Path, image_name: str) -> Path:
    """Accept both the released hierarchy and a flattened ``image/`` directory."""
    relative = Path(image_name)
    candidates = (
        root / relative,
        root / "image" / relative.name,
        root / "image" / f"{relative.stem}.png",
        root / "image" / f"{relative.stem}.tif",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[1]


def inspect_lrs(root: Path) -> DatasetInspection:
    annotation = root / "LRS_VQA_merged.jsonl"
    if not annotation.is_file():
        raise FileNotFoundError(f"LRS-VQA annotation was not found: {annotation}")
    annotations = 0
    missing = 0
    with annotation.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            annotations += 1
            row = json.loads(line)
            if not resolve_lrs_image(root, str(row["image"])).is_file():
                missing += 1
    return DatasetInspection("lrs-vqa", annotations, annotations - missing, missing, str(root))


def iter_lrs(root: Path) -> Iterator[EvalSample]:
    annotation = root / "LRS_VQA_merged.jsonl"
    with annotation.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = resolve_lrs_image(root, str(row["image"]))
            if not image_path.is_file():
                continue
            question_id = str(row["question_id"])
            yield EvalSample(
                dataset="lrs-vqa",
                sample_id=question_id,
                image_ref=image_path,
                image_display_path=str(image_path),
                question=str(row["text"]),
                ground_truth=str(row["ground_truth"]),
                choices=None,
                metadata={
                    "category": row.get("category", "unknown"),
                    "source": question_id.split("_", maxsplit=1)[0],
                    "size_bin": row.get("size_bin", "unknown"),
                    "image_size": row.get("image_size"),
                },
            )


def _mme_remote_sensing_rows(root: Path) -> list[dict[str, Any]]:
    annotation = root / "MME_RealWorld.json"
    if not annotation.is_file():
        raise FileNotFoundError(f"MME-RealWorld annotation was not found: {annotation}")
    with annotation.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return [
        row
        for row in rows
        if row.get("Task") == "Perception" and row.get("Subtask") == "Remote Sensing"
    ]


def inspect_mme_remote_sensing(root: Path) -> DatasetInspection:
    rows = _mme_remote_sensing_rows(root)
    missing = sum(not (root / str(row["Image"])).is_file() for row in rows)
    return DatasetInspection("mme-realworld-rs", len(rows), len(rows) - missing, missing, str(root))


def iter_mme_remote_sensing(root: Path) -> Iterator[EvalSample]:
    for row in _mme_remote_sensing_rows(root):
        image_path = root / str(row["Image"])
        if not image_path.is_file():
            continue
        yield EvalSample(
            dataset="mme-realworld-rs",
            sample_id=str(row["Question_id"]),
            image_ref=image_path,
            image_display_path=str(image_path),
            question=str(row["Text"]),
            ground_truth=str(row["Ground truth"]),
            choices=[str(choice) for choice in row["Answer choices"]],
            metadata={
                "task": row.get("Task"),
                "subtask": row.get("Subtask"),
                "category": row.get("Category", "unknown"),
                "question_type": row.get("Question Type"),
                "dataset_source": row.get("Dataset"),
                "language": "en",
            },
        )


def _load_xlrs(root: Path):
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise ImportError("XLRS-Bench evaluation requires 'datasets'. Install it with: pip install datasets") from error

    if not root.is_dir():
        raise FileNotFoundError(f"XLRS-Bench dataset was not found: {root}")
    dataset_dict = load_from_disk(str(root))
    if "train" not in dataset_dict:
        raise ValueError(f"XLRS-Bench has no train split: {root}")
    return dataset_dict["train"]


def inspect_xlrs(root: Path) -> DatasetInspection:
    dataset = _load_xlrs(root)
    return DatasetInspection("xlrs-bench", len(dataset), len(dataset), None, str(root))


def iter_xlrs(root: Path) -> Iterator[EvalSample]:
    dataset = _load_xlrs(root)
    for row in dataset:
        images = row["image"]
        if not images:
            continue
        category = str(row["category"])
        yield EvalSample(
            dataset="xlrs-bench",
            sample_id=f"xlrs-{row['index']}",
            image_ref=images[0],
            image_display_path=str(row.get("path", "")),
            question=str(row["question"]),
            ground_truth=str(row["answer"]),
            choices=[str(choice) for choice in row["multi-choice options"]],
            metadata={
                "category": category,
                "l2_category": row.get("l2-category", "unknown"),
                "is_multi_choice": category == "Land use classification/Overall Land use classification",
            },
        )


def inspect_dataset(dataset: str, root: Path) -> DatasetInspection:
    if dataset == "lrs-vqa":
        return inspect_lrs(root)
    if dataset == "mme-realworld-rs":
        return inspect_mme_remote_sensing(root)
    if dataset == "xlrs-bench":
        return inspect_xlrs(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def iter_dataset(dataset: str, root: Path) -> Iterator[EvalSample]:
    if dataset == "lrs-vqa":
        return iter_lrs(root)
    if dataset == "mme-realworld-rs":
        return iter_mme_remote_sensing(root)
    if dataset == "xlrs-bench":
        return iter_xlrs(root)
    raise ValueError(f"Unsupported dataset: {dataset}")
