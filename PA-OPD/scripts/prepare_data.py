"""Prepare Vision-OPD-6K data for PA-OPD training.

Usage:
    # Download and process into one directory.
    python scripts/prepare_data.py --data-dir ./data --hf-repo <dataset_repo>

    # Process a local dataset without modifying it.  The provided dataset keeps
    # original images in split tar archives, so extraction is explicit.
    python scripts/prepare_data.py \
        --source-data-dir /path/to/Vision-OPD-6K \
        --data-dir ./cache/pa-opd \
        --extract-source-images

With --source-data-dir, the source is read-only. train.parquet and any
materialized original images are written under --data-dir.  Crop paths continue
to point at the source directory when those crops are already unpacked there.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import snapshot_download


REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)

FORMAT_INSTRUCTION = (
    "\n\nContinue the already-open <think> block with reasoning, then emit "
    "</think><answer>the option label only</answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PA-OPD training data.")
    parser.add_argument("--data-dir", default="./data", help="Output directory for processed data")
    parser.add_argument(
        "--source-data-dir",
        help="Read an existing Vision-OPD-6K directory without modifying it",
    )
    parser.add_argument(
        "--extract-source-images",
        action="store_true",
        help="Extract split original-image archives into --data-dir/images",
    )
    parser.add_argument("--hf-repo", default=os.environ.get("OPDV_DATASET_REPO"), help="Hugging Face dataset repo")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only preprocess")
    args = parser.parse_args()
    if not args.source_data_dir and not args.skip_download and not args.hf_repo:
        parser.error("--hf-repo or OPDV_DATASET_REPO is required unless --skip-download is set")
    if args.extract_source_images and not args.source_data_dir:
        parser.error("--extract-source-images is only valid with --source-data-dir")
    return args


def _extract_split_archives(archive_parts: list[Path], output_dir: Path) -> None:
    """Concatenate split gzip parts and extract without invoking a shell."""
    with subprocess.Popen(["cat", *(str(part) for part in archive_parts)], stdout=subprocess.PIPE) as cat_process:
        assert cat_process.stdout is not None
        try:
            subprocess.run(
                ["tar", "--no-same-owner", "-xzf", "-", "-C", str(output_dir)],
                stdin=cat_process.stdout,
                check=True,
            )
        finally:
            cat_process.stdout.close()
            cat_returncode = cat_process.wait()
    if cat_returncode != 0:
        raise subprocess.CalledProcessError(cat_returncode, ["cat", *(str(part) for part in archive_parts)])


def download_dataset(repo_id: str, data_dir: str) -> None:
    print(f"Downloading dataset from {repo_id} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=data_dir)

    images_dir = Path(data_dir) / "images"
    teacher_dir = Path(data_dir) / "teacher_images"

    tar_files = sorted(images_dir.glob("images.tar.gz*"))
    if tar_files:
        print("Extracting student images ...")
        _extract_split_archives(tar_files, images_dir)
        for archive in tar_files:
            archive.unlink()

    teacher_tar = teacher_dir / "teacher_images.tar.gz"
    if teacher_tar.exists():
        print("Extracting teacher images ...")
        subprocess.run(["tar", "--no-same-owner", "-xzf", str(teacher_tar), "-C", str(teacher_dir)], check=True)
        teacher_tar.unlink()

    print("Image extraction complete.")


def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def _contains_png(directory: Path) -> bool:
    return next(directory.glob("*.png"), None) is not None


def materialize_source_images(source_data_dir: Path, output_data_dir: Path, extract: bool) -> Path:
    """Return a directory of directly readable original images.

    Some local snapshots retain only images.tar.gz00...05.  Extracting those
    archives is deliberately opt-in because it can consume substantial disk.
    """
    source_images = source_data_dir / "images"
    if _contains_png(source_images):
        return source_images

    output_images = output_data_dir / "images"
    if _contains_png(output_images):
        return output_images

    archive_parts = sorted(source_images.glob("images.tar.gz*"))
    if not archive_parts:
        raise FileNotFoundError(
            f"No readable original images or split archives found in {source_images}"
        )
    if not extract:
        raise RuntimeError(
            "Original images are stored as split archives. Re-run with "
            "--extract-source-images to materialize them under --data-dir/images."
        )

    output_images.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(archive_parts)} read-only image archive parts to {output_images} ...")
    _extract_split_archives(archive_parts, output_images)

    if not _contains_png(output_images):
        raise RuntimeError(f"Archive extraction produced no PNG files in {output_images}")
    return output_images


def build_record(item: dict[str, Any], image_dir: Path, source_data_dir: Path) -> dict[str, Any]:
    image_rel = item["images"][0]
    teacher_rel = item["teacher_images"][0]
    image_path = image_dir / Path(image_rel).name
    teacher_path = source_data_dir / teacher_rel
    if not image_path.is_file():
        raise FileNotFoundError(f"Original image referenced by record is missing: {image_path}")
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Evidence crop referenced by record is missing: {teacher_path}")

    question = clean_question(item.get("problem", ""))
    answer = str(item.get("answer", "")).strip().upper()

    return {
        "data_source": "pa_opd_mcq",
        "prompt": [{"role": "user", "content": f"<image>\n{question}{FORMAT_INSTRUCTION}"}],
        "images": [{"path": str(image_path)}],
        "bbox_images": [{"path": str(teacher_path)}],
        "ability": "visual_question_answering",
        "reward_model": {
            "style": "none",
            "ground_truth": answer,
        },
        "extra_info": {
            "answer": answer,
            "question": question,
            "source_extra_info": item.get("extra_info", {}),
        },
    }


def convert_to_parquet(source_data_dir: Path, output_data_dir: Path, image_dir: Path) -> None:
    jsonl_path = source_data_dir / "train.jsonl"
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Converting train.jsonl to train.parquet ...")
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append(build_record(item, image_dir, source_data_dir))

    dataset = datasets.Dataset.from_list(records)
    output_path = output_data_dir / "train.parquet"
    dataset.to_parquet(str(output_path))
    print(f"Saved {len(records)} records to {output_path}")


def main() -> None:
    args = parse_args()
    output_data_dir = Path(args.data_dir).resolve()
    source_data_dir = Path(args.source_data_dir or output_data_dir).resolve()
    if args.source_data_dir and source_data_dir == output_data_dir:
        raise ValueError("--source-data-dir and --data-dir must differ to keep the source read-only")
    output_data_dir.mkdir(parents=True, exist_ok=True)

    if not args.source_data_dir and not args.skip_download:
        download_dataset(args.hf_repo, str(output_data_dir))

    if args.source_data_dir:
        image_dir = materialize_source_images(
            source_data_dir, output_data_dir, extract=args.extract_source_images
        )
    else:
        image_dir = source_data_dir / "images"
    convert_to_parquet(source_data_dir, output_data_dir, image_dir)
    print(f"\nData preparation complete. Training data at: {output_data_dir / 'train.parquet'}")


if __name__ == "__main__":
    main()
