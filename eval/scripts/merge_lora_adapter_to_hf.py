#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a standalone Hugging Face model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a standalone Hugging Face model."
    )
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Override base_model_name_or_path in adapter_config.json.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"ERROR: required file not found: {path}")


def model_weight_files(model_dir: Path) -> Iterable[Path]:
    yield from model_dir.glob("model*.safetensors")
    yield from model_dir.glob("pytorch_model*.bin")


def has_hf_weights(model_dir: Path) -> bool:
    return (model_dir / "config.json").is_file() and any(model_weight_files(model_dir))


def copy_adapter_tokenizer_artifacts(adapter_dir: Path, output_dir: Path) -> None:
    """Keep adapter-side text template/tokenizer but base image preprocessing."""
    for name in TOKENIZER_ARTIFACTS:
        source = adapter_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def validate_model(model_dir: Path) -> tuple[object, int]:
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    safetensor_files = sorted(model_dir.glob("model*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(f"No safetensors model weights found in {model_dir}")
    with safe_open(safetensor_files[0], framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    if tensor_count == 0:
        raise RuntimeError(f"Empty safetensors shard: {safetensor_files[0]}")
    return config, tensor_count


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    require_file(adapter_dir / "adapter_config.json")
    require_file(adapter_dir / "adapter_model.safetensors")

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    base_value = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_value:
        raise SystemExit("ERROR: adapter has no base_model_name_or_path; pass --base-model")
    base_model = Path(base_value).expanduser().resolve()
    if not base_model.is_dir():
        raise SystemExit(f"ERROR: base model directory not found: {base_model}")
    require_file(base_model / "config.json")

    if output_dir.exists():
        if has_hf_weights(output_dir):
            config, tensor_count = validate_model(output_dir)
            print(f"Merged LoRA model already exists: {output_dir}")
            print(f"Architecture: {config.architectures}; first-shard tensors: {tensor_count}")
            return
        raise SystemExit(
            f"ERROR: merge target already exists but is incomplete: {output_dir}. "
            "It was not modified; inspect it or choose a different output directory."
        )

    partial_dir = output_dir.with_name(f"{output_dir.name}.partial-{os.getpid()}")
    if partial_dir.exists():
        raise SystemExit(f"ERROR: partial merge directory already exists: {partial_dir}")

    print("Merging PEFT LoRA adapter")
    print(f"  adapter: {adapter_dir}")
    print(f"  base:    {base_model}")
    print(f"  target:  {output_dir}")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(
            partial_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code=True, local_files_only=True
        )
        processor.save_pretrained(partial_dir)
        copy_adapter_tokenizer_artifacts(adapter_dir, partial_dir)
        if not has_hf_weights(partial_dir):
            raise RuntimeError(f"Merge completed without HF weights in {partial_dir}")
        config, tensor_count = validate_model(partial_dir)
        partial_dir.rename(output_dir)
        print(f"Verified merged LoRA model: {output_dir}")
        print(f"Architecture: {config.architectures}; first-shard tensors: {tensor_count}")
    except Exception:
        print(f"ERROR: merge failed; partial output preserved at: {partial_dir}", file=sys.stderr)
        raise
    finally:
        for name in ("merged_model", "model"):
            if name in locals():
                del locals()[name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
