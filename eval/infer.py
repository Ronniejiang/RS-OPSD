"""Run one image/question through a local RS-OPSD Hugging Face checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .model import TransformersGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True, help="Question text, including options when applicable.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=16_777_216)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking (disabled by default).")
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser()
    args.image = args.image.expanduser()
    if not (args.model_path / "config.json").is_file():
        parser.error("--model-path must be a local Hugging Face model directory containing config.json")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.question.strip():
        parser.error("--question must not be empty")
    if args.max_tokens <= 0 or args.max_pixels <= 0:
        parser.error("--max-tokens and --max-pixels must be positive")
    # Remote-sensing images can exceed Pillow's default area limit.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    generator = TransformersGenerator(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        max_pixels=args.max_pixels,
    )
    print(generator.generate(image, args.question))


if __name__ == "__main__":
    main()
