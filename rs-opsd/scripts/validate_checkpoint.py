#!/usr/bin/env python3
"""Fail fast when a RS-OPSD FSDP checkpoint cannot be resumed safely."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _nonempty(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    return None


def validate(checkpoint: Path, world_size: int) -> list[str]:
    errors: list[str] = []
    if world_size < 1:
        return ["world size must be positive"]
    if not checkpoint.is_dir():
        return [f"checkpoint directory does not exist: {checkpoint}"]

    actor = checkpoint / "actor"
    for required in (checkpoint / "data.pt", actor / "fsdp_config.json", actor / "teacher" / "fsdp_config.json"):
        status = _nonempty(required)
        if status:
            errors.append(f"{status}: {required}")
    for rank in range(world_size):
        required = (
            actor / f"model_world_size_{world_size}_rank_{rank}.pt",
            actor / f"optim_world_size_{world_size}_rank_{rank}.pt",
            actor / f"extra_state_world_size_{world_size}_rank_{rank}.pt",
            actor / "teacher" / f"model_world_size_{world_size}_rank_{rank}.pt",
        )
        for path in required:
            status = _nonempty(path)
            if status:
                errors.append(f"rank {rank}: {status}: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    errors = validate(args.checkpoint.resolve(), args.world_size)
    if errors:
        print("RS-OPSD resume checkpoint is incomplete:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"RS-OPSD resume checkpoint is complete: {args.checkpoint.resolve()} (world_size={args.world_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
