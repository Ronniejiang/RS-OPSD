"""Tests for portable strict resume checkpoint validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.validate_checkpoint import validate


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"state")


def _complete_checkpoint(root: Path, world_size: int) -> None:
    _write(root / "data.pt")
    _write(root / "actor/fsdp_config.json")
    _write(root / "actor/teacher/fsdp_config.json")
    for rank in range(world_size):
        _write(root / f"actor/model_world_size_{world_size}_rank_{rank}.pt")
        _write(root / f"actor/optim_world_size_{world_size}_rank_{rank}.pt")
        _write(root / f"actor/extra_state_world_size_{world_size}_rank_{rank}.pt")
        _write(root / f"actor/teacher/model_world_size_{world_size}_rank_{rank}.pt")


def test_complete_checkpoint_is_accepted() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "global_step_10"
        _complete_checkpoint(checkpoint, 2)
        assert validate(checkpoint, 2) == []


def test_missing_or_empty_rank_state_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "global_step_10"
        _complete_checkpoint(checkpoint, 2)
        (checkpoint / "actor/optim_world_size_2_rank_1.pt").write_bytes(b"")
        errors = validate(checkpoint, 2)
        assert any("rank 1: empty" in error and "optim" in error for error in errors)
