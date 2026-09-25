"""Zero-dependency test runner for PA-OPD's preflight smoke tests."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest as _pytest


if hasattr(_pytest, "main"):
    raise SystemExit(_pytest.main())


def _run_module(path: Path) -> tuple[int, int]:
    spec = importlib.util.spec_from_file_location(f"_pa_opd_smoke_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed = 0
    failed = 0
    for name, fn in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if inspect.signature(fn).parameters:
            print(f"SKIPPED {path.name}::{name} (pytest fixture required)")
            continue
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAILED {path.name}::{name}: {exc}", file=sys.stderr)
        else:
            passed += 1
            print(f"PASSED {path.name}::{name}")
    return passed, failed


def main() -> int:
    test_paths = [Path(arg).resolve() for arg in sys.argv[1:] if arg.endswith(".py")]
    if not test_paths:
        print("PA-OPD fallback pytest runner needs explicit test-file paths", file=sys.stderr)
        return 2
    passed = failed = 0
    for path in test_paths:
        one_passed, one_failed = _run_module(path)
        passed += one_passed
        failed += one_failed
    print(f"{passed} passed, {failed} failed")
    return int(failed > 0)


raise SystemExit(main())
