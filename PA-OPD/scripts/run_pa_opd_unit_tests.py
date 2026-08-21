"""Run PA-OPD unit tests without requiring pytest in the training venv."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    test_paths = [
        root / "tests" / "test_pa_opd_runtime.py",
        root / "tests" / "test_pa_opd_legacy_resume.py",
    ]
    failures = []
    total = 0
    for path in test_paths:
        namespace = runpy.run_path(str(path))
        for name in sorted(namespace):
            candidate = namespace[name]
            if not name.startswith("test_") or not callable(candidate):
                continue
            total += 1
            try:
                candidate()
            except Exception as exc:
                failures.append((path.name, name, exc))
                print(f"FAIL {path.name}::{name}: {exc}")
            else:
                print(f"PASS {path.name}::{name}")
    print(f"PA-OPD unit tests: {total - len(failures)}/{total} passed")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
