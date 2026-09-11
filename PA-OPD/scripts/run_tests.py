"""Run PA-OPDVR core tests without requiring pytest in the training environment."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    test_paths = [
        root / "tests" / "test_pa_opd_direct.py",
        root / "tests" / "test_pa_opd_direct_kl.py",
        root / "tests" / "test_pa_opd_topk_jsd.py",
        root / "tests" / "test_pa_opd_three_image.py",
        root / "tests" / "test_resume_checkpoint.py",
    ]
    failures: list[tuple[str, str, Exception]] = []
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
            except Exception as exc:  # pragma: no cover - reports test failures
                failures.append((path.name, name, exc))
                print(f"FAIL {path.name}::{name}: {exc}")
            else:
                print(f"PASS {path.name}::{name}")
    print(f"PA-OPDVR core tests: {total - len(failures)}/{total} passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
