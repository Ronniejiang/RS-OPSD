#!/usr/bin/env bash
set -euo pipefail

# Read-only runtime probe for a Fuyao PPU image. It intentionally does not
# load a model or access benchmark data. A successful exit proves that the
# image can initialize PPU CUDA compatibility and carries a native Qwen3-VL
# vLLM implementation.

PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || {
  echo "ERROR: Python not found: ${PYTHON}" >&2
  exit 127
}

"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import os
import sys
from packaging.version import Version


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


packages = {
    name: package_version(name)
    for name in ("torch", "vllm", "transformers", "acext", "flashinfer-python")
}
report: dict[str, object] = {
    "python": sys.executable,
    "packages": packages,
    "environment": {
        key: os.environ.get(key)
        for key in ("SAIL_PYPI_SDK_VESION", "CUDA_SDK_VER")
        if os.environ.get(key) is not None
    },
}
failures: list[str] = []

try:
    import torch

    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
    }
    if not cuda["available"] or not cuda["device_count"]:
        failures.append("torch.cuda reports no visible PPU device")
    else:
        try:
            cuda["device_0"] = torch.cuda.get_device_name(0)
            tensor = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            cuda["kernel_probe"] = float(tensor.item())
        except Exception as error:
            cuda["initialization_error"] = f"{type(error).__name__}: {error}"
            failures.append("PPU CUDA initialization failed")
    report["cuda"] = cuda
except Exception as error:
    report["cuda"] = {"initialization_error": f"{type(error).__name__}: {error}"}
    failures.append("unable to import or initialize torch")

vllm_version = packages["vllm"]
if vllm_version is None:
    failures.append("vllm is not installed")
else:
    try:
        if Version(vllm_version.split("+", maxsplit=1)[0]) < Version("0.18.0"):
            failures.append(f"vLLM {vllm_version} is older than 0.18.0")
    except Exception:
        failures.append(f"could not parse vLLM version: {vllm_version}")

ppu2_packages = [
    f"{name}=={version}"
    for name, version in packages.items()
    if version is not None and "ppu2" in version.lower()
]
report["ppu2_packages"] = ppu2_packages
if not ppu2_packages:
    failures.append("no installed runtime package identifies as PPU 2.x")

try:
    qwen3_module = importlib.import_module("vllm.model_executor.models.qwen3_vl")
    report["qwen3_vl"] = {"available": True, "module": qwen3_module.__file__}
except Exception as error:
    report["qwen3_vl"] = {"available": False, "error": f"{type(error).__name__}: {error}"}
    failures.append("vLLM has no importable native qwen3_vl implementation")

report["passed"] = not failures
report["failures"] = failures
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if failures:
    raise SystemExit(2)
PY
