#!/usr/bin/env python3
"""Validate the PPU CUDA-compatibility runtime before RS-OPSD training."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


# The script is also meant to be runnable directly by a user before submit;
# add the local PA-OPD source tree instead of requiring a globally installed
# ``verl`` package in the PPU SDK image.
PA_OPD_ROOT = Path(__file__).resolve().parents[1]
if str(PA_OPD_ROOT) not in sys.path:
    sys.path.insert(0, str(PA_OPD_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-devices", type=int, default=0)
    args = parser.parse_args()

    import torch

    required_modules = (
        "tensordict",
        "codetiming",
        "datasets",
        "hydra",
        "omegaconf",
        "peft",
        "torchdata",
        "wandb",
        "ray",
        "transformers",
        "vllm",
        "verl",
    )
    missing_modules: list[str] = []
    for name in required_modules:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            missing_modules.append(f"{name} ({type(error).__name__}: {error})")
            print(f"{name}=MISSING ({type(error).__name__}: {error})")
            continue
        print(f"{name}={getattr(module, '__version__', 'local-source')}")

    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"nccl_available={torch.distributed.is_nccl_available()}")
    if args.require_devices:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA-compatible PPU device is visible. Run this only inside the requested PPU job; "
                "the PPU SDK intentionally exposes devices through torch.cuda."
            )
        if torch.cuda.device_count() != args.require_devices:
            raise RuntimeError(
                f"Expected exactly {args.require_devices} visible PPU devices, got {torch.cuda.device_count()}"
            )
        for index in range(args.require_devices):
            print(f"device[{index}]={torch.cuda.get_device_name(index)}")
    if missing_modules:
        raise RuntimeError("Missing or unloadable training dependencies: " + "; ".join(missing_modules))
    from verl.utils.profiler.nvtx_profile import mark_end_range, mark_start_range
    for color in ("red", "green", "blue", "yellow", "purple", "cyan", "olive", "brown", "pink"):
        mark_end_range(mark_start_range(message="PA-OPD preflight", color=color))
    print("NVTX training colors: passed (no matplotlib required)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
