#!/usr/bin/env bash
set -euo pipefail

# Read-only training preflight.  It runs in the same image and mounted volume
# as the eight-PPU training command, but requests one device and never starts
# Ray or the trainer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PA_OPD_ROOT="${PA_OPD_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face model directory}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the GeoEvidence dataset directory}"
REQUIRE_DEVICES="${REQUIRE_DEVICES:-1}"

if [[ "${PYTHON}" == */* ]]; then
  [[ -x "${PYTHON}" ]] || { echo "ERROR: Python executable not found: ${PYTHON}" >&2; exit 1; }
else
  command -v "${PYTHON}" >/dev/null 2>&1 || { echo "ERROR: Python command not found: ${PYTHON}" >&2; exit 1; }
fi
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATA_ROOT}/train.jsonl" ]] || { echo "ERROR: missing ${DATA_ROOT}/train.jsonl" >&2; exit 1; }
[[ -f "${DATA_ROOT}/final_annotations.json" ]] || { echo "ERROR: missing ${DATA_ROOT}/final_annotations.json" >&2; exit 1; }

export PYTHONPATH="${PA_OPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "PA_OPD_ROOT=${PA_OPD_ROOT}"
echo "PYTHON=$(command -v "${PYTHON}")"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_ROOT=${DATA_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/preflight_ppu_runtime.py" --require-devices "${REQUIRE_DEVICES}"
echo "PPU training preflight passed."
