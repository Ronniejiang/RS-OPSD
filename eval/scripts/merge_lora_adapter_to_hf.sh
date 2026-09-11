#!/usr/bin/env bash
set -euo pipefail

# Convert a PEFT LoRA adapter into a standalone Hugging Face model directory
# that can be loaded by vLLM. The adapter itself is never changed.
#
# Example:
#   PYTHON=/path/to/python bash "$0" /path/to/checkpoint-639

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

ADAPTER_DIR="${1:-${ADAPTER_DIR:-}}"
OUTPUT_DIR="${2:-${MERGED_PATH:-}}"
[[ -n "${ADAPTER_DIR}" ]] || { echo "Usage: $0 /path/to/lora-adapter [merged-output-dir]" >&2; exit 2; }
ADAPTER_DIR="${ADAPTER_DIR%/}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADAPTER_DIR}/merged_huggingface}"
OUTPUT_DIR="${OUTPUT_DIR%/}"

[[ -f "${ADAPTER_DIR}/adapter_config.json" ]] || { echo "ERROR: adapter_config.json missing: ${ADAPTER_DIR}" >&2; exit 1; }
[[ -f "${ADAPTER_DIR}/adapter_model.safetensors" ]] || { echo "ERROR: adapter_model.safetensors missing: ${ADAPTER_DIR}" >&2; exit 1; }
[[ -x "${PYTHON}" ]] || { echo "ERROR: Python environment not found: ${PYTHON}" >&2; exit 1; }

"${PYTHON}" "${SCRIPT_DIR}/merge_lora_adapter_to_hf.py" \
  --adapter-dir "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}"
