#!/usr/bin/env bash
set -euo pipefail

# Convert a verl/FSDP actor checkpoint into a vLLM-loadable Hugging Face model.
# The original shards are read-only; the merged model is written to a separate
# directory and atomically renamed only after the expected model file exists.
#
# Example:
#   VERL_ROOT=/path/to/verl PYTHON=/path/to/python bash eval/scripts/merge_fsdp_checkpoint_to_hf.sh \
#     /path/to/checkpoints/global_step_195

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_ROOT="${VERL_ROOT:-}"
PYTHON="${PYTHON:-python3}"

CHECKPOINT_DIR="${1:-${CHECKPOINT_DIR:-}}"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Usage: $0 /path/to/global_step_N [merged-output-dir]" >&2
  exit 2
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR%/}"
ACTOR_DIR="${CHECKPOINT_DIR}/actor"
MERGED_PATH="${2:-${MERGED_PATH:-${CHECKPOINT_DIR}/merged_huggingface}}"
MERGED_PATH="${MERGED_PATH%/}"

[[ -d "${ACTOR_DIR}" ]] || { echo "ERROR: actor checkpoint not found: ${ACTOR_DIR}" >&2; exit 1; }
[[ -f "${ACTOR_DIR}/fsdp_config.json" ]] || { echo "ERROR: FSDP configuration missing: ${ACTOR_DIR}/fsdp_config.json" >&2; exit 1; }
[[ -d "${ACTOR_DIR}/huggingface" ]] || { echo "ERROR: Hugging Face config directory missing: ${ACTOR_DIR}/huggingface" >&2; exit 1; }
[[ -x "${PYTHON}" ]] || { echo "ERROR: Python environment not found: ${PYTHON}" >&2; exit 1; }
[[ -d "${VERL_ROOT}/verl" ]] || { echo "ERROR: set VERL_ROOT to a source tree containing verl/: ${VERL_ROOT:-<unset>}" >&2; exit 1; }

world_size="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["world_size"])' "${ACTOR_DIR}/fsdp_config.json")"
for ((rank = 0; rank < world_size; rank++)); do
  shard="${ACTOR_DIR}/model_world_size_${world_size}_rank_${rank}.pt"
  [[ -f "${shard}" ]] || { echo "ERROR: missing FSDP model shard: ${shard}" >&2; exit 1; }
done

if [[ -f "${MERGED_PATH}/model.safetensors" ]]; then
  echo "Merged model already exists: ${MERGED_PATH}"
else
  [[ ! -e "${MERGED_PATH}" ]] || {
    echo "ERROR: merge target exists but has no model.safetensors: ${MERGED_PATH}" >&2
    echo "Choose another MERGED_PATH or inspect the incomplete target; it was not modified." >&2
    exit 1
  }
  temp_path="${MERGED_PATH}.partial-$$"
  trap '[[ -d "${temp_path:-}" ]] && echo "Partial merge preserved at: ${temp_path}" >&2' EXIT
  echo "Merging ${world_size} FSDP shards"
  echo "  source: ${ACTOR_DIR}"
  echo "  target: ${MERGED_PATH}"
  PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${temp_path}" \
      --trust-remote-code
  [[ -f "${temp_path}/model.safetensors" ]] || {
    echo "ERROR: merge completed without model.safetensors: ${temp_path}" >&2
    exit 1
  }
  mv "${temp_path}" "${MERGED_PATH}"
  trap - EXIT
fi

"${PYTHON}" - "${MERGED_PATH}" <<'PY'
from pathlib import Path
import sys
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

path = Path(sys.argv[1])
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
AutoProcessor.from_pretrained(path, trust_remote_code=True)
with safe_open(path / "model.safetensors", framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
if not keys:
    raise SystemExit("model.safetensors contains no tensors")
print(f"Verified merged model: {path}")
print(f"Architecture: {config.architectures}; tensors: {len(keys)}")
PY
