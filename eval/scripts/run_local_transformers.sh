#!/usr/bin/env bash
set -euo pipefail

# Local single-process Hugging Face evaluation. This is the compatibility
# fallback for machines whose installed vLLM does not support the target VLM.
# It deliberately uses one worker because the model is resident in one process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_ID="${MODEL_ID:-${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH:-model}")}}"
BENCHMARK="${BENCHMARK:-all}"
LRS_ROOT="${LRS_ROOT:-}"
MME_ROOT="${MME_ROOT:-}"
XLRS_ROOT="${XLRS_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval/results}"
RUN_NAME="${RUN_NAME:-${MODEL_ID}}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_PIXELS="${MAX_PIXELS:-16777216}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
RESUME="${RESUME:-0}"
LIMIT="${LIMIT:-}"
LRS_SEMANTIC_MODEL="${LRS_SEMANTIC_MODEL:-}"
LRS_SEMANTIC_THRESHOLD="${LRS_SEMANTIC_THRESHOLD:-0.85}"
LRS_SEMANTIC_BATCH_SIZE="${LRS_SEMANTIC_BATCH_SIZE:-64}"

[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: config.json missing from: ${MODEL_PATH}" >&2; exit 1; }
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "ERROR: Python not found: ${PYTHON}" >&2; exit 1; }

DATASET_ARGS=(--dataset "${BENCHMARK}")
[[ -n "${LRS_ROOT}" ]] && DATASET_ARGS+=(--lrs-root "${LRS_ROOT}")
[[ -n "${MME_ROOT}" ]] && DATASET_ARGS+=(--mme-root "${MME_ROOT}")
[[ -n "${XLRS_ROOT}" ]] && DATASET_ARGS+=(--xlrs-root "${XLRS_ROOT}")

EVAL_ARGS=(
  "${DATASET_ARGS[@]}"
  --backend transformers
  --model-path "${MODEL_PATH}"
  --model-id "${MODEL_ID}"
  --output-root "${OUT_ROOT}"
  --run-name "${RUN_NAME}"
  --max-tokens "${MAX_TOKENS}"
  --parallel-workers 1
  --max-pixels "${MAX_PIXELS}"
  --enable-thinking "${ENABLE_THINKING}"
)
[[ "${RESUME}" == "1" ]] && EVAL_ARGS+=(--resume)
[[ -n "${LIMIT}" ]] && EVAL_ARGS+=(--limit "${LIMIT}")
[[ -n "${LRS_SEMANTIC_MODEL}" ]] && EVAL_ARGS+=(--lrs-semantic-model "${LRS_SEMANTIC_MODEL}" --lrs-semantic-threshold "${LRS_SEMANTIC_THRESHOLD}" --lrs-semantic-batch-size "${LRS_SEMANTIC_BATCH_SIZE}")

cd "${REPO_ROOT}"
echo "Starting local Transformers evaluation from ${MODEL_PATH} (one worker)."
exec "${PYTHON}" -m eval.run "${EVAL_ARGS[@]}"
