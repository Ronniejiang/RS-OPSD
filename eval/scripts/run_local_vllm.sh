#!/usr/bin/env bash
set -euo pipefail

# One-command evaluation for a local Hugging Face vision model. It validates
# datasets, starts vLLM, verifies /v1/models, evaluates, and stops the server.
# All model and serving parameters are supplied through environment variables.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 BENCHMARK=lrs-vqa \
#   MODEL_PATH=/path/to/model LRS_ROOT=/path/to/lrs-vqa bash eval/scripts/run_local_vllm.sh
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 BENCHMARK=all \
#   LRS_ROOT=/path/to/lrs MME_ROOT=/path/to/mme XLRS_ROOT=/path/to/xlrs \
#   MODEL_PATH=/path/to/model bash eval/scripts/run_local_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH:-model}")}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"
API_BASE="${API_BASE:-}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
SERVER_LOG="${SERVER_LOG:-${REPO_ROOT}/eval/results/${SERVED_MODEL_NAME}_server.log}"
SKIP_SERVE="${SKIP_SERVE:-0}"
KEEP_SERVER="${KEEP_SERVER:-0}"

BENCHMARK="${BENCHMARK:-all}"
LRS_ROOT="${LRS_ROOT:-}"
MME_ROOT="${MME_ROOT:-}"
XLRS_ROOT="${XLRS_ROOT:-}"
RUN_NAME="${RUN_NAME:-${SERVED_MODEL_NAME}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval/results}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_RETRIES="${MAX_RETRIES:-3}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-32}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
MAX_PIXELS="${MAX_PIXELS:-16777216}"
IMAGE_FORMAT="${IMAGE_FORMAT:-png}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
RESUME="${RESUME:-0}"
LIMIT="${LIMIT:-}"
LRS_SEMANTIC_MODEL="${LRS_SEMANTIC_MODEL:-}"
LRS_SEMANTIC_THRESHOLD="${LRS_SEMANTIC_THRESHOLD:-0.85}"
LRS_SEMANTIC_BATCH_SIZE="${LRS_SEMANTIC_BATCH_SIZE:-64}"

[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: config.json missing from: ${MODEL_PATH}" >&2; exit 1; }
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "ERROR: Python not found: ${PYTHON}" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required for the server readiness check." >&2; exit 1; }

# Never pass empty roots: eval.run treats an empty string as the current directory.
DATASET_ARGS=(--dataset "${BENCHMARK}")
[[ -n "${LRS_ROOT}" ]] && DATASET_ARGS+=(--lrs-root "${LRS_ROOT}")
[[ -n "${MME_ROOT}" ]] && DATASET_ARGS+=(--mme-root "${MME_ROOT}")
[[ -n "${XLRS_ROOT}" ]] && DATASET_ARGS+=(--xlrs-root "${XLRS_ROOT}")

cd "${REPO_ROOT}"
# Ensure local inference and readiness checks bypass inherited proxies.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${HOST}"
export no_proxy="${NO_PROXY}"
if [[ "${SKIP_SERVE}" != "1" ]]; then
  PORT="$("${PYTHON}" -m eval.server_readiness port --host "${HOST}" --port "${PORT:-0}")"
  EXPECTED_API_BASE="http://${HOST}:${PORT}/v1"
  if [[ -n "${API_BASE}" && "${API_BASE%/}" != "${EXPECTED_API_BASE}" ]]; then
    echo "ERROR: API_BASE must match the local server: ${EXPECTED_API_BASE}" >&2
    exit 1
  fi
  API_BASE="${EXPECTED_API_BASE}"
else
  API_BASE="${API_BASE:-http://${HOST}:${PORT:-8000}/v1}"
fi
echo "Model endpoint: ${API_BASE}; expected model: ${SERVED_MODEL_NAME}"
# Model identity is verified before evaluation, not just HTTP availability.

echo "Checking benchmark layouts before loading the model..."
"${PYTHON}" -m eval.run "${DATASET_ARGS[@]}" --dry-run

mkdir -p "${OUT_ROOT}" "$(dirname "${SERVER_LOG}")"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" && "${KEEP_SERVER}" != "1" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "Stopping vLLM server (pid=${server_pid})..."
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local probe_args=(wait --api-base "${API_BASE}" --model-id "${SERVED_MODEL_NAME}" --timeout "${WAIT_SECONDS}")
  [[ -n "${server_pid}" ]] && probe_args+=(--pid "${server_pid}")
  if ! "${PYTHON}" -m eval.server_readiness "${probe_args[@]}"; then
    tail -n 80 "${SERVER_LOG}" >&2 || true
    return 1
  fi
}

if [[ "${SKIP_SERVE}" == "1" ]]; then
  echo "Using already-running model server: ${API_BASE}"
else
  command -v "${VLLM_BIN}" >/dev/null 2>&1 || { echo "ERROR: vLLM not found: ${VLLM_BIN}" >&2; exit 1; }
  VLLM_ARGS=(
    serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TP_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
  )
  [[ -n "${MAX_MODEL_LEN}" ]] && VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
  # Extra options are explicitly supplied by the caller, e.g. VLLM_EXTRA_ARGS='--enforce-eager'.
  [[ -n "${VLLM_EXTRA_ARGS}" ]] && read -r -a EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS}" && VLLM_ARGS+=("${EXTRA_ARGS[@]}")
  echo "Starting vLLM from ${MODEL_PATH}; log: ${SERVER_LOG}"
  "${VLLM_BIN}" "${VLLM_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  server_pid=$!
fi

wait_for_server
echo "Model server ready: ${API_BASE}"

EVAL_ARGS=(
  "${DATASET_ARGS[@]}"
  --api-base "${API_BASE}"
  --api-key "${API_KEY}"
  --model-id "${SERVED_MODEL_NAME}"
  --output-root "${OUT_ROOT}"
  --run-name "${RUN_NAME}"
  --max-tokens "${MAX_TOKENS}"
  --max-retries "${MAX_RETRIES}"
  --parallel-workers "${PARALLEL_WORKERS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --max-pixels "${MAX_PIXELS}"
  --image-format "${IMAGE_FORMAT}"
  --jpeg-quality "${JPEG_QUALITY}"
  --enable-thinking "${ENABLE_THINKING}"
)
[[ "${RESUME}" == "1" ]] && EVAL_ARGS+=(--resume)
[[ -n "${LIMIT}" ]] && EVAL_ARGS+=(--limit "${LIMIT}")
[[ -n "${LRS_SEMANTIC_MODEL}" ]] && EVAL_ARGS+=(--lrs-semantic-model "${LRS_SEMANTIC_MODEL}" --lrs-semantic-threshold "${LRS_SEMANTIC_THRESHOLD}" --lrs-semantic-batch-size "${LRS_SEMANTIC_BATCH_SIZE}")

"${PYTHON}" -m eval.run "${EVAL_ARGS[@]}"
