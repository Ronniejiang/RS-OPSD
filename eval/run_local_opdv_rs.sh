#!/usr/bin/env bash
set -euo pipefail

# Start the local OPD-V Qwen3-VL-8B checkpoint with vLLM, then evaluate it on
# LRS-VQA, MME-RealWorld Remote Sensing, and/or XLRS-Bench.
#
# Required dataset roots are determined by BENCHMARK. For example:
#   BENCHMARK=lrs-vqa LRS_ROOT=/path/to/lrs-vqa bash eval/run_local_opdv_rs.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 BENCHMARK=all \
#   LRS_ROOT=/path/to/lrs MME_ROOT=/path/to/mme XLRS_ROOT=/path/to/xlrs \
#   bash eval/run_local_opdv_rs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OPD-V-Qwen3-VL-8B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_BASE="${API_BASE:-http://${HOST}:${PORT}/v1}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
SERVER_LOG="${SERVER_LOG:-${REPO_ROOT}/eval/results/${SERVED_MODEL_NAME}_server.log}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
KEEP_SERVER="${KEEP_SERVER:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"

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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: config.json was not found in model directory: ${MODEL_PATH}" >&2
  exit 1
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: Python executable was not found: ${PYTHON}" >&2
  exit 1
fi

mkdir -p "$(dirname "${SERVER_LOG}")" "${OUT_ROOT}"

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
  local elapsed=0
  while (( elapsed < WAIT_SECONDS )); do
    if curl --fail --silent --show-error "${API_BASE}/models" >/dev/null; then
      return 0
    fi
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "ERROR: vLLM server exited during startup. See ${SERVER_LOG}" >&2
      tail -n 80 "${SERVER_LOG}" >&2 || true
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "ERROR: vLLM did not become ready within ${WAIT_SECONDS}s: ${API_BASE}/models" >&2
  return 1
}

if [[ "${SKIP_SERVE}" != "1" ]]; then
  if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
    echo "ERROR: vLLM executable was not found: ${VLLM_BIN}" >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to wait for the vLLM server." >&2
    exit 1
  fi

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
  # Deliberately allow advanced vLLM options through this explicit user input.
  [[ -n "${VLLM_EXTRA_ARGS}" ]] && read -r -a EXTRA_VLLM_ARGS <<< "${VLLM_EXTRA_ARGS}" && VLLM_ARGS+=("${EXTRA_VLLM_ARGS[@]}")

  echo "Starting vLLM with model: ${MODEL_PATH}"
  echo "Server log: ${SERVER_LOG}"
  "${VLLM_BIN}" "${VLLM_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  server_pid=$!
  wait_for_server
  echo "vLLM is ready at ${API_BASE} (pid=${server_pid})"
else
  echo "SKIP_SERVE=1: using existing server at ${API_BASE}"
  wait_for_server
fi

EVAL_ARGS=(
  --dataset "${BENCHMARK}"
  --lrs-root "${LRS_ROOT}"
  --mme-root "${MME_ROOT}"
  --xlrs-root "${XLRS_ROOT}"
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

cd "${REPO_ROOT}"
"${PYTHON}" -m eval.run "${EVAL_ARGS[@]}"
