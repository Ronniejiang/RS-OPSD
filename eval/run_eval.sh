#!/usr/bin/env bash
set -euo pipefail

# OPD-V OpenAI-compatible evaluation launcher for the remote-sensing suites.
# BENCHMARK accepts lrs-vqa, mme-realworld-rs, xlrs-bench, or a comma-separated list.

PYTHON="${PYTHON:-python3}"
API_BASE="${API_BASE:?ERROR: API_BASE must be set (for example http://localhost:8000/v1)}"
OPENAI_MODEL_ID="${OPENAI_MODEL_ID:?ERROR: OPENAI_MODEL_ID must be set}"

BENCHMARK="${BENCHMARK:-lrs-vqa}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
LRS_ROOT="${LRS_ROOT:-}"
MME_ROOT="${MME_ROOT:-}"
XLRS_ROOT="${XLRS_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-eval/results}"
RUN_NAME="${RUN_NAME:-${OPENAI_MODEL_ID//\//_}}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_RETRIES="${MAX_RETRIES:-3}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-32}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
MAX_PIXELS="${MAX_PIXELS:-16777216}"
IMAGE_FORMAT="${IMAGE_FORMAT:-png}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
RESUME="${RESUME:-0}"
LIMIT="${LIMIT:-}"

ARGS=(
  --dataset "${BENCHMARK}"
  --api-base "${API_BASE}"
  --api-key "${API_KEY}"
  --model-id "${OPENAI_MODEL_ID}"
  --output-root "${OUT_ROOT}"
  --run-name "${RUN_NAME}"
  --max-tokens "${MAX_TOKENS}"
  --max-retries "${MAX_RETRIES}"
  --parallel-workers "${PARALLEL_WORKERS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --max-pixels "${MAX_PIXELS}"
  --image-format "${IMAGE_FORMAT}"
  --jpeg-quality "${JPEG_QUALITY}"
)

IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARK}"
for BENCH in "${BENCHMARKS[@]}"; do
  BENCH="$(echo "${BENCH}" | xargs)"
  case "${BENCH}" in
    lrs-vqa)
      [[ -n "${LRS_ROOT}" ]] || { echo "ERROR: LRS_ROOT must be set for lrs-vqa" >&2; exit 1; }
      ;;
    mme-realworld-rs)
      [[ -n "${MME_ROOT}" ]] || { echo "ERROR: MME_ROOT must be set for mme-realworld-rs" >&2; exit 1; }
      ;;
    xlrs-bench)
      [[ -n "${XLRS_ROOT}" ]] || { echo "ERROR: XLRS_ROOT must be set for xlrs-bench" >&2; exit 1; }
      ;;
    *)
      echo "ERROR: Unsupported benchmark: ${BENCH}" >&2
      exit 1
      ;;
  esac
done

[[ -n "${LRS_ROOT}" ]] && ARGS+=(--lrs-root "${LRS_ROOT}")
[[ -n "${MME_ROOT}" ]] && ARGS+=(--mme-root "${MME_ROOT}")
[[ -n "${XLRS_ROOT}" ]] && ARGS+=(--xlrs-root "${XLRS_ROOT}")
[[ "${RESUME}" == "1" ]] && ARGS+=(--resume)
[[ -n "${LIMIT}" ]] && ARGS+=(--limit "${LIMIT}")
[[ -n "${ENABLE_THINKING}" ]] && ARGS+=(--enable-thinking "${ENABLE_THINKING}")

"${PYTHON}" -m eval.run "${ARGS[@]}"
