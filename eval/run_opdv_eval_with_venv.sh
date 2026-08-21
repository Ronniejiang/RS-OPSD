#!/usr/bin/env bash
set -euo pipefail

# Bootstrap/reuse the dedicated vLLM environment, then invoke the normal local
# OPD-V evaluator. All benchmark and serving environment variables are passed
# through unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_VENV="${EVAL_VENV:-${REPO_ROOT}/.venv-eval}"

bash "${SCRIPT_DIR}/setup_opdv_eval_env.sh"
export PYTHON="${EVAL_VENV}/bin/python"
export VLLM_BIN="${EVAL_VENV}/bin/vllm"
exec bash "${SCRIPT_DIR}/eval_local_opdv_qwen3vl8b.sh"
