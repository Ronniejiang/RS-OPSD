#!/usr/bin/env bash
set -euo pipefail

# Create (once) and reuse a minimal, persistent vLLM environment for OPD-V
# inference. FOVIS/.venv-fovis is intentionally not used: it has neither vLLM
# nor the OpenAI Python client required by this evaluator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_VENV="${EVAL_VENV:-${REPO_ROOT}/.venv-eval}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.cache/pip}"
VLLM_VERSION="${VLLM_VERSION:-0.18.0}"
OPENAI_VERSION="${OPENAI_VERSION:-2.24.0}"
DATASETS_VERSION="${DATASETS_VERSION:-4.4.2}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
mkdir -p "$(dirname "${EVAL_VENV}")" "${PIP_CACHE_DIR}"

if [[ ! -x "${EVAL_VENV}/bin/python" ]]; then
  echo "Creating OPD-V evaluation environment: ${EVAL_VENV}"
  "${PYTHON_BIN}" -m venv "${EVAL_VENV}"
fi

needs_install=0
if ! "${EVAL_VENV}/bin/python" - <<PY
import importlib.metadata as metadata
import sys

expected = {
    "vllm": "${VLLM_VERSION}",
    "openai": "${OPENAI_VERSION}",
    "datasets": "${DATASETS_VERSION}",
}
for package, version in expected.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        sys.exit(1)
    if installed != version:
        print(f"{package}: found {installed}, expected {version}")
        sys.exit(1)
PY
then
  needs_install=1
fi

if [[ "${needs_install}" == "1" ]]; then
  echo "Installing pinned OPD-V inference dependencies. This happens only once per ${EVAL_VENV}."
  "${EVAL_VENV}/bin/python" -m pip install --upgrade pip
  "${EVAL_VENV}/bin/python" -m pip install \
    "vllm==${VLLM_VERSION}" \
    "openai==${OPENAI_VERSION}" \
    "datasets==${DATASETS_VERSION}"
fi

"${EVAL_VENV}/bin/python" - <<'PY'
import datasets
import openai
import vllm

print(f"Ready: vllm={vllm.__version__}, openai={openai.__version__}, datasets={datasets.__version__}")
PY
echo "Use these executables: ${EVAL_VENV}/bin/python and ${EVAL_VENV}/bin/vllm"
