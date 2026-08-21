#!/usr/bin/env bash
set -euo pipefail

# The sole PA-OPD training entrypoint. Training semantics live in
# verl/trainer/config/pa_opd.yaml; pass optional Hydra overrides as "$@".
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_TRAIN_VENV="${PROJECT_ROOT}/.venv"
PYTHON="${PYTHON:-${DEFAULT_TRAIN_VENV}/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-}"
RUN_NAME="${RUN_NAME:-pa-opd-qwen3-vl-8b-4gpu}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${PROJECT_ROOT}/outputs}"
WORK_ROOT="${WORK_ROOT:-${CHECKPOINTS_ROOT}/${RUN_NAME}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${WORK_ROOT}/huggingface}"
# Ray uses AF_UNIX sockets, whose paths cannot exceed 107 bytes. Keep its
# ephemeral socket directory short; persistent run artifacts remain in WORK_ROOT.
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/paopd-ray}"
TMPDIR="${TMPDIR:-/tmp/paopd-tmp}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen3vl.jinja}"

[[ -x "${PYTHON}" ]] || { echo "ERROR: Python executable not found: ${PYTHON}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: Qwen3-VL backbone not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${SOURCE_DATA_DIR}/train.jsonl" ]] || { echo "ERROR: Vision-OPD-6K train.jsonl not found: ${SOURCE_DATA_DIR}/train.jsonl" >&2; exit 1; }
[[ -d "${SOURCE_DATA_DIR}/images" && -d "${SOURCE_DATA_DIR}/teacher_images" ]] || { echo "ERROR: Vision-OPD-6K image directories not found under: ${SOURCE_DATA_DIR}" >&2; exit 1; }
[[ -f "${CHAT_TEMPLATE_FILE}" ]] || { echo "ERROR: chat template not found: ${CHAT_TEMPLATE_FILE}" >&2; exit 1; }

mkdir -p "${WORK_ROOT}" "${HF_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export RAY_TMPDIR
export TMPDIR
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
# Let Ray assign a unique CUDA_VISIBLE_DEVICES value to every FSDP worker.
# A truthy NOSET flag is incompatible with the one-GPU-per-worker PA-OPD
# resource pool and produces NCCL "Duplicate GPU detected" at FSDP startup.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

export PA_OPD_ROOT="${PROJECT_ROOT}"
export PA_OPD_MODEL_PATH="${MODEL_PATH}"
export PA_OPD_TRAIN_FILE="${SOURCE_DATA_DIR}/train.jsonl"
export PA_OPD_WORK_DIR="${WORK_ROOT}"
export PA_OPD_RUN_NAME="${RUN_NAME}"
export PA_OPD_CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE}"

printf 'PA-OPD config: %s\n' "${PROJECT_ROOT}/verl/trainer/config/pa_opd.yaml"
printf 'Backbone: %s\n' "${MODEL_PATH}"
printf 'Training data source: %s\n' "${PA_OPD_TRAIN_FILE}"
printf 'Runtime output: %s\n' "${WORK_ROOT}"
printf 'Ray temporary directory: %s\n' "${RAY_TMPDIR}"
printf 'Expected GPUs: 4\n'

exec "${PYTHON}" -m verl.trainer.main_ppo --config-name pa_opd "$@"
