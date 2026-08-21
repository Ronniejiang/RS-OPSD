#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PA_OPD_CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PA_OPD_MODE="${PA_OPD_MODE:-adaptive_format}"

case "${PA_OPD_MODE}" in
  adaptive_format|opd_rlvr) ;;
  *)
    echo "PA_OPD_MODE must be adaptive_format or opd_rlvr, got: ${PA_OPD_MODE}" >&2
    exit 2
    ;;
esac

PA_OPD_VENV="${PA_OPD_VENV:-${PA_OPD_CODE_ROOT}/.venv}"
PA_OPD_RUN_NAME="${PA_OPD_RUN_NAME:-pa-opd-qwen3-vl-8b-4gpu-${PA_OPD_MODE}}"
PA_OPD_WORK_ROOT="${PA_OPD_WORK_ROOT:-${PA_OPD_CODE_ROOT}/outputs/${PA_OPD_RUN_NAME}}"
PA_OPD_SHORT_TMP="${PA_OPD_SHORT_TMP:-/tmp/po-${RANDOM}-$$}"

mkdir -p \
  "${PA_OPD_WORK_ROOT}/checkpoints" \
  "${PA_OPD_WORK_ROOT}/rollouts" \
  "${PA_OPD_WORK_ROOT}/wandb" \
  "${PA_OPD_WORK_ROOT}/cache/huggingface" \
  "${PA_OPD_WORK_ROOT}/cache/torch" \
  "${PA_OPD_SHORT_TMP}/ray" \
  "${PA_OPD_SHORT_TMP}/tmp"

export PA_OPD_CODE_ROOT PA_OPD_MODE PA_OPD_RUN_NAME PA_OPD_WORK_ROOT
export PA_OPD_FORMAT_REWARD_COEF="${PA_OPD_FORMAT_REWARD_COEF:-0.1}"
export PA_OPD_ACCURACY_REWARD_COEF="${PA_OPD_ACCURACY_REWARD_COEF:-1.0}"
# Four FSDP workers must receive distinct CUDA_VISIBLE_DEVICES assignments
# from Ray. Some cluster images export this opt-in Ray flag globally; leaving
# it set makes every child actor inherit the TaskRunner's single GPU and NCCL
# rejects the process group as duplicate-GPU usage.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PA_OPD_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_TMPDIR="${PA_OPD_SHORT_TMP}/ray"
export TMPDIR="${PA_OPD_SHORT_TMP}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export WANDB_DIR="${PA_OPD_WORK_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${PA_OPD_WORK_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${PA_OPD_WORK_ROOT}/cache/torch"
export XDG_CACHE_HOME="${PA_OPD_WORK_ROOT}/cache"

echo "PA-OPD mode: ${PA_OPD_MODE}"
echo "PA-OPD config: ${PA_OPD_CODE_ROOT}/verl/trainer/config/pa_opd_modes.yaml"
echo "Runtime output: ${PA_OPD_WORK_ROOT}"
echo "Expected GPUs: 4"

exec "${PA_OPD_VENV}/bin/python" -m verl.trainer.main_pa_opd \
  --config-name pa_opd_modes \
  "$@"
