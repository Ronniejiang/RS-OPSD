#!/usr/bin/env bash
set -euo pipefail

PA_OPD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PA_OPD_VENV="${PA_OPD_VENV:-${PA_OPD_DIR}/.venv}"
PA_OPD_MODE="${PA_OPD_MODE:-adaptive_format}"
PA_OPD_RUN_NAME="${PA_OPD_RUN_NAME:-pa-opd-qwen3-vl-8b-4gpu}"
PA_OPD_WORK_ROOT="${PA_OPD_WORK_ROOT:-${PA_OPD_DIR}/outputs/${PA_OPD_RUN_NAME}}"
PA_OPD_RESUME_STEP="${PA_OPD_RESUME_STEP:-30}"
PA_OPD_SHORT_TMP="${PA_OPD_SHORT_TMP:-/tmp/po-${RANDOM}-$$}"
PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME="${PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME:-0}"

mkdir -p \
  "${PA_OPD_WORK_ROOT}/checkpoints" \
  "${PA_OPD_WORK_ROOT}/rollouts" \
  "${PA_OPD_WORK_ROOT}/wandb" \
  "${PA_OPD_WORK_ROOT}/cache/huggingface" \
  "${PA_OPD_WORK_ROOT}/cache/torch" \
  "${PA_OPD_SHORT_TMP}/ray" \
  "${PA_OPD_SHORT_TMP}/tmp"

export PA_OPD_CODE_ROOT="${PA_OPD_DIR}"
export PA_OPD_MODE PA_OPD_RUN_NAME PA_OPD_WORK_ROOT
export PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME
export PA_OPD_FORMAT_REWARD_COEF="0.1"
export PA_OPD_ACCURACY_REWARD_COEF="1.0"
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PA_OPD_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
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

echo "PA-OPD Ray temporary root: ${RAY_TMPDIR}"

CHECKPOINT_PATH="${PA_OPD_WORK_ROOT}/checkpoints/global_step_${PA_OPD_RESUME_STEP}"
ENTRYPOINT="verl.trainer.main_pa_opd"
HYDRA_LEGACY_ARGS=()

case "${PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME,,}" in
  1|true|yes|on)
    ENTRYPOINT="verl.trainer.main_pa_opd_resume"
    HYDRA_LEGACY_ARGS=(pa_opd_legacy_rollout_resume=true)
    echo "PA-OPD legacy migration source: ${CHECKPOINT_PATH}"
    echo "Migration is enabled only for this explicitly requested legacy checkpoint."
    ;;
  0|false|no|off|"")
    echo "PA-OPD strict resume source: ${CHECKPOINT_PATH}"
    ;;
  *)
    echo "ERROR: PA_OPD_ALLOW_LEGACY_ROLLOUT_RESUME must be 0 or 1" >&2
    exit 2
    ;;
esac

exec "${PA_OPD_VENV}/bin/python" -m "${ENTRYPOINT}" \
  --config-name pa_opd_modes \
  "${HYDRA_LEGACY_ARGS[@]}" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${CHECKPOINT_PATH}" \
  "$@"
