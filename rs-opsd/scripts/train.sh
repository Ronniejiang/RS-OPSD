#!/usr/bin/env bash
set -euo pipefail

# Portable RS-OPSD launcher. It deliberately contains no cluster, queue, or
# repository-specific path assumptions.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PA_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  MODEL_PATH=/path/to/model DATA_ROOT=/path/to/data OUTPUT_DIR=/path/to/output \
    bash scripts/train.sh {direct-2k-kl|direct-2k-kl-mixed|direct-2k-topk64-jsd-kl|direct-2k-three-image-kl|direct-2k-three-image-topk64-jsd-kl} [Hydra overrides...]

Optional environment variables:
  PYTHON=python              Python executable (default: python)
  PA_OPD_NUM_GPUS=4          GPUs visible to this job (default: 4)
  WANDB_MODE=offline         W&B mode (default: offline)
  RESUME_FROM_PATH=/path/to/global_step_N
  PA_OPD_SHORT_TMP=/tmp/...  Short temporary path for Ray sockets
  PA_OPD_VISIONOPD_ROOT=/path/to/VisionOPD  Additional source for direct-2k-kl-mixed
  PA_OPD_SAVE_FREQ=30         Required by direct-2k-three-image-kl-32gpu

Multi-node recipe: direct-2k-three-image-kl-32gpu (run once per Fuyao pod).
USAGE
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
recipe="$1"
shift
case "${recipe}" in
  direct-2k-kl) config_name="rs_opsd_direct_2k_kl" ;;
  direct-2k-kl-mixed) config_name="rs_opsd_direct_2k_kl_mixed" ;;
  direct-2k-topk64-jsd-kl) config_name="rs_opsd_direct_2k_topk64_jsd_kl" ;;
  direct-2k-three-image-kl) config_name="rs_opsd_direct_2k_three_image_kl" ;;
  direct-2k-three-image-kl-32gpu) config_name="rs_opsd_direct_2k_three_image_kl_32gpu" ;;
  direct-2k-three-image-topk64-jsd-kl) config_name="rs_opsd_direct_2k_three_image_topk64_jsd_kl" ;;
  *) echo "ERROR: unsupported recipe: ${recipe}" >&2; usage >&2; exit 2 ;;
esac

: "${MODEL_PATH:?MODEL_PATH must point to a local Hugging Face model directory}"
: "${DATA_ROOT:?DATA_ROOT must contain train.jsonl, images/, and teacher_images/}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required and receives checkpoints, rollouts, caches, and W&B files}"
PYTHON="${PYTHON:-python}"
PA_OPD_NUM_GPUS="${PA_OPD_NUM_GPUS:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${PYTHON}" == */* ]]; then
  [[ -x "${PYTHON}" ]] || { echo "ERROR: Python executable not found: ${PYTHON}" >&2; exit 1; }
else
  command -v "${PYTHON}" >/dev/null || { echo "ERROR: Python command not found: ${PYTHON}" >&2; exit 1; }
fi
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model directory not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATA_ROOT}/train.jsonl" ]] || { echo "ERROR: missing ${DATA_ROOT}/train.jsonl" >&2; exit 1; }
[[ -d "${DATA_ROOT}/images" ]] || { echo "ERROR: missing ${DATA_ROOT}/images" >&2; exit 1; }
[[ -d "${DATA_ROOT}/teacher_images" ]] || { echo "ERROR: missing ${DATA_ROOT}/teacher_images" >&2; exit 1; }
if [[ "${recipe}" == direct-2k-three-image-kl || "${recipe}" == direct-2k-three-image-kl-32gpu || "${recipe}" == direct-2k-three-image-topk64-jsd-kl ]]; then
  [[ -f "${DATA_ROOT}/derived/train.jsonl" ]] || { echo "ERROR: three-image recipe requires derived/train.jsonl" >&2; exit 1; }
  [[ -d "${DATA_ROOT}/derived/teacher_images" ]] || { echo "ERROR: three-image recipe requires derived/teacher_images" >&2; exit 1; }
fi

if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/validate_checkpoint.py" \
    --checkpoint "${RESUME_FROM_PATH}" --world-size "${PA_OPD_NUM_GPUS}"
fi

short_tmp="${PA_OPD_SHORT_TMP:-/tmp/pao-${UID:-0}-${RANDOM}}"
local_cache_root="${PA_OPD_LOCAL_CACHE_ROOT:-${short_tmp}/cache}"
mkdir -p "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/rollouts" "${OUTPUT_DIR}/wandb" \
  "${short_tmp}/ray" "${short_tmp}/tmp" "${local_cache_root}/huggingface" \
  "${local_cache_root}/torch" "${local_cache_root}/xdg" "${local_cache_root}/vllm"

export PYTHONPATH="${PA_OPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PA_OPD_MODEL_PATH="${MODEL_PATH}"
export PA_OPD_TRAIN_FILE="${DATA_ROOT}/train.jsonl"
export PA_OPD_WORK_DIR="${OUTPUT_DIR}"
export PA_OPD_RUN_NAME="${PA_OPD_RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
export PA_OPD_NUM_GPUS
# The output directory is a network filesystem. vLLM/Inductor materializes
# and dlopens many transient shared objects during its first profile run; doing
# that there can produce stale file handles when eight rollout workers compile
# concurrently. Keep all rebuildable caches on node-local storage instead.
export PA_OPD_LOCAL_CACHE_ROOT="${local_cache_root}"
export HF_HOME="${local_cache_root}/huggingface"
export TORCH_HOME="${local_cache_root}/torch"
export XDG_CACHE_HOME="${local_cache_root}/xdg"
export PA_OPD_VLLM_CACHE_ROOT="${local_cache_root}/vllm"
export VLLM_CACHE_ROOT="${local_cache_root}/vllm/driver"
export TORCHINDUCTOR_CACHE_DIR="${local_cache_root}/torchinductor-driver"
export TRITON_CACHE_DIR="${local_cache_root}/triton-driver"
export WANDB_MODE WANDB_DIR="${OUTPUT_DIR}/wandb"
export RAY_TMPDIR="${short_tmp}/ray"
export TMPDIR="${short_tmp}/tmp"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"
export PYTHONPYCACHEPREFIX="${short_tmp}/pycache"
export PYTHONUNBUFFERED=1
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND

resume_args=()
if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
  resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

printf 'RS-OPSD recipe: %s\nmodel: %s\ndata: %s\noutput: %s\nGPUs: %s\n' \
  "${recipe}" "${MODEL_PATH}" "${PA_OPD_TRAIN_FILE}" "${OUTPUT_DIR}" "${PA_OPD_NUM_GPUS}"

runner=("${PYTHON}")
if (( ${PA_OPD_NNODES:-1} > 1 )); then
  runner+=("${SCRIPT_DIR}/run_multinode_ray.py" -- "${PYTHON}")
fi
exec "${runner[@]}" -m verl.trainer.main_pa_opd_direct \
  --config-name "${config_name}" \
  "${resume_args[@]}" \
  "$@"
