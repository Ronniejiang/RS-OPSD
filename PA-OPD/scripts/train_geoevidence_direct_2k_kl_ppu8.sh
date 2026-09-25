#!/usr/bin/env bash
set -euo pipefail

# PPU uses a CUDA/NCCL compatibility layer.  Do not replace CUDA_VISIBLE_DEVICES
# or trainer.device=cuda with a hypothetical torch.ppu backend.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PA_OPD_ROOT="${PA_OPD_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/ppu_comm_env.sh"
# The submitted source is unpacked at /code in Fuyao.  Use the selected PPU
# image's interpreter by default; a workstation-only virtualenv path is not
# available inside that container.
PYTHON="${PYTHON:-python3}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face model directory}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the GeoEvidence dataset directory}"
RUN_NAME="${RUN_NAME:-qwen3-vl-8b-instruct-gpu8-2k-kl-orig+crop}"
TRAIN_RECIPE="${TRAIN_RECIPE:-direct-2k-kl}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  : "${CHECKPOINTS_ROOT:?Set OUTPUT_DIR or CHECKPOINTS_ROOT explicitly}"
  OUTPUT_DIR="${CHECKPOINTS_ROOT}/${RUN_NAME}"
fi
# PA_OPD_NUM_GPUS is the global FSDP/Ray world size. Keep it distinct from
# PA_OPD_GPUS_PER_NODE: the latter is what the scheduled Fuyao node exposes
# locally. The target PPU partition offers both 1x8 and 1x16 nodes.
PA_OPD_GPUS_PER_NODE="${PA_OPD_GPUS_PER_NODE:-8}"
PA_OPD_NNODES="${PA_OPD_NNODES:-1}"
PA_OPD_NUM_GPUS="${PA_OPD_NUM_GPUS:-$((PA_OPD_GPUS_PER_NODE * PA_OPD_NNODES))}"
WANDB_MODE="${WANDB_MODE:-offline}"
PA_OPD_ROLLOUT_TP="${PA_OPD_ROLLOUT_TP:-1}"
PA_OPD_STUDENT_IMAGE_MODE="${PA_OPD_STUDENT_IMAGE_MODE:-images}"
PA_OPD_TEACHER_FULL_IMAGE_MODE="${PA_OPD_TEACHER_FULL_IMAGE_MODE:-images}"
export PA_OPD_STUDENT_IMAGE_MODE PA_OPD_TEACHER_FULL_IMAGE_MODE
[[ "${PA_OPD_ROLLOUT_TP}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: PA_OPD_ROLLOUT_TP must be a positive integer" >&2; exit 2; }
(( PA_OPD_GPUS_PER_NODE % PA_OPD_ROLLOUT_TP == 0 )) || { echo "ERROR: rollout TP must divide devices per node" >&2; exit 2; }
export PYTHONUNBUFFERED=1
echo "Starting PA-OPD PPU launcher: rollout TP=${PA_OPD_ROLLOUT_TP}, devices=${PA_OPD_NUM_GPUS}"

[[ "${PA_OPD_GPUS_PER_NODE}" == "8" || "${PA_OPD_GPUS_PER_NODE}" == "16" ]] || {
  echo "ERROR: this PPU recipe supports 8 or 16 devices per node, got ${PA_OPD_GPUS_PER_NODE}" >&2
  exit 2
}
[[ "${PA_OPD_NNODES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: PA_OPD_NNODES must be a positive integer" >&2; exit 2; }
[[ "${PA_OPD_NUM_GPUS}" == "$((PA_OPD_GPUS_PER_NODE * PA_OPD_NNODES))" ]] || {
  echo "ERROR: PA_OPD_NUM_GPUS=${PA_OPD_NUM_GPUS} must equal PA_OPD_GPUS_PER_NODE * PA_OPD_NNODES (${PA_OPD_GPUS_PER_NODE} * ${PA_OPD_NNODES})" >&2
  exit 2
}
case "${TRAIN_RECIPE}" in
  direct-2k-kl|direct-2k-kl-mixed|direct-2k-topk64-jsd-kl|direct-2k-three-image-kl-32gpu|direct-2k-three-image-topk64-jsd-kl) ;;
  *)
    echo "ERROR: unsupported direct GeoEvidence recipe: ${TRAIN_RECIPE}" >&2
    exit 2
    ;;
esac
if [[ "${PYTHON}" == */* ]]; then
  [[ -x "${PYTHON}" ]] || { echo "ERROR: Python executable not found: ${PYTHON}" >&2; exit 1; }
else
  command -v "${PYTHON}" >/dev/null 2>&1 || { echo "ERROR: Python command not found: ${PYTHON}" >&2; exit 1; }
fi
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: model not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATA_ROOT}/train.jsonl" ]] || { echo "ERROR: missing ${DATA_ROOT}/train.jsonl" >&2; exit 1; }
[[ -f "${DATA_ROOT}/final_annotations.json" ]] || {
  echo "ERROR: missing ${DATA_ROOT}/final_annotations.json; rebuild the manifest from the released annotations" >&2
  exit 1
}

export PYTHONPATH="${PA_OPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# Correct device placement allows normal rank-0 state synchronization. Meta
# initialization on other ranks requires this broadcast; disabling it now
# explicitly selects real CPU checkpoint loading on every rank instead.
export PA_OPD_FSDP_SYNC_MODULE_STATES="${PA_OPD_FSDP_SYNC_MODULE_STATES:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
# A hybrid WorkerDict already contains Actor/Reference/Teacher/rollout. The
# vendor Ray 2.31 runtime packed three 1/3-GPU WorkerDicts onto each device,
# leaving 10 of 16 devices idle. Reserve a full GPU for each training rank.
export PA_OPD_VALIDATE_GPU_ASSIGNMENT=1
# ``low`` combines activation offload, one concurrent rollout sequence and a
# 2K visual profiling budget. Historical memory fractions measured with three
# training ranks on one device are not valid capacity estimates after fixing
# the Ray placement; monitor the full-GPU-per-rank job before tuning upward.
PA_OPD_MEMORY_PROFILE="${PA_OPD_MEMORY_PROFILE:-balanced}"
rollout_profile_args=()
case "${PA_OPD_MEMORY_PROFILE}" in
  balanced)
    : "${PA_OPD_VLLM_GPU_MEMORY_UTILIZATION:=0.70}"
    : "${PA_OPD_VLLM_MAX_NUM_SEQS:=8}"
    : "${PA_OPD_VLLM_MAX_MODEL_LEN:=10240}"
    : "${PA_OPD_VLLM_MAX_NUM_BATCHED_TOKENS:=10240}"
    : "${PA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU:=10240}"
    : "${PA_OPD_MAX_PROMPT_LENGTH:=8192}"
    : "${PA_OPD_ENABLE_ACTIVATION_OFFLOAD:=false}"
    ;;
  low)
    # GeoEvidence's audited maximum Student sequence is 4,398 prompt + 16
    # generated tokens. A 5,120-token engine retains a >700-token margin;
    # Teacher prompt length is controlled separately and remains unchanged.
    : "${PA_OPD_VLLM_GPU_MEMORY_UTILIZATION:=0.66}"
    : "${PA_OPD_VLLM_MAX_NUM_SEQS:=1}"
    : "${PA_OPD_VLLM_MAX_MODEL_LEN:=5120}"
    : "${PA_OPD_VLLM_MAX_NUM_BATCHED_TOKENS:=5120}"
    : "${PA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU:=5120}"
    : "${PA_OPD_MAX_PROMPT_LENGTH:=4608}"
    : "${PA_OPD_ENABLE_ACTIVATION_OFFLOAD:=true}"
    # Student sees exactly one existing 2K image. Qwen's default 4096^2
    # processor budget makes vLLM profile 16,384 visual tokens unnecessarily.
    # All 6,150 distinct manifest images were checked: the 2048^2 cap changes
    # none of their smart-resize dimensions. Teacher/FSDP processing is untouched.
    rollout_profile_args+=(
      '+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.size={shortest_edge:65536,longest_edge:4194304}'
      '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:1,video:0}'
    )
    ;;
  *)
    echo "ERROR: PA_OPD_MEMORY_PROFILE must be balanced or low, got: ${PA_OPD_MEMORY_PROFILE}" >&2
    exit 2
    ;;
esac
export PA_OPD_MEMORY_PROFILE PA_OPD_VLLM_GPU_MEMORY_UTILIZATION PA_OPD_VLLM_MAX_NUM_SEQS
export PA_OPD_VLLM_MAX_MODEL_LEN PA_OPD_VLLM_MAX_NUM_BATCHED_TOKENS
export PA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU PA_OPD_MAX_PROMPT_LENGTH PA_OPD_ENABLE_ACTIVATION_OFFLOAD
export PA_OPD_VLLM_ENFORCE_EAGER="${PA_OPD_VLLM_ENFORCE_EAGER:-true}"

validation_recipe="${TRAIN_RECIPE}"
[[ "${TRAIN_RECIPE}" == direct-2k-three-image-kl-32gpu ]] && validation_recipe=direct-2k-three-image-kl
[[ "${TRAIN_RECIPE}" == direct-2k-kl-mixed ]] && validation_recipe=direct-2k-kl
"${PYTHON}" "${SCRIPT_DIR}/validate_geoevidence_views.py" \
  --data-root "${DATA_ROOT}" --recipe "${validation_recipe}" \
  --student-view "${PA_OPD_STUDENT_IMAGE_MODE}" --teacher-full-view "${PA_OPD_TEACHER_FULL_IMAGE_MODE}"
if [[ "${TRAIN_RECIPE}" == direct-2k-kl-mixed ]]; then
  : "${PA_OPD_VISIONOPD_ROOT:?Mixed recipe requires PA_OPD_VISIONOPD_ROOT}"
  "${PYTHON}" "${SCRIPT_DIR}/validate_mixed_2k_data.py" \
    --geoevidence-root "${DATA_ROOT}" --visionopd-root "${PA_OPD_VISIONOPD_ROOT}"
fi
"${PYTHON}" "${SCRIPT_DIR}/preflight_ppu_runtime.py" --require-devices "${PA_OPD_GPUS_PER_NODE}"
if [[ "${PA_OPD_TP_PREFLIGHT:-0}" == "1" ]]; then
  PA_OPD_NUM_GPUS="${PA_OPD_NUM_GPUS}" PA_OPD_ROLLOUT_TP="${PA_OPD_ROLLOUT_TP}" \
    NCCL_DEBUG=INFO "${PYTHON}" "${SCRIPT_DIR}/preflight_vllm_tp_ppu.py" --phase ray
fi
"${PYTHON}" - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
annotations = json.loads((root / "final_annotations.json").read_text(encoding="utf-8"))
if not isinstance(annotations, list) or not annotations:
    raise RuntimeError("final_annotations.json must contain a non-empty list")
expected_rows = len(annotations)
rows = [json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
if len(rows) != expected_rows:
    raise RuntimeError(
        f"Manifest has {len(rows)} rows but final_annotations.json has {expected_rows}; "
        "run scripts/rebuild_geoevidence_train_jsonl.py"
    )
for index, row in enumerate(rows, start=1):
    for field in ("images", "teacher_images"):
        value = row.get(field)
        if not isinstance(value, list) or len(value) != 1 or not (root / value[0]).is_file():
            raise RuntimeError(f"Manifest row {index} has no usable {field}: {value!r}")
print(f"GeoEvidence direct 2K manifest: {len(rows)} rows, all Student/Teacher views present")
PY

mkdir -p "${OUTPUT_DIR}"
# ``train.sh`` is deliberately a separate shell process.  Export every
# recipe input, including defaults set above, so this entrypoint behaves the
# same when launched directly and when launched through ``fuyao deploy env``.
export MODEL_PATH DATA_ROOT RUN_NAME CHECKPOINTS_ROOT OUTPUT_DIR PA_OPD_NUM_GPUS PA_OPD_GPUS_PER_NODE PA_OPD_NNODES TRAIN_RECIPE
export WANDB_MODE
export PA_OPD_RUN_NAME="${RUN_NAME}"
export PA_OPD_SHORT_TMP="${PA_OPD_SHORT_TMP:-/tmp/pao-${UID:-0}-${RANDOM}}"

echo "PPU memory profile: ${PA_OPD_MEMORY_PROFILE}"
echo "  topology: ${PA_OPD_NNODES} node(s) x ${PA_OPD_GPUS_PER_NODE} PPU = ${PA_OPD_NUM_GPUS} total devices"
echo "  vLLM: utilization=${PA_OPD_VLLM_GPU_MEMORY_UTILIZATION}, eager=${PA_OPD_VLLM_ENFORCE_EAGER}, max_num_seqs=${PA_OPD_VLLM_MAX_NUM_SEQS}, max_model_len=${PA_OPD_VLLM_MAX_MODEL_LEN}, max_batched_tokens=${PA_OPD_VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  actor: max_tokens_per_gpu=${PA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU}, activation_offload=${PA_OPD_ENABLE_ACTIVATION_OFFLOAD}; data.max_prompt_length=${PA_OPD_MAX_PROMPT_LENGTH}"
exec bash "${SCRIPT_DIR}/train.sh" "${TRAIN_RECIPE}" \
  '+trainer.ray_max_colocate_count=1' \
  "data.pa_opd_student_image_mode=${PA_OPD_STUDENT_IMAGE_MODE}" \
  "data.pa_opd_teacher_full_image_mode=${PA_OPD_TEACHER_FULL_IMAGE_MODE}" \
  'data.pa_opd_separate_teacher_views=true' \
  'actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images' \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${PA_OPD_ROLLOUT_TP}" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${PA_OPD_VLLM_GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.enforce_eager=${PA_OPD_VLLM_ENFORCE_EAGER}" \
  "actor_rollout_ref.rollout.max_num_seqs=${PA_OPD_VLLM_MAX_NUM_SEQS}" \
  "actor_rollout_ref.rollout.max_model_len=${PA_OPD_VLLM_MAX_MODEL_LEN}" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=${PA_OPD_VLLM_MAX_NUM_BATCHED_TOKENS}" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.model.enable_activation_offload=${PA_OPD_ENABLE_ACTIVATION_OFFLOAD}" \
  "data.max_prompt_length=${PA_OPD_MAX_PROMPT_LENGTH}" \
  "${rollout_profile_args[@]}" \
  "$@"
