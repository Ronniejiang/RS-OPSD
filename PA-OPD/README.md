# PA-OPD

PA-OPD is an independent implementation of privileged-advantage on-policy
visual distillation, derived from the local OPD-V baseline. The original
`other_methods/OPD-V` directory is not modified.

## Method

- **Student rollout:** sees only the original image and produces an explicit
  `<think>...</think><answer>...</answer>` response.
- **EMA teacher:** sees the original image followed by the evidence-centered
  crop, then evaluates the same Student-generated prefix token by token.
- **Pre-rollout answer probe:** both models are prompted directly at
  `<answer>`; candidate `A/B/C/D` logits are normalized within the option set.
  The detached PA weight is `1[teacher argmax = GT] * max(V_teacher - V_student, 0)`.
- **Reasoning-only OPD:** that sample weight is applied only to content tokens
  in a strictly well-formed `<think>` block.
- **Format-RLVR:** a separate binary reward checks only one complete and ordered
  set of the four tags. Its GRPO loss is weighted by `0.1`; answer correctness
  is never used as RL reward.

## Quick start

All training semantics are in verl/trainer/config/pa_opd.yaml: the Qwen3-VL
backbone, four-GPU PA-OPD teacher/probe settings, thinking, format reward, and
output paths. Create a CUDA-compatible environment, then run:

    cd PA-OPD
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

    MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct SOURCE_DATA_DIR=/path/to/Vision-OPD-6K CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_pa_opd.sh

MODEL_PATH is the local Qwen3-VL model directory. SOURCE_DATA_DIR must contain
train.jsonl, images/, and teacher_images/; data is read in place and is not
copied into this repository. Generated state (checkpoints, rollouts, Hugging
Face cache, and W&B cache) is written to outputs/<run-name>/ by default; set
WORK_ROOT=/path/to/output to use another location. RAY_TMPDIR defaults to
/tmp/paopd-ray to avoid the Unix-socket path-length limit.

Optional Hydra overrides remain available, for example:

    MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct SOURCE_DATA_DIR=/path/to/Vision-OPD-6K bash scripts/run_pa_opd.sh trainer.total_epochs=2
