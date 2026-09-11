# PA-OPDVR

PA-OPDVR is a reward-free, direct-answer implementation of privileged visual
on-policy distillation for multiple-choice vision-language tasks. The Student
receives the released full image; the EMA Teacher receives that same full image
and additional visual evidence. Both are evaluated on the Student's sampled
answer prefix.

This public directory intentionally supports only the following no-thinking,
single-rollout recipes:

| Recipe | Student input | Privileged Teacher input | Objective |
| --- | --- | --- | --- |
| `direct-2k-kl` | full image | full image + evidence crop | OPDVR policy-gradient loss + frozen-reference low-variance KL (1e-3) |
| `direct-2k-topk64-jsd-kl` | full image | full image + evidence crop | `direct-2k-kl` plus GT-safe shared-support top-64 JSD |
| `direct-2k-three-image-kl` | full image | full image + derived red-box crop + tight crop | `direct-2k-kl`, with a larger Teacher re-prompt budget |

The teacher gate is the constrained multiple-choice GT probe: a privileged
Teacher contribution is used only when its argmax over legal answer
continuations equals the ground-truth answer. OPDVR then turns the sampled-token
Teacher/Student log-probability difference into a detached, GT-signed policy
gradient. The optional JSD recipe additionally distils a shared top-64 support
that always includes every legal GT continuation; non-GT Teacher option mass is
removed before the JSD is evaluated.

There is deliberately no thinking-mode protocol, format reward, RLVR, adaptive
rollout controller, platform submit script, or internal path default.

## Dataset contract

For `direct-2k-kl` and `direct-2k-topk64-jsd-kl`, `DATA_ROOT` must contain:

```text
DATA_ROOT/
  train.jsonl
  images/           # paths referenced by train.jsonl
  teacher_images/   # evidence crop paths referenced by train.jsonl
```

Each JSONL record must provide one `images` path, one `teacher_images` path, a
multiple-choice `problem`, and an `answer`. The launcher checks the directory
layout; the loader validates every referenced image and the option labels.

The three-image recipe additionally requires a positionally aligned derived
manifest and crop directory:

```text
DATA_ROOT/
  derived/
    train.jsonl
    teacher_images/
```

Its Teacher image order is always full image, derived red-box crop, then tight
crop. The Student still sees only the full image.

## Train

Install the project dependencies in your preferred environment, then invoke
the portable launcher from this directory:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/rs_opd \
OUTPUT_DIR=/path/to/output \
PA_OPD_NUM_GPUS=4 \
bash scripts/train.sh direct-2k-kl
```

Select the other supported recipes by replacing the final argument with
`direct-2k-topk64-jsd-kl` or `direct-2k-three-image-kl`. `PYTHON` defaults to
`python`; `WANDB_MODE` defaults to `offline`. Any remaining arguments are
passed through as Hydra overrides, for example:

```bash
bash scripts/train.sh direct-2k-kl trainer.total_epochs=2
```

To resume, point `RESUME_FROM_PATH` at a complete checkpoint. The launcher
checks every rank's actor model, optimizer, extra-state, EMA Teacher shard, and
data state before starting the distributed job:

```bash
RESUME_FROM_PATH=/path/to/checkpoints/global_step_100 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/rs_opd \
OUTPUT_DIR=/path/to/output \
bash scripts/train.sh direct-2k-kl
```

## Validate the installation

The core regression suite does not require `pytest`:

```bash
PYTHONPATH=. python scripts/run_tests.py
```

It covers the direct option protocol, frozen-reference KL configuration,
GT-safe top-k JSD, three-image loader, and checkpoint validator.
