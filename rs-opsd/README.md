# RS-OPSD

RS-OPSD combines **Context-Preserving Visual Privilege (CPVP)** and
**Correctness-Aligned Distillation (CAD)** for remote-sensing VQA. The current
training protocol is reward-free, direct-answer, single-rollout, and no-thinking.
The Student generates an on-policy answer from global evidence. The Teacher
scores that same answer prefix with privileged views; it does not generate a
separate distillation trajectory. The Teacher can be EMA-updated or frozen,
including a larger frozen backbone; the KL Reference remains frozen.

The three-view CAD + KL recipe is the main method. Two-view recipes are visual
privilege ablations; JSD recipes are optional objective extensions:

| Recipe | Student input | Privileged Teacher input | Objective |
| --- | --- | --- | --- |
| `direct-2k-kl` | full image | full image + evidence crop | CAD policy-gradient loss + frozen-reference low-variance KL (1e-3) |
| `direct-2k-topk64-jsd-kl` | full image | full image + evidence crop | `direct-2k-kl` plus GT-safe shared-support top-64 JSD |
| `direct-2k-three-image-kl` | full image | full image + derived red-box crop + tight crop | `direct-2k-kl`, with a larger Teacher re-prompt budget |

## CPVP: What Should the Teacher See?

The ordered views are Global Evidence $I_G$ (full image), Contextual Evidence
$I_C$ (expanded red-box crop), and Fine-grained Evidence $I_F$ (tight crop).
The method defines $I_C$ by doubling the tight region's width and height about
the same center. Training loads pre-generated images; it neither regenerates
them nor verifies the expansion geometry. Student and Reference see only
$I_G$; Teacher sees $(I_G,z)$ with $z=\{I_C,I_F\}$.
The current boxed-input recipes use `bbox_images` for $I_G$.

## CAD: When Should the Student Trust It?

The sample-level gate $g_i=\operatorname{Probe}_T(a_i^*)$ is a **GT-prefix**
reliability check, not free-generation accuracy. At the first answer position,
the Teacher chooses among option labels; at every subsequent position it may
stop with EOS or append a later label in canonical order. Every GT transition,
including the final EOS, must win the constrained argmax. After the first label,
ties prefer EOS. No GT answer is inserted into the user question.

For sampled Student token $y_{i,t}$, define

$$r_{i,t}=\operatorname{sg}[\log p^T_{i,t}-\log p^S_{i,t}],\qquad
A_{i,t}=g_iR_i[R_ir_{i,t}]_+,$$

where $R_i=+1$ for an exactly correct normalized answer set, otherwise $-1$.
Correct answers retain positive Teacher preferences; incorrect answers retain
negative preferences. These gates only mask CAD, not Reference KL:

$$\mathcal L_{\mathrm{CAD}}=-\frac{\sum_{i,t}m^{\mathrm{ans}}_{i,t}
\operatorname{sg}[\rho_{i,t}]\operatorname{sg}[A_{i,t}]\log p^S_{i,t}}
{\sum_{i,t}m^{\mathrm{ans}}_{i,t}},\qquad
\mathcal L=\mathcal L_{\mathrm{CAD}}+\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.$$

The answer mask selects label-bearing BPE tokens in valid direct answers,
excluding EOS and punctuation-only tokens. Invalid answers have an empty mask.
The denominator includes the Teacher-presence mask but not reliability or sign
gates, and is global across DP ranks and accumulated micro-batches per optimizer
mini-batch. Importance weights retain their existing detached, clipped policy
ratio and rollout correction. KL uses the response mask, not the CAD mask.
Gradient-norm clipping uses `actor_rollout_ref.actor.grad_clip` (not PPO ratio clipping).

The optional JSD extension uses shared top-64 support including legal GT
continuations; non-GT legal Teacher continuation mass is removed before JSD.
It adds a separate weighted term and is not part of the main CAD + KL equation.

The active direct recipes have no reasoning tags, format reward, RLVR, or adaptive
rollout controller. `scripts/train.sh` is portable; cluster submission wrappers
are local-only and are not included in the public repository.

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
checks every rank's actor model, optimizer, extra-state, separate Teacher shard, and
data state before starting the distributed job:

```bash
RESUME_FROM_PATH=/path/to/checkpoints/global_step_100 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/rs_opd \
OUTPUT_DIR=/path/to/output \
bash scripts/train.sh direct-2k-kl
```

## Validate the installation

Check the runtime dependencies and visible devices before training:

```bash
python scripts/preflight_runtime.py --require-devices 4
```

Set `--require-devices` to the number of devices allocated on the local node.
This preflight is not an end-to-end training test.

For breaking interface changes, metric names and the intermediate-EOS probe
fix, see [the terminology migration notes](docs/rs_opsd_cad_migration.md).

PA-OPD loss reduction now uses optimizer-step global token means across DP
ranks and micro-batches. See [the reduction specification](docs/global_token_mean.md)
for the separate CAD/JSD/KL denominators and prior validation coverage.

## Portable GPU training

Private scheduler submission scripts, container-build wrappers, experiment logs,
and local environments are intentionally excluded from the public repository.
Use the runtime launcher inside an environment with the required dependencies
and already allocated devices. Supply all model, data and output paths explicitly:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/GeoEvidence-6K \
OUTPUT_DIR=/path/to/output \
PA_OPD_GPUS_PER_NODE=16 PA_OPD_NNODES=1 \
PA_OPD_STUDENT_IMAGE_MODE=bbox_images PA_OPD_TEACHER_FULL_IMAGE_MODE=bbox_images \
PA_OPD_MEMORY_PROFILE=low PA_OPD_ROLLOUT_TP=2 \
bash scripts/train_geoevidence.sh
```

For the 32-device, batch-96, three-epoch three-view recipe, run the same
launcher on each of two allocated 16-device nodes, setting
`TRAIN_RECIPE=direct-2k-three-image-kl-32gpu`, `PA_OPD_NNODES=2`,
`PA_OPD_SAVE_FREQ=30`, and a shared unique `PA_OPD_LAUNCH_ID`.
Set each node's `NODE_RANK` and the shared `MASTER_ADDR` / `MASTER_PORT`.
Keep model/data/output paths consistent across nodes.

The boxed-full baseline uses `bbox_images/` for Student and Teacher global
views and `teacher_images/` for the Teacher's fine-grained evidence.
For manifest rebuilding, explicitly pass `--data-root` to the rebuild scripts.
Runtime configs and logs contain the paths supplied at launch; review and
redact them separately before publishing logs or syncing them to a service.

The low-memory launch profile uses 16 FSDP ranks, rollout TP=2 (8 replicas),
one concurrent rollout sequence, a 5120-token rollout limit, activation
offload, and `gpu_memory_utilization=0.66`. Student visual profiling is bounded
to one image / 2048² pixels; Teacher's two-image processing remains unchanged.
These memory settings are a launch profile, not a guarantee against OOM.

The public training stack uses PyTorch's CUDA device API, NCCL-compatible
collectives, and vLLM. Install versions compatible with your accelerator and
driver; this is not a promise of CPU or arbitrary accelerator support.
`PA_OPD_GPUS_PER_NODE` accepts any positive count, with rollout TP dividing
that count. Adjust recipe batch sizes and memory budgets for your topology.
The 32-GPU recipe remains an explicit topology preset, not a requirement for
all recipes. Run `scripts/preflight_geoevidence.sh` before starting training.

No vendor-specific communication statistics are changed by default. If your
runtime requires a documented workaround, explicitly set
`RS_OPSD_COMM_STATS_ENV` to its `*_STATS_MODE` variable name and
`RS_OPSD_COMM_STATS_MODE` to the required value; the default `inherit` leaves
the environment untouched. The bounded `scripts/probe_collectives.py` tests
torch/vLLM collectives using the currently configured runtime, without loading
a hard-coded vendor library. Run it only with two allocated, idle devices.

Each hybrid training rank reserves **one whole Ray GPU** through
`trainer.ray_max_colocate_count=1` to prevent unintended rank colocation. A loading-time
assignment check now rejects repeated devices before model weights are loaded.
The optional TP preflight tests collectives without model weights, then exits
and releases its Ray actors before training starts.

The launcher enables FSDP `sync_module_states`: nonzero ranks may initialize
on `meta` only when rank 0 subsequently broadcasts the loaded weights. If
synchronization is explicitly disabled, every rank now loads real CPU weights
instead. Disabling the broadcast while retaining empty meta parameters can
produce uniform logits and zero gradients without a startup exception.
NVTX profiling also maps extended color names to RGB integers, avoiding an
optional matplotlib dependency in the runtime environment.

When monitoring, require the requested number of unique training devices and
a completed training step with sensible outputs and finite gradients; a
scheduler's running status alone does not prove that training has started.
For a read-only device/process memory snapshot on systems providing
`nvidia-smi`, run `python3 scripts/inspect_job_devices.py`.
