<p align="center">
  <img src="figures/logo_01.png" width="72" alt="RS-OPSD logo: a globe with a magnifying glass" />
</p>

<h1 align="center">
  <img src="figures/rs_opsd_title.svg" width="1000" alt="RS-OPSD: Reliable Privileged On-Policy Self-Distillation for Ultra-High-Resolution Remote Sensing VQA" />
</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/ronniejiangC/GeoEvidence-6K"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-GeoEvidence--6K-6F7A3A" alt="Hugging Face dataset: GeoEvidence-6K" /></a>
  <a href="https://huggingface.co/ronniejiangC/RS-OPD-Lite"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-RS--OPD--Lite%20%282B%29-6F7A3A" alt="Hugging Face model: RS-OPD-Lite (2B)" /></a>
  <a href="https://huggingface.co/ronniejiangC/RS-OPSD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-RS--OPSD%20%288B%29-6F7A3A" alt="Hugging Face model: RS-OPSD (8B)" /></a>
  <a href="#paper-and-project-page"><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-B31B1B?logo=arxiv&amp;logoColor=white" alt="arXiv paper: coming soon" /></a>
  <a href="#paper-and-project-page"><img src="https://img.shields.io/badge/Project%20Page-Coming%20Soon-777777?logo=githubpages&amp;logoColor=white" alt="Project page: coming soon" /></a>
</p>

## Leaderboard

<p align="center">
  <img src="figures/leaderboard_01.png" width="1200" alt="Leaderboard comparing RS-OPSD (49.3), RS-OPD-Lite (44.2), and other models" />
</p>

Average scores across **XLRS-Bench**, **MME-RealWorld-RS**, and **LRS-VQA**.
RS-OPSD achieves the best results on all three benchmarks among the methods
compared in the paper, improving the average score over the strongest competing
method, ZoomSearch, by **4.0 percentage points**.

| Model | Inference model size | XLRS-Bench | MME-RealWorld-RS | LRS-VQA | Average | Latency (s/sample) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B-Instruct | 8B | 50.5 | 41.9 | 30.1 | 40.8 | 1.75 |
| RS-OPD-Lite | 2B | 45.8 | 56.2 | 30.5 | 44.2 | **1.09** |
| RS-OPSD | 8B | **53.1** | **61.5** | **33.3** | **49.3** | 1.58 |

Latency is measured on a **single NVIDIA H100 with batch size 1**, averaged
across the three benchmarks under the paper's evaluation protocol.
RS-OPD-Lite surpasses most evaluated 8B-scale models and achieves the lowest
measured latency—**12.8% lower** than the second-fastest method.
Neither variant requires additional visual search or external tool calls at
inference time.

## Method overview

<table>
  <tr>
    <td><em>Can the benefit of zoom-in evidence be internalized during training, rather than requiring visual search or tool use at inference time?</em></td>
  </tr>
</table>

<p align="center">
  <img src="figures/RS-OPSD_01.png" width="1200" alt="RS-OPSD architecture: global, contextual and fine-grained evidence for CPVP; teacher reliability probing, correctness-aligned token filtering and reference KL regularization for CAD" />
</p>

**Context-Preserving Visual Privilege (CPVP): what should the teacher see?**
The teacher receives three complementary views: **Global Evidence** preserves
scene-level context, **Contextual Evidence** retains the target and its
surroundings, and **Fine-grained Evidence** provides a tight target crop.
The student and frozen reference model receive only the global view.

**Correctness-Aligned Distillation (CAD): when should the student trust it?**
The student generates one on-policy response. Instead of generating a separate
trajectory, the privileged teacher evaluates the same student-generated prefixes.
CAD filters supervision at two levels:

- **Sample-level reliability:** a teacher-forced ground-truth-prefix probe keeps
  samples only when the teacher selects the GT continuation at every probe
  position within the valid candidate set.
- **Token-level alignment:** for a correct student answer, retain teacher
  preferences that reinforce the sampled tokens; for an incorrect answer,
  retain preferences that suppress them. Unreliable or conflicting signals are
  discarded.

## GeoEvidence-6K

### Dataset construction

<p align="center">
  <img src="figures/HF-SR_01.png" width="1200" alt="GeoEvidence-6K construction using human feedback-guided skill refinement, automated annotation, and evidence-grounded remote-sensing questions" />
</p>

**GeoEvidence-6K** pairs UHR remote sensing questions with explicit
question-relevant evidence regions. It contains **6,750 VQA samples** derived
from **1,050 source images** across five public sources: MiniFrance, HRSCD,
SWISSIMAGE, GeoNRW, and Beeldmateriaal. Source images average
**10,714 × 10,714 pixels**, while annotated target regions average
**1,095 × 1,074 pixels**—roughly **1%** of the full-image area.

**Human Feedback-Guided Skill Refinement (HF-SR)** turns human corrections into
reusable annotation skills, rather than only correcting individual samples.
The annotation agent first examines a downsampled overview, then inspects
candidate regions at native resolution to produce evidence boxes, questions,
and reference answers. Human reviewers check the image, target region, and
annotation together. Recurring errors and corrections are consolidated into
procedural rules for subsequent annotation, with frequent early refinement and
progressively longer intervals as the skill stabilizes.

### Dataset statistics

<p align="center">
  <img src="figures/GeoEvidence6K_01.png" width="1200" alt="GeoEvidence-6K statistics: word cloud, token-length distribution, and task distribution across shape, color, position, category, counting, route planning and land use" />
</p>

The seven task categories span five **local-evidence tasks**—counting, position,
color, category, and shape—and two **global-context tasks**—land use and route
planning. Each local-evidence task accounts for approximately **15.6%** of the
dataset; each global-context task accounts for approximately **11.1%**.
The reported question-token distribution has a mean of **56.87**, median of
**54**, and 95th percentile of **84**.

The standardized release contains **6,000 single-choice** and **750
multiple-choice** questions, together with four processed image views:
`plain`, `global`, `contextual`, and `evidence`. Original large images are not
included. Download the dataset and find its usage guide on
[Hugging Face](https://huggingface.co/datasets/ronniejiangC/GeoEvidence-6K).

## Paper and project page

The arXiv paper and project page links are **coming soon**. Their badges above
are placeholders and will be updated when the pages are available.

## Repository layout

The repository is organized around the proposed method and a standalone
evaluator. It intentionally excludes datasets, checkpoints,
experiment logs, and other generated artifacts.

| Path | Purpose |
| --- | --- |
| `rs-opsd/` | RS-OPSD training implementation, configurations, dependencies, and runtime launchers. |
| `eval/` | OpenAI-compatible evaluator for LRS-VQA, MME-RealWorld Remote Sensing, and XLRS-Bench. |
| `eval/scripts/` | Local evaluation and environment launchers. |


## RS-OPSD training

The main experiments train exclusively on GeoEvidence-6K with **one rollout
per sample**, a **global batch size of 96**, **KL coefficient 0.001**, and
**gradient-norm clipping threshold 5**. RS-OPSD is trained for **150 steps**;
RS-OPD-Lite is trained for **120 steps**. These are the paper's reported settings;
select the appropriate model, teacher-update policy, and runtime configuration
when reproducing either variant.

The released training recipes use **2K images + CAD + reference KL** only:
direct answers, one rollout per sample, and no reasoning tags or format reward.
The two-view recipe is a visual-privilege ablation; the three-view recipe is
the main method. Teacher weights can be EMA-updated or frozen; the reference
model stays frozen and receives the Student's input.

### Environment

Run from the repository root, using Python 3.12 and a compatible accelerator:

```bash
cd rs-opsd
python3.12 -m venv .venv
.venv/bin/python -m pip install --use-pep517 -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/preflight_runtime.py --require-devices 4
```

`requirements-core.txt` lists application dependencies; `requirements-runtime.txt`
pins the upstream PyTorch/vLLM/Ray stack; `requirements.txt` includes both.
Upstream vLLM 0.18 uses Transformers 4.x (at least 4.57.3). The runtime uses
PyTorch's CUDA API, NCCL-compatible collectives and vLLM; choose builds matched
to your hardware and driver. SDPA with padding removal disabled is used for
the actor, so FlashAttention is not an unconditional actor dependency.

For a prebuilt accelerator environment, keep its compatible runtime. First
check `python -m pip install --dry-run -r requirements-core.txt`, applying
your platform's constraints with `-c` if needed. Install core dependencies only
after checking transitive runtime changes, then register this fork using
`python -m pip install --no-deps -e .` and run `python -m pip check`.
The preflight checks dependencies/devices, not end-to-end training.

### Data and image views

Training loads existing 2K views; it does not generate images. The local
training layout is:

```text
DATA_ROOT/
  final_annotations.json
  train.jsonl
  images/                    # plain full images
  bbox_images/               # boxed full images, when selected
  teacher_images/            # tight evidence crops
  derived/                   # required for three-view recipes
    train.jsonl
    teacher_images/          # contextual crops with evidence boxes
```

Each training JSONL row provides one `images` path, one `teacher_images` path,
a multiple-choice `problem`, and an `answer`. Paths are relative to the dataset
root. The derived manifest must be positionally aligned with the main manifest.
Teacher view order is global, contextual (three-view only), then evidence.
The Student and reference use only the selected global view.

The public dataset calls these views `plain`, `global`, `contextual`, and
`evidence`, respectively. Its standardized `metadata.jsonl` is **not** a
drop-in replacement for the training manifest. Prepare the local layout and
manifests before launching; do not pass release metadata as `train.jsonl`.
For original annotation-based manifests, see
`scripts/rebuild_geoevidence_train_jsonl.py --help` and
`scripts/rebuild_geoevidence_derived_train_jsonl.py --help` (inside `rs-opsd/`).

### Launch

From `rs-opsd/`, after activating the training environment:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/GeoEvidence-6K \
OUTPUT_DIR=/path/to/output \
PYTHON=.venv/bin/python \
TRAIN_RECIPE=direct-2k-three-image-kl \
PA_OPD_GPUS_PER_NODE=4 PA_OPD_NNODES=1 \
PA_OPD_STUDENT_IMAGE_MODE=bbox_images \
PA_OPD_TEACHER_FULL_IMAGE_MODE=bbox_images \
bash scripts/train_geoevidence.sh
```

The image-mode variables accept `images` (plain) or `bbox_images` (boxed).
The launcher explicitly routes Teacher views separately from Student views.
All paths and resources are supplied at runtime; no scheduler submission
scripts are required.

| Recipe | Teacher views | Notes |
| --- | --- | --- |
| `direct-2k-kl` | Global + evidence | Two-view ablation |
| `direct-2k-three-image-kl` | Global + contextual + evidence | Main method |
| `direct-2k-three-image-kl-32gpu` | Global + contextual + evidence | Batch 96, three epochs, 2 × 16 GPUs |
| `direct-2k-kl-mixed` | Global + evidence | Optional extra data via `PA_OPD_VISIONOPD_ROOT` |

All four use the same CAD + KL objective. Paper settings above are not all
launcher defaults: pass Hydra overrides for the desired experiment, e.g.
`trainer.total_training_steps=150 actor_rollout_ref.actor.grad_clip=5`.
The base two-/three-view configs use batch 32 and one epoch; the 32-GPU preset
sets batch 96 and three epochs. KL defaults to 0.001.

For 32 GPUs, run the launcher on each of two allocated 16-GPU nodes with
`TRAIN_RECIPE=direct-2k-three-image-kl-32gpu`, `PA_OPD_GPUS_PER_NODE=16`,
`PA_OPD_NNODES=2`, and `PA_OPD_SAVE_FREQ=30` (optimizer steps).
Set a shared unique `PA_OPD_LAUNCH_ID`, shared `MASTER_ADDR` / `MASTER_PORT`,
and each node's `NODE_RANK`. Keep model/data/output paths identical across nodes.
Batch sizes must be compatible with the topology.

For a 2B Student with a frozen 8B Teacher, use the 2B path as `MODEL_PATH`
and append:

```text
actor_rollout_ref.actor.self_distillation.teacher_model_path=/path/to/Qwen3-VL-8B-Instruct
actor_rollout_ref.actor.self_distillation.fixed_teacher_ema=false
```

`PA_OPD_MEMORY_PROFILE=low` enables activation offload, one concurrent rollout
sequence and a smaller rollout context. `PA_OPD_ROLLOUT_TP` must divide GPUs
per node. These settings reduce memory pressure but do not guarantee freedom
from OOM. Each training rank reserves a whole Ray GPU; an assignment check
rejects duplicate devices before model loading.

### Loss reduction and checkpoints

For sampled-token probabilities, the retained objective is

$$r_{i,t}=\operatorname{sg}[\log p^T_{i,t}-\log p^S_{i,t}],\qquad
A_{i,t}=g_iR_i[R_ir_{i,t}]_+,$$

$$\mathcal L_{\mathrm{CAD}}=
-\frac{\sum_{i,t}m^{\mathrm{ans}}_{i,t}\operatorname{sg}[\rho_{i,t}]
\operatorname{sg}[A_{i,t}]\log p^S_{i,t}}{\sum_{i,t}m^{\mathrm{ans}}_{i,t}},
\qquad \mathcal L=\mathcal L_{\mathrm{CAD}}+\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.$$

Here \(g_i\) is Teacher reliability, \(R_i\) is +1 for an exactly correct
normalized answer set and −1 otherwise, and \(\rho_{i,t}\) is the detached,
clipped policy importance weight including rollout correction.

CAD uses a Teacher GT-prefix probe (including the final EOS), then keeps only
sampled-token preferences aligned with Student answer correctness. Valid
label-bearing answer tokens form its denominator; EOS and punctuation-only
tokens do not. Teacher presence masks the denominator, while reliability,
sign selection and importance weights affect only the numerator.
Reference KL uses the full response mask.

Both terms use **global token mean per optimizer mini-batch**, across all DP
ranks and accumulated micro-batches. Two token counts are reduced once before
micro-batch splitting. Each rank backpropagates
`DP_size × local_masked_sum / max(global_token_count, 1)`; FSDP's gradient
average cancels the DP-size factor. There is no extra division by accumulation
steps. The supported reduction path requires Ulysses SP=1.
Monitor `actor/cad_loss`, `actor/kl_loss`, `actor/grad_norm` and
`pa_opd/global_{semantic,response}_tokens`.

Resume by adding `RESUME_FROM_PATH=/path/to/checkpoints/global_step_N`
to the launch environment with the same topology and matching configuration.
The launcher validates actor/optimizer/extra state, Teacher shards and data
state. A standalone inference export cannot replace a full resume checkpoint.
The 32-GPU preset also exports Hugging Face weights under `OUTPUT_DIR/inference`;
keep full checkpoints separately for resuming. Model/optimizer state layout
is unchanged by this cleanup.

`WANDB_MODE` defaults to `offline`. Persistent checkpoints, rollouts and logs
go under `OUTPUT_DIR`; rebuildable caches and Ray sockets use short local
temporary paths. Internal `pa_opd_*` fields and `PA_OPD_*` variables remain
for compatibility. Use a fresh output directory for a new experiment.

## Evaluation

Run evaluation commands from the repository root. Install the lightweight API
client dependencies with `python -m pip install -r eval/requirements.txt`.
Remote evaluation does not need local PyTorch or vLLM. For local Transformers
inference, LoRA merging or optional semantic scoring, install hardware-matched
PyTorch/torchvision and `eval/requirements-local.txt`. Local vLLM serving
additionally needs a compatible vLLM build.

| Benchmark | Local layout |
| --- | --- |
| LRS-VQA | `LRS_VQA_merged.jsonl`; released image paths or flattened `image/` |
| MME-RealWorld-RS | `MME_RealWorld.json` with relative image paths; English Perception / Remote Sensing subset |
| XLRS-Bench | `datasets.load_from_disk` directory containing the benchmark's `train` split |

```bash
python -m eval.run --dataset lrs-vqa --lrs-root /path/to/LRS-VQA --dry-run

python -m eval.run \
  --dataset lrs-vqa,mme-realworld-rs,xlrs-bench \
  --lrs-root /path/to/LRS-VQA \
  --mme-root /path/to/MME-RealWorld \
  --xlrs-root /path/to/XLRS-Bench \
  --api-base http://localhost:8000/v1 \
  --model-id your-served-model-name \
  --parallel-workers 32 --run-name evaluation
```

Results are written incrementally to `eval/results/<run-name>/` as per-benchmark
JSONL files plus `summary.json`. Rerun with `--resume` to skip completed IDs.
Use `python -m eval.run --help` and `eval/.env.example` for all options.
The default image-area cap is 16,777,216 pixels; a 4K input can require about
15.7K visual tokens, so allow sufficient model context or reduce `--max-pixels`.

Local launchers:

- `eval/scripts/run_local_vllm.sh`: serve and evaluate a Hugging Face checkpoint.
- `eval/scripts/run_local_transformers.sh`: direct single-sample inference fallback.
- `eval/scripts/run_openai_compatible.sh`: evaluate an existing endpoint.
- `eval/scripts/probe_vllm_runtime.sh`: check local devices and vLLM support.

For example:

```bash
MODEL_PATH=/path/to/huggingface-model \
SERVED_MODEL_NAME=rs-opsd BENCHMARK=lrs-vqa \
LRS_ROOT=/path/to/LRS-VQA TP_SIZE=1 \
bash eval/scripts/run_local_vllm.sh
```

To export an FSDP actor checkpoint for inference:

```bash
VERL_ROOT="$PWD/rs-opsd" PYTHON=/path/to/training-python \
bash eval/scripts/merge_fsdp_checkpoint_to_hf.sh /path/to/global_step_N
```

`eval/scripts/merge_lora_adapter_to_hf.sh` handles PEFT adapters instead.
Optional LRS-VQA tolerant scoring accepts
`--lrs-semantic-model /path/to/bge-base-en-v1.5 --lrs-semantic-threshold 0.85`;
download that model separately. Strict accuracy stays unchanged, and boolean
or numeric answers remain exact-only.

## Reproducibility and artifacts

Datasets, model weights, checkpoints, W&B data, Ray output, and evaluation
results are deliberately ignored by Git. Supply local paths through the
environment variables shown above. This repository contains no cluster- or
scheduler-specific submission scripts.

## License

RS-OPSD is distributed under the [Apache-2.0 license](rs-opsd/LICENSE).

## Acknowledgements

We thank the teams behind the following works for their contributions and open resources:

- **XLRS-Bench** — [Paper](https://arxiv.org/abs/2503.23771) · [Code and dataset](https://github.com/AI9Stars/XLRS-Bench)
- **MME-RealWorld** — [Paper](https://arxiv.org/abs/2408.13257) · [Project page](https://mme-realworld.github.io/)
- **LRS-VQA** — [Paper](https://arxiv.org/abs/2503.07588) · [Code and dataset](https://github.com/VisionXLab/LRS-VQA)
- **Vision-OPD** — [Paper](https://arxiv.org/abs/2605.18740) · [Code](https://github.com/VisionOPD/Vision-OPD)
