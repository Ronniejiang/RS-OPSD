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
  <a href="#paper"><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-B31B1B?logo=arxiv&amp;logoColor=white" alt="arXiv paper: coming soon" /></a>
  <a href="#paper"><img src="https://img.shields.io/badge/Project%20Page-Coming%20Soon-777777?logo=githubpages&amp;logoColor=white" alt="Project page: coming soon" /></a>
</p>

## Paper

### Results

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

The paper and project page are coming soon; the badges above link to the
released dataset and models.

### Method

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

### GeoEvidence-6K

<p align="center">
  <img src="figures/HF-SR_01.png" width="1200" alt="GeoEvidence-6K construction using human feedback-guided skill refinement, automated annotation, and evidence-grounded remote-sensing questions" />
</p>

**GeoEvidence-6K** pairs UHR remote sensing questions with explicit
question-relevant evidence regions. It contains **6,750 VQA samples** derived
from **1,050 source images** across five public sources: MiniFrance, HRSCD,
SWISSIMAGE, GeoNRW, and Beeldmateriaal. Source images average
**10,714 × 10,714 pixels**, while annotated target regions average
**1,095 × 1,074 pixels**—roughly **1%** of the full-image area.

**Human Feedback-Guided Skill Refinement (HF-SR)** turns annotation feedback
into reusable rules for constructing evidence-grounded questions.

<p align="center">
  <img src="figures/GeoEvidence6K_01.png" width="1200" alt="GeoEvidence-6K statistics: word cloud, token-length distribution, and task distribution across shape, color, position, category, counting, route planning and land use" />
</p>

The dataset covers counting, position, color, category, shape, land use and
route planning.

The standardized release contains **6,000 single-choice** and **750
multiple-choice** questions, together with four processed image views:
`plain`, `global`, `contextual`, and `evidence`. Original large images are not
included. Download the dataset and find its usage guide on
[Hugging Face](https://huggingface.co/datasets/ronniejiangC/GeoEvidence-6K).

## Environment setup

Use Python 3.12 and a CUDA/NCCL-compatible GPU runtime. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --use-pep517 -r rs-opsd/requirements.txt
python -m pip install --no-deps -e rs-opsd
python -m pip install -r eval/requirements.txt
```

Training dependencies are split into application packages
(`requirements-core.txt`) and the PyTorch/vLLM runtime
(`requirements-runtime.txt`); `requirements.txt` installs both.
For a prebuilt accelerator environment, preserve its compatible runtime
and review a dry run before installing core dependencies instead.

Check the allocated devices before training; replace `8` with your local GPU count:

```bash
python rs-opsd/scripts/preflight_runtime.py --require-devices 8
```

## Training

### Prepare the data

The main method uses 2K images: the student sees the boxed global view, and
the teacher sees global, contextual and fine-grained evidence in that order.
Prepare this local training layout:

```text
GeoEvidence-6K/
├── final_annotations.json
├── train.jsonl
├── images/                   # plain
├── bbox_images/              # global
├── teacher_images/           # evidence
└── derived/
    ├── train.jsonl
    └── teacher_images/       # contextual
```

Each training record contains `problem`, `answer`, `images` and
`teacher_images`, with image paths relative to the dataset root. The derived
manifest must preserve the main manifest's row order.

The Hugging Face release's `metadata.jsonl` is not directly accepted as a
training manifest. The commands below assume the training annotation schema
and image layout above. If `final_annotations.json` and the processed images
are already available, rebuild the manifests with:

```bash
python rs-opsd/scripts/rebuild_geoevidence_train_jsonl.py \
  --data-root /path/to/GeoEvidence-6K
python rs-opsd/scripts/rebuild_geoevidence_derived_train_jsonl.py \
  --data-root /path/to/GeoEvidence-6K
```

These commands replace the corresponding manifests; they do not generate images.

### Train RS-OPSD

Run from the repository root with the environment activated:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/GeoEvidence-6K \
OUTPUT_DIR=/path/to/outputs/rs-opsd \
TRAIN_RECIPE=direct-2k-three-image-kl \
PA_OPD_GPUS_PER_NODE=8 PA_OPD_NNODES=1 \
PA_OPD_STUDENT_IMAGE_MODE=bbox_images \
PA_OPD_TEACHER_FULL_IMAGE_MODE=bbox_images \
bash rs-opsd/scripts/train_geoevidence.sh \
  data.train_batch_size=96 \
  actor_rollout_ref.actor.ppo_mini_batch_size=96 \
  actor_rollout_ref.actor.grad_clip=5 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  trainer.total_epochs=3 \
  trainer.total_training_steps=150
```

This uses one rollout per sample, global batch size 96, CAD with global
token-mean reduction, and KL regularization toward a frozen reference.
The privileged teacher is updated by EMA. Adjust the GPU count and memory
settings to your hardware; `PA_OPD_MEMORY_PROFILE=low` and
`PA_OPD_ROLLOUT_TP=2` are available for memory-constrained runs
(the TP size must divide GPUs per node).

For RS-OPD-Lite, use Qwen3-VL-2B-Instruct as `MODEL_PATH`, set the training
step limit to `120`, and append these overrides to use a frozen 8B teacher:

```text
actor_rollout_ref.actor.self_distillation.teacher_model_path=/path/to/Qwen3-VL-8B-Instruct
actor_rollout_ref.actor.self_distillation.fixed_teacher_ema=false
```

### Resume and export

Checkpoints, rollouts and offline W&B logs are saved under `OUTPUT_DIR`.
To resume, rerun the same training command with
`RESUME_FROM_PATH=/path/to/checkpoints/global_step_N` set in the environment.
Keep the topology and configuration consistent. Resume requires the complete
checkpoint, including optimizer and teacher state, not just inference weights.

Export a training checkpoint to Hugging Face format for inference:

```bash
VERL_ROOT="$PWD/rs-opsd" PYTHON="$(command -v python)" \
bash eval/scripts/merge_fsdp_checkpoint_to_hf.sh \
  /path/to/checkpoints/global_step_N \
  /path/to/exported-model
```

## Evaluation

Evaluation supports XLRS-Bench, MME-RealWorld-RS and LRS-VQA using their
original benchmark inputs.

| Benchmark | Expected local files |
| --- | --- |
| XLRS-Bench | A `datasets.load_from_disk` directory with the benchmark's `train` split |
| MME-RealWorld-RS | `MME_RealWorld.json` and its referenced images |
| LRS-VQA | `LRS_VQA_merged.jsonl` and the released image paths or `image/` directory |

With the environment above activated, this command starts a local vLLM server,
evaluates all three benchmarks and stops the server afterward:

```bash
MODEL_PATH=/path/to/RS-OPSD-or-exported-model \
SERVED_MODEL_NAME=rs-opsd \
BENCHMARK=all \
LRS_ROOT=/path/to/LRS-VQA \
MME_ROOT=/path/to/MME-RealWorld \
XLRS_ROOT=/path/to/XLRS-Bench \
TP_SIZE=1 MAX_MODEL_LEN=32768 \
RUN_NAME=rs-opsd-eval \
bash eval/scripts/run_local_vllm.sh
```

Set `TP_SIZE` for the available GPUs. Results are written to
`eval/results/rs-opsd-eval/` as per-benchmark JSONL files and `summary.json`.
Set `RESUME=1` to skip completed samples when rerunning.

For an existing OpenAI-compatible endpoint, use `python -m eval.run` with
`--api-base` and `--model-id`; only `eval/requirements.txt` is required locally.
See `python -m eval.run --help` and `eval/.env.example` for dataset paths,
image limits and other evaluation options.

## Acknowledgements

We thank the teams behind the following works for their contributions and open resources:

- **XLRS-Bench** — [Paper](https://arxiv.org/abs/2503.23771) · [Code and dataset](https://github.com/AI9Stars/XLRS-Bench)
- **MME-RealWorld** — [Paper](https://arxiv.org/abs/2408.13257) · [Project page](https://mme-realworld.github.io/)
- **LRS-VQA** — [Paper](https://arxiv.org/abs/2503.07588) · [Code and dataset](https://github.com/VisionXLab/LRS-VQA)
- **Vision-OPD** — [Paper](https://arxiv.org/abs/2605.18740) · [Code](https://github.com/VisionOPD/Vision-OPD)

## License

RS-OPSD is distributed under the [Apache-2.0 license](rs-opsd/LICENSE).
