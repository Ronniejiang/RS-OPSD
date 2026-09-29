<p align="center">
  <img src="figures/logo_01.png" width="180" alt="RS-OPSD logo: a globe with a magnifying glass" />
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

<p align="center">
  <a href="rs-opsd/README.md">Training</a> ·
  <a href="eval/README.md">Evaluation</a> ·
  <a href="#method-overview">Method</a> ·
  <a href="#geoevidence-6k">Dataset</a>
</p>

## Leaderboard

<p align="center">
  <img src="figures/leaderboard_01.png" width="1200" alt="Leaderboard comparing RS-OPSD (49.3), RS-OPD-Lite (44.2), and other models" />
</p>

RS-OPSD and its lightweight variant, RS-OPD-Lite, achieve scores of **49.3** and
**44.2**, respectively, in the comparison shown above.

## Method overview

RS-OPSD combines **Context-Preserving Visual Privilege (CPVP)** and
**Correctness-Aligned Distillation (CAD)** for remote-sensing visual question
answering. This repository contains its training and evaluation code.

<p align="center">
  <img src="figures/RS-OPSD_01.png" width="1200" alt="RS-OPSD architecture: global, contextual and fine-grained evidence for CPVP; teacher reliability probing, correctness-aligned token filtering and reference KL regularization for CAD" />
</p>

CPVP provides the teacher with global, contextual, and fine-grained evidence,
while the student observes only the global view. CAD evaluates the
student-generated trajectory using the privileged teacher and filters
supervision through a ground-truth-prefix teacher reliability probe and
correctness-aligned token signals. A frozen reference model regularizes the
student through a KL term.

## GeoEvidence-6K

### Dataset construction

<p align="center">
  <img src="figures/HF-SR_01.png" width="1200" alt="GeoEvidence-6K construction using human feedback-guided skill refinement, automated annotation, and evidence-grounded remote-sensing questions" />
</p>

GeoEvidence-6K is constructed with **Human Feedback-Guided Skill Refinement
(HF-SR)**, using feedback to refine the annotation process and encourage
questions that require fine-grained visual evidence. The figure illustrates
the construction workflow and example evidence regions and questions.

### Dataset statistics

<p align="center">
  <img src="figures/GeoEvidence6K_01.png" width="1200" alt="GeoEvidence-6K statistics: word cloud, token-length distribution, and task distribution across shape, color, position, category, counting, route planning and land use" />
</p>

The release includes **6,750 QA samples**: **6,000 single-choice** and
**750 multiple-choice** questions. The figure summarizes question vocabulary,
token lengths, and task coverage. Download the dataset and find its usage guide
on [Hugging Face](https://huggingface.co/datasets/ronniejiangC/GeoEvidence-6K).

## Paper and project page

The arXiv paper and project page links are **coming soon**. Their badges above
are placeholders and will be updated when the pages are available.

## Repository layout

The repository is organized around the proposed method and a standalone
evaluator. It intentionally excludes datasets, checkpoints,
experiment logs, and other generated artifacts.

| Path | Purpose |
| --- | --- |
| `rs-opsd/` | RS-OPSD training implementation, configurations, unit tests, and runtime launchers. |
| `eval/` | OpenAI-compatible evaluator for LRS-VQA, MME-RealWorld Remote Sensing, and XLRS-Bench. |
| `eval/scripts/` | Local evaluation and environment launchers. |


## RS-OPSD training

Use a compatible PyTorch/vLLM environment, a local Qwen3-VL model, and the
GeoEvidence dataset layout described in [the training guide](rs-opsd/README.md).
Install dependencies appropriate for your accelerator platform, then run:

    cd rs-opsd
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

    MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
    DATA_ROOT=/path/to/GeoEvidence-6K \
    OUTPUT_DIR=/path/to/output \
    PYTHON=.venv/bin/python PA_OPD_NUM_GPUS=4 \
    bash scripts/train.sh direct-2k-three-image-kl

`DATA_ROOT` must contain `train.jsonl`, `images/`, and `teacher_images/`;
the three-view recipe additionally needs `derived/train.jsonl` and
`derived/teacher_images/`. Set `OUTPUT_DIR` explicitly for generated artifacts.
The directory is now `rs-opsd/`; internal `pa_opd_*` fields and `PA_OPD_*`
environment variables retain their names for compatibility. Existing data and
checkpoint paths are unchanged.

## Evaluation

The evaluator sends requests to an OpenAI-compatible model server and supports
LRS-VQA, MME-RealWorld Remote Sensing, and XLRS-Bench. Install its lightweight
client dependencies and point it at a running server:

    python3 -m venv .venv-eval
    .venv-eval/bin/pip install openai pillow datasets
    .venv-eval/bin/python -m eval.run --dataset lrs-vqa --lrs-root /path/to/lrs-vqa --api-base http://localhost:8000/v1 --model-id your-served-model-name

To create a dedicated vLLM environment and serve a local checkpoint first, run
MODEL_PATH=/path/to/model LRS_ROOT=/path/to/lrs-vqa BENCHMARK=lrs-vqa bash
eval/scripts/run_local_vllm.sh. Use --dry-run with eval.run to validate a
dataset layout before starting a server. Full options are in eval/README.md.

## Reproducibility and artifacts

Datasets, model weights, checkpoints, W&B data, Ray output, and evaluation
results are deliberately ignored by Git. Supply local paths through the
environment variables shown above. This repository contains no cluster- or
scheduler-specific submission scripts.

## License

RS-OPSD is distributed under the [Apache-2.0 license](rs-opsd/LICENSE).
