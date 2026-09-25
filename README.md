# RS-OPSD

RS-OPSD combines **Context-Preserving Visual Privilege (CPVP)** and
**Correctness-Aligned Distillation (CAD)** for remote-sensing visual question
answering. This repository contains its training and evaluation code.

The repository is organized around the proposed method, a standalone evaluator,
and a pinned OPD-V baseline. It intentionally excludes datasets, checkpoints,
experiment logs, and other generated artifacts.

## Repository layout

| Path | Purpose |
| --- | --- |
| `rs-opsd/` | RS-OPSD training implementation, configurations, unit tests, and runtime launchers. |
| `eval/` | OpenAI-compatible evaluator for LRS-VQA, MME-RealWorld Remote Sensing, and XLRS-Bench. |
| `other_methods/OPD-V/` | OPD-V baseline, included as a Git submodule at a fixed upstream revision. |
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

RS-OPSD is distributed under the [Apache-2.0 license](rs-opsd/LICENSE). The
OPD-V baseline remains subject to its upstream license and notices.
