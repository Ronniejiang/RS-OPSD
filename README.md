# RS-OPSD

RS-OPSD contains the training and evaluation code used to study **PA-OPD**
(privileged-advantage on-policy visual distillation) for remote-sensing visual
question answering.

The repository is organized around the proposed method, a standalone evaluator,
and a pinned OPD-V baseline. It intentionally excludes datasets, checkpoints,
experiment logs, and other generated artifacts.

## Repository layout

| Path | Purpose |
| --- | --- |
| `PA-OPD/` | PA-OPD training implementation, configurations, unit tests, and training launchers. |
| `eval/` | OpenAI-compatible evaluator for LRS-VQA, MME-RealWorld Remote Sensing, and XLRS-Bench. |
| `other_methods/OPD-V/` | OPD-V baseline, included as a Git submodule at a fixed upstream revision. |
| `eval/*.sh` | Local evaluation and environment launchers. |


## PA-OPD training

PA-OPD requires a CUDA-compatible PyTorch/vLLM environment, a local Qwen3-VL
model, and the Vision-OPD-6K dataset. Create PA-OPD/.venv (or set PYTHON to
its executable), install the dependencies appropriate for your CUDA platform,
and run:

    cd PA-OPD
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

    MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct SOURCE_DATA_DIR=/path/to/Vision-OPD-6K CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_pa_opd.sh

SOURCE_DATA_DIR must contain train.jsonl, images/, and teacher_images/. The
default output directory is PA-OPD/outputs/<run-name>/; set WORK_ROOT to store
run artifacts elsewhere. See PA-OPD/README.md and PA-OPD/PA_OPD_MODES.md for
method and mode details.

## Evaluation

The evaluator sends requests to an OpenAI-compatible model server and supports
LRS-VQA, MME-RealWorld Remote Sensing, and XLRS-Bench. Install its lightweight
client dependencies and point it at a running server:

    python3 -m venv .venv-eval
    .venv-eval/bin/pip install openai pillow datasets
    .venv-eval/bin/python -m eval.run --dataset lrs-vqa --lrs-root /path/to/lrs-vqa --api-base http://localhost:8000/v1 --model-id OPD-V

To create a dedicated vLLM environment and serve a local checkpoint first, run
MODEL_PATH=/path/to/model LRS_ROOT=/path/to/lrs-vqa BENCHMARK=lrs-vqa bash
eval/run_opdv_eval_with_venv.sh. Use --dry-run with eval.run to validate a
dataset layout before starting a server. Full options are in eval/README.md.

## Reproducibility and artifacts

Datasets, model weights, checkpoints, W&B data, Ray output, and evaluation
results are deliberately ignored by Git. Supply local paths through the
environment variables shown above. This repository contains no cluster- or
scheduler-specific submission scripts.

## License

PA-OPD is distributed under the [Apache-2.0 license](PA-OPD/LICENSE). The
OPD-V baseline remains subject to its upstream license and notices.
