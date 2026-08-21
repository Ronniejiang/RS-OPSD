# Remote-sensing benchmark evaluation

This evaluator runs an OPD-V-served model through the same OpenAI-compatible
chat-completions interface used by `other_methods/OPD-V/eval/infer.py`, while
using the XLRS-Bench, MME-RealWorld Remote Sensing, and LRS-VQA adapters and
metrics from the FOVIS evaluation design.

It writes one JSONL file per benchmark and `summary.json` under
`eval/results/<run-name>/`. Results are flushed after each finished request;
run the same command with `--resume` to skip completed sample IDs.

## Dependencies

The evaluator itself needs Pillow and the OpenAI client. XLRS-Bench also needs
Hugging Face Datasets:

```bash
pip install openai pillow datasets
```

## Dataset layouts

The expected local layouts are the same as the FOVIS evaluator:

- LRS-VQA: `<LRS_ROOT>/LRS_VQA_merged.jsonl`, with images either at their
  released relative paths or flattened under `<LRS_ROOT>/image/`.
- MME-RealWorld: `<MME_ROOT>/MME_RealWorld.json`; only English
  `Perception / Remote Sensing` rows are evaluated. `Image` paths in the JSON
  are resolved relative to `<MME_ROOT>`.
- XLRS-Bench: a directory readable by `datasets.load_from_disk` containing a
  `train` split.

Check a dataset layout without importing the OpenAI client or starting a model:

```bash
python -m eval.run \
  --dataset lrs-vqa \
  --lrs-root /path/to/lrs-vqa \
  --dry-run
```

## Run inference

Start the OPD-V model with its normal OpenAI-compatible server, then pass its
base URL and exposed model name:

```bash
python -m eval.run \
  --dataset lrs-vqa,mme-realworld-rs,xlrs-bench \
  --lrs-root /path/to/lrs-vqa \
  --mme-root /path/to/mme-realworld \
  --xlrs-root /path/to/xlrs-bench \
  --api-base http://localhost:8000/v1 \
  --model-id OPD-V \
  --parallel-workers 32 \
  --run-name opd-v-rs
```

`--enable-thinking True` passes
`chat_template_kwargs.enable_thinking=true`, exactly as the existing OPD-V
evaluator does. Images are converted to PNG data URIs before the request so
that TIFF LRS-VQA images and in-memory XLRS images work with the same model
interface. Use `--image-format jpeg --jpeg-quality 95` if request payloads are
too large. `--max-pixels` bounds the image area before encoding (default:
16,777,216 pixels).

There is also an environment-variable launcher:

```bash
API_BASE=http://localhost:8000/v1 \
OPENAI_MODEL_ID=OPD-V \
BENCHMARK=lrs-vqa,mme-realworld-rs,xlrs-bench \
LRS_ROOT=/path/to/lrs-vqa \
MME_ROOT=/path/to/mme-realworld \
XLRS_ROOT=/path/to/xlrs-bench \
bash eval/run_eval.sh
```
