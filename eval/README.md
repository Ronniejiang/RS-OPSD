# Remote-sensing benchmark evaluation
This portable evaluator runs any OpenAI-compatible vision model on XLRS-Bench,
MME-RealWorld Remote Sensing, and LRS-VQA. It supports vLLM directly, but can
also target another compliant chat-completions endpoint.

It writes one JSONL file per benchmark and `summary.json` under
`eval/results/<run-name>/`. Results are flushed after each finished request;
run the same command with `--resume` to skip completed sample IDs.

## Dependencies

The evaluator itself needs Pillow and the OpenAI client. XLRS-Bench also needs
Hugging Face Datasets:

```bash
pip install openai pillow datasets
```

Optional LRS-VQA tolerant semantic scoring additionally uses the existing
`torch` and `transformers` packages with a local BGE model. Download it
once before submitting an evaluation; the submitted job uses local files only:

```bash
EVAL_PYTHON=python
MODEL_DIR=/path/to/bge-base-en-v1.5
"${EVAL_PYTHON}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-base-en-v1.5', local_dir='${MODEL_DIR}')"
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

Start a model with an OpenAI-compatible server, then pass its
base URL and exposed model name:

```bash
python -m eval.run \
  --dataset lrs-vqa,mme-realworld-rs,xlrs-bench \
  --lrs-root /path/to/lrs-vqa \
  --mme-root /path/to/mme-realworld \
  --xlrs-root /path/to/xlrs-bench \
  --api-base http://localhost:8000/v1 \
  --model-id your-served-model-name \
  --parallel-workers 32 \
  --run-name baseline
```

For LRS-VQA, add a local BGE path to retain strict accuracy and also calculate
a tolerant metric:

```bash
  --lrs-semantic-model /path/to/bge-base-en-v1.5 \
  --lrs-semantic-threshold 0.85
```

The existing `correct` and `accuracy` fields remain strict normalized-string
matches. The semantic pass writes `semantic_similarity`, `semantic_match`,
`tolerant_correct`, and `tolerant_match_source`; summary adds
`tolerant_accuracy`, `canonical_alias_rescued`, and `semantic_rescued`. Boolean and numeric answers
remain exact-only to prevent antonyms or different counts from being accepted.

`--enable-thinking True` passes
`chat_template_kwargs.enable_thinking=true`, when supported by the endpoint.
Images are converted to PNG data URIs before the request so that TIFF LRS-VQA images and in-memory XLRS images work with the same model
interface. Use `--image-format jpeg --jpeg-quality 95` if request payloads are
too large. `--max-pixels` bounds the image area before encoding (default:
16,777,216 pixels).

### Local Transformers fallback

`run_local_vllm.sh` remains the preferred launcher when the installed vLLM
version natively supports the model. On a machine with an older or
vendor-customized vLLM that cannot load the VLM, use the direct Transformers
backend instead. It preserves the same adapters, prompts, image pixel limit,
answer parsing, and per-record JSONL output, but deliberately evaluates one
sample at a time on the local GPU.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON=/path/to/python-with-torch-and-transformers \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
MODEL_ID=qwen3-vl-8b-instruct \
BENCHMARK=lrs-vqa,mme-realworld-rs,xlrs-bench \
LRS_ROOT=/path/to/LRS-VQA \
MME_ROOT=/path/to/MME-RealWorld \
XLRS_ROOT=/path/to/XLRS-Bench \
bash eval/scripts/run_local_transformers.sh
```

There is also an environment-variable launcher:

```bash
API_BASE=http://localhost:8000/v1 \
OPENAI_MODEL_ID=your-served-model-name \
BENCHMARK=lrs-vqa,mme-realworld-rs,xlrs-bench \
LRS_ROOT=/path/to/lrs-vqa \
MME_ROOT=/path/to/mme-realworld \
XLRS_ROOT=/path/to/xlrs-bench \
LRS_SEMANTIC_MODEL=/path/to/bge-base-en-v1.5 \
LRS_SEMANTIC_THRESHOLD=0.85 \
bash eval/scripts/run_openai_compatible.sh

```
## Layout and launchers

The public evaluator consists of adapters, scoring, and inference code at the
top level of `eval/`. Canonical runnable helpers are in `eval/scripts/`:

- `run_openai_compatible.sh`: evaluate an already-running model endpoint.
- `run_local_vllm.sh`: validate layouts, start vLLM, verify the expected model
  identity, evaluate, then stop the server.
- `run_local_transformers.sh`: directly run a local Hugging Face VLM when a
  compatible vLLM is unavailable.
- `create_vllm_env.sh`: optionally create a reproducible vLLM environment.
- `probe_ppu_vllm_runtime.sh`: verify that a PPU image has a usable PPU2
  runtime, vLLM, and native Qwen3-VL implementation.
- Scheduler-specific submission scripts and historical result snapshots are
  local-only and are not distributed with the public evaluator.
- `merge_fsdp_checkpoint_to_hf.sh`: merge a `verl` FSDP actor checkpoint.
- `merge_lora_adapter_to_hf.sh`: merge a PEFT LoRA adapter into a standalone
  Hugging Face model directory.

Use `eval/.env.example` as the variable reference. All paths, model names,
and server settings are supplied by command-line arguments or environment
variables; no organization-specific path, username, queue, or dataset path is
embedded in the public evaluator.

For a local vLLM evaluation:

```bash
MODEL_PATH=/path/to/huggingface-model \
SERVED_MODEL_NAME=your-served-model-name \
BENCHMARK=lrs-vqa,mme-realworld-rs,xlrs-bench \
LRS_ROOT=/path/to/LRS-VQA \
MME_ROOT=/path/to/MME-RealWorld \
XLRS_ROOT=/path/to/XLRS-Bench \
TP_SIZE=1 \
bash eval/scripts/run_local_vllm.sh
```

FSDP merging requires a compatible `verl` checkout and Python environment:

```bash
VERL_ROOT=/path/to/verl \
PYTHON=/path/to/python \
bash eval/scripts/merge_fsdp_checkpoint_to_hf.sh /path/to/global_step_N
```

### PPU evaluation

Inside an allocated PPU runtime, use `probe_ppu_vllm_runtime.sh` to verify
the installed stack, then run `run_local_vllm.sh` with explicit paths as above.
Scheduler configuration, image repositories and submission wrappers are local
deployment details and are not included in this repository.

For a full benchmark run, use `BENCHMARK=all LIMIT=` and a unique `RUN_NAME`.
A 4K image at a 16 MP cap can require roughly 15.7K visual tokens; choose
`MAX_MODEL_LEN=32768` or reduce `MAX_PIXELS` to fit a smaller context budget.

The strict `correct` / `accuracy` fields are never changed by tolerant LRS-VQA
scoring. The optional tolerant score first applies a small audited alias table
(for example, `rectangle` / `rectangular`) and only then uses local BGE cosine
similarity for remaining non-boolean, non-numeric answers.
