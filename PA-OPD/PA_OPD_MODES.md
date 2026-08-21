# PA-OPD training modes

Run from this directory with the target vLLM 0.18 environment:

```bash
cd PA-OPD
PA_OPD_MODE=adaptive_format bash train_pa_opd.sh
```

The default `adaptive_format` mode starts with eight responses per prompt.
After three consecutive steps whose strict format-valid rate is at least 90%,
the next step uses one response per prompt and only PA-OPD contributes a
gradient.  A later step below 90% switches the next step back to eight
responses and format GRPO.

For fixed eight-response PA-OPD plus accuracy RLVR:

```bash
PA_OPD_MODE=opd_rlvr bash train_pa_opd.sh
```

Its scalar rule reward is `0.1 * format_valid + 1.0 * answer_correct`.

Outputs are separated automatically:

- `outputs/pa-opd-qwen3-vl-8b-4gpu-adaptive_format`
- `outputs/pa-opd-qwen3-vl-8b-4gpu-opd_rlvr`

Both modes save every ten steps.  Resume is automatic only within the same
mode and exact runtime configuration.  In addition to Student/EMA Teacher,
optimizer, scheduler, RNG and dataloader state, every new checkpoint contains
`pa_opd_runtime.json`.  A checkpoint without that file is intentionally
rejected by this entrypoint.

The existing `scripts/run_pa_opd.sh` remains the legacy fixed-n entrypoint.
