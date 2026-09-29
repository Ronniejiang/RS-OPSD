# RS-OPSD terminology and CAD probe migration

## Names and interfaces

RS-OPSD is the complete method. CPVP denotes Global / Contextual / Fine-grained
Evidence, in that order. CAD denotes the correctness-aligned sampled-answer
objective, not Reference KL or the optional top-k JSD extension.

The canonical Python loss entry is `compute_cad_loss`. Six Hydra configs use
the `rs_opsd_direct_2k_` prefix; the recipe suffixes remain unchanged. Old loss
function/config names are removed, without aliases. `scripts/train.sh` and
repository callers use the new names. The training directory is `rs-opsd/`.
The `pa_opd_*` fields, environment variables, recipe CLI choices and local-only
submission filenames retain their existing names for compatibility.
Explicit `RUN_NAME` / `OUTPUT_DIR` values keep their existing semantics.

New training logs emit only the new CAD names:

| Metric | Meaning |
| --- | --- |
| `actor/cad_loss` | Pure sampled-answer CAD, before optional JSD |
| `actor/distillation_loss` | CAD + JSD coefficient times JSD, excluding KL |
| `actor/topk_jsd_loss` | Unweighted JSD extension, when enabled |
| `actor/kl_loss` | Unweighted frozen-Reference KL |
| `cad/teacher_student_gap_token_mean` | Detached sampled-token log-probability gap |
| `cad/advantage_abs_token_mean` | Absolute gated CAD signal |
| `cad/reinforce_token_fraction`, `cad/suppress_token_fraction`, `cad/active_token_fraction` | Token-gate diagnostics |
| `cad/teacher_reliable_fraction`, `cad/trajectory_correct_fraction` | Probe-pass and Student-correct fractions |
| `cad/num_answer_tokens`, `cad/empty_target_batch` | Valid answer mask diagnostics |

Former sampled-only loss maps to `actor/cad_loss`; former combined distillation
loss maps to `actor/distillation_loss`. Existing diagnostics otherwise retain
their aggregation behavior. Historical dashboards require selecting the new
keys for new runs; existing offline logs and stored metrics are not rewritten.

## Behavioral correction

The Teacher probe is evaluated on canonical GT answer prefixes, separately from
the Student-prefix distillation forward. The first step allows option labels
only. Every later step now allows EOS **as well as** later unselected labels;
EOS is first, so ties prefer stopping. The final step still checks EOS against
remaining continuations. This is constrained-greedy GT-path agreement, not
unconstrained Teacher generation accuracy.

Previously, intermediate positions omitted EOS. A Teacher that preferred
stopping at `A` could pass a GT `A,B` check. Such a sample is now rejected.
JSD shares this candidate grammar, so its mandatory support and non-GT
continuation masking now include intermediate EOS. No JSD formula changed.

For fixed masks/probabilities/gates, CAD loss and Student gradients are unchanged.
Reliability and direction gates only affect the numerator, not the optimizer-step
global answer-token denominator. KL remains active independently of CAD gating.
Prompts, dataset image files, optimizer settings and Teacher update policies are
unchanged. CPVP views are loaded, not generated; the loader does not verify that
the contextual view was created by doubling the evidence region's dimensions.

## Resume and historical results

No checkpoint format changes or weight conversions are needed. Use the renamed
config with the same explicit model paths, output path and resume checkpoint.
Automatic resume must be pointed at the existing output rather than a newly
named default directory. Changing probe behavior means continued training is
not step-for-step equivalent to the previous algorithm. No running jobs,
existing checkpoints, datasets, historical archives or result directories are
modified by this migration.

## Migration verification

- 103 regression tests passed, covering probes, token alignment, JSD, global
  token-mean gradients, checkpoint handling and recipe configuration.
- Eight direct comparisons against the pre-migration loss implementation
  matched both loss values and Student gradients exactly for fixed inputs.
- All six renamed YAML configurations retain their non-naming values.
- Shell syntax checks and `git diff --check` passed.

These are CPU/configuration checks, not a full accelerator training run. No training
jobs were submitted or restarted for this migration.
