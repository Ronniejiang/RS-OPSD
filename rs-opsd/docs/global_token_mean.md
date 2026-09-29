# PA-OPD global token mean

Implemented 2026-09-14 following the gradient-reduction audit. All direct
2K+KL and 2K+KL+JSD recipes inherit `actor.loss_agg_mode: token-mean`.
The optimizer mini-batch, not the rollout batch or an individual micro-batch,
defines the averaging boundary.

For each loss term k:

    N_k = sum_over_DP_ranks_and_micro_batches(sum(M_k))
    L_k = sum(M_k * token_loss_k) / max(N_k, 1)

The masks retain their previous meaning:

| Term | Denominator mask |
| --- | --- |
| Sampled CAD | Semantic-token mask times Teacher-present mask |
| Optional safe top-k JSD | GT-compatible prefix mask times Teacher-present mask |
| Frozen-reference KL | Response mask |

Teacher reliability, signed-advantage selection and importance weights remain
in the numerator only. In particular, this is valid-token mean, not mean over
tokens with nonzero loss. Each term has its own denominator; JSD prefix/EOS
positions must not be counted using CAD's semantic mask.

Before micro-batch splitting, `collect_global_token_mean` performs one SUM
collective on three scalar counts. With averaged FSDP gradients, each local
micro-batch backpropagates `DP_size * local_masked_sum / max(N_k, 1)`.
There is no additional sample-count ratio or division by accumulation steps.
FSDP's average cancels that single DP-size factor. Gradient clipping, learning
rate, loss coefficients and Teacher EMA are unchanged.

CAD, JSD and KL loss logs sum these local micro contributions and then use
the trainer's rank mean. Thus for the current one-mini-batch, one-epoch update
they report the global objectives. Other diagnostic rates from the loss
kernels still retain their existing micro-batch averaging semantics. Added
`pa_opd/global_token_mean` and `pa_opd/global_{semantic,jsd_prefix,response}_tokens`
expose the mode and denominators (counts are averaged if an update includes
multiple optimizer mini-batches).

This path supports the current FSDP/SP=1 recipes. Ulysses SP>1 explicitly fails
instead of assuming its replication/reduction semantics. Non-PA-OPD policies
and explicitly selected non-token-mean reductions retain their existing path.

## Prior validation

The development tests (not included in this release) compared the actual loss kernels and gradients with full-batch reference
calculations for unequal lengths, empty masks, Teacher presence/reliability,
simulated 1/2/16 ranks and varying micro-batch partitions. The real actor
update loop was tested with tiny CPU model-forward doubles, both dynamic and
fixed batches, with and without JSD. A separate real two-process CPU/Gloo FSDP
test checked reduced accumulated gradients, all-empty and empty-rank cases,
loss values, preclip norms and clipped gradients. This is not a full-model
16-device end-to-end test.

Existing checkpoints remain loadable: no model/optimizer state layout or
runtime sidecar was changed. However, resuming with this code deliberately
changes the reduction objective, so it is not numerically equivalent to
continuing with the old code. Existing submitted job snapshots do not change;
using this fix requires a newly submitted snapshot. No jobs were restarted
by the implementation/test work.
