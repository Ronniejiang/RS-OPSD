#!/usr/bin/env bash
# Source only from the PPU launcher, before Python/Ray initializes PCCL.
# PCCL 2.1.0-11's C4 CONN statistics exhaust nccl_call_pool in the bounded
# torch-all-gather + PyNCCL-all-reduce probe. This disables only C4 performance
# statistics, not collectives, training metrics, or error reporting.
# See docs/pccl_call_pool_debug_20260913.md for the real-device A/B result.
PA_OPD_PCCL_STATS_MODE="${PA_OPD_PCCL_STATS_MODE:-none}"
case "${PA_OPD_PCCL_STATS_MODE}" in
  none|CONN|QP)
    export ACCL_C4_STATS_MODE="${PA_OPD_PCCL_STATS_MODE}"
    ;;
  inherit) ;;
  *)
    echo "ERROR: PA_OPD_PCCL_STATS_MODE must be none, CONN, QP, or inherit" >&2
    return 2
    ;;
esac
export PA_OPD_PCCL_STATS_MODE
echo "PPU C4 statistics: ACCL_C4_STATS_MODE=${ACCL_C4_STATS_MODE:-unset} (policy=${PA_OPD_PCCL_STATS_MODE})"
