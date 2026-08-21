#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PA_OPD_CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PA_OPD_VENV="${PA_OPD_VENV:-${PA_OPD_CODE_ROOT}/.venv}"
PA_OPD_TEST_TMP="${PA_OPD_TEST_TMP:-/tmp/paopd-preflight-${UID:-0}}"

mkdir -p "${PA_OPD_TEST_TMP}/pycache" "${PA_OPD_TEST_TMP}/work/checkpoints" \
  "${PA_OPD_TEST_TMP}/work/rollouts"

export PYTHONPATH="${PA_OPD_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="${PA_OPD_TEST_TMP}/pycache"
export PA_OPD_CODE_ROOT
export PA_OPD_MODE="${PA_OPD_MODE:-adaptive_format}"
export PA_OPD_RUN_NAME="${PA_OPD_RUN_NAME:-pa-opd-preflight}"
export PA_OPD_WORK_ROOT="${PA_OPD_TEST_TMP}/work"

"${PA_OPD_VENV}/bin/python" -m py_compile \
  "${PA_OPD_CODE_ROOT}/verl/trainer/main_ppo.py" \
  "${PA_OPD_CODE_ROOT}/verl/trainer/main_pa_opd.py" \
  "${PA_OPD_CODE_ROOT}/verl/trainer/main_pa_opd_resume.py" \
  "${PA_OPD_CODE_ROOT}/verl/trainer/ppo/pa_opd_runtime.py" \
  "${PA_OPD_CODE_ROOT}/verl/trainer/ppo/ray_trainer.py" \
  "${PA_OPD_CODE_ROOT}/verl/utils/reward_score/pa_opd_protocol.py" \
  "${PA_OPD_CODE_ROOT}/verl/utils/reward_score/pa_opd_dynamic_reward.py"

"${PA_OPD_VENV}/bin/python" "${PA_OPD_CODE_ROOT}/scripts/run_pa_opd_unit_tests.py"

for mode in adaptive_format opd_rlvr; do
  PA_OPD_MODE="${mode}" "${PA_OPD_VENV}/bin/python" -m verl.trainer.main_pa_opd \
    --config-name pa_opd_modes --cfg job --resolve >/dev/null
done

echo "PA-OPD preflight passed: syntax, protocol/controller tests, and both Hydra modes"
