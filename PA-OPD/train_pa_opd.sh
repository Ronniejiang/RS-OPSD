#!/usr/bin/env bash
set -euo pipefail

PA_OPD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${PA_OPD_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  bash "${PA_OPD_DIR}/scripts/test_pa_opd_modes.sh"
fi

exec bash "${PA_OPD_DIR}/scripts/run_pa_opd_modes.sh" "$@"
