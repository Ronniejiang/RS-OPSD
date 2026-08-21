#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias. PA-OPD is configured exclusively by pa_opd.yaml.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/run_pa_opd.sh" "$@"
