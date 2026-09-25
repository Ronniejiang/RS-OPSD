"""PPU communication workaround is selected before any Python/Ray startup."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/ppu_comm_env.sh'


def run_policy(policy=None):
    env = dict(os.environ, ACCL_C4_STATS_MODE='CONN')
    env.pop('PA_OPD_PCCL_STATS_MODE', None)
    if policy is not None:
        env['PA_OPD_PCCL_STATS_MODE'] = policy
    # A child shell verifies export/inheritance, not only a shell-local value.
    return subprocess.run(
        ['bash', '-c', 'set -euo pipefail; source "$1"; bash -c \'echo CHILD_MODE=$ACCL_C4_STATS_MODE\'',
         'test', str(HELPER)], env=env, text=True, capture_output=True, timeout=5)


def test_default_disables_platform_conn_statistics_in_child():
    result = run_policy()
    assert result.returncode == 0, result.stderr
    assert 'CHILD_MODE=none' in result.stdout


def test_explicit_diagnostic_modes_and_inherit():
    for policy, expected in [('none', 'none'), ('CONN', 'CONN'), ('QP', 'QP'), ('inherit', 'CONN')]:
        result = run_policy(policy)
        assert result.returncode == 0, result.stderr
        assert f'CHILD_MODE={expected}' in result.stdout


def test_invalid_policy_fails_before_child():
    result = run_policy('invalid')
    assert result.returncode == 2
    assert 'CHILD_MODE' not in result.stdout


def test_ppu_launchers_wire_workaround_before_runtime():
    launcher = (ROOT / 'scripts/train_geoevidence_direct_2k_kl_ppu8.sh').read_text()
    assert launcher.index('source "${SCRIPT_DIR}/ppu_comm_env.sh"') < launcher.index('preflight_ppu_runtime.py')
    assert 'PA_OPD_PCCL_STATS_MODE' in HELPER.read_text()
