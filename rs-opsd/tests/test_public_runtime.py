"""Public entrypoints require explicit paths, never a developer's private mount."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    'train_geoevidence_direct_2k_kl_ppu8.sh', 'preflight_geoevidence_ppu.sh',
])
def test_runtime_requires_explicit_model_path(name):
    env = dict(os.environ)
    env.pop('MODEL_PATH', None)
    result = subprocess.run(['bash', str(ROOT / 'scripts' / name)], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'Set MODEL_PATH' in result.stderr


@pytest.mark.parametrize('name', [
    'rebuild_geoevidence_train_jsonl.py', 'rebuild_geoevidence_derived_train_jsonl.py',
])
def test_manifest_builder_requires_explicit_data_root(name):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert '--data-root' in result.stderr
