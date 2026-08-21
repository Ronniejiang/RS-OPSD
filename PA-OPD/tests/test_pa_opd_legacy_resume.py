import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from verl.trainer.main_pa_opd_resume import _load_rollout_format_rate, _reconstruct_runtime_state
from verl.trainer.ppo.pa_opd_runtime import PAOPDRuntimeController


def _write_rollout(path, valid, invalid):
    good = {"output": "reasoning</think><answer>A</answer>"}
    bad = {"output": "reasoning answer A"}
    with path.open("w", encoding="utf-8") as stream:
        for row in [good] * valid + [bad] * invalid:
            stream.write(json.dumps(row) + "\n")


def _assert_runtime_error(action, expected_fragment):
    try:
        action()
    except RuntimeError as exc:
        assert expected_fragment in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_load_rollout_format_rate_requires_complete_file():
    with tempfile.TemporaryDirectory() as directory:
        rollout = Path(directory) / "30.jsonl"
        _write_rollout(rollout, valid=31, invalid=1)
        assert _load_rollout_format_rate(rollout, expected_rows=32) == 31 / 32
        _assert_runtime_error(
            lambda: _load_rollout_format_rate(rollout, expected_rows=33), "incomplete rollout"
        )


def test_load_rollout_format_rate_rejects_missing_output():
    with tempfile.TemporaryDirectory() as directory:
        rollout = Path(directory) / "30.jsonl"
        rollout.write_text('{"score": 1}\n', encoding="utf-8")
        _assert_runtime_error(
            lambda: _load_rollout_format_rate(rollout, expected_rows=1), "no output field"
        )


def test_legacy_opd_rlvr_migration_initializes_fixed_runtime_without_rollouts():
    controller = PAOPDRuntimeController(
        {
            "mode": "opd_rlvr",
            "rlvr_loss_coef": 0.1,
            "adaptive_format": {
                "recovery_rollout_n": 8,
                "opd_only_rollout_n": 1,
                "valid_threshold": 0.90,
                "stable_steps": 3,
            },
        }
    )
    trainer = SimpleNamespace(pa_opd_runtime_controller=controller)
    _reconstruct_runtime_state(trainer)
    assert controller.state.phase == "opd_rlvr"
    assert controller.state.effective_rollout_n == 8
    assert controller.effective_rollout_n == 8
    assert controller.rlvr_active
