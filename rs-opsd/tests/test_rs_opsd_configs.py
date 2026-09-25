"""Canonical RS-OPSD configuration names and portable launcher routing."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('recipe,suffix', [
    ('direct-2k-kl', 'kl'),
    ('direct-2k-kl-mixed', 'kl_mixed'),
    ('direct-2k-topk64-jsd-kl', 'topk64_jsd_kl'),
    ('direct-2k-three-image-kl', 'three_image_kl'),
    ('direct-2k-three-image-kl-32gpu', 'three_image_kl_32gpu'),
    ('direct-2k-three-image-topk64-jsd-kl', 'three_image_topk64_jsd_kl'),
])
def test_canonical_recipe_composition(recipe, suffix, monkeypatch):
    for name, value in {'PA_OPD_MODEL_PATH': '/tmp/base-model',
                        'PA_OPD_WORK_DIR': '/tmp/existing-explicit-output',
                        'PA_OPD_RUN_NAME': 'existing-explicit-run',
                        'PA_OPD_TRAIN_FILE': '/tmp/data/train.jsonl',
                        'PA_OPD_VISIONOPD_ROOT': '/tmp/visionopd',
                        'PA_OPD_SAVE_FREQ': '30'}.items():
        monkeypatch.setenv(name, value)
    name = 'rs_opsd_direct_2k_' + suffix
    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name=name)
    assert f'{recipe}) config_name="{name}"' in (ROOT / 'scripts/train.sh').read_text()
    assert cfg.trainer.group_name.startswith('rs-opsd-')
    assert cfg.trainer.experiment_name == 'existing-explicit-run'
    assert cfg.trainer.default_local_dir == '/tmp/existing-explicit-output/checkpoints'
    assert cfg.actor_rollout_ref.model.path == '/tmp/base-model'
    actor = cfg.actor_rollout_ref.actor
    assert actor.use_kl_loss and actor.kl_loss_coef == .001
    assert actor.loss_agg_mode == 'token-mean'
    assert actor.self_distillation.teacher_model_path == '/tmp/base-model'
    assert actor.self_distillation.teacher_input_mode == (
        'global_plus_derived_plus_crop' if 'three_image' in suffix else 'global_plus_crop')
    assert actor.self_distillation.get('pa_opd_topk_jsd_enabled', False) == ('jsd' in suffix)
    assert instantiate(actor.checkpoint) is not None
    OmegaConf.to_container(actor.self_distillation, resolve=True, throw_on_missing=True)
