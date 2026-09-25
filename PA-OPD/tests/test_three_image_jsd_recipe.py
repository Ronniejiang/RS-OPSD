from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
RECIPE = "direct-2k-three-image-topk64-jsd-kl"


def test_three_image_jsd_composition_keeps_views_losses_and_fresh_start():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="rs_opsd_direct_2k_three_image_topk64_jsd_kl")
        jsd = compose(config_name="rs_opsd_direct_2k_topk64_jsd_kl")
    sd = config.actor_rollout_ref.actor.self_distillation
    for key in ("full_logit_distillation", "distillation_topk", "distillation_add_tail",
                "alpha", "pa_opd_topk_jsd_enabled", "pa_opd_topk_jsd_coef"):
        assert sd[key] == jsd.actor_rollout_ref.actor.self_distillation[key]
    assert sd.teacher_input_mode == "global_plus_derived_plus_crop"
    assert sd.teacher_image_key == "teacher_images"
    assert sd.max_reprompt_len == 65536
    assert config.actor_rollout_ref.actor.use_kl_loss
    assert config.actor_rollout_ref.actor.kl_loss_coef == 0.001
    assert config.data.pa_opd_direct_three_image_jsonl
    assert not config.data.pa_opd_direct_jsonl
    assert config.data.pa_opd_student_image_mode == "images"
    assert config.data.pa_opd_teacher_full_image_mode == "bbox_images"
    assert config.data.pa_opd_separate_teacher_views
    assert len(config.data.train_files) == 1
    assert config.trainer.resume_mode == "disable"
    assert config.trainer.resume_from_path is None


def test_bbox_student_override_keeps_three_image_jsd():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="rs_opsd_direct_2k_three_image_topk64_jsd_kl",
                         overrides=["data.pa_opd_student_image_mode=bbox_images"])
    assert config.data.pa_opd_student_image_mode == "bbox_images"
    assert config.data.pa_opd_teacher_full_image_mode == "bbox_images"
    assert config.data.pa_opd_direct_three_image_jsonl
    assert config.actor_rollout_ref.actor.self_distillation.pa_opd_topk_jsd_enabled


def test_new_recipe_is_routed_by_all_required_entrypoints():
    for script in ("train.sh", "train_geoevidence_direct_2k_kl_ppu8.sh",
                   "validate_geoevidence_views.py"):
        assert RECIPE in (ROOT / "scripts" / script).read_text()
