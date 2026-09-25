from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.run_multinode_ray import topology, read_status, write_status

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('kl,clip,student_size,fixed_teacher', [
    (.01, 1., '8b', False),
    (.001, 5., '8b', False),
    (.001, 20., '8b', False),
    (.001, 100., '8b', False),
    (.001, 5., '2b', False),
    (.001, 5., '2b', True),
])
def test_single_variable_ablation_configs(kl, clip, student_size, fixed_teacher, tmp_path, monkeypatch):
    # Public recipes must be testable without private cluster submission scripts.
    model_path = str(tmp_path / f'student-{student_size}')
    teacher_path = str(tmp_path / 'teacher-8b')
    monkeypatch.setenv('PA_OPD_MODEL_PATH', model_path)
    monkeypatch.setenv('PA_OPD_WORK_DIR', str(tmp_path / 'output'))
    monkeypatch.setenv('PA_OPD_SAVE_FREQ', '30')
    overrides = [
        f'actor_rollout_ref.actor.kl_loss_coef={kl}',
        f'actor_rollout_ref.actor.grad_clip={clip}',
    ]
    if fixed_teacher:
        overrides += [
            f'actor_rollout_ref.actor.self_distillation.teacher_model_path={teacher_path}',
            'actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed',
            'actor_rollout_ref.actor.self_distillation.fixed_teacher_ema=false',
            'actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0',
        ]
    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='rs_opsd_direct_2k_three_image_kl_32gpu', overrides=overrides)
    actor = cfg.actor_rollout_ref.actor
    assert cfg.actor_rollout_ref.model.path == model_path
    assert actor.self_distillation.teacher_model_path == (teacher_path if fixed_teacher else model_path)
    if fixed_teacher:
        assert actor.self_distillation.teacher_model_source == 'fixed'
        assert not actor.self_distillation.fixed_teacher_ema
        assert actor.self_distillation.teacher_update_rate == 0.
        assert cfg.actor_rollout_ref.ref.get('model', {}).get('path', model_path) == model_path
    assert actor.kl_loss_coef == kl and actor.grad_clip == clip
    assert actor.clip_ratio_low == .2 and actor.clip_ratio_high == .3
    assert actor.ppo_mini_batch_size == cfg.data.train_batch_size == 96
    assert cfg.trainer.nnodes * cfg.trainer.n_gpus_per_node == 32
    assert cfg.trainer.total_epochs == 3 and cfg.trainer.save_freq == 30
    assert cfg.trainer.resume_mode == 'disable' and cfg.trainer.resume_from_path is None
    assert not actor.self_distillation.pa_opd_topk_jsd_enabled
    assert actor.self_distillation.teacher_input_mode == 'global_plus_derived_plus_crop'


def test_32gpu_batch_epochs_views_and_checkpoint_configuration(monkeypatch):
    monkeypatch.setenv('PA_OPD_WORK_DIR', '/tmp/test-pao32-output')
    monkeypatch.setenv('PA_OPD_SAVE_FREQ', '30')
    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='rs_opsd_direct_2k_three_image_kl_32gpu')
    actor = cfg.actor_rollout_ref.actor
    assert cfg.trainer.nnodes * cfg.trainer.n_gpus_per_node == 32
    assert actor.ppo_mini_batch_size == cfg.data.train_batch_size == 96
    assert actor.ppo_mini_batch_size // 32 == 3
    assert actor.ppo_micro_batch_size_per_gpu == actor.ppo_epochs == cfg.actor_rollout_ref.rollout.n == 1
    assert actor.loss_agg_mode == 'token-mean'
    assert cfg.trainer.total_epochs == 3 and cfg.trainer.save_freq == 30
    assert actor.use_kl_loss and actor.kl_loss_coef == .001
    assert not actor.self_distillation.pa_opd_topk_jsd_enabled
    assert actor.self_distillation.teacher_input_mode == 'global_plus_derived_plus_crop'
    assert cfg.data.pa_opd_student_image_mode == cfg.data.pa_opd_teacher_full_image_mode == 'bbox_images'
    assert cfg.data.pa_opd_direct_three_image_jsonl and len(cfg.data.train_files) == 1
    assert cfg.trainer.resume_mode == 'disable'
    assert not cfg.trainer.skip_checkpoint and cfg.trainer.max_actor_ckpt_to_keep == 2
    assert set(actor.checkpoint.save_contents) == {'model', 'optimizer', 'extra', 'hf_model'}
    assert actor.checkpoint.hf_model_dir == '/tmp/test-pao32-output/inference'
    assert set(actor.checkpoint.load_contents) == {'model', 'optimizer', 'extra'}
    # Hydra must be able to instantiate the typed checkpoint config.
    from hydra.utils import instantiate
    assert instantiate(actor.checkpoint).hf_model_dir == '/tmp/test-pao32-output/inference'
    monkeypatch.setenv('PA_OPD_SAVE_FREQ', '-1')
    assert cfg.trainer.save_freq == -1  # trainer's last-step save still applies


def test_multinode_topology_rejects_unsafe_fallbacks():
    env = dict(PA_OPD_NNODES='2', PA_OPD_GPUS_PER_NODE='16', NODE_RANK='1', MASTER_ADDR='head', MASTER_PORT='23456')
    assert topology(env) == (2, 16, 1, 'head', 23456)
    for bad in ({'NODE_RANK': '-1'}, {'NODE_RANK': '2'}, {'MASTER_ADDR': 'localhost'}):
        with pytest.raises(ValueError):
            topology(dict(env, **bad))


def test_launcher_status_survives_head_exit(tmp_path):
    path = tmp_path / 'launch-id/status.json'
    assert read_status(path) == {}
    write_status(path, 'training')
    assert read_status(path)['state'] == 'training'
    write_status(path, 'finished', returncode=0)
    assert read_status(path)['returncode'] == 0


def test_only_head_launches_training_after_both_nodes_register(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts import run_multinode_ray as runner

    for key, value in dict(PA_OPD_NNODES='2', PA_OPD_GPUS_PER_NODE='16', NODE_RANK='0',
                           MASTER_ADDR='head', MASTER_PORT='23456', PA_OPD_LAUNCH_ID='test-launch',
                           OUTPUT_DIR=str(tmp_path), RAY_TMPDIR=str(tmp_path/'ray')).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, 'argv', ['run_multinode_ray.py', '--', 'python', '-m', 'trainer'])
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(runner.socket, 'gethostbyname', lambda name: '10.0.0.1')
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    def start(cmd):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(runner.subprocess, 'Popen', start)
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(
        init=lambda **kw: None, shutdown=lambda: None,
        nodes=lambda: [{'Alive':True, 'Resources':{'GPU':16}} for _ in range(2)]))
    assert runner.main() == 0
    assert len(calls) == 2 and '--head' in calls[0]
    assert calls[1] == ['python', '-m', 'trainer']
    status_path = tmp_path/'launcher/test-launch/status.json'
    assert read_status(status_path)['returncode'] == 0
    # The other pod can observe successful completion after the head exits;
    # it must not run another training driver or overwrite the final status.
    monkeypatch.setenv('NODE_RANK', '1')
    monkeypatch.setenv('MASTER_ADDR', 'head')
    monkeypatch.setenv('MASTER_PORT', '23456')
    assert runner.main() == 0
    assert len(calls) == 2


def test_final_validator_requires_32_rank_resume_and_inference_files(tmp_path):
    from scripts.run_multinode_ray import verify_final_checkpoint

    checkpoint = tmp_path/'checkpoints/global_step_210'
    checkpoint.mkdir(parents=True)
    (tmp_path/'checkpoints/latest_checkpointed_iteration.txt').write_text('210')
    paths = [checkpoint/'data.pt', checkpoint/'actor/fsdp_config.json', checkpoint/'actor/teacher/fsdp_config.json']
    for rank in range(32):
        for prefix in ('model', 'optim', 'extra_state'):
            paths.append(checkpoint/f'actor/{prefix}_world_size_32_rank_{rank}.pt')
        paths.append(checkpoint/f'actor/teacher/model_world_size_32_rank_{rank}.pt')
    for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'preprocessor_config.json'):
        paths.append(tmp_path/'inference/global_step_210'/name)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'test fixture')
    verify_final_checkpoint(tmp_path, 32)
    (checkpoint/'actor/teacher/model_world_size_32_rank_31.pt').write_bytes(b'')
    with pytest.raises(RuntimeError, match='teacher'):
        verify_final_checkpoint(tmp_path, 32)


def test_hf_exports_survive_resume_checkpoint_rotation_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path/'triton'))
    monkeypatch.setenv('TORCH_EXTENSIONS_DIR', str(tmp_path/'extensions'))
    # This test wraps FSDP, not DeepSpeed. Avoid loading unrelated vendor
    # DeepSpeed autotune extensions/atexit hooks on the CPU-only test runner.
    monkeypatch.setattr('accelerate.utils.other.is_deepspeed_available', lambda: False)
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoModelForImageTextToText, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from verl.trainer.config import CheckpointConfig
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    dist.init_process_group('gloo', rank=0, world_size=1, init_method=f'file://{tmp_path / "init"}')
    try:
        torch.manual_seed(123)
        cfg = Qwen3VLConfig(
            text_config={'vocab_size':16, 'hidden_size':16, 'intermediate_size':32,
                         'num_hidden_layers':1, 'num_attention_heads':2, 'num_key_value_heads':2,
                         'head_dim':8, 'rope_scaling':{'rope_type':'default', 'mrope_section':[1,1,2]}},
            vision_config={'hidden_size':16, 'intermediate_size':32, 'out_hidden_size':16,
                           'num_heads':2, 'depth':1, 'num_position_embeddings':16, 'patch_size':2,
                           'temporal_patch_size':1, 'spatial_merge_size':2, 'deepstack_visual_indexes':[]},
            architectures=['Qwen3VLForConditionalGeneration'])
        model = FSDP(Qwen3VLForConditionalGeneration(cfg), device_id=torch.device('cpu'), use_orig_params=True)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'<bos>':0,'<eos>':1,'A':2,'B':3})))
        manager = FSDPCheckpointManager(model=model, optimizer=optimizer, lr_scheduler=scheduler,
            processing_class=tokenizer, checkpoint_config=CheckpointConfig(
                save_contents=['model','optimizer','extra','hf_model'], hf_model_dir=str(tmp_path/'inference')))
        for step in (30, 60, 70):
            manager.save_checkpoint(str(tmp_path / 'checkpoints' / f'global_step_{step}' / 'actor'),
                                    global_step=step, max_ckpt_to_keep=2)
        assert not (tmp_path/'checkpoints/global_step_30/actor').exists()
        for step in (30, 60, 70):
            export = tmp_path/'inference'/f'global_step_{step}'
            assert (export/'tokenizer.json').is_file()
            loaded = AutoModelForImageTextToText.from_pretrained(export, dtype=torch.float32)
            with FSDP.summon_full_params(model):
                for key, value in model.module.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value)
        latest = tmp_path/'checkpoints/global_step_70/actor'
        assert (latest/'optim_world_size_1_rank_0.pt').is_file()
        assert (latest/'extra_state_world_size_1_rank_0.pt').is_file()
    finally:
        dist.destroy_process_group()
