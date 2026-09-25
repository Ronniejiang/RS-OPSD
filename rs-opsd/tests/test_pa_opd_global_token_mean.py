"""Global masked-token reduction must not depend on micro-batch/DP partitioning."""
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss, compute_cad_loss, compute_pa_opd_safe_topk_jsd_loss, kl_penalty
from verl.trainer.ppo.pa_opd_reduction import GlobalTokenMean, collect_global_token_mean


def fixture_batch(n=32):
    generator = torch.Generator().manual_seed(2026)
    logits = torch.randn(n, 5, 6, generator=generator, dtype=torch.float64) * .2
    teacher = logits.clone()
    teacher[..., 0] += 1.5
    teacher = teacher.log_softmax(-1)
    ref = (logits + torch.randn(n, 5, 6, generator=generator, dtype=torch.float64) * .3).log_softmax(-1)
    lengths = torch.arange(n) % 4  # includes zero-semantic rows
    positions = torch.arange(5)[None, :]
    batch = {
        "response_mask": (positions < (lengths + 1)[:, None]).double(),
        "pa_opd_semantic_token_mask": (positions < lengths[:, None]).double(),
        "pa_opd_jsd_prefix_mask": (positions < ((lengths % 2) + 1)[:, None]).double(),
        "self_distillation_mask": (torch.arange(n) % 5 != 0).double(),
    }
    return logits, teacher, ref, batch


def terms(logits, teacher, ref, batch, norm, reliable=None):
    logp = logits.log_softmax(-1)
    sampled = logp[..., 0]
    n, length, vocab = logp.shape
    if reliable is None:
        reliable = torch.ones(n)
    ids = torch.arange(vocab).expand(n, length, vocab)
    cfg = SimpleNamespace(is_clip=2.)
    shared = dict(teacher_present_mask=batch["self_distillation_mask"], old_log_probs=sampled.detach())
    cad = compute_cad_loss(sampled, teacher[..., 0], batch["pa_opd_semantic_token_mask"],
                              torch.ones(n, dtype=torch.bool), reliable, cfg,
                              **shared, **norm.kwargs("semantic"))[0]
    jsd = compute_pa_opd_safe_topk_jsd_loss(
        logp, teacher, ids, torch.ones_like(ids, dtype=torch.bool),
        torch.zeros((n, length), dtype=torch.long), torch.arange(4).expand(n, length, 4),
        batch["pa_opd_jsd_prefix_mask"], reliable, cfg, student_log_probs=sampled,
        **shared, **norm.kwargs("jsd_prefix"),
    )[0]
    kl = agg_loss(kl_penalty(sampled, ref[..., 0], 'low_var_kl'), batch['response_mask'],
                  'token-mean', **norm.kwargs('response'))
    return torch.stack((cad, jsd, kl))


@pytest.mark.parametrize('world,micro', [(1,1), (1,3), (2,1), (2,5), (16,1), (16,2)])
def test_unequal_length_rank_and_micro_partition_matches_global_batch(world, micro):
    base, teacher, ref, batch = fixture_batch()
    norm = collect_global_token_mean(batch, device='cpu')
    expected_z = base.clone().requires_grad_()
    expected = terms(expected_z, teacher, ref, batch, norm)
    expected_grad = torch.autograd.grad(expected @ torch.tensor([1.,1.,.001], dtype=torch.float64), expected_z)[0]
    z = base.clone().requires_grad_()
    actual = torch.zeros(3, dtype=torch.float64)
    # Global counts are shared; FSDP averages the resulting local gradients.
    rank_norm = GlobalTokenMean(norm.counts, world)
    ordering = torch.randperm(32, generator=torch.Generator().manual_seed(5))
    for rank_rows in ordering.chunk(world):
        for rows in rank_rows.split(micro):
            actual = actual + terms(z[rows], teacher[rows], ref[rows],
                                    {k:v[rows] for k,v in batch.items()}, rank_norm) / world
    actual_grad = torch.autograd.grad(actual @ torch.tensor([1.,1.,.001], dtype=torch.float64), z)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-10)


def test_counts_include_presence_not_reliability_and_single_collective(monkeypatch):
    from verl.trainer.ppo import pa_opd_reduction as reduction
    _, _, _, batch = fixture_batch(8)
    batch['pa_opd_teacher_reliable'] = torch.zeros(8)  # not a denominator gate
    local = collect_global_token_mean(batch, device='cpu').counts
    calls = []
    monkeypatch.setattr(reduction.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(reduction.dist, 'get_world_size', lambda group: 16)
    def all_reduce(counts, **kwargs):
        calls.append(counts.clone())
        counts.add_(torch.tensor([7., 11., 19.]))  # remote ranks' total
    monkeypatch.setattr(reduction.dist, 'all_reduce', all_reduce)
    norm = collect_global_token_mean(batch, device='cpu')
    assert len(calls) == 1 and norm.dp_size == 16
    torch.testing.assert_close(norm.counts, local + torch.tensor([7., 11., 19.]))
    assert local[2] == batch['response_mask'].sum()
    assert local[0] == (batch['pa_opd_semantic_token_mask'] * batch['self_distillation_mask'][:,None]).sum()


def test_empty_and_unreliable_terms_are_finite_with_zero_gradients():
    base, teacher, ref, batch = fixture_batch(8)
    for empty in (False, True):
        z = base.clone().requires_grad_()
        masks = {k: torch.zeros_like(v) if empty else v for k,v in batch.items()}
        norm = collect_global_token_mean(masks, device='cpu')
        losses = terms(z, teacher, ref, masks, norm, reliable=torch.zeros(8))
        assert torch.isfinite(losses).all()
        torch.testing.assert_close(losses[:2], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(torch.autograd.grad(losses[:2].sum(), z)[0], torch.zeros_like(z))
        if empty:
            assert losses[2].item() == 0


@pytest.mark.parametrize('dynamic,micro', [(True,1), (True,3), (False,1), (False,2), (False,8)])
@pytest.mark.parametrize('jsd_enabled', [False, True])
def test_actor_update_backward_and_loss_metrics_match_full_batch(monkeypatch, dynamic, micro, jsd_enabled):
    """Run the real update loop; only model forward/optimizer are tiny CPU doubles."""
    from pathlib import Path
    import numpy as np
    from hydra import compose, initialize_config_dir
    from verl import DataProto
    from verl.workers.actor import dp_actor
    from verl.utils.profiler import performance

    base, teacher, ref, masks = fixture_batch(8)
    expected_z = base.clone().requires_grad_()
    expected_terms = terms(expected_z, teacher, ref, masks, collect_global_token_mean(masks, device='cpu'))
    expected_loss = expected_terms @ torch.tensor([1.,float(jsd_enabled),.001], dtype=torch.float64)
    expected_grad = torch.autograd.grad(expected_loss, expected_z)[0]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='rs_opsd_direct_2k_three_image_topk64_jsd_kl').actor_rollout_ref.actor
    config.ppo_mini_batch_size = 8
    config.ppo_micro_batch_size_per_gpu = micro
    config.use_dynamic_bsz = dynamic
    config.self_distillation.pa_opd_topk_jsd_enabled = jsd_enabled
    actor = dp_actor.DataParallelPPOActor.__new__(dp_actor.DataParallelPPOActor)
    actor.config = config
    actor.actor_module = torch.nn.Module()
    actor.actor_module.register_parameter('logits', torch.nn.Parameter(base.clone()))
    actor.teacher_module = torch.nn.Linear(1, 1)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.)
    actor.scaler = None
    actor.ulysses_sequence_parallel_size = 1
    actor.use_prefix_grouper = False
    actor.use_fused_kernels = False
    actor._update_teacher = lambda: None
    saved = {}
    def step():
        saved['grad'] = actor.actor_module.logits.grad.clone()
        return saved['grad'].norm()
    actor._optimizer_step = step
    def forward(inputs, **kwargs):
        rows = inputs['input_ids'][:,0].long()
        lp = (actor.actor_module.logits[rows].log_softmax(-1)
              if kwargs.get('module') is None else teacher[rows])
        ids = torch.arange(6).expand(len(rows),5,6)
        return {'log_probs':lp[...,0], 'entropys':None, 'all_logps':lp,
                'topk_logps':lp, 'topk_indices':ids, 'topk_valid_mask':torch.ones_like(ids, dtype=torch.bool)}
    actor._forward_micro_batch = forward
    monkeypatch.setattr(dp_actor, 'get_device_id', lambda: 'cpu')
    monkeypatch.setattr(performance, '_get_current_mem_info', lambda: (0,0,0,0))
    monkeypatch.setattr(dp_actor, 'prepare_dynamic_batch', lambda batch, **kw: (batch.split(micro), None))
    row_ids = torch.arange(8)[:,None].expand(8,5)
    tensors = dict(masks)
    for key in ('input_ids', 'teacher_input_ids', 'pa_opd_teacher_probe_input_ids'):
        tensors[key] = row_ids
    for key in ('position_ids', 'teacher_position_ids', 'pa_opd_teacher_probe_position_ids',
                'responses', 'pa_opd_probe_responses', 'pa_opd_jsd_target_token_ids'):
        tensors[key] = torch.zeros(8,5, dtype=torch.long)
    for key in ('attention_mask', 'teacher_attention_mask', 'pa_opd_teacher_probe_attention_mask', 'pa_opd_probe_response_mask'):
        tensors[key] = torch.ones(8,5)
    for key in ('teacher_response_start_idx', 'pa_opd_teacher_probe_response_start_idx'):
        tensors[key] = torch.zeros(8, dtype=torch.long)
    for key in ('pa_opd_jsd_allowed_token_ids', 'pa_opd_probe_allowed_token_ids'):
        tensors[key] = torch.arange(4).expand(8,5,4)
    tensors['pa_opd_answer_token_mask'] = masks['pa_opd_semantic_token_mask']
    tensors['pa_opd_trajectory_correct'] = torch.ones(8, dtype=torch.bool)
    tensors['old_log_probs'] = base.log_softmax(-1)[...,0]
    tensors['ref_log_prob'] = ref[...,0]
    data = DataProto.from_dict(tensors=tensors,
                              non_tensors={'pa_opd_teacher_probe_multi_modal_inputs':np.array([None]*8, dtype=object)},
                              meta_info={'temperature':1.0})
    metrics = actor.update_policy(data)
    torch.testing.assert_close(saved['grad'], expected_grad, atol=1e-12, rtol=1e-10)
    assert metrics['actor/cad_loss'] == pytest.approx(expected_terms[0].item())
    assert metrics['actor/distillation_loss'] == pytest.approx((expected_terms[0] + jsd_enabled * expected_terms[1]).item())
    if jsd_enabled:
        assert metrics['actor/topk_jsd_loss'] == pytest.approx(expected_terms[1].item())
    assert metrics['actor/kl_loss'] == pytest.approx(expected_terms[2].item())
    assert metrics['actor/pg_loss'] == pytest.approx((expected_terms[0] + jsd_enabled * expected_terms[1]).item())
    assert metrics['pa_opd/global_token_mean'] == 1.
    expected_counts = collect_global_token_mean(masks, device='cpu').counts
    assert metrics['pa_opd/global_semantic_tokens'] == expected_counts[0].item()
    assert metrics['pa_opd/global_jsd_prefix_tokens'] == (expected_counts[1].item() if jsd_enabled else 0.)
    assert metrics['pa_opd/global_response_tokens'] == expected_counts[2].item()


def _fsdp_global_token_mean_worker(rank, init_file):
    """Exercise real reduce-scatter/gradient accumulation, not a simulated AVG."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method=f'file://{init_file}')
    try:
        for case in ('unequal', 'empty_rank', 'all_empty'):
            features, teacher, ref, masks = fixture_batch()
            features, teacher, ref = features.float(), teacher.float(), ref.float()
            masks = {key: mask.float() for key, mask in masks.items()}
            if case == 'empty_rank':
                for mask in masks.values():
                    mask[16:] = 0
            elif case == 'all_empty':
                masks = {key: torch.zeros_like(mask) for key, mask in masks.items()}
            present = masks['self_distillation_mask'][:, None]
            counts = torch.stack(((masks['pa_opd_semantic_token_mask'] * present).sum(),
                                  (masks['pa_opd_jsd_prefix_mask'] * present).sum(),
                                  masks['response_mask'].sum()))
            torch.manual_seed(37)
            reference = torch.nn.Linear(6, 6, bias=False)
            local_model = torch.nn.Linear(6, 6, bias=False)
            local_model.load_state_dict(reference.state_dict())
            model = FSDP(local_model, device_id=torch.device('cpu'), use_orig_params=True)
            coefficients = torch.tensor([1., 1., .001])
            expected = terms(reference(features), teacher, ref, masks, GlobalTokenMean(counts, 1))
            (expected @ coefficients).backward()
            expected_grad = reference.weight.grad.clone()
            rows = torch.arange(rank * 16, (rank + 1) * 16)
            norm = collect_global_token_mean({k: v[rows] for k, v in masks.items()}, device='cpu')
            torch.testing.assert_close(norm.counts, counts)
            actual = torch.zeros(3)
            for micro_rows in rows.split(4):
                losses = terms(model(features[micro_rows]), teacher[micro_rows], ref[micro_rows],
                               {k: v[micro_rows] for k, v in masks.items()}, norm)
                (losses @ coefficients).backward()
                actual += losses.detach()
            dist.all_reduce(actual)
            torch.testing.assert_close(actual / 2, expected, atol=1e-7, rtol=1e-5)
            with FSDP.summon_full_params(model, with_grads=True):
                torch.testing.assert_close(model.module.weight.grad, expected_grad, atol=1e-7, rtol=1e-5)
            reported_norm = model.clip_grad_norm_(max_norm=.01)
            torch.testing.assert_close(reported_norm, expected_grad.norm(), atol=1e-7, rtol=1e-5)
            with FSDP.summon_full_params(model, with_grads=True):
                clipped = expected_grad * min(1., .01 / (expected_grad.norm().item() + 1e-6))
                torch.testing.assert_close(model.module.weight.grad, clipped, atol=1e-7, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_real_two_process_fsdp_global_token_mean(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_fsdp_global_token_mean_worker, args=(str(tmp_path / 'gloo_init'),), nprocs=2, join=True)
