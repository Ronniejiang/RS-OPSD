import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from transformers import AutoProcessor, AutoTokenizer

from verl.trainer.ppo.pa_opd_direct import (
    build_direct_option_token_mask,
    build_direct_probe_plan,
    compute_constrained_teacher_reliability,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.dataset.pa_opd_dataset import load_pa_opd_direct_jsonl
from verl.utils.reward_score.pa_opd_direct_protocol import (
    canonicalize_option_set,
    parse_direct_option_response,
)


def _model_path():
    model_path = os.environ.get("PA_OPD_MODEL_PATH")
    if not model_path:
        pytest.skip("Set PA_OPD_MODEL_PATH to a local Qwen3-VL checkpoint for tokenizer tests")
    return model_path


def _tokenizer():
    return AutoTokenizer.from_pretrained(_model_path(), trust_remote_code=True, local_files_only=True)


def test_direct_protocol_normalizes_multi_select_order_and_rejects_prose():
    assert canonicalize_option_set("B,A,C", ["A", "B", "C", "D"]) == "A,B,C"
    parsed = parse_direct_option_response(" Answer: B, A, C ", ["A", "B", "C", "D"])
    assert parsed.valid
    assert parsed.canonical_answer == "A,B,C"
    assert not parse_direct_option_response("The answer is A,B,C", ["A", "B", "C", "D"]).valid
    assert not parse_direct_option_response("A,A", ["A", "B", "C", "D"]).valid


def test_direct_qwen_bpe_mask_covers_labels_but_not_eos():
    tokenizer = _tokenizer()
    token_ids = tokenizer.encode("B,A,C", add_special_tokens=False)
    ids = torch.tensor([token_ids + [tokenizer.eos_token_id]])
    response_mask = torch.ones_like(ids)
    answer_mask, valid, parsed = build_direct_option_token_mask(
        ids,
        response_mask,
        tokenizer=tokenizer,
        option_labels_per_row=[["A", "B", "C", "D"]],
    )
    assert valid.tolist() == [True]
    assert parsed[0].canonical_answer == "A,B,C"
    assert answer_mask.sum().item() == len(token_ids)
    assert answer_mask[0, -1].item() == 0


def test_direct_constrained_probe_checks_full_set_and_terminal_eos():
    tokenizer = _tokenizer()
    plan = build_direct_probe_plan(tokenizer, "Question\n", ["A", "B", "C", "D"], "B,A,C")
    assert plan.canonical_answer == "A,B,C"
    assert len(plan.response_token_ids) == 4  # A, ,B, ,C, EOS
    assert tokenizer.eos_token_id not in plan.allowed_token_ids[0]
    assert all(candidates[0] == tokenizer.eos_token_id for candidates in plan.allowed_token_ids[1:])

    max_candidates = max(len(row) for row in plan.allowed_token_ids)
    allowed = torch.full((1, len(plan.response_token_ids), max_candidates), -1, dtype=torch.long)
    for step, candidates in enumerate(plan.allowed_token_ids):
        allowed[0, step, : len(candidates)] = torch.tensor(candidates)
    target = torch.tensor([plan.response_token_ids], dtype=torch.long)
    target_mask = torch.ones_like(target)
    vocab = max(max(plan.response_token_ids), int(allowed.max().item())) + 1
    all_log_probs = torch.full((1, target.shape[1], vocab), -30.0)
    for step, token_id in enumerate(plan.response_token_ids):
        all_log_probs[0, step, token_id] = 0.0
    reliable, probability = compute_constrained_teacher_reliability(
        all_log_probs, target, target_mask, allowed
    )
    assert reliable.tolist() == [True]
    assert probability.item() > 0.99

    # Real Qwen BPE: A -> EOS must not pass an A,B,C target, even if ,B
    # has the highest probability among the remaining option labels.
    all_log_probs[0, 1, tokenizer.eos_token_id] = 1.0
    reliable, _ = compute_constrained_teacher_reliability(all_log_probs, target, target_mask, allowed)
    assert reliable.tolist() == [False]
    all_log_probs[0, 1, tokenizer.eos_token_id] = -30.0

    wrong_token = next(token for token in plan.allowed_token_ids[-1] if token != plan.response_token_ids[-1])
    all_log_probs[0, -1, wrong_token] = 1.0
    reliable, _ = compute_constrained_teacher_reliability(all_log_probs, target, target_mask, allowed)
    assert reliable.tolist() == [False]


def test_direct_loader_keeps_original_and_teacher_image_paths():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "images").mkdir()
        (root / "teacher_images").mkdir()
        (root / "images" / "original.png").write_bytes(b"original")
        (root / "teacher_images" / "crop.png").write_bytes(b"crop")
        record = {
            "images": ["images/original.png"],
            "teacher_images": ["teacher_images/crop.png"],
            "problem": (
                "<image> Which categories apply?\n\n"
                "A. alpha\nB. beta\nC. gamma\nD. delta\n\n"
                "Answer with the option letters from the given choices, separated by commas."
            ),
            "answer": "B,A,C",
        }
        (root / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        dataset = load_pa_opd_direct_jsonl(root / "train.jsonl")
        item = dataset[0]
        assert item["images"] == [{"path": str(root / "images" / "original.png")}]
        assert item["bbox_images"] == [{"path": str(root / "teacher_images" / "crop.png")}]
        assert item["reward_model"]["ground_truth"] == "A,B,C"
        assert item["extra_info"]["option_labels"] == ["A", "B", "C", "D"]
        assert "<think>" not in item["prompt"][0]["content"]
        assert "<answer>" not in item["prompt"][0]["content"]


def test_stock_qwen_template_disables_thinking_prefix():
    processor = AutoProcessor.from_pretrained(_model_path(), trust_remote_code=True, local_files_only=True)
    text = processor.apply_chat_template(
        [{"role": "user", "content": "Choose one option."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        pa_opd_direct_answer=True,
    )
    assert "<think>" not in text
def test_direct_probe_pads_variable_length_gt_paths_before_teacher_forward():
    tokenizer = _tokenizer()
    class FakeTrainer:
        pass
    class Batch:
        def __init__(self):
            self.non_tensor_batch = {"reward_model": [{"ground_truth": "A"}, {"ground_truth": "A,B,C"}], "extra_info": [{"option_labels": ["A", "B", "C", "D"]}, {"option_labels": ["A", "B", "C", "D"]}]}
        def __len__(self):
            return len(self.non_tensor_batch["reward_model"])
    fake = FakeTrainer()
    fake.tokenizer = tokenizer
    fake.processor = None
    fake.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(self_distillation=SimpleNamespace(max_reprompt_len=10240))), data=SimpleNamespace(apply_chat_template_kwargs={"enable_thinking": False, "pa_opd_direct_answer": True}))
    def build_teacher_inputs(messages, responses, response_mask, max_prompt_len, apply_kwargs):
        prompt_len = 8522 if messages[-1]["content"] == "long" else 8
        prompt_ids = torch.full((prompt_len,), 42, dtype=torch.long)
        attention = torch.cat([torch.ones(prompt_len, dtype=torch.long), response_mask])
        input_ids = torch.cat([prompt_ids, responses])
        position_ids = torch.arange(input_ids.numel(), dtype=torch.long)
        return input_ids, attention, position_ids, torch.tensor(prompt_len), None
    fake._build_teacher_prompt_inputs = build_teacher_inputs
    fake._pad_pa_opd_prompt_inputs = lambda prefix, records, device: RayPPOTrainer._pad_pa_opd_prompt_inputs(fake, prefix, records, device)
    probe, _ = RayPPOTrainer._build_pa_opd_direct_probe_batch(fake, Batch(), [[{"role": "user", "content": "long"}], [{"role": "user", "content": "short"}]], torch.device("cpu"))
    input_ids = probe.batch["pa_opd_teacher_probe_input_ids"]
    starts = probe.batch["pa_opd_teacher_probe_response_start_idx"]
    responses = probe.batch["pa_opd_probe_responses"]
    response_mask = probe.batch["pa_opd_probe_response_mask"]
    positions = DataParallelPPOActor._build_response_positions(starts, responses.size(1), input_ids.size(1))
    assert positions.max().item() < input_ids.size(1)
    assert response_mask.sum(dim=-1).tolist() == [2, 4]
    assert input_ids.size(1) == 8526
