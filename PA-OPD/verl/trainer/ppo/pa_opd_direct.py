"""Direct-answer RS-OPSD helpers for RS-OPD multi-select questions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from verl.utils.reward_score.pa_opd_direct_protocol import (
    DirectOptionResult,
    canonicalize_option_set,
    normalize_option_labels,
    parse_direct_option_response,
)


def _input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [int(item) for item in ids]


def _one_token_suffix(tokenizer: Any, prefix: str, continuation: str) -> int:
    prefix_ids = _input_ids(tokenizer, prefix)
    full_ids = _input_ids(tokenizer, prefix + continuation)
    suffix = full_ids[len(prefix_ids) :]
    if len(suffix) != 1:
        raise ValueError(
            "Direct RS-OPSD constrained probe requires every option continuation "
            f"to be one Qwen BPE token; prefix={prefix[-32:]!r}, continuation={continuation!r}, "
            f"suffix={suffix!r}"
        )
    return suffix[0]


@dataclass(frozen=True)
class DirectProbePlan:
    """Teacher-forced GT path and legal constrained-decoding choices."""

    response_token_ids: tuple[int, ...]
    allowed_token_ids: tuple[tuple[int, ...], ...]
    canonical_answer: str


@dataclass(frozen=True)
class DirectJSDTokenMetadata:
    """GT-aligned direct-answer positions used by safe top-k JSD."""

    target_token_ids: torch.Tensor
    allowed_token_ids: torch.Tensor
    prefix_mask: torch.Tensor


def build_direct_probe_plan(
    tokenizer: Any,
    prompt_text: str,
    option_labels: Iterable[Any],
    ground_truth: Any,
) -> DirectProbePlan:
    """Build CAD's GT-prefix constrained-greedy Teacher reliability probe.

    At every generated label position the Teacher competes only against labels
    that make a valid canonical option set.  After each selected label it may
    either emit EOS or append a later unselected label.  This checks the exact
    multi-select answer, including whether Teacher stops at the correct set,
    without generating a separate Teacher trajectory. The first step requires
    a label (no empty answer); every later step allows EOS. Ties after the
    first label prefer EOS, consistently with the terminal step.
    """

    labels = normalize_option_labels(option_labels)
    canonical = canonicalize_option_set(ground_truth, labels)
    selected = tuple(canonical.split(","))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Direct RS-OPSD probe requires a tokenizer EOS token")

    response_ids: list[int] = []
    allowed_ids: list[tuple[int, ...]] = []
    prefix_answer = ""
    selected_positions = [labels.index(label) for label in selected]
    for selected_index, label in enumerate(selected):
        if selected_index == 0:
            legal_text = list(labels)
            target_text = label
        else:
            previous_position = selected_positions[selected_index - 1]
            legal_text = ["," + candidate for candidate in labels[previous_position + 1 :]]
            target_text = "," + label
        legal_ids = tuple(_one_token_suffix(tokenizer, prompt_text + prefix_answer, text) for text in legal_text)
        if selected_index > 0:
            # Early termination must compete with the next GT label; otherwise
            # a Teacher preferring only A could incorrectly pass a GT A,B probe.
            legal_ids = (int(eos_token_id),) + legal_ids
        target_id = _one_token_suffix(tokenizer, prompt_text + prefix_answer, target_text)
        if target_id not in legal_ids:
            raise RuntimeError("GT answer transition is absent from its constrained probe grammar")
        allowed_ids.append(legal_ids)
        response_ids.append(target_id)
        prefix_answer += target_text

    last_position = selected_positions[-1]
    terminal_legal = [int(eos_token_id)] + [
        _one_token_suffix(tokenizer, prompt_text + prefix_answer, "," + candidate)
        for candidate in labels[last_position + 1 :]
    ]
    allowed_ids.append(tuple(terminal_legal))
    response_ids.append(int(eos_token_id))
    return DirectProbePlan(tuple(response_ids), tuple(allowed_ids), canonical)


def build_direct_option_token_mask(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    tokenizer: Any,
    option_labels_per_row: Sequence[Sequence[Any]],
) -> tuple[torch.Tensor, torch.Tensor, list[DirectOptionResult]]:
    """Mask direct-answer BPE tokens, excluding EOS and punctuation-only tokens."""

    if response_ids.shape != response_mask.shape:
        raise ValueError("response_ids and response_mask must have identical shapes")
    if len(option_labels_per_row) != response_ids.shape[0]:
        raise ValueError("option_labels_per_row must provide one option set per response")

    answer_mask = torch.zeros_like(response_mask)
    valid = torch.zeros(response_ids.shape[0], dtype=torch.bool, device=response_ids.device)
    parsed_results: list[DirectOptionResult] = []
    special_token_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}

    for row_idx, row_labels in enumerate(option_labels_per_row):
        active_positions = response_mask[row_idx].bool().nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        if not active_positions:
            parsed_results.append(DirectOptionResult(False, (), None, ()))
            continue
        active_ids = response_ids[row_idx, active_positions].detach().cpu().tolist()
        content_pairs = [
            (position, int(token_id))
            for position, token_id in zip(active_positions, active_ids, strict=True)
            if int(token_id) not in special_token_ids
        ]
        if not content_pairs:
            parsed_results.append(DirectOptionResult(False, (), None, ()))
            continue
        last_content_position = content_pairs[-1][0]
        if any(
            int(token_id) in special_token_ids and position < last_content_position
            for position, token_id in zip(active_positions, active_ids, strict=True)
        ):
            parsed_results.append(DirectOptionResult(False, (), None, ()))
            continue

        content_positions = [position for position, _ in content_pairs]
        token_ids = [token_id for _, token_id in content_pairs]
        text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        parsed = parse_direct_option_response(text, row_labels)
        parsed_results.append(parsed)
        if not parsed.valid:
            continue

        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        roundtrip_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        if roundtrip_ids != token_ids:
            raise ValueError(
                "Direct RS-OPSD cannot align decoded response text back to Qwen BPE ids; "
                f"row={row_idx}, original_len={len(token_ids)}, roundtrip_len={len(roundtrip_ids)}"
            )
        for content_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in parsed.label_spans):
                answer_mask[row_idx, content_positions[content_index]] = 1
        valid[row_idx] = bool(answer_mask[row_idx].any())

    return answer_mask.to(dtype=response_mask.dtype), valid, parsed_results


def build_direct_jsd_token_metadata(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    answer_token_mask: torch.Tensor,
    plans: Sequence[DirectProbePlan],
) -> DirectJSDTokenMetadata:
    """Map a rollout onto the GT grammar until its first answer divergence."""

    if response_ids.shape != response_mask.shape or response_ids.shape != answer_token_mask.shape:
        raise ValueError("response ids, mask, and answer mask must have identical shapes")
    if len(plans) != response_ids.shape[0]:
        raise ValueError("plans must provide one GT grammar per response")

    batch_size, response_length = response_ids.shape
    max_candidates = max(len(candidates) for plan in plans for candidates in plan.allowed_token_ids)
    target_token_ids = torch.full_like(response_ids, -1)
    allowed_token_ids = torch.full(
        (batch_size, response_length, max_candidates),
        -1,
        dtype=torch.long,
        device=response_ids.device,
    )
    prefix_mask = torch.zeros_like(response_mask)

    for row_idx, plan in enumerate(plans):
        plan_step = 0
        prefix_matches_gt = True
        active_positions = response_mask[row_idx].bool().nonzero(as_tuple=False).flatten().tolist()
        for position in active_positions:
            token_id = int(response_ids[row_idx, position].item())
            is_label = bool(answer_token_mask[row_idx, position].item())
            is_eos = token_id == plan.response_token_ids[-1]
            if not is_label and not is_eos:
                continue
            if plan_step >= len(plan.response_token_ids):
                break
            if prefix_matches_gt:
                expected_id = plan.response_token_ids[plan_step]
                prefix_mask[row_idx, position] = 1
                target_token_ids[row_idx, position] = expected_id
                candidates = plan.allowed_token_ids[plan_step]
                allowed_token_ids[row_idx, position, : len(candidates)] = torch.tensor(
                    candidates, dtype=torch.long, device=response_ids.device
                )
                if token_id != expected_id:
                    prefix_matches_gt = False
            if is_eos:
                break
            if is_label:
                plan_step += 1

    return DirectJSDTokenMetadata(target_token_ids, allowed_token_ids, prefix_mask)


def compute_constrained_teacher_reliability(
    all_log_probs: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_mask: torch.Tensor,
    allowed_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard constrained-greedy reliability and GT token probability.

    ``allowed_token_ids`` is padded with ``-1``.  The final target position is
    EOS, so a reliable Teacher must also stop rather than append an extra label.
    """

    if all_log_probs.ndim != 3:
        raise ValueError("all_log_probs must be [batch, response, vocab]")
    if target_token_ids.shape != target_mask.shape or target_token_ids.shape != all_log_probs.shape[:2]:
        raise ValueError("target ids/mask must match all_log_probs [batch, response]")
    if allowed_token_ids.shape[:2] != target_token_ids.shape or allowed_token_ids.ndim != 3:
        raise ValueError("allowed_token_ids must be [batch, response, candidates]")

    active = target_mask.to(dtype=torch.bool)
    safe_allowed = allowed_token_ids.clamp_min(0).to(device=all_log_probs.device, dtype=torch.long)
    candidate_log_probs = all_log_probs.gather(dim=-1, index=safe_allowed)
    candidate_log_probs = candidate_log_probs.masked_fill(
        allowed_token_ids.to(device=all_log_probs.device).lt(0), float("-inf")
    )
    predicted = candidate_log_probs.argmax(dim=-1)
    predicted_ids = safe_allowed.gather(dim=-1, index=predicted.unsqueeze(-1)).squeeze(-1)
    target_ids = target_token_ids.to(device=all_log_probs.device, dtype=torch.long)
    reliable = ((predicted_ids.eq(target_ids)) | ~active).all(dim=-1).detach()
    target_log_probs = all_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    token_probability = torch.exp(target_log_probs.masked_select(active).mean()) if active.any() else torch.tensor(
        0.0, device=all_log_probs.device
    )
    return reliable, token_probability.detach()


__all__ = [
    "DirectJSDTokenMetadata",
    "DirectProbePlan",
    "build_direct_jsd_token_metadata",
    "build_direct_option_token_mask",
    "build_direct_probe_plan",
    "compute_constrained_teacher_reliability",
]
