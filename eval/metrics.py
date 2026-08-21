"""Prompt construction, answer extraction, and result aggregation."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Iterable

from .adapters import EvalSample


CHOICE_PATTERN = re.compile(r"(?<![A-Z])\(?([A-E])\)?(?![A-Z])")
COMPACT_MULTI_CHOICE_PATTERN = re.compile(r"(?<![A-Z])([A-E]{2,5})(?![A-Z])")


def build_prompt(sample: EvalSample) -> str:
    if sample.dataset == "lrs-vqa":
        return sample.question

    choices = "\n".join(sample.choices or [])
    if sample.dataset == "mme-realworld-rs":
        return (
            f"{sample.question}\n"
            "The choices are listed below:\n"
            f"{choices}\n"
            "Select the best answer to the above multiple-choice question based on the image. "
            "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
            "The best answer is:"
        )

    if sample.metadata.get("is_multi_choice"):
        instruction = (
            "This is a multiple-choice question with more than one correct answer. "
            "Respond with all correct letters only.\nThe answer is:"
        )
    else:
        instruction = (
            "Select the answer for the multiple-choice question based on the image. "
            "Respond with only the letter corresponding to the correct answer.\nThe answer is:"
        )
    return f"{sample.question}\nThe choices are listed below:\n{choices}\n{instruction}"


def normalize_freeform(text: str) -> str:
    return " ".join(text.casefold().strip(" \t\r\n.,;:!?\"'").split())


def extract_choice(text: str, choices: list[str] | None = None) -> str:
    upper = text.upper()
    match = CHOICE_PATTERN.search(upper)
    if match:
        return match.group(1)
    normalized = normalize_freeform(text)
    for choice in choices or []:
        choice_match = re.match(r"\(?([A-E])\)?[.)\s]+(.*)", choice.strip(), flags=re.IGNORECASE)
        if choice_match and normalized == normalize_freeform(choice_match.group(2)):
            return choice_match.group(1).upper()
    return ""


def extract_multi_choices(text: str) -> set[str]:
    """Extract separated and compact answers, for example ``B, C, D`` and ``BCD``."""
    upper = text.upper()
    separated = {match.group(1) for match in CHOICE_PATTERN.finditer(upper)}
    compact = {
        letter
        for match in COMPACT_MULTI_CHOICE_PATTERN.finditer(upper)
        for letter in match.group(1)
    }
    return separated | compact


def score_prediction(sample: EvalSample, prediction: str) -> dict[str, Any]:
    if sample.dataset == "lrs-vqa":
        parsed = normalize_freeform(prediction)
        return {"parsed_answer": parsed, "correct": parsed == normalize_freeform(sample.ground_truth), "metrics": {}}

    if sample.dataset == "mme-realworld-rs":
        parsed = extract_choice(prediction, sample.choices)
        return {"parsed_answer": parsed, "correct": parsed == sample.ground_truth.strip().upper(), "metrics": {}}

    if sample.metadata.get("is_multi_choice"):
        parsed_set = extract_multi_choices(prediction)
        expected_set = extract_multi_choices(sample.ground_truth)
        union = parsed_set | expected_set
        return {
            "parsed_answer": ",".join(sorted(parsed_set)),
            "correct": parsed_set == expected_set,
            "metrics": {"jaccard": len(parsed_set & expected_set) / len(union) if union else 1.0},
        }

    parsed = extract_choice(prediction, sample.choices)
    return {"parsed_answer": parsed, "correct": parsed == sample.ground_truth.strip().upper(), "metrics": {}}


def _group_summary(records: Iterable[dict[str, Any]], metadata_key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("metadata", {}).get(metadata_key, "unknown"))].append(record)
    return {name: summarize_records(items, include_breakdowns=False) for name, items in sorted(groups.items())}


def summarize_records(records: list[dict[str, Any]], *, include_breakdowns: bool = True) -> dict[str, Any]:
    total = len(records)
    successful = sum(record.get("status") == "ok" for record in records)
    correct = sum(bool(record.get("correct")) for record in records)
    inference_elapsed = sum(
        max(float(record.get("elapsed_sec", 0.0)), 0.0)
        for record in records
        if record.get("status") == "ok"
    )
    summary: dict[str, Any] = {
        "samples": total,
        "successful": successful,
        "errors": total - successful,
        "coverage": successful / total if total else 0.0,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_inference_sec": round(inference_elapsed, 3),
        "mean_inference_sec_per_sample": inference_elapsed / successful if successful else 0.0,
        "mean_inference_samples_per_sec": successful / inference_elapsed if inference_elapsed else 0.0,
    }
    jaccards = [
        float(record["metrics"]["jaccard"])
        for record in records
        if "jaccard" in record.get("metrics", {})
    ]
    if jaccards:
        summary["mean_jaccard"] = sum(jaccards) / len(jaccards)
    if include_breakdowns:
        summary["by_category"] = _group_summary(records, "category")
        if records and records[0].get("dataset") == "lrs-vqa":
            summary["by_source"] = _group_summary(records, "source")
            summary["by_size_bin"] = _group_summary(records, "size_bin")
    return summary
