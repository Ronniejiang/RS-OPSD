"""Offline, optional semantic scoring for free-form LRS-VQA answers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol, Sequence


_BOOLEAN_ANSWERS = frozenset({"yes", "no", "true", "false"})
_NUMBER_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", "first", "second", "third",
        "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    }
)
_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:st|nd|rd|th)?")


@dataclass(frozen=True)
class LRSSemanticConfig:
    """Configuration recorded with an optional LRS-VQA semantic pass."""

    model_path: Path
    threshold: float = 0.85
    batch_size: int = 64

    def __post_init__(self) -> None:
        has_config = self.model_path.is_dir() and (self.model_path / "config.json").is_file()
        has_weights = any(
            (self.model_path / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        if not has_config or not has_weights:
            raise ValueError(
                f"LRS semantic model must be a complete local Hugging Face directory: {self.model_path}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("LRS semantic threshold must be in [0, 1].")
        if self.batch_size <= 0:
            raise ValueError("LRS semantic batch size must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model_path": str(self.model_path),
            "threshold": self.threshold,
            "batch_size": self.batch_size,
            "device": "cpu",
            "protected_answers": "boolean_or_number",
        }


class PairSimilarityScorer(Protocol):
    """A batched, deterministic text-pair similarity interface."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return cosine similarities, one for each input pair."""


class BGEEmbedder:
    """BGE CLS embeddings loaded locally, without network fallback."""

    def __init__(self, config: LRSSemanticConfig) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "LRS semantic scoring requires torch and transformers in the evaluation environment."
            ) from error

        self._torch = torch
        self._batch_size = config.batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        )
        self._model = AutoModel.from_pretrained(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        ).to("cpu")
        self._model.eval()
        self._cache: dict[str, Any] = {}

    def _encode(self, texts: Sequence[str]):
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            with self._torch.inference_mode():
                hidden_state = self._model(**encoded).last_hidden_state[:, 0]
                embeddings = self._torch.nn.functional.normalize(hidden_state, p=2, dim=1).cpu()
            self._cache.update(zip(batch, embeddings))
        return [self._cache[text] for text in texts]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        left = self._encode([pair[0] for pair in pairs])
        right = self._encode([pair[1] for pair in pairs])
        return [float((a * b).sum().item()) for a, b in zip(left, right)]


def normalize_freeform(text: str) -> str:
    """Match the evaluator's stable strict-normalization semantics."""
    return " ".join(text.casefold().strip(" \t\r\n.,;:!?\"'").split())


def canonicalize_lrs_answer(text: str) -> str:
    """Return a vetted LRS-VQA answer canonical form without broad stemming."""
    normalized = normalize_freeform(text)
    return _CANONICAL_LRS_ANSWERS.get(normalized, normalized)


def is_protected_answer(text: str) -> bool:
    """Keep potentially opposite boolean and numeric answers strictly exact."""
    answer = normalize_freeform(text)
    return answer in _BOOLEAN_ANSWERS or answer in _NUMBER_WORDS or bool(_NUMBER_PATTERN.fullmatch(answer))


def annotate_lrs_semantic_records(
    records: list[dict[str, Any]], scorer: PairSimilarityScorer, threshold: float
) -> None:
    """Add tolerant LRS-VQA fields in place without changing strict correctness."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("LRS semantic threshold must be in [0, 1].")

    pending: list[tuple[dict[str, Any], tuple[str, str]]] = []
    for record in records:
        if record.get("dataset") != "lrs-vqa":
            continue

        strict = bool(record.get("correct"))
        base = {
            "semantic_scored": False,
            "semantic_similarity": None,
            "semantic_match": False,
            "tolerant_correct": strict,
        }
        if record.get("status") != "ok":
            record.update(base, tolerant_match_source="error")
            continue
        if strict:
            record.update(base, tolerant_match_source="strict")
            continue

        prediction = normalize_freeform(str(record.get("prediction", "")))
        ground_truth = normalize_freeform(str(record.get("ground_truth", "")))
        if not prediction or not ground_truth:
            record.update(base, tolerant_match_source="empty")
            continue
        if is_protected_answer(prediction) or is_protected_answer(ground_truth):
            record.update(base, tolerant_match_source="strict_guard")
            continue
        canonical_prediction = canonicalize_lrs_answer(prediction)
        canonical_ground_truth = canonicalize_lrs_answer(ground_truth)
        if canonical_prediction == canonical_ground_truth:
            record.update(
                base,
                canonical_prediction=canonical_prediction,
                canonical_ground_truth=canonical_ground_truth,
                tolerant_correct=True,
                tolerant_match_source="canonical_alias",
            )
            continue
        record.update(
            canonical_prediction=canonical_prediction,
            canonical_ground_truth=canonical_ground_truth,
        )
        pending.append((record, (prediction, ground_truth)))

    similarities = scorer.score_pairs([pair for _, pair in pending])
    if len(similarities) != len(pending):
        raise ValueError("Semantic scorer returned a different number of scores than pairs.")

    for (record, _), similarity in zip(pending, similarities):
        similarity = float(similarity)
        matched = similarity >= threshold
        record.update(
            {
                "semantic_scored": True,
                "semantic_similarity": round(similarity, 6),
                "semantic_match": matched,
                "tolerant_correct": matched,
                "tolerant_match_source": "embedding" if matched else "not_matched",
            }
        )


def write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically replace a result JSONL after offline semantic annotation."""
    temporary = path.with_name(f".{path.name}.semantic-{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

# These are deliberately small, audited groups of answers that are interchangeable
# for LRS-VQA. Matching is only against a complete normalized answer: we do not
# stem arbitrary words or remove tokens, which would make the tolerant metric too
# permissive for short visual answers.
_CANONICAL_ALIAS_GROUPS = (
    ("rectangle", "rectangular"),
    ("circle", "circular"),
    ("triangle", "triangular"),
    ("cylinder", "cylindrical"),
    ("t-shaped", "t shaped"),
    ("gray", "grey"),
    ("airplane", "airplanes", "aeroplane", "aeroplanes"),
    ("taxiing", "taxying"),
    ("building", "buildings"),
    ("road", "roads"),
    ("tree", "trees"),
    ("car", "cars"),
    ("vehicle", "vehicles"),
    ("bridge", "bridges"),
    ("parked", "stationary"),
)
_CANONICAL_LRS_ANSWERS = {
    alias: group[0]
    for group in _CANONICAL_ALIAS_GROUPS
    for alias in group
}
