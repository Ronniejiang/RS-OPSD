"""Direct option-set protocol used by reward-free RS-OPD OPDVR.

Unlike the adaptive PA-OPD recipe, this protocol does not create or validate
``<think>``/``<answer>`` tags.  A completion is useful only when it consists
of a comma-separated set of option labels (optionally prefixed by ``Answer:``).
The set is canonicalised to the option order from the individual question, so
the RS-OPD annotation ``B,A,C`` and a model completion ``A,B,C`` are equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_DIRECT_ANSWER_RE = re.compile(
    r"\A\s*(?:answer\s*:\s*)?(?P<labels>[A-Za-z](?:\s*,\s*[A-Za-z])*)\s*\Z",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class DirectOptionResult:
    """Parsed direct-option completion with spans in the source string."""

    valid: bool
    labels: tuple[str, ...]
    canonical_answer: str | None
    label_spans: tuple[tuple[int, int], ...]


def normalize_option_labels(option_labels: Iterable[Any]) -> tuple[str, ...]:
    """Validate and normalise the ordered labels exposed by one question."""

    labels = tuple(str(label).strip().upper() for label in option_labels)
    if not labels or any(not re.fullmatch(r"[A-Z]", label) for label in labels):
        raise ValueError(f"Option labels must be non-empty single letters, got {labels!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Option labels must be unique, got {labels!r}")
    return labels


def canonicalize_option_set(value: Any, option_labels: Iterable[Any]) -> str:
    """Return an order-canonical comma-separated option set or raise.

    This is intentionally shared by dataset GT processing and generated-answer
    evaluation so that their equality criterion cannot drift.
    """

    labels = normalize_option_labels(option_labels)
    raw = "" if value is None else str(value)
    match = _DIRECT_ANSWER_RE.fullmatch(raw)
    if match is None:
        raise ValueError(f"Expected comma-separated option labels, got {value!r}")
    selected = tuple(item.upper() for item in _LABEL_RE.findall(match.group("labels")))
    if not selected:
        raise ValueError("An answer must select at least one option label")
    unknown = sorted(set(selected).difference(labels))
    if unknown:
        raise ValueError(f"Answer labels {unknown!r} are absent from question labels {labels!r}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Answer labels must not repeat, got {selected!r}")
    selected_set = set(selected)
    return ",".join(label for label in labels if label in selected_set)


def parse_direct_option_response(value: Any, option_labels: Iterable[Any]) -> DirectOptionResult:
    """Parse a direct model completion and expose selected-label character spans."""

    raw = "" if value is None else str(value)
    match = _DIRECT_ANSWER_RE.fullmatch(raw)
    if match is None:
        return DirectOptionResult(False, (), None, ())
    try:
        labels = tuple(item.upper() for item in _LABEL_RE.findall(match.group("labels")))
        canonical = canonicalize_option_set(match.group("labels"), option_labels)
    except ValueError:
        return DirectOptionResult(False, (), None, ())

    start = match.start("labels")
    spans = tuple((start + item.start(), start + item.end()) for item in _LABEL_RE.finditer(match.group("labels")))
    return DirectOptionResult(True, labels, canonical, spans)


__all__ = [
    "DirectOptionResult",
    "canonicalize_option_set",
    "normalize_option_labels",
    "parse_direct_option_response",
]
