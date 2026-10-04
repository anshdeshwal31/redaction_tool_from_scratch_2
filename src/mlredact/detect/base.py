"""Detector interface.  Every detector is pure and deterministic: same views -> same candidates."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Protocol

from mlredact.core.model import Candidate, Evidence
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.text.views import TextView


class Detector(Protocol):
    detector_id: str
    version: str

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]: ...


def candidate(
    detector: Detector,
    entity_type: EntityType,
    view: ViewKind,
    start: int,
    end: int,
    rule_id: str,
    strength: EvidenceStrength,
    score: float = 1.0,
) -> Candidate:
    return Candidate(
        entity_type=entity_type,
        view=view,
        start=start,
        end=end,
        evidence=Evidence(detector.detector_id, detector.version, rule_id, strength, round(score, 4)),
    )


def trim_span(text: str, start: int, end: int, strip: str = " \t,;:.()[]{}\"'") -> tuple[int, int]:
    """Shrink a span so it does not start/end with punctuation or spaces."""
    while start < end and text[start] in strip:
        start += 1
    while end > start and text[end - 1] in strip:
        end -= 1
    return start, end


def finditer_lines(pattern: re.Pattern[str], text: str) -> Iterable[re.Match[str]]:
    """``pattern.finditer`` – patterns are written with ``[ \\t]`` so matches never span lines."""
    return pattern.finditer(text)


def line_prefix(text: str, pos: int, max_chars: int = 40) -> str:
    """Text before ``pos`` on the same line (up to ``max_chars``) – used for context cues."""
    line_start = text.rfind("\n", 0, pos) + 1
    return text[max(line_start, pos - max_chars) : pos]


def line_suffix(text: str, pos: int, max_chars: int = 40) -> str:
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    return text[pos : min(line_end, pos + max_chars)]
