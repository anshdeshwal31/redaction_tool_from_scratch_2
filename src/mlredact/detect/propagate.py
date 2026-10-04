"""Document-level propagation (plan §8/§9): one confident hit finds every other occurrence.

Values of confirmed mentions (and job-supplied seeds) are searched over every line:

* **exact** comparison keys (case/punctuation/space-insensitive) over token windows of size
  n-1..n+1, because OCR splits and merges tokens ("WC 1234567" vs "WC1234567");
* **alternate** readings from the OCR decoder (per-token runner-up characters);
* **fuzzy** matching for longer values with a length-scaled edit budget, on windows of the *same*
  token count only, so a match can never absorb neighbouring label words.

Windows are minimal: a window that contains an already-matched window for the same target, or that
has a token contributing no alphanumerics, is never emitted.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from mlredact.core.model import Candidate, Evidence, Mention, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, ViewKind
from mlredact.text.normalize import match_key
from mlredact.text.views import TextView

DETECTOR_ID = "propagation"
DETECTOR_VERSION = "2"
_MAX_ALT_COMBOS = 64


@dataclass(frozen=True, slots=True)
class Target:
    key: str
    n_tokens: int
    entity_type: EntityType
    source: str  # "mention" | "seed"


def max_edits(key_len: int) -> int:
    """Edit budget for OCR-tolerant matching, scaled with length (short values: exact only)."""
    if key_len < 6:
        return 0
    if key_len < 10:
        return 1
    if key_len < 16:
        return 2
    return 3


def line_ranges(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(tokens) + 1):
        if i == len(tokens) or (tokens[i].page, tokens[i].line_no) != (tokens[start].page, tokens[start].line_no):
            ranges.append((start, i))
            start = i
    return ranges


def targets_from_mentions(
    mentions: Sequence[Mention], actions: Mapping[EntityType, ActionKind], min_len: int
) -> list[Target]:
    out: dict[tuple[str, EntityType], Target] = {}
    for m in mentions:
        if actions.get(m.entity_type, ActionKind.KEEP) is ActionKind.KEEP:
            continue
        if not any(e.strength is EvidenceStrength.STRONG for e in m.evidence):
            continue
        key = match_key(m.value)
        if len(key) < min_len:
            continue
        out.setdefault((key, m.entity_type), Target(key, len(m.token_ids), m.entity_type, "mention"))
    return [out[k] for k in sorted(out, key=lambda k: (k[0], k[1].value))]


def _readings(window: Sequence[Token]) -> list[str]:
    choices = [(t.text, *t.alternates) for t in window]
    return [match_key(" ".join(c)) for c in itertools.islice(itertools.product(*choices), _MAX_ALT_COMBOS)]


def propagate(
    targets: Sequence[Target],
    view: TextView,
    *,
    fuzzy_ratio: float = 0.0,
) -> list[Candidate]:
    """Find occurrences of ``targets``.  ``fuzzy_ratio`` is an additional similarity floor applied
    on top of the edit budget (kept for configuration compatibility)."""
    tokens = view.tokens
    token_keys = [match_key(t.text) for t in tokens]
    # Index targets: exact lookups by (window size, key); fuzzy candidates by window size.
    exact_index: dict[tuple[int, str], list[int]] = {}
    fuzzy_by_size: dict[int, list[int]] = {}
    sizes: set[int] = set()
    for ti, t in enumerate(targets):
        for size in {max(1, t.n_tokens - 1), t.n_tokens, t.n_tokens + 1}:
            exact_index.setdefault((size, t.key), []).append(ti)
            sizes.add(size)
        if max_edits(len(t.key)):
            fuzzy_by_size.setdefault(t.n_tokens, []).append(ti)
    out: list[Candidate] = []
    for a, b in line_ranges(tokens):
        matched: dict[int, list[tuple[int, int]]] = {}
        for size in sorted(sizes):
            for i in range(a, b - size + 1):
                j = i + size
                hits: list[tuple[int, str, float]] = []
                readings = _readings(tokens[i:j])
                seen_targets: set[int] = set()
                for r_i, reading in enumerate(readings):
                    for ti in exact_index.get((size, reading), ()):
                        if ti not in seen_targets:
                            seen_targets.add(ti)
                            hits.append((ti, "exact" if r_i == 0 else "alternate", 1.0))
                primary = readings[0]
                for ti in fuzzy_by_size.get(size, ()):
                    if ti in seen_targets:
                        continue
                    key = targets[ti].key
                    budget = max_edits(len(key))
                    if abs(len(primary) - len(key)) > budget:
                        continue
                    dist = Levenshtein.distance(primary, key, score_cutoff=budget)
                    similarity = 1.0 - dist / max(len(primary), len(key), 1)
                    if dist <= budget and similarity >= fuzzy_ratio:
                        hits.append((ti, "fuzzy", similarity))
                for ti, rule, score in sorted(hits):
                    target = targets[ti]
                    windows = matched.setdefault(ti, [])
                    if any(s >= i and e <= j for s, e in windows):
                        continue  # minimal windows only
                    if size != target.n_tokens and any(not token_keys[k] for k in range(i, j)):
                        continue  # a resized window must not absorb punctuation-only tokens
                    windows.append((i, j))
                    out.append(
                        Candidate(
                            entity_type=target.entity_type,
                            view=ViewKind.LINE,
                            start=view.starts[i],
                            end=view.ends[j - 1],
                            evidence=Evidence(
                                DETECTOR_ID if target.source == "mention" else "seed",
                                DETECTOR_VERSION,
                                f"{target.source}.{rule}",
                                EvidenceStrength.STRONG,
                                round(score, 4),
                            ),
                        )
                    )
    return out
