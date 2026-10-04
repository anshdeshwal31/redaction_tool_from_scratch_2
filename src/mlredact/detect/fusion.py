"""Evidence fusion (plan §7.3): candidates -> mentions over canonical tokens.

* Candidates from every view are mapped to token index ranges of the shared token sequence.
* Overlapping candidates are merged (union of token extents); the mention type is chosen by a
  fixed priority table, then evidence strength, then rule id — all deterministic.
* Decision rule: any strong evidence, or >= 2 independent weak detectors, or a single weak hit of a
  high-risk type (person / identifier / contact / address) makes a mention.  Everything else is
  recorded as a low-evidence candidate (kept, but visible in the manifest).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlredact.core.canonical import content_id
from mlredact.core.model import Candidate, Evidence, Mention, Token
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.text.views import TextView

T = EntityType

TYPE_PRIORITY: tuple[EntityType, ...] = (
    T.MEDICARE,
    T.IHI,
    T.HPI,
    T.TFN,
    T.CRN,
    T.NDIS,
    T.DVA_FILE,
    T.PASSPORT,
    T.DRIVER_LICENCE,
    T.PROVIDER_NUMBER,
    T.AHPRA,
    T.CARD_NUMBER,
    T.BANK_ACCOUNT,
    T.ABN,
    T.ACN,
    T.VIN,
    T.VEHICLE_REG,
    T.CLINICAL_ID,
    T.COURT_FILE,
    T.EMPLOYEE_ID,
    T.REFERENCE,
    T.EMAIL,
    T.URL,
    T.PHONE,
    T.IP_ADDRESS,
    T.DATE_OF_BIRTH,
    T.DATE_OF_DEATH,
    T.PERSON,
    T.STREET_ADDRESS,
    T.ORGANISATION,
    T.LOCALITY,
    T.POSTCODE,
    T.OTHER_ID,
    T.DATE,
    T.AGE,
)
_RANK = {t: i for i, t in enumerate(TYPE_PRIORITY)}
HIGH_RISK_TYPES = frozenset(set(TYPE_PRIORITY) - {T.DATE, T.AGE, T.ORGANISATION, T.LOCALITY, T.POSTCODE})


@dataclass(frozen=True, slots=True)
class LowEvidence:
    entity_type: EntityType
    token_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class _Hit:
    first: int  # token index (inclusive)
    last: int  # token index (inclusive)
    cand: Candidate


def _to_hits(candidates: Sequence[Candidate], views: Mapping[ViewKind, TextView]) -> list[_Hit]:
    hits: list[_Hit] = []
    for c in candidates:
        rng = views[c.view].token_range(c.start, c.end)
        if len(rng) == 0:
            continue
        hits.append(_Hit(rng.start, rng.stop - 1, c))
    hits.sort(key=lambda h: (h.first, h.last, _RANK[h.cand.entity_type], h.cand.evidence))
    return hits


def _accept(hits: Sequence[_Hit]) -> bool:
    if any(h.cand.evidence.strength is EvidenceStrength.STRONG for h in hits):
        return True
    detectors = {h.cand.evidence.detector_id for h in hits}
    if len(detectors) >= 2:
        return True
    return any(h.cand.entity_type in HIGH_RISK_TYPES for h in hits)


def _choose_type(hits: Sequence[_Hit]) -> EntityType:
    best = min(
        hits,
        key=lambda h: (
            -int(h.cand.evidence.strength),
            _RANK[h.cand.entity_type],
            h.cand.evidence.rule_id,
        ),
    )
    return best.cand.entity_type


def fuse(
    candidates: Sequence[Candidate],
    views: Mapping[ViewKind, TextView],
    doc_key: str,
) -> tuple[list[Mention], list[LowEvidence]]:
    tokens = views[ViewKind.LINE].tokens
    hits = _to_hits(candidates, views)
    # Group transitively-overlapping hits (sweep over sorted token ranges).
    groups: list[list[_Hit]] = []
    current: list[_Hit] = []
    current_last = -1
    for h in hits:
        if current and h.first <= current_last:
            current.append(h)
            current_last = max(current_last, h.last)
        else:
            if current:
                groups.append(current)
            current, current_last = [h], h.last
    if current:
        groups.append(current)

    mentions: list[Mention] = []
    low: list[LowEvidence] = []
    for group in groups:
        # Within a group, keep only the hits that pass the decision rule on their own type:
        # a weak DATE overlapping a strong DOB must not decide the type.
        first = min(h.first for h in group)
        last = max(h.last for h in group)
        evidence = tuple(sorted({h.cand.evidence for h in group}))
        token_ids = tuple(tokens[i].token_id for i in range(first, last + 1))
        if not _accept(group):
            low.append(LowEvidence(_choose_type(group), token_ids, evidence))
            continue
        etype = _choose_type(group)
        value = _join(tokens[first : last + 1])
        mentions.append(
            Mention(
                mention_id=content_id("M", doc_key, token_ids[0], token_ids[-1], etype.value),
                entity_type=etype,
                token_ids=token_ids,
                evidence=evidence,
                value=value,
            )
        )
    return mentions, low


def _join(tokens: Sequence[Token]) -> str:
    return " ".join(t.text for t in tokens)
