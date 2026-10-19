"""Detection orchestration: views -> detectors -> fusion -> propagation (fixpoint) -> extension."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mlredact.config.schema import AppConfig
from mlredact.core.canonical import content_id
from mlredact.core.model import Candidate, Mention, PageAnalysis
from mlredact.core.types import EntityType, ViewKind
from mlredact.detect.base import Detector
from mlredact.detect.fusion import LowEvidence, fuse
from mlredact.detect.propagate import DETECTOR_ID as PROPAGATION_ID
from mlredact.detect.propagate import DETECTOR_VERSION as PROPAGATION_VERSION
from mlredact.detect.propagate import Target, propagate, targets_from_mentions
from mlredact.detect.rules.address import AddressDetector
from mlredact.detect.rules.ages import AgeDetector
from mlredact.detect.rules.au_ids import AuIdentifierDetector
from mlredact.detect.rules.contact import ContactDetector
from mlredact.detect.rules.dates import DateDetector
from mlredact.detect.rules.generic_id import GenericIdDetector
from mlredact.detect.rules.labels import LabelValueDetector
from mlredact.detect.rules.persons import PersonCueDetector
from mlredact.resolve.persons import PersonResolution, component_candidates, resolve_persons
from mlredact.security.sensitive import Sensitive
from mlredact.text.normalize import match_key
from mlredact.text.views import TextView, build_views

_MAX_PROPAGATION_ROUNDS = 4


@dataclass(frozen=True, slots=True)
class Seed:
    """A job-supplied known identifier (e.g. the claimant's name or claim number)."""

    entity_type: EntityType
    value: Sensitive[str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    mentions: tuple[Mention, ...]
    low_evidence: tuple[LowEvidence, ...]
    detectors: tuple[tuple[str, str], ...]
    propagation_rounds: int
    persons: PersonResolution
    line_view: TextView = field(repr=False)


def default_detectors() -> tuple[Detector, ...]:
    return (
        AuIdentifierDetector(),
        ContactDetector(),
        DateDetector(),
        LabelValueDetector(),
        PersonCueDetector(),
        AddressDetector(),
        GenericIdDetector(),
        AgeDetector(),
    )


def _extend_uncertain(mentions: Sequence[Mention], view: TextView, threshold: float, doc_key: str) -> list[Mention]:
    """Neighbour rule: absorb adjacent same-line tokens whose weakest character is uncertain."""
    index = {t.token_id: i for i, t in enumerate(view.tokens)}
    tokens = view.tokens
    out: list[Mention] = []
    for m in mentions:
        first, last = index[m.token_ids[0]], index[m.token_ids[-1]]
        line = (tokens[first].page, tokens[first].line_no)
        while (
            first > 0
            and (tokens[first - 1].page, tokens[first - 1].line_no) == line
            and tokens[first - 1].min_confidence < threshold
        ):
            first -= 1
        line = (tokens[last].page, tokens[last].line_no)
        while (
            last + 1 < len(tokens)
            and (tokens[last + 1].page, tokens[last + 1].line_no) == line
            and tokens[last + 1].min_confidence < threshold
        ):
            last += 1
        ids = tuple(t.token_id for t in tokens[first : last + 1])
        if ids == m.token_ids:
            out.append(m)
            continue
        out.append(
            Mention(
                mention_id=content_id("M", doc_key, ids[0], ids[-1], m.entity_type.value),
                entity_type=m.entity_type,
                token_ids=ids,
                evidence=m.evidence,
                value=" ".join(t.text for t in tokens[first : last + 1]),
            )
        )
    return out


def detect_document(
    pages: Sequence[PageAnalysis],
    config: AppConfig,
    doc_key: str,
    seeds: Sequence[Seed] = (),
    detectors: Sequence[Detector] | None = None,
) -> DetectionResult:
    views = build_views(pages)
    line_view = views[ViewKind.LINE]
    dets = tuple(detectors) if detectors is not None else default_detectors()
    candidates: list[Candidate] = []
    for d in dets:
        candidates.extend(d.detect(views))
    actions = config.policy.entity_actions
    det_cfg = config.detection
    seed_targets = [
        Target(match_key(s.value.reveal()), len(s.value.reveal().split()), s.entity_type, "seed")
        for s in seeds
        if len(match_key(s.value.reveal())) >= 2
    ]
    if seed_targets:
        candidates.extend(propagate(seed_targets, line_view, fuzzy_ratio=det_cfg.propagation_fuzzy_ratio))
    mentions, low = fuse(candidates, views, doc_key)
    rounds = 0
    if det_cfg.propagate:
        seen: set[tuple[int, int, EntityType]] = {
            (c.start, c.end, c.entity_type) for c in candidates if c.view is ViewKind.LINE
        }
        for round_no in range(1, _MAX_PROPAGATION_ROUNDS + 1):
            rounds = round_no
            targets = targets_from_mentions(mentions, actions, det_cfg.propagation_min_len)
            new = [
                c
                for c in propagate(targets, line_view, fuzzy_ratio=det_cfg.propagation_fuzzy_ratio)
                if (c.start, c.end, c.entity_type) not in seen
            ]
            # Component-level person propagation: standalone surnames / given names of resolved
            # people ("Smith reports...", "John attended...").
            persons = resolve_persons(mentions, line_view)
            new.extend(
                c
                for c in component_candidates(persons, line_view, det_cfg.propagation_min_len)
                if (c.start, c.end, c.entity_type) not in seen
            )
            if not new:
                break
            seen.update((c.start, c.end, c.entity_type) for c in new)
            candidates.extend(new)
            mentions, low = fuse(candidates, views, doc_key)
    if det_cfg.extend_into_uncertain_neighbours:
        mentions = _extend_uncertain(mentions, line_view, config.ocr.uncertain_char_confidence, doc_key)
    ordered = sorted(mentions, key=lambda m: (m.token_ids[0], m.token_ids[-1], m.entity_type.value))
    return DetectionResult(
        mentions=tuple(ordered),
        low_evidence=tuple(low),
        detectors=tuple(
            sorted(
                {(d.detector_id, d.version) for d in dets}
                | {(PROPAGATION_ID, PROPAGATION_VERSION), ("resolve.persons", "1")}
            )
        ),
        propagation_rounds=rounds,
        persons=resolve_persons(ordered, line_view),
        line_view=line_view,
    )
