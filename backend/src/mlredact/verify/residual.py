"""V4: residual PII in the *re-rendered, re-OCR'd output* (plan §12).

Three probes over the re-read output:

1. every removed value (exact / OCR-tolerant / alternate readings) must be gone,
2. the rule detectors must find no new strong PII outside the planned redaction boxes, and
3. no barcode or face may still be detectable outside the planned redaction boxes.

Residual hits become additional redaction boxes for one re-plan; if they persist the job fails.

Address units (``addresses``) are probed by their per-part targets in place of their removed values (compound and
whole-token targets, form parts as 'label value', exempt positions apart), and the address join runs on the re-read
output too: a unit or an anchored place whose identifying words are not this run's fakes is a residual even though
LOCALITY and POSTCODE are KEEP types.  A residual that repeats a found unit is re-planned with that unit's fake.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from mlredact.config.schema import AppConfig
from mlredact.core.canonical import content_id
from mlredact.core.geometry import BBox
from mlredact.core.model import Mention, PageAnalysis, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, RegionKind, VerificationCheck, ViewKind
from mlredact.detect.engine import default_detectors
from mlredact.detect.fusion import fuse
from mlredact.detect.propagate import Target, propagate
from mlredact.policy.addresses import ADDRESS_TYPES, fold_word, join_addresses
from mlredact.text.address import STRUCTURE_ROLES, KeptKind
from mlredact.text.normalize import match_key
from mlredact.text.views import TextView, build_views
from mlredact.verify.addresses import AddressReread, swapped
from mlredact.verify.checks import CheckResult
from mlredact.verify.needles import MIN_WHOLE

_COVERED_FRACTION = 0.5
_VISUAL_RESIDUALS = frozenset({RegionKind.BARCODE, RegionKind.FACE})
_DESCENDERS = frozenset("gjpqy,;()[]{}_")  # as the planner's: they calibrate the font of a drawn fake


@dataclass(frozen=True, slots=True)
class ResidualHit:
    """``fake``: for a residual that repeats a found address unit, the text of that unit's fake to draw over ``box``
    (the union of the residual words on one line) instead of a black box; ``original`` is the re-read text it covers
    (it calibrates the drawing, finds the words in a re-typeset line, and never leaves the process)."""

    page: int
    box: BBox
    reason: str  # entity type or region kind value
    source: str  # "removed_value" | "redetect" | "region"
    fake: str | None = field(default=None, repr=False)
    original: str | None = field(default=None, repr=False)


def _covered(box: BBox, ops: Sequence[RenderOp]) -> bool:
    covered = 0
    for op in ops:
        inter = box.intersection(op.box)
        if inter is not None:
            covered += inter.area
    return box.area > 0 and covered / box.area >= _COVERED_FRACTION


def _explained(value: str, replacement_keys: Sequence[str], etype: EntityType) -> bool:
    """A re-detected mention that lies within one of our own surrogates / labels is not residual.

    Safe because V5 guarantees no surrogate equals or nearly equals any original value; the
    removed-value probe (which this never suppresses) still catches every visible original.  An address mention that
    merely contains a surrogate is not explained: 'Address: 40 Bridge Street, Penrith NSW 2750' with only the street
    line replaced contains the fake street line, and the place after it is still there.  A mention of another type is:
    OCR glues a fake phone to the next label ('0491 570 760Email:') and re-reads the run as one phone."""
    key = match_key(value)
    if len(key) < 3:
        return False
    if any(key in r for r in replacement_keys):
        return True
    return etype not in ADDRESS_TYPES and any(len(r) >= 3 and r in key for r in replacement_keys)


def _inside_any(box: BBox, regions: Sequence[BBox]) -> bool:
    covered = sum(inter.area for r in regions if (inter := box.intersection(r)) is not None)
    return box.area > 0 and covered / box.area >= _COVERED_FRACTION


def find_residuals(
    output_pages: Sequence[PageAnalysis],
    removed: Sequence[Mention],
    planned: Mapping[int, Sequence[RenderOp]],
    config: AppConfig,
    replacements: Sequence[str] = (),
    kept: Mapping[int, Sequence[BBox]] | None = None,
    *,
    addresses: AddressReread | None = None,
) -> list[ResidualHit]:
    """``planned`` must carry the boxes that were *actually* erased; ``replacements`` are the
    surrogate / label strings drawn on the pages; ``kept`` are regions the policy deliberately left
    visible although their type is normally acted on (e.g. professionals under the claimant-focused
    profile).  ``kept`` only exempts re-detection - the removed-value probe is never suppressed.

    ``addresses``: the document's address units (``removed`` then holds no unit mention).  Their targets are probed in
    place of removed values, with the exempt positions of ``addresses.exempt`` (kept places and kept mentions) left
    alone, and the address join runs on the re-read output."""
    kept = kept or {}
    replacement_keys = sorted({match_key(r) for r in replacements if match_key(r)})
    views = build_views(output_pages)
    line_view = views[ViewKind.LINE]
    tokens: Sequence[Token] = line_view.tokens
    vcfg = config.verification
    hits: list[ResidualHit] = []

    def record(token_slice: Sequence[Token], etype: EntityType, source: str) -> None:
        for tok in token_slice:
            if not _covered(tok.bbox, planned.get(tok.page, ())):
                hits.append(ResidualHit(tok.page, tok.bbox, etype.value, source, original=tok.text))

    targets: dict[tuple[str, EntityType], Target] = {}
    for m in removed:
        key = match_key(m.value)
        if len(key) >= vcfg.residual_min_len:
            targets.setdefault((key, m.entity_type), Target(key, len(m.token_ids), m.entity_type, "mention"))
    ordered_targets = [targets[k] for k in sorted(targets, key=lambda k: (k[0], k[1].value))]
    for cand in propagate(ordered_targets, line_view, fuzzy_ratio=vcfg.residual_fuzzy_ratio):
        rng = line_view.token_range(cand.start, cand.end)
        record(tokens[rng.start : rng.stop], cand.entity_type, "removed_value")

    candidates = [c for d in default_detectors() for c in d.detect(views)]
    mentions, _low = fuse(candidates, views, "residual")
    index = {t.token_id: t for t in tokens}
    if addresses is not None:
        hits.extend(_address_hits(addresses, output_pages, line_view, index, mentions, planned, config))
    actions = config.policy.entity_actions
    for m in mentions:
        if actions[m.entity_type] is ActionKind.KEEP:
            continue
        if not any(e.strength is EvidenceStrength.STRONG for e in m.evidence):
            continue
        if _explained(m.value, replacement_keys, m.entity_type):
            continue
        # A mention that runs on into kept words (a fake street line read with the kept place after it) leaves them.
        record(
            [t for t in (index[t] for t in m.token_ids) if not _inside_any(t.bbox, kept.get(t.page, ()))],
            m.entity_type,
            "redetect",
        )
    for out_page in output_pages:
        for region in out_page.regions:
            if region.kind in _VISUAL_RESIDUALS and not _covered(region.bbox, planned.get(region.page, ())):
                hits.append(ResidualHit(region.page, region.bbox, region.kind.value, "region"))
    # Deterministic, de-duplicated order.
    unique = {(h.page, h.box, h.reason, h.source): h for h in hits}
    return [unique[k] for k in sorted(unique, key=lambda k: (k[0], k[1], k[2], k[3]))]


def _address_hits(
    addresses: AddressReread,
    output_pages: Sequence[PageAnalysis],
    line_view: TextView,
    index: Mapping[str, Token],
    mentions: Sequence[Mention],
    planned: Mapping[int, Sequence[RenderOp]],
    config: AppConfig,
) -> list[ResidualHit]:
    """The address residuals of the re-read output.  A token is left alone when a planned operation covers it, when it
    lies in an exempt position (kept places, kept mentions of other types), or, in a unit, when it is a word of a fake
    this run drew.  Windows that repeat a found unit become that unit's fake; any other residual is a black box."""
    tokens = line_view.tokens
    vcfg = config.verification
    hits: list[ResidualHit] = []
    done: set[str] = set()
    reason = EntityType.STREET_ADDRESS.value

    # A re-typeset line is drawn again whole, the words it kept included: its operation covers a word only when the
    # word is one this run drew (a kept original left in it is as readable as before).
    drawn = {page: [op for op in ops if op.kind is not ActionKind.RETYPESET_REGION] for page, ops in planned.items()}

    def covered(tok: Token) -> bool:
        if _covered(tok.bbox, drawn.get(tok.page, ())):
            return True
        return _covered(tok.bbox, planned.get(tok.page, ())) and fold_word(tok.text) in addresses.fake_words

    retypeset = {page: [op for op in ops if op.kind is ActionKind.RETYPESET_REGION] for page, ops in planned.items()}

    def exempt_on_its_line(tok: Token) -> bool:
        """A kept word of a re-typeset line, wherever the line drew it."""
        word = fold_word(tok.text)
        return bool(word) and any(
            op.ink_box is not None
            and word in addresses.line_exempt.get((op.page, op.ink_box), ())
            and _covered(tok.bbox, (op,))
            for op in retypeset.get(tok.page, ())
        )

    def eligible(tok: Token) -> bool:
        return (
            tok.token_id not in done
            and not covered(tok)
            and not _inside_any(tok.bbox, addresses.exempt.get(tok.page, ()))
            and not exempt_on_its_line(tok)
        )

    def record(window: Sequence[Token], focus: Sequence[Token], source: str) -> None:
        """``window``: consecutive tokens of one line; ``focus``: the ones that are residual."""
        todo = [t for t in focus if eligible(t)]
        if not todo:
            return
        fake = swapped([t.text for t in window], addresses) if all(eligible(t) for t in window) else None
        if fake is not None:
            box = window[0].bbox
            for t in window[1:]:
                box = box.union(t.bbox)
            hits.append(ResidualHit(window[0].page, box, reason, source, fake, " ".join(t.text for t in window)))
            done.update(t.token_id for t in window)
        else:
            hits.extend(ResidualHit(t.page, t.bbox, reason, source, original=t.text) for t in todo)
            done.update(t.token_id for t in todo)

    # The probe: the units' targets in the re-read output (a repeat of a found address too).
    probe: dict[str, Target] = {}
    for unit_targets in addresses.targets:
        for words in unit_targets.compound:
            probe.setdefault(match_key(" ".join(words)), Target("", len(words), EntityType.STREET_ADDRESS, "mention"))
        for word in unit_targets.whole:
            probe.setdefault(match_key(word), Target("", 1, EntityType.STREET_ADDRESS, "mention"))
    ordered = [
        Target(key, t.n_tokens, t.entity_type, t.source) for key, t in sorted(probe.items()) if len(key) >= MIN_WHOLE
    ]
    found = propagate(ordered, line_view, fuzzy_ratio=vcfg.residual_fuzzy_ratio)
    windows = [tokens[r.start : r.stop] for r in (line_view.token_range(c.start, c.end) for c in found)]
    for window in sorted(windows, key=lambda w: (-len(w), w[0].token_id if w else "")):
        if window:
            record(window, window, "removed_value")

    # The join on the re-read output: a unit, or an anchored place, whose identifying words are not this run's fakes.
    if config.policy.addresses == "whole":
        reread = join_addresses(mentions, output_pages, config.policy)
        for unit in reread.units.values():
            owned = [index[t] for t in unit.roles if t in index]
            focus = [
                index[t]
                for t, role in unit.roles.items()
                if t in index and (role is None or role not in STRUCTURE_ROLES) and t not in unit.excluded
            ]
            _unit_residuals(owned, focus, addresses, record)
        for place in reread.kept:
            if place.kind is KeptKind.PLACE_AFTER_CUE:
                owned = [index[t] for t in place.token_ids if t in index]
                _unit_residuals(owned, owned, addresses, record)
    return hits


def _unit_residuals(
    owned: Sequence[Token],
    focus: Sequence[Token],
    addresses: AddressReread,
    record: Callable[[Sequence[Token], Sequence[Token], str], None],
) -> None:
    """The residual words of one re-read unit (``focus``: its identifying words; ``owned``: all its words), line by
    line.  A word of a fake this run drew is no residual."""
    todo = [t for t in focus if (w := fold_word(t.text)) and w not in addresses.fake_words and w not in addresses.plain]
    by_line: dict[tuple[int, int], list[Token]] = {}
    for tok in todo:
        by_line.setdefault((tok.page, tok.line_no), []).append(tok)
    for key in sorted(by_line):
        group = sorted(by_line[key], key=lambda t: t.word_no)
        lo, hi = group[0].word_no, group[-1].word_no
        span = sorted(
            (t for t in owned if (t.page, t.line_no) == key and lo <= t.word_no <= hi), key=lambda t: t.word_no
        )
        record(span if [t.word_no for t in span] == list(range(lo, hi + 1)) else group, group, "redetect")


def residual_ops(hits: Sequence[ResidualHit], doc_key: str, iteration: int) -> dict[int, list[RenderOp]]:
    """The re-plan's operations: a black box over each hit, or, where the hit repeats a found address unit, that
    unit's fake drawn over it.  Each carries the re-read text it covers as ``source_text`` (in memory only), so a hit
    on a re-typeset line can be applied to the line's words (``Redactor._merge_ops``)."""
    out: dict[int, list[RenderOp]] = {}
    for h in hits:
        ref_id = content_id("R", doc_key, iteration, h.page, *h.box.as_tuple())
        op = (
            RenderOp(
                h.page,
                ActionKind.SURROGATE,
                h.box,
                h.reason,
                ref_id,
                h.fake,
                h.box,
                any(c in _DESCENDERS for c in h.original or ""),
                h.original,
            )
            if h.fake is not None
            else RenderOp(h.page, ActionKind.BLACKOUT, h.box, h.reason, ref_id, source_text=h.original)
        )
        out.setdefault(h.page, []).append(op)
    return out


def residual_result(hits: Sequence[ResidualHit], replans: int) -> CheckResult:
    return CheckResult(
        VerificationCheck.RESIDUAL,
        not hits,
        {
            "residual_tokens": len(hits),
            "from_removed_values": sum(h.source == "removed_value" for h in hits),
            "from_redetection": sum(h.source == "redetect" for h in hits),
            "from_regions": sum(h.source == "region" for h in hits),
            "replans": replans,
        },
    )
