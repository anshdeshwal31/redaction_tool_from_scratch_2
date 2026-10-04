"""V4: residual PII in the *re-rendered, re-OCR'd output* (plan §12).

Three probes over the re-read output:

1. every removed value (exact / OCR-tolerant / alternate readings) must be gone,
2. the rule detectors must find no new strong PII outside the planned redaction boxes, and
3. no barcode or face may still be detectable outside the planned redaction boxes.

Residual hits become additional redaction boxes for one re-plan; if they persist the job fails.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlredact.config.schema import AppConfig
from mlredact.core.canonical import content_id
from mlredact.core.geometry import BBox
from mlredact.core.model import Mention, PageAnalysis, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, RegionKind, VerificationCheck, ViewKind
from mlredact.detect.engine import default_detectors
from mlredact.detect.fusion import fuse
from mlredact.detect.propagate import Target, propagate
from mlredact.text.normalize import match_key
from mlredact.text.views import build_views
from mlredact.verify.checks import CheckResult

_COVERED_FRACTION = 0.5
_VISUAL_RESIDUALS = frozenset({RegionKind.BARCODE, RegionKind.FACE})


@dataclass(frozen=True, slots=True)
class ResidualHit:
    page: int
    box: BBox
    reason: str  # entity type or region kind value
    source: str  # "removed_value" | "redetect" | "region"


def _covered(box: BBox, ops: Sequence[RenderOp]) -> bool:
    covered = 0
    for op in ops:
        inter = box.intersection(op.box)
        if inter is not None:
            covered += inter.area
    return box.area > 0 and covered / box.area >= _COVERED_FRACTION


def _explained(value: str, replacement_keys: Sequence[str]) -> bool:
    """A re-detected mention that is (part of) one of our own surrogates / labels is not residual.

    Safe because V5 guarantees no surrogate equals or nearly equals any original value; the
    removed-value probe (which this never suppresses) still catches every visible original."""
    key = match_key(value)
    if len(key) < 3:
        return False
    return any(key in r or (len(r) >= 3 and r in key) for r in replacement_keys)


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
) -> list[ResidualHit]:
    """``planned`` must carry the boxes that were *actually* erased; ``replacements`` are the
    surrogate / label strings drawn on the pages; ``kept`` are regions the policy deliberately left
    visible although their type is normally acted on (e.g. professionals under the claimant-focused
    profile).  ``kept`` only exempts re-detection - the removed-value probe is never suppressed."""
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
                hits.append(ResidualHit(tok.page, tok.bbox, etype.value, source))

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
    actions = config.policy.entity_actions
    for m in mentions:
        if actions[m.entity_type] is ActionKind.KEEP:
            continue
        if not any(e.strength is EvidenceStrength.STRONG for e in m.evidence):
            continue
        if _explained(m.value, replacement_keys):
            continue
        toks = [index[t] for t in m.token_ids]
        if all(_inside_any(t.bbox, kept.get(t.page, ())) for t in toks):
            continue
        record(toks, m.entity_type, "redetect")
    for out_page in output_pages:
        for region in out_page.regions:
            if region.kind in _VISUAL_RESIDUALS and not _covered(region.bbox, planned.get(region.page, ())):
                hits.append(ResidualHit(region.page, region.bbox, region.kind.value, "region"))
    # Deterministic, de-duplicated order.
    unique = {(h.page, h.box, h.reason, h.source): h for h in hits}
    return [unique[k] for k in sorted(unique, key=lambda k: (k[0], k[1], k[2], k[3]))]


def residual_ops(hits: Sequence[ResidualHit], doc_key: str, iteration: int) -> dict[int, list[RenderOp]]:
    out: dict[int, list[RenderOp]] = {}
    for h in hits:
        out.setdefault(h.page, []).append(
            RenderOp(
                page=h.page,
                kind=ActionKind.BLACKOUT,
                box=h.box,
                reason=h.reason,
                ref_id=content_id("R", doc_key, iteration, h.page, *h.box.as_tuple()),
            )
        )
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
