"""The non-sensitive manifest (Appendix A of the plan).

It contains geometry, types, actions, evidence and provenance — never document text.  It is
deterministic: no timestamps, no random IDs, canonical ordering.  Operational data (timings, job id)
goes to ``run.json`` instead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from mlredact import __version__
from mlredact.config.schema import AppConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import Evidence, PageAnalysis, PageGeometry, RenderOp
from mlredact.core.types import ActionKind, JobStatus
from mlredact.detect.fusion import LowEvidence
from mlredact.ingest.inspect import InspectionReport
from mlredact.policy.engine import Decision
from mlredact.runtime.profile import RuntimeProfile
from mlredact.surrogate.assign import MentionSurrogate
from mlredact.verify.checks import CheckResult

SCHEMA_VERSION = "1.1.0"


def _evidence(ev: Sequence[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "detector": e.detector_id,
            "version": e.detector_version,
            "rule": e.rule_id,
            "strength": "strong" if int(e.strength) == 2 else "weak",
            "score": e.score,
        }
        for e in sorted(ev)
    ]


def _mean_confidence(page: PageAnalysis) -> float:
    confidences = [t.confidence for line in page.lines for t in line.tokens]
    return round(sum(confidences) / len(confidences), 4) if confidences else 0.0


def _box(geom: PageGeometry, op: RenderOp, erased: Mapping[tuple[str, BBox], BBox]) -> dict[str, Any]:
    out: dict[str, Any] = {"page": op.page, "px": list(op.box.as_tuple()), "pt": list(geom.box_to_pt(op.box))}
    actual = erased.get((op.ref_id, op.box))
    if actual is not None and actual != op.box:
        out["erased_px"] = list(actual.as_tuple())
    return out


def build_manifest(
    *,
    status: JobStatus,
    input_sha256: str,
    config: AppConfig,
    config_hash: str,
    runtime: RuntimeProfile,
    models: Sequence[Mapping[str, Any]],
    detectors: Sequence[tuple[str, str]],
    inspection: InspectionReport | None,
    pages: Sequence[PageAnalysis],
    decisions: Sequence[Decision],
    ops: Mapping[int, Sequence[RenderOp]],
    low_evidence: Sequence[LowEvidence],
    checks: Sequence[CheckResult],
    reasons: Sequence[str],
    propagation_rounds: int,
    surrogates: Mapping[str, MentionSurrogate] | None = None,
    replacement_field: str = "surrogate",
    surrogate_key_id: str | None = None,
    surrogate_scope: str = "document",
    erased: Mapping[tuple[str, BBox], BBox] | None = None,
    notices: Mapping[int, bool] | None = None,
    source_format: str = "pdf",
) -> dict[str, Any]:
    erased = erased or {}
    notices = notices or {}
    geoms = {p.geometry.index: p.geometry for p in pages}
    surrogates = surrogates or {}
    ops_by_ref: dict[str, list[RenderOp]] = defaultdict(list)
    for page_ops in ops.values():
        for op in page_ops:
            ops_by_ref[op.ref_id].append(op)
            for mention_id in op.covers:  # a re-typeset line carries these mentions
                ops_by_ref[mention_id].append(op)
    mentions = []
    for d in decisions:
        m = d.mention
        rendered = ops_by_ref.get(m.mention_id, [])
        entry: dict[str, Any] = {
            "mention_id": m.mention_id,
            "type": m.entity_type.value,
            "action": d.action.value,
            "render": sorted({op.kind.value for op in rendered}),
            "boxes": [_box(geoms[op.page], op, erased) for op in sorted(rendered, key=lambda o: o.sort_key)],
            "token_ids": list(m.token_ids),
            "evidence": _evidence(m.evidence),
        }
        sur = surrogates.get(m.mention_id)
        if sur is not None:
            # Surrogates / labels are visible in the released PDF, so they are not sensitive.
            entry[replacement_field] = " ".join(s for s in sur.tokens if s)
        mentions.append(entry)
    mention_ids = {d.mention.mention_id for d in decisions}
    region_actions = config.policy.region_actions
    regions = []
    for p in pages:
        for r in p.regions:
            rendered_ops = ops_by_ref.get(r.region_id, [])
            box = (
                _box(geoms[r.page], rendered_ops[0], erased)
                if rendered_ops
                else {"page": r.page, "px": list(r.bbox.as_tuple()), "pt": list(geoms[r.page].box_to_pt(r.bbox))}
            )
            regions.append(
                {
                    "region_id": r.region_id,
                    "kind": r.kind.value,
                    "action": region_actions.get(r.kind, ActionKind.REMOVE).value,
                    "detector": r.detector,
                    "rule": r.rule,
                    "score": r.score,
                    "box": box,
                }
            )
    mention_ids |= {r.region_id for p in pages for r in p.regions}
    residual = [
        {**_box(geoms[op.page], op, erased), "type": op.reason, "ref_id": op.ref_id}
        for page_ops in ops.values()
        for op in page_ops
        if op.ref_id not in mention_ids and not op.covers
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status.value,
        "reasons": sorted(reasons),
        "job": {
            "input_sha256": input_sha256,
            "page_count": len(pages) if pages else (inspection.page_count if inspection else 0),
            "profile": config.policy.profile_id,
            "render_style": config.redaction.style,
            "source_format": source_format,
        },
        "provenance": {
            "mlredact_version": __version__,
            "config_hash": config_hash,
            "runtime_profile": {"id": runtime.profile_id, "libraries": runtime.libraries},
            "models": list(models),
            "detectors": [{"id": i, "version": v} for i, v in detectors],
            # The caller's scope id is never recorded (it may itself be a claim number).
            "surrogate": {"key_id": surrogate_key_id, "scope": surrogate_scope} if surrogate_key_id else None,
        },
        "input": inspection.as_manifest() if inspection else None,
        "pages": [
            {
                "index": p.geometry.index,
                "width_pt": p.geometry.width_pt,
                "height_pt": p.geometry.height_pt,
                "dpi": p.geometry.dpi,
                "width_px": p.geometry.width_px,
                "height_px": p.geometry.height_px,
                "kind": p.kind.value,
                "lines": len(p.lines),
                "tokens": sum(len(line.tokens) for line in p.lines),
                "notice": notices.get(p.geometry.index, False),
                "unexplained_ink": p.unexplained_ink,
                "mean_ocr_confidence": _mean_confidence(p),
            }
            for p in pages
        ],
        "mentions": mentions,
        "regions": regions,
        "residual_boxes": residual,
        "low_evidence": [
            {"type": le.entity_type.value, "token_ids": list(le.token_ids), "evidence": _evidence(le.evidence)}
            for le in low_evidence
        ],
        "detection": {"propagation_rounds": propagation_rounds},
        "verification": {
            "passed": bool(checks) and all(c.passed for c in checks),
            "checks": [c.manifest() for c in checks],
        },
    }
