"""Benchmark stage files from a job trace: for the evaluator, never production.

The files follow the benchmark's adapter contract (``rb.text/0.1``, ``rb.mentions/0.1``,
``rb.entities/0.1``, ``rb.plan/0.1``, ``rb.pseudonyms/0.1``, ``rb.runinfo/0.1``).  The benchmark rejects
unknown keys, so only contract keys appear here.  Unlike the manifest, these files hold document text by
design (the evaluator scores OCR and detection against its ground truth): only ``mlredact bench`` writes
them, into the evaluator's run folder, and nothing here is logged.

Coordinates are ``display_pt``: PDF points of the page as displayed (CropBox after /Rotate, origin
top-left, y down).  The canonical raster is the displayed page (``mlredact.ingest.render``) and every
reader's boxes are mapped back into it, so a pixel box converts with its page's own dpi alone:
pt = px * 72 / dpi.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from mlredact.core.errors import ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import Mention, PageAnalysis, PageGeometry
from mlredact.core.types import ActionKind, EntityType, RegionKind
from mlredact.ocr.fusion import ENGINE_ID as TEXT_LAYER_ENGINE
from mlredact.pipeline.runner import JobTrace

FRAME = "display_pt"
CANDIDATE = "mlredact"

_REPLACE_KINDS = frozenset(
    {
        ActionKind.SURROGATE,
        ActionKind.RETYPESET_REGION,
        ActionKind.LABEL,
        ActionKind.SYNTHESISE,
        ActionKind.GENERALISE,
        ActionKind.DATE_SHIFT,
    }
)
_REMOVE_KINDS = frozenset({ActionKind.REMOVE, ActionKind.BLACKOUT})
# Operations that draw replacement text: their mention's surrogate or label, or a re-typeset line with the
# replacements of the mentions it covers.
_TYPESET_KINDS = frozenset({ActionKind.SURROGATE, ActionKind.LABEL, ActionKind.RETYPESET_REGION})
# Region kinds that are PII.  Faint ink (show-through, pencil) is only lightened where it is faint, and
# graphics (figures, body charts) are layout: neither is a mention, whatever the policy does with them.
PII_REGION_KINDS = frozenset(
    {
        RegionKind.SIGNATURE,
        RegionKind.FACE,
        RegionKind.PHOTO,
        RegionKind.HANDWRITING,
        RegionKind.BARCODE,
        RegionKind.LOGO,
        RegionKind.STAMP,
        RegionKind.ILLEGIBLE,
    }
)
# Every mention label bench can emit: the candidate manifest's label_map must cover all of them.
LABELS = tuple(sorted({t.value for t in EntityType} | {k.value for k in PII_REGION_KINDS}))
# Reason codes of the verification checks (V1-V6); any other code is a step that failed.
_VERIFICATION_CODES = frozenset(c.value for c in ReasonCode if c.name.startswith("VERIFY_"))


def display_poly(box: BBox, geometry: PageGeometry) -> list[list[float]]:
    """Canonical-raster pixel box -> rectangle in display points, using the page's own dpi."""
    s = 72.0 / geometry.dpi
    x0, y0, x1, y1 = (round(v * s, 3) for v in box.as_tuple())
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def plan_action(kind: ActionKind) -> str | None:
    """``replace`` where the operation puts something new in, ``remove`` where it only erases.  None
    for faint-ink suppression: it keeps every strong-ink pixel, so it neither removes nor replaces."""
    if kind in _REPLACE_KINDS:
        return "replace"
    if kind in _REMOVE_KINDS:
        return "remove"
    return None


def _unit(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 5)


def _mention_conf(mention: Mention) -> float | None:
    """The best evidence score, when it is a probability."""
    best = max((e.score for e in mention.evidence), default=None)
    return round(best, 5) if best is not None and 0.0 <= best <= 1.0 else None


def _page_sources(page: PageAnalysis, text_layer: bool) -> set[str]:
    """How a page was read.  OCR reads every page; the PDF's own text layer is fused in where the page
    has one (its words replace uncertain OCR readings, so such a page is hybrid unless every token came
    from the text layer)."""
    if not (text_layer and page.embedded_char_count > 0):
        return {"ocr"}
    engines = {t.engine for t in page.tokens()}
    return {"text_layer"} if engines == {TEXT_LAYER_ENGINE} else {"ocr", "text_layer"}


def text_artifact(trace: JobTrace, *, text_layer: bool) -> dict[str, Any]:
    """``rb.text/0.1``: every token in reading order.  ``text_layer``: the configuration fuses the PDF's
    text layer (``ocr.embedded_text``).  Injected text (the oracle modes) keeps the source it was injected
    with."""
    pages = []
    sources: set[str] = set()
    for page in trace.pages:
        geometry = page.geometry
        tokens: list[dict[str, Any]] = []
        for line in page.lines:
            for tok in line.tokens:
                tokens.append(
                    {
                        "id": tok.token_id,
                        "text": tok.text,
                        "poly": display_poly(tok.bbox, geometry),
                        "conf": _unit(tok.confidence),
                        "line": line.line_id,
                        "order": len(tokens),
                    }
                )
        pages.append({"index": geometry.index, "tokens": tokens})
        sources |= _page_sources(page, text_layer)
    source = "hybrid" if len(sources) > 1 else next(iter(sources), "ocr")
    if trace.injected is not None:
        source = trace.injected.text_source
    return {"schema": "rb.text/0.1", "frame": FRAME, "source": source, "pages": pages}


def mentions_artifact(trace: JobTrace) -> dict[str, Any]:
    """``rb.mentions/0.1``: what the tool decided to replace or remove.  Detected values the policy keeps
    (dates, places) are left out: the benchmark scores a flagged kept word as a false positive."""
    geoms = {p.geometry.index: p.geometry for p in trace.pages}
    tokens = {t.token_id: t for p in trace.pages for t in p.tokens()}
    mentions: list[dict[str, Any]] = []
    for d in trace.decisions:
        if d.action is ActionKind.KEEP:
            continue
        m = d.mention
        entry: dict[str, Any] = {
            "id": m.mention_id,
            "label": m.entity_type.value,
            "text": m.value,
            "tokens": list(m.token_ids),
            "regions": [
                {"page": tokens[t].page, "poly": display_poly(tokens[t].bbox, geoms[tokens[t].page])}
                for t in m.token_ids
            ],
        }
        conf = _mention_conf(m)
        if conf is not None:
            entry["conf"] = conf
        mentions.append(entry)
    for rd in trace.regions:
        r = rd.region
        if rd.action in (ActionKind.KEEP, ActionKind.SUPPRESS_INK) or r.kind not in PII_REGION_KINDS:
            continue
        entry = {
            "id": r.region_id,
            "label": r.kind.value,
            "regions": [{"page": r.page, "poly": display_poly(r.bbox, geoms[r.page])}],
        }
        if 0.0 <= r.score <= 1.0:
            entry["conf"] = round(r.score, 5)
        mentions.append(entry)
    return {"schema": "rb.mentions/0.1", "frame": FRAME, "mentions": mentions}


def entities_artifact(trace: JobTrace, mention_ids: AbstractSet[str]) -> dict[str, Any]:
    """``rb.entities/0.1``: one cluster per resolved person, with its mentions in ``mentions.json``."""
    clusters = []
    for cluster, members in sorted((trace.persons.clusters if trace.persons else {}).items()):
        listed = [m for m in members if m in mention_ids]
        if listed:
            clusters.append({"id": cluster, "mentions": listed})
    return {"schema": "rb.entities/0.1", "clusters": clusters}


def plan_artifact(trace: JobTrace, mention_ids: AbstractSet[str]) -> dict[str, Any]:
    """``rb.plan/0.1``: one region per final render operation, with the box it actually erased."""
    geoms = {p.geometry.index: p.geometry for p in trace.pages}
    seen: Counter[str] = Counter()
    regions = []
    for page in sorted(trace.ops):
        for op in trace.ops[page]:
            action = plan_action(op.kind)
            if action is None:
                continue
            n = seen[op.ref_id]
            seen[op.ref_id] += 1
            box = trace.erased.get((op.ref_id, op.box), op.box)
            entry: dict[str, Any] = {
                "id": f"{op.ref_id}.{n}",
                "page": op.page,
                "poly": display_poly(box, geoms[op.page]),
                "action": action,
                "mentions": [i for i in dict.fromkeys((op.ref_id, *op.covers)) if i in mention_ids],
            }
            if op.text:
                entry["replacement"] = op.text
            regions.append(entry)
    return {"schema": "rb.plan/0.1", "frame": FRAME, "regions": regions}


def pseudonyms_artifact(trace: JobTrace, mention_ids: AbstractSet[str]) -> dict[str, Any]:
    """``rb.pseudonyms/0.1``: every replacement text drawn, per mention and as an original -> replacement
    table.  A surrogate that was never drawn (its mention was blacked out instead, e.g. on a sideways line)
    is left out."""
    typeset = [op for ops in trace.ops.values() for op in ops if op.kind in _TYPESET_KINDS]
    drawn = {ref for op in typeset for ref in (op.ref_id, *op.covers)}
    parsed = trace.persons.mentions if trace.persons else {}
    mentions: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    listed: set[tuple[str | None, str, str]] = set()
    for d in trace.decisions:
        m = d.mention
        sur = trace.surrogates.get(m.mention_id)
        if sur is None or m.mention_id not in mention_ids or m.mention_id not in drawn:
            continue
        replacement = " ".join(s for s in sur.tokens if s)
        mentions.append({"mention": m.mention_id, "replacement": replacement})
        person = parsed.get(m.mention_id)
        cluster = person.cluster if person is not None else None
        if (cluster, m.value, replacement) not in listed:
            listed.add((cluster, m.value, replacement))
            entries.append({"cluster": cluster, "original": m.value, "replacement": replacement})
    return {"schema": "rb.pseudonyms/0.1", "entries": entries, "mentions": mentions}


def stage_files(trace: JobTrace, *, text_layer: bool) -> dict[str, dict[str, Any]]:
    """Every stage file by file name.  Entities, plan and pseudonyms only name mentions that
    ``mentions.json`` lists.  Where the mentions were injected (oracle modes from ``oracle_mentions`` on),
    they are reported under their injected ids (``mlredact.bench.oracle_files``)."""
    injected = trace.injected
    if injected is not None and injected.mentions is not None:
        from mlredact.bench import oracle_files  # it builds on this module's writers

        return oracle_files.stage_files(trace, injected, text_artifact(trace, text_layer=text_layer))
    mentions = mentions_artifact(trace)
    ids = frozenset(m["id"] for m in mentions["mentions"])
    return {
        "text.json": text_artifact(trace, text_layer=text_layer),
        "mentions.json": mentions,
        "entities.json": entities_artifact(trace, ids),
        "plan.json": plan_artifact(trace, ids),
        "pseudonyms.json": pseudonyms_artifact(trace, ids),
    }


def run_info(
    *,
    version: str,
    config: Mapping[str, Any],
    seed: int | None,
    models: Sequence[Mapping[str, Any]],
    started_at: str,
    finished_at: str,
    notes: Sequence[str],
    released: bool,
    checks_failed: Sequence[str],
) -> dict[str, Any]:
    """``rb.runinfo/0.1``.  ``models``: the manifest's model provenance; ``checks_failed``: the reason
    codes the document was not released for."""
    return {
        "schema": "rb.runinfo/0.1",
        "candidate": CANDIDATE,
        "candidate_version": version,
        "config": dict(config),
        "seed": seed,
        "models": [{"name": m["id"], "version": m["version"]} for m in models],
        "started_at": started_at,
        "finished_at": finished_at,
        "notes": "; ".join(notes) if notes else None,
        "release": {"released": released, "checks_failed": sorted(checks_failed)},
    }


def unreleased_note(reasons: Sequence[str]) -> str:
    """The ``run_info`` note for an output the tool did not release: held back by its own verification, or
    built before a later step failed (the checks then never finished, so nothing was held back)."""
    if reasons and all(r in _VERIFICATION_CODES for r in reasons):
        return "held back by the tool's own verification: output.pdf is the copy it would not release"
    codes = ", ".join(sorted(reasons)) or "no reason recorded"
    return f"a step failed after the output was built ({codes}): output.pdf is that unreleased copy"
