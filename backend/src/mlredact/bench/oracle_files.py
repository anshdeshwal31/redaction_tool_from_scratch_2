"""Stage files of the oracle modes that inject mentions (``mlredact bench``): for the evaluator, never production.

From ``oracle_mentions`` on, the stage files report the injected mentions under their injected ids, as tool 2
redacted them (``JobTrace.injected``): ``mentions.json`` lists every one that was placed on tool 2's tokens or became
a region, mentions inside another included; ``entities.json``, ``plan.json`` and ``pseudonyms.json`` name those ids,
so the benchmark can follow each injected mention through every stage.  ``text.json`` is written as in ``e2e``,
with the injected text's source.  Like every stage file these hold document text; nothing here is logged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from mlredact.bench.artifacts import _TYPESET_KINDS, FRAME, display_poly, plan_action
from mlredact.core.types import EntityType
from mlredact.pipeline.injection import InjectReport, ReportedMention
from mlredact.pipeline.runner import JobTrace


def mentions_artifact(trace: JobTrace, mentions: Sequence[ReportedMention]) -> dict[str, Any]:
    """``rb.mentions/0.1``: one entry per injected mention, with its own tokens and the regions it became."""
    geoms = {p.geometry.index: p.geometry for p in trace.pages}
    out: list[dict[str, Any]] = []
    for m in mentions:
        entry: dict[str, Any] = {"id": m.mention_id, "label": m.label}
        if m.token_ids:
            entry["text"] = m.text
            entry["tokens"] = list(m.token_ids)
        entry["regions"] = [{"page": page, "poly": display_poly(box, geoms[page])} for page, box in m.boxes]
        if m.conf is not None:
            entry["conf"] = m.conf
        out.append(entry)
    return {"schema": "rb.mentions/0.1", "frame": FRAME, "mentions": out}


def entities_artifact(trace: JobTrace, report: InjectReport, persons: AbstractSet[str]) -> dict[str, Any]:
    """``rb.entities/0.1`` over the listed person mentions (``persons``): the injected clusters where they were
    injected (``oracle_entities``), otherwise tool 2's own, each member translated to the injected mentions it
    stands for.  A mention is in one cluster at most (the first, in id order)."""
    clusters: list[dict[str, Any]] = []
    if report.clusters is not None:
        for cluster, members in report.clusters:
            listed = [m for m in members if m in persons]
            if listed:
                clusters.append({"id": cluster, "mentions": listed})
        return {"schema": "rb.entities/0.1", "clusters": clusters}
    taken: set[str] = set()
    for cluster, members in sorted((trace.persons.clusters if trace.persons else {}).items()):
        listed = sorted({a for m in members for a in report.aliases.get(m, ()) if a in persons} - taken)
        taken.update(listed)
        if listed:
            clusters.append({"id": cluster, "mentions": listed})
    return {"schema": "rb.entities/0.1", "clusters": clusters}


def plan_artifact(trace: JobTrace, aliases: Mapping[str, Sequence[str]], listed: AbstractSet[str]) -> dict[str, Any]:
    """``rb.plan/0.1``: one region per final render operation, with the box it actually erased (as in ``e2e``),
    naming the injected mentions the operation stands for: the owner of shared tokens first.  A re-plan's
    operations name none."""
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
            named = (a for ref in (op.ref_id, *op.covers) for a in aliases.get(ref, ()))
            entry: dict[str, Any] = {
                "id": f"{op.ref_id}.{n}",
                "page": op.page,
                "poly": display_poly(box, geoms[op.page]),
                "action": action,
                "mentions": [a for a in dict.fromkeys(named) if a in listed],
            }
            if op.text:
                entry["replacement"] = op.text
            regions.append(entry)
    return {"schema": "rb.plan/0.1", "frame": FRAME, "regions": regions}


def pseudonyms_artifact(trace: JobTrace, mentions: Sequence[ReportedMention]) -> dict[str, Any]:
    """``rb.pseudonyms/0.1``: the replacement drawn for each injected mention that owns its tokens, the surrogate
    pieces at its own tokens.  A mention inside another has none of its own (its pixels show the owner's), and a
    surrogate that was never drawn (blacked out instead, e.g. on a sideways line) is left out, as in ``e2e``.  The
    cluster of a person is the one its surrogate was drawn for: tool 2's, or the injected one."""
    typeset = [op for ops in trace.ops.values() for op in ops if op.kind in _TYPESET_KINDS]
    drawn = {ref for op in typeset for ref in (op.ref_id, *op.covers)}
    parsed = trace.persons.mentions if trace.persons else {}
    out: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    listed: set[tuple[str | None, str, str]] = set()
    for m in mentions:
        internal = list(dict.fromkeys(ref for ref, _pos in m.slots))
        if not m.owner or not internal or not all(ref in drawn and ref in trace.surrogates for ref in internal):
            continue
        replacement = " ".join(p for ref, pos in m.slots if (p := trace.surrogates[ref].tokens[pos]))
        if not replacement:
            continue
        out.append({"mention": m.mention_id, "replacement": replacement})
        person = parsed.get(internal[0])
        cluster = person.cluster if person is not None else None
        if (cluster, m.text, replacement) not in listed:
            listed.add((cluster, m.text, replacement))
            entries.append({"cluster": cluster, "original": m.text, "replacement": replacement})
    return {"schema": "rb.pseudonyms/0.1", "entries": entries, "mentions": out}


def stage_files(trace: JobTrace, report: InjectReport, text: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every stage file by file name, from the injected mentions ``report`` lists; ``text``: ``text.json``."""
    mentions = report.mentions or ()
    listed = frozenset(m.mention_id for m in mentions)
    persons = frozenset(m.mention_id for m in mentions if m.label == EntityType.PERSON.value)
    return {
        "text.json": text,
        "mentions.json": mentions_artifact(trace, mentions),
        "entities.json": entities_artifact(trace, report, persons),
        "plan.json": plan_artifact(trace, report.aliases, listed),
        "pseudonyms.json": pseudonyms_artifact(trace, mentions),
    }
