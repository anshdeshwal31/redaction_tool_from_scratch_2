"""The injected plan as tool 2's render operations (``mlredact bench`` mode ``oracle_regions``): for the evaluator,
never production.

In ``oracle_regions`` the benchmark also injects the plan (``plan.json``): the regions to apply, each naming the
mentions it stands for, ``replace`` where a fake is required and ``remove`` where removal is allowed.  Tool 2 applies
exactly these regions, without padding, instead of planning its own operations; its decisions and surrogates still
choose what is drawn, as in its own planning (:func:`mlredact.render.plan.plan_ops`).  By what lies under it, a region
becomes:

* the operation of a non-text region (a signature, a logo) of the mentions it names, when it names a region mention
  or matches one of their regions better than any word: the profile's action on that kind of region (a stand-in, or
  a removal);
* where its words are one mention's on a line: that mention's operation there, joined with the mention's other
  regions on the line where no other word lies between them, so one fake is drawn over the regions and their whole
  words (tool 2 always replaces whole words);
* where its words are several mentions', or a mention's and other words: each line re-typeset, the mentions' words
  as their fakes and the other words as they read;
* where it has no word of any mention: a removal (an illegible region).

A ``remove`` region removes instead of drawing.  Text on a sideways line, and a mention without a fake, are blacked
out, as tool 2 always does.  An injected replacement text is not used: tool 2 draws its own fakes.

Pure, no I/O.  Nothing here logs, and the note carries counts only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from mlredact.bench.inject import Injection, PlanRegion
from mlredact.bench.labels import family_of
from mlredact.core.canonical import content_id
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import PageAnalysis, Region, RenderOp, Token
from mlredact.core.types import ActionKind, RegionKind
from mlredact.pipeline.injection import PlanEcho, ReportedMention, on_page, pt_box
from mlredact.policy.engine import Decision
from mlredact.render.plan import _DESCENDERS, _TEXT_KINDS, effective_kind
from mlredact.surrogate.assign import MentionSurrogate

# A word lies under a plan region when at least this much of it is inside the region, or it holds this much of the
# region (characters inside the word).
_UNDER = 0.5
# A non-text region of the mentions a plan region names is that plan region when they overlap at least this much
# (IoU) and no word under it matches it as well (IoU, or how much of it the word holds).
_SAME_REGION = 0.5
# Region actions that apply a region as such.  The plan says to apply every region it lists, so any other one (a
# region the profile would keep) removes it.
_REGION_OPS = frozenset({ActionKind.SYNTHESISE, ActionKind.REMOVE})
_ILLEGIBLE = RegionKind.ILLEGIBLE.value
_RETYPESET = "retypeset"  # the reason of a re-typeset line, as in tool 2's own planning
_Key = tuple[str, BBox]  # an operation's ref_id and box: the key of the box it erased


@dataclass(frozen=True, slots=True)
class InjectedPlan:
    """The injected plan as render operations, by page; how each plan region on the document's pages was applied,
    by page and id; and the count note for ``run_info.json``."""

    ops: dict[int, tuple[RenderOp, ...]]
    echoes: tuple[PlanEcho, ...]
    note: str


@dataclass(frozen=True, slots=True)
class _Part:
    """A plan region's words of one mention on one line: their positions in the mention, and the box applied there."""

    region: str
    action: str
    mention_id: str
    page: int
    line_no: int
    positions: tuple[int, ...]
    box: BBox


def injected_ops(
    injection: Injection,
    reported: Sequence[ReportedMention],
    aliases: Mapping[str, Sequence[str]],
    *,
    decisions: Sequence[Decision],
    surrogates: Mapping[str, MentionSurrogate],
    tokens: Mapping[str, Token],
    pages: Sequence[PageAnalysis],
    region_actions: Mapping[str, ActionKind],
    rotated: AbstractSet[str],
    style: str,
) -> InjectedPlan:
    """The operations that apply the injected plan (``injection.plan``) to the staged pages.  ``reported``: the
    injected mentions as ``mentions.json`` lists them; ``aliases``: per region id, the injected mentions it came from;
    ``region_actions``: the action on each region, by region id; ``rotated``: the tokens on sideways lines;
    ``style``: the render style."""
    planner = _Planner(reported, aliases, decisions, surrogates, tokens, pages, region_actions, rotated, style)
    return planner.run(injection)


class _Planner:
    """One run of :func:`injected_ops`."""

    def __init__(
        self,
        reported: Sequence[ReportedMention],
        aliases: Mapping[str, Sequence[str]],
        decisions: Sequence[Decision],
        surrogates: Mapping[str, MentionSurrogate],
        tokens: Mapping[str, Token],
        pages: Sequence[PageAnalysis],
        region_actions: Mapping[str, ActionKind],
        rotated: AbstractSet[str],
        style: str,
    ) -> None:
        self.surrogates = surrogates
        self.tokens = tokens
        self.region_actions = region_actions
        self.rotated = rotated
        self.style = style
        self.geoms = {p.geometry.index: p.geometry for p in pages}
        self.words = {p.geometry.index: tuple(p.tokens()) for p in pages}
        self.decisions = {d.mention.mention_id: d for d in decisions if d.action is not ActionKind.KEEP}
        # Each token belongs to one internal mention at most (the stage redacts shared tokens once).
        self.owner = {
            tid: (mention_id, pos)
            for mention_id, d in sorted(self.decisions.items())
            for pos, tid in enumerate(d.mention.token_ids)
        }
        self.listed = frozenset(m.mention_id for m in reported)
        regions = {r.region_id: r for p in pages for r in p.regions}
        self.regions_of: dict[str, list[Region]] = defaultdict(list)  # injected mention id -> the regions it became
        for region_id, named in sorted(aliases.items()):
            region = regions.get(region_id)
            if region is not None:
                for mention_id in named:
                    self.regions_of[mention_id].append(region)
        self.counts: Counter[str] = Counter()

    def run(self, injection: Injection) -> InjectedPlan:
        non_text = frozenset(m.id for m in injection.mentions or () if not family_of(m.label).types)
        placed: list[tuple[PlanRegion, BBox]] = []
        for r in injection.plan or ():
            geometry = self.geoms.get(r.page)
            if geometry is None:
                self.counts["beyond"] += 1
                continue
            if not on_page(r.poly, geometry):
                self.counts["off_page"] += 1
                continue
            placed.append((r, pt_box(r.poly, geometry)))
        placed.sort(key=lambda p: (p[0].page, p[1].y0, p[1].x0, p[1].y1, p[1].x1, p[0].id))
        ops: dict[_Key, RenderOp] = {}
        applied: dict[str, set[_Key]] = defaultdict(set)
        parts: dict[tuple[str, int, int, str], list[_Part]] = defaultdict(list)
        for r, box in placed:
            self.counts["replacements"] += bool(r.replacement)
            direct, joined = self._apply(r, box, non_text)
            applied[r.id].update(self._add(ops, op) for op in direct)
            for part in joined:
                parts[(part.mention_id, part.page, part.line_no, part.action)].append(part)
        for _key, group in sorted(parts.items()):
            for op, members in self._runs(group):
                key = self._add(ops, op)
                self.counts["joined"] += len({p.region for p in members}) > 1
                for p in members:
                    applied[p.region].add(key)
        by_page: dict[int, list[RenderOp]] = defaultdict(list)
        for op in ops.values():
            by_page[op.page].append(op)
        echoes = tuple(
            PlanEcho(
                region_id=r.id,
                page=r.page,
                poly=tuple((x, y) for x, y in r.poly),
                mentions=tuple(sorted({m for m in r.mentions if m in self.listed})),
                ops=tuple(sorted(applied[r.id], key=lambda k: ops[k].sort_key)),
            )
            for r, _box in sorted(placed, key=lambda p: (p[0].page, p[0].id))
        )
        return InjectedPlan(
            ops={page: tuple(sorted(v, key=lambda o: o.sort_key)) for page, v in sorted(by_page.items())},
            echoes=echoes,
            note=self._note(injection, len(placed), len(ops)),
        )

    def _add(self, ops: dict[_Key, RenderOp], op: RenderOp) -> _Key:
        """Plan an operation.  One already planned with the same ref and box stands for both: the box it erased is
        known by that key."""
        key = (op.ref_id, op.box)
        if key in ops:
            self.counts["merged"] += 1
        else:
            ops[key] = op
        return key

    # --------------------------------------------------------------------------------------- regions
    def _apply(self, r: PlanRegion, box: BBox, non_text: AbstractSet[str]) -> tuple[list[RenderOp], list[_Part]]:
        """A plan region's own operations, and its parts that join a mention's operation on a line."""
        words = [t for t in self.words.get(r.page, ()) if _under(t.bbox, box)]
        region = self._region(r, box, words, non_text)
        if region is not None:
            self.counts["regions"] += 1
            action = self.region_actions.get(region.region_id, ActionKind.REMOVE)
            kind = action if r.action == "replace" and action in _REGION_OPS else ActionKind.REMOVE
            return [RenderOp(r.page, kind, box, region.kind.value, region.region_id)], []
        owners = sorted({self.owner[t.token_id][0] for t in words if t.token_id in self.owner})
        if not owners:
            self.counts["illegible"] += 1
            removed = union_all([box, *(t.bbox for t in words)])
            return [RenderOp(r.page, ActionKind.REMOVE, removed, _ILLEGIBLE, _ref(r.page, removed))], []
        lines: dict[int, list[Token]] = defaultdict(list)
        for t in words:
            lines[t.line_no].append(t)
        direct: list[RenderOp] = []
        parts: list[_Part] = []
        for line_no, line in sorted(lines.items()):
            # On one line the operation covers the region and its whole words; over several lines each line's
            # operation covers that line's words, so that each draws in its own band.
            part = union_all([box, *(t.bbox for t in line)]) if len(lines) == 1 else union_all([t.bbox for t in line])
            owned = [self.owner[t.token_id] for t in line if t.token_id in self.owner]
            on_line = sorted({mention_id for mention_id, _pos in owned})
            if len(owned) == len(line) and len(on_line) == 1:
                positions = tuple(pos for _mention_id, pos in owned)
                parts.append(_Part(r.id, r.action, on_line[0], r.page, line_no, positions, part))
            else:
                direct.append(self._retypeset(r, line, part, on_line or owners))
        self.counts["mixed"] += bool(direct)
        return direct, parts

    def _region(self, r: PlanRegion, box: BBox, words: Sequence[Token], non_text: AbstractSet[str]) -> Region | None:
        """The non-text region a plan region applies, if any: the closest region of the region mentions it names (a
        signature); else the closest region of the text mentions it names (a logo around an organisation's name),
        when they overlap well and no word under the plan region matches it as well."""
        named = sorted(set(r.mentions))
        own = [g for m in named if m in non_text for g in self.regions_of.get(m, ()) if g.page == r.page]
        if own:
            return _closest(own, box)
        found = [g for m in named for g in self.regions_of.get(m, ()) if g.page == r.page]
        if not found:
            return None
        best = _closest(found, box)
        overlap = best.bbox.iou(box)
        word = max((max(t.bbox.iou(box), box.overlap_fraction(t.bbox)) for t in words), default=0.0)
        return best if overlap >= _SAME_REGION and overlap > word else None

    # ------------------------------------------------------------------------------------ operations
    def _kind(self, d: Decision) -> ActionKind:
        """A mention's operation as tool 2 plans it: its action in the render style; text on a sideways line, or
        without a surrogate, is blacked out (``render.plan.plan_ops``)."""
        kind = effective_kind(d.action, self.style)
        if kind in _TEXT_KINDS and any(t in self.rotated for t in d.mention.token_ids):
            return ActionKind.BLACKOUT
        if kind is ActionKind.SURROGATE and d.mention.mention_id not in self.surrogates:
            return ActionKind.BLACKOUT
        return kind

    def _runs(self, parts: Sequence[_Part]) -> list[tuple[RenderOp, list[_Part]]]:
        """A mention's parts on one line (one action) as operations: one per run of its words with no other word
        between them, over the parts' boxes and those whole words.  A part's words count with the mention's words
        between them (its box holds them)."""
        first = parts[0]
        d = self.decisions[first.mention_id]
        m = d.mention
        kind = self._kind(d) if first.action == "replace" else ActionKind.REMOVE
        sur = self.surrogates.get(m.mention_id) if kind in _TEXT_KINDS else None
        filled = sorted({i for p in parts for i in range(min(p.positions), max(p.positions) + 1)})
        runs: list[list[int]] = []
        for i in filled:
            if runs and runs[-1][-1] == i - 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        out = []
        for run in runs:
            members = [p for p in parts if run[0] <= min(p.positions) <= run[-1]]
            words = [self.tokens[m.token_ids[i]] for i in run]
            ink = union_all([t.bbox for t in words])
            texts = [t.text for t in words]
            pieces = [sur.tokens[i] for i in run if sur.tokens[i]] if sur is not None else []
            op = RenderOp(
                page=first.page,
                kind=kind,
                box=union_all([ink, *(p.box for p in members)]),
                reason=m.entity_type.value,
                ref_id=m.mention_id,
                text=" ".join(pieces) if kind in _TEXT_KINDS else None,
                ink_box=ink,
                has_descender=any(c in _DESCENDERS for c in "".join(texts)),
                source_text=" ".join(texts) if kind in _TEXT_KINDS else None,
            )
            out.append((op, members))
        return out

    def _drawn(self, t: Token) -> str | None:
        """What a re-typeset line draws for a word: its mention's piece of the fake, or the word itself where no
        mention has it; None for a mention's word that is not drawn as text."""
        owned = self.owner.get(t.token_id)
        if owned is None:
            return t.text
        mention_id, pos = owned
        sur = self.surrogates.get(mention_id)
        if sur is None or self._kind(self.decisions[mention_id]) not in _TEXT_KINDS:
            return None
        return sur.tokens[pos]

    def _retypeset(self, r: PlanRegion, line: Sequence[Token], box: BBox, covers: Sequence[str]) -> RenderOp:
        """A plan region's line over several mentions, or over a mention and other words: re-typeset, each mention's
        words as their fakes and the other words as they read.  Where that cannot be drawn (a ``remove`` region, a
        sideways line, the blackout style, a mention not drawn as text) the line is blacked out, or removed."""
        ref = _ref(r.page, box)
        drawn = [self._drawn(t) for t in line]
        if (
            r.action == "replace"
            and self.style != "blackout"
            and not any(t.token_id in self.rotated for t in line)
            and None not in drawn
        ):
            source = " ".join(t.text for t in line)
            return RenderOp(
                page=r.page,
                kind=ActionKind.RETYPESET_REGION,
                box=box,
                reason=_RETYPESET,
                ref_id=ref,
                text=" ".join(w for w in drawn if w),
                ink_box=union_all([t.bbox for t in line]),
                has_descender=any(c in _DESCENDERS for c in source),
                source_text=source,
                covers=tuple(covers),
            )
        kind = ActionKind.BLACKOUT if r.action == "replace" else ActionKind.REMOVE
        return RenderOp(r.page, kind, box, _ILLEGIBLE, ref, covers=tuple(covers))

    # ------------------------------------------------------------------------------------------- note
    def _note(self, injection: Injection, applied: int, ops: int) -> str:
        c, n = self.counts, injection.counts
        return (
            f"injected plan: {n['plan_regions']} regions read ({n['plan_regions_duplicate']} with a repeated id), "
            f"{applied} applied by {ops} operations ({c['joined']} joining regions of a mention on a line, "
            f"{c['merged']} the same as another), {c['regions']} applied as non-text regions, {c['mixed']} over "
            f"several mentions or other words, {c['illegible']} over no mention's word removed as illegible, "
            f"{c['beyond']} beyond the last page, {c['replacements']} replacement texts not used, {c['off_page']} "
            "off their page"
        )


def _under(word: BBox, region: BBox) -> bool:
    """The word lies under the region: mostly inside it, or holding most of it (characters inside the word)."""
    return word.overlap_fraction(region) >= _UNDER or region.overlap_fraction(word) >= _UNDER


def _closest(regions: Sequence[Region], box: BBox) -> Region:
    return min(regions, key=lambda g: (-g.bbox.iou(box), g.region_id))


def _ref(page: int, box: BBox) -> str:
    """The id of an operation no single mention or region owns, from its position (``I``: an injected region)."""
    return content_id("I", page, *box.as_tuple())
