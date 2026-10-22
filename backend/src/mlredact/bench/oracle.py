"""The benchmark's oracle modes (``mlredact bench``): injected stage inputs in the pipeline, for the evaluator, never
production.

:class:`OracleInputs` answers the runner's hooks (``mlredact.pipeline.injection.StageInputs``) from what
:func:`mlredact.bench.inject.load_injection` read:

* ``oracle_text``: the injected tokens replace OCR.  Region detection, detection, the policy and planning are tool
  2's own, over the injected text;
* ``oracle_mentions``: the injected mentions also replace region detection, detection and the policy's keeps: every
  one of them is redacted, with the profile's action, or a surrogate where the profile would keep the value (words
  without a letter or digit, which no fake can stand in for, are removed);
* ``oracle_entities``: the injected clusters also replace tool 2's clustering of person names;
* ``oracle_regions``: the injected plan also replaces tool 2's planning: exactly its regions are applied
  (:mod:`mlredact.bench.oracle_plan`).

An injected mention becomes tool 2's mentions over whole tokens, and regions where its anchors are not text (a
signature) or are not explained by its tokens (a logo around an organisation's name).  Where injected mentions share
a token, the tokens are redacted once, with the type of the mention that has the most of them; every injected mention
is still reported under its own id (``InjectReport``).  Verification checks the output as usual, but from
``oracle_mentions`` on it re-plans only removed values that are still readable, and in ``oracle_regions`` nothing:
whatever else it finds was not injected, and only counts against the release.

Nothing here logs.  Errors and notes carry counts and fixed names only, never injected text.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Any

from mlredact.bench.inject import Injection, PredMention
from mlredact.bench.labels import (
    FALLBACK,
    FAMILIES,
    LABEL_MAP,
    REGION_KINDS_USED,
    Family,
    family_of,
    refine,
    split_location,
)
from mlredact.bench.oracle_plan import InjectedPlan, injected_ops
from mlredact.config.schema import AppConfig
from mlredact.core.canonical import content_id
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import Candidate, Evidence, Mention, PageAnalysis, Region, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, RegionKind, VerificationCheck
from mlredact.detect.engine import DetectionResult, default_detectors
from mlredact.detect.fusion import TYPE_PRIORITY
from mlredact.pipeline.injection import InjectedPage, InjectedStage, InjectReport, ReportedMention, on_page, pt_box
from mlredact.policy.engine import Decision, decide, decide_injected
from mlredact.resolve.persons import resolve_persons, with_clusters
from mlredact.surrogate.assign import MentionSurrogate
from mlredact.text.views import TextView, build_line_view, build_views
from mlredact.verify.residual import ResidualHit

# The injected mentions' evidence, and the injected regions' detector (manifest provenance).
DETECTOR = ("bench.inject", "1")
REGION_DETECTOR = "inject"
# The oracle modes, each injecting one stage input more than the one before.
MODES = ("oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions")
# Residual hit sources (``ResidualHit.source``) verification may re-plan, per mode; every hit still counts for the
# release.  From oracle_mentions on, what the rules find again was not injected, and stays as it was.
REPLAN_SOURCES: dict[str, tuple[str, ...]] = {
    "oracle_text": ("removed_value", "redetect", "region"),
    "oracle_mentions": ("removed_value",),
    "oracle_entities": ("removed_value",),
    "oracle_regions": (),
}
_SOURCE_NAMES = {"removed_value": "removed values", "redetect": "re-detection", "region": "regions"}
# A region anchor that one of the mention's token boxes matches this closely (IoU), or covers this much of, is that
# token: a word box, or a character box inside the word.  Any other anchor of a text mention becomes a region.
_EXPLAINED = 0.9
# A mention that names neither tokens nor a span has the tokens at least this much inside its region anchors, and the
# token that is an anchor none of those is (``_EXPLAINED``: a character box inside a word is the word).
_PROJECTED = 0.5
# Labels of a fixed vocabulary (the benchmark's types, tool 2's own labels): only these name a rule in the manifest.
# Any other label is injected text.
_RULE_LABELS = frozenset({*FAMILIES, *LABEL_MAP, "QUASI_IDENTIFIER"})
_RANK = {t: i for i, t in enumerate(TYPE_PRIORITY)}
# An injected mention placed on the pages: the mention, its family, its tokens, how they were found, its regions.
_Placed = tuple[PredMention, Family, tuple[int, ...], str, tuple[Region, ...]]


def check_region_actions(config: AppConfig) -> None:
    """Refuse a profile that keeps a kind of region injected mentions become (a signature, a logo, ...): it would
    leave injected PII on the page.  No packaged profile does."""
    actions = config.policy.region_actions
    kept = sorted(kind for kind in REGION_KINDS_USED if actions[kind] is ActionKind.KEEP)
    if kept:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, inject=True, kept_region=kept[0])


class OracleInputs:
    """The runner's hooks for one document's injected stage inputs (``StageInputs``).

    ``stage`` and ``plan`` keep what the stage files report (the injected mentions as tool 2 redacted them, their
    aliases and clusters, how each plan region was applied, count notes); a repeated run replaces it."""

    def __init__(self, injection: Injection) -> None:
        if injection.mode not in MODES:
            raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, inject=True, mode_supported=False)
        self._injection = injection
        self._beyond = 0  # injected tokens on pages the document does not have
        self._staged: _Staged | None = None
        self._planned: InjectedPlan | None = None

    @property
    def mode(self) -> str:
        return self._injection.mode

    @property
    def detect_regions(self) -> bool:
        """Tool 2 looks for regions around the injected text while it detects mentions itself (``oracle_text``)."""
        return self._injection.mentions is None

    def pages(self, page_count: int) -> list[InjectedPage]:
        self._beyond = sum(len(p.tokens) for i, p in sorted(self._injection.pages.items()) if i >= page_count)
        return [self._injection.page(i) for i in range(page_count)]

    def stage(self, pages: Sequence[PageAnalysis], config: AppConfig, doc_key: str) -> InjectedStage | None:
        """Detection and policy from the injected mentions; None in ``oracle_text`` (tool 2 detects)."""
        if self._injection.mentions is None:
            return None
        self._planned = None
        self._staged = _Stager(self._injection, pages, doc_key).run(config)
        return self._staged.stage

    def plan(
        self,
        *,
        decisions: Sequence[Decision],
        surrogates: Mapping[str, MentionSurrogate],
        tokens: Mapping[str, Token],
        pages: Sequence[PageAnalysis],
        region_actions: Mapping[str, ActionKind],
        rotated: AbstractSet[str],
        config: AppConfig,
    ) -> dict[int, tuple[RenderOp, ...]] | None:
        """The operations that apply the injected plan (``oracle_regions``); None in the other modes (tool 2
        plans)."""
        if self._injection.plan is None:
            return None
        staged = self._staged
        self._planned = injected_ops(
            self._injection,
            staged.mentions if staged is not None else (),
            staged.aliases if staged is not None else {},
            decisions=decisions,
            surrogates=surrogates,
            tokens=tokens,
            pages=pages,
            region_actions=region_actions,
            rotated=rotated,
            style=config.redaction.style,
        )
        return self._planned.ops

    def replan_hits(self, hits: Sequence[ResidualHit]) -> list[ResidualHit]:
        if self.mode == "oracle_text":
            return list(hits)
        allowed = REPLAN_SOURCES[self.mode]
        return [h for h in hits if h.source in allowed]

    def report(self) -> InjectReport:
        injection, staged, planned = self._injection, self._staged, self._planned
        notes = (
            _text_note(injection, self._beyond),
            *(staged.notes if staged is not None else ()),
            *([planned.note] if planned is not None else []),
        )
        if staged is None:
            return InjectReport(injection.mode, injection.text_source, notes=notes)
        echoes = planned.echoes if planned is not None else None
        return InjectReport(
            injection.mode, injection.text_source, staged.mentions, staged.aliases, staged.clusters, echoes, notes
        )

    def residual_note(self, manifest: Mapping[str, Any]) -> str:
        """The ``run_info`` note on verification's residual check (V4): its hits by source after the re-plans, and
        the sources this mode re-plans."""
        allowed = ", ".join(_SOURCE_NAMES[s] for s in REPLAN_SOURCES[self.mode]) or "none"
        checks = manifest.get("verification", {}).get("checks", ())
        metrics = next((c["metrics"] for c in checks if c.get("check") == VerificationCheck.RESIDUAL.value), None)
        if metrics is None:
            return f"V4 residual check not run (re-plans from: {allowed})"
        return (
            f"V4 residual hits: {metrics['residual_tokens']} ({metrics['from_removed_values']} removed values, "
            f"{metrics['from_redetection']} re-detection, {metrics['from_regions']} regions) after "
            f"{metrics['replans']} re-plans (re-plans from: {allowed})"
        )


@dataclass(frozen=True, slots=True)
class _Staged:
    """What one ``stage`` run produced: the runner's stage, and the report's parts."""

    stage: InjectedStage
    mentions: tuple[ReportedMention, ...]
    aliases: dict[str, tuple[str, ...]]
    clusters: tuple[tuple[str, tuple[str, ...]], ...] | None
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Anchored:
    """An injected mention on tool 2's pages.  ``tokens``: its line-view token indices, in reading order;
    ``found_by``: how its tokens, or else its regions, were found (``tokens``, ``span``, ``regions``; empty for
    neither); ``kind``: its own type, for a text label with tokens; ``regions``: the regions it became."""

    source: PredMention
    family: Family
    tokens: tuple[int, ...]
    found_by: str
    kind: EntityType | None
    regions: tuple[Region, ...]


@dataclass(frozen=True, slots=True)
class _Text:
    """An injected text mention with tokens, and its own type."""

    anchored: _Anchored
    kind: EntityType

    @property
    def id(self) -> str:
        return self.anchored.source.id

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.anchored.tokens


class _Stager:
    """One run of ``OracleInputs.stage``: the injected mentions on the analysed pages."""

    def __init__(self, injection: Injection, pages: Sequence[PageAnalysis], doc_key: str) -> None:
        self.injection = injection
        self.pages = pages
        self.doc_key = doc_key
        self.view = build_line_view(pages)
        self.index = {t.token_id: i for i, t in enumerate(self.view.tokens)}
        self.geoms = {p.geometry.index: p.geometry for p in pages}
        self.counts: Counter[str] = Counter()

    def run(self, config: AppConfig) -> _Staged:
        sources = sorted(self.injection.mentions or (), key=lambda m: m.id)
        placed = [self._place(m) for m in sources]
        candidates = self._candidates(placed)
        anchored = [self._typed(p, candidates) for p in placed]

        texts = [_Text(a, a.kind) for a in anchored if a.kind is not None]
        internal: list[Mention] = []
        aliases: dict[str, tuple[str, ...]] = {}
        slots: dict[int, tuple[str, int]] = {}
        groups: dict[str, list[_Text]] = {}  # internal mention id -> its group, ranked (the owner first)
        for group in _groups(texts):
            ranked = sorted(group, key=self._rank)
            winner = ranked[0]
            runs = _runs(sorted({i for t in group for i in t.tokens}), self.view)
            self.counts["nested"] += len(group) - 1
            self.counts["split"] += len(runs) > 1
            for run in runs:
                parts = self._parts(run, winner)
                self.counts["location_split"] += len(parts) > 1
                for kind, start, end in parts:
                    mention = self._mention(run[start:end], kind, winner)
                    covered = set(run[start:end])
                    internal.append(mention)
                    aliases[mention.mention_id] = tuple(t.id for t in ranked if covered.intersection(t.tokens))
                    groups[mention.mention_id] = ranked
                    slots.update((i, (mention.mention_id, pos)) for pos, i in enumerate(run[start:end]))
        mentions = sorted(internal, key=lambda m: (m.token_ids[0], m.token_ids[-1], m.entity_type.value))

        persons = resolve_persons(mentions, self.view)
        clusters = None
        if self.injection.clusters is not None:
            first = self._first_clusters({a.source.id: a for a in anchored})
            persons = with_clusters(persons, _person_clusters(mentions, groups, first))
            members: dict[str, list[str]] = defaultdict(list)
            for mention_id, cluster in sorted(first.items()):
                members[cluster].append(mention_id)
            clusters = tuple((cluster, tuple(ids)) for cluster, ids in sorted(members.items()))
        # A run with no letter or digit (a dash, a bullet, a barcode's bars read as "|||") has nothing a fake could
        # stand in for: it is removed.
        decisions = [
            d if _fakeable(d.mention.value) else Decision(d.mention, ActionKind.REMOVE)
            for d in decide_injected(mentions, config.policy)
        ]
        self.counts["kept_acted"] = sum(d.action is ActionKind.KEEP for d in decide(mentions, config.policy))
        self.counts["unfakeable"] = sum(not _fakeable(m.value) for m in mentions)

        regions: dict[str, Region] = {}
        for a in anchored:
            for r in a.regions:
                regions.setdefault(r.region_id, r)
                named = aliases.get(r.region_id, ())
                aliases[r.region_id] = named if a.source.id in named else (*named, a.source.id)
        detection = DetectionResult(
            mentions=tuple(mentions),
            low_evidence=(),
            detectors=(DETECTOR, ("resolve.persons", "1")),
            propagation_rounds=0,
            persons=persons,
            line_view=self.view,
        )
        stage = InjectedStage(_with_regions(self.pages, regions.values()), detection, tuple(decisions))
        owners = {group[0].id for group in groups.values()}
        reported = [self._report(a, slots, owners) for a in anchored if a.tokens or a.regions]
        reported.sort(key=lambda r: (min(page for page, _box in r.boxes), r.mention_id))
        notes = (self._mention_note(), *([self._cluster_note()] if clusters is not None else []))
        return _Staged(stage, tuple(reported), dict(sorted(aliases.items())), clusters, notes)

    # ------------------------------------------------------------------------------------- anchors
    def _place(self, m: PredMention) -> _Placed:
        """A mention's tokens and the regions it becomes."""
        family = family_of(m.label)
        self.counts["fallback"] += family is FALLBACK
        anchors = self._anchors(m)
        tokens, found_by = self._tokens(m, family, anchors)
        regions = self._regions(m, family, tokens, anchors)
        if not tokens:
            found_by = "regions" if regions else ""
        self.counts[f"by_{found_by}" if found_by else "unanchored"] += 1
        return m, family, tokens, found_by, regions

    def _anchors(self, m: PredMention) -> list[tuple[int, BBox]]:
        """The mention's region anchors as (page, pixel box), on the document's pages."""
        out = []
        for a in m.regions or ():
            geometry = self.geoms.get(a.page)
            if geometry is None:
                self.counts["anchors_beyond"] += 1
                continue
            if not on_page(a.poly, geometry):
                self.counts["anchors_off_page"] += 1
                continue
            out.append((a.page, pt_box(a.poly, geometry)))
        return out

    def _tokens(
        self, m: PredMention, family: Family, anchors: Sequence[tuple[int, BBox]]
    ) -> tuple[tuple[int, ...], str]:
        """The mention's tokens and how they were found: by its token ids, else by its span, else (a text label) as
        the tokens mostly inside its region anchors, and the token that holds an anchor none of those is (part of a
        word: the box of some of its characters)."""
        if m.tokens:
            ids = [self.injection.token_ids.get(t) for t in dict.fromkeys(m.tokens)]
            found = {self.index[i] for i in ids if i is not None and i in self.index}
            self.counts["token_ids_not_found"] += sum(i is None or i not in self.index for i in ids)
            if found:
                return tuple(sorted(found)), "tokens"
        if m.span is not None:
            spanned = self.injection.span_token_ids(m.span.page, m.span.start, m.span.end)
            found = {self.index[i] for i in spanned if i in self.index}
            if found:
                return tuple(sorted(found)), "span"
        if family.types and anchors:
            boxes: dict[int, list[BBox]] = defaultdict(list)
            for page, box in anchors:
                boxes[page].append(box)
            tokens = self.view.tokens
            projected = {i for i, t in enumerate(tokens) if _covered(t.bbox, boxes.get(t.page, ())) >= _PROJECTED}
            for page, box in anchors:
                if not self._explained(page, box, sorted(projected)):
                    projected.update(
                        i for i, t in enumerate(tokens) if t.page == page and box.overlap_fraction(t.bbox) >= _EXPLAINED
                    )
            if projected:
                return tuple(sorted(projected)), "regions"
        return (), ""

    def _regions(
        self, m: PredMention, family: Family, tokens: Sequence[int], anchors: Sequence[tuple[int, BBox]]
    ) -> tuple[Region, ...]:
        """The regions a mention becomes.  A region label (a signature): every anchor, and its tokens line by line.
        A text label: the anchors its tokens do not explain (a logo around the name), as its family's region kind."""
        rule, score = _rule(m.label), m.conf if m.conf is not None else 1.0
        boxes: list[tuple[int, BBox]]
        if family.types:
            boxes = [(page, box) for page, box in anchors if not self._explained(page, box, tokens)]
            self.counts["extra_regions"] += len(boxes)
        else:
            lines: dict[tuple[int, int], list[BBox]] = defaultdict(list)
            for i in tokens:
                t = self.view.tokens[i]
                lines[(t.page, t.line_no)].append(t.bbox)
            boxes = [*anchors, *((page, union_all(b)) for (page, _line), b in sorted(lines.items()))]
        return tuple(_region(page, family.region_kind, box, rule, score) for page, box in sorted(set(boxes)))

    def _explained(self, page: int, anchor: BBox, tokens: Sequence[int]) -> bool:
        """One of the tokens is the anchor: their boxes match (IoU), or the token's box holds the anchor (a character
        box inside the word)."""
        for i in tokens:
            box = self.view.tokens[i].bbox
            if self.view.tokens[i].page == page and (
                box.iou(anchor) >= _EXPLAINED or anchor.overlap_fraction(box) >= _EXPLAINED
            ):
                return True
        return False

    # ---------------------------------------------------------------------------------------- types
    def _candidates(self, placed: Sequence[_Placed]) -> list[tuple[range, Candidate]]:
        """Tool 2's rule detectors over the injected text, once, when an identifier's type is to be chosen."""
        if not any(family.ambiguous and tokens for _m, family, tokens, _f, _r in placed):
            return []
        views = build_views(self.pages)
        return [(views[c.view].token_range(c.start, c.end), c) for d in default_detectors() for c in d.detect(views)]

    def _typed(self, placed: _Placed, candidates: Sequence[tuple[range, Candidate]]) -> _Anchored:
        """The mention with its own type: a location's is its locality, or a lone postcode's; an identifier's is
        the one tool 2's rules recognise in it; any other text label's is its family's."""
        m, family, tokens, found_by, regions = placed
        kind = None
        if family.types and tokens:
            if family.location:
                parts = split_location([self.view.tokens[i].text for i in tokens])
                kind = EntityType.POSTCODE if [p[0] for p in parts] == [EntityType.POSTCODE] else EntityType.LOCALITY
            else:
                kind = refine(family, range(tokens[0], tokens[-1] + 1), candidates)
        return _Anchored(m, family, tokens, found_by, kind, regions)

    def _rank(self, t: _Text) -> tuple[int, int, int, str]:
        """Who owns shared tokens: the most tokens, then the longest text, then type priority, then the id."""
        text = t.anchored.source.text or " ".join(self.view.tokens[i].text for i in t.tokens)
        return (-len(t.tokens), -len(text), _RANK[t.kind], t.id)

    # ------------------------------------------------------------------------------------- mentions
    def _parts(self, run: Sequence[int], winner: _Text) -> list[tuple[EntityType, int, int]]:
        """A run's mentions as (type, start, end): a location's trailing postcode is a mention of its own."""
        if winner.anchored.family.location:
            return split_location([self.view.tokens[i].text for i in run])
        return [(winner.kind, 0, len(run))]

    def _mention(self, run: Sequence[int], kind: EntityType, winner: _Text) -> Mention:
        tokens = [self.view.tokens[i] for i in run]
        ids = tuple(t.token_id for t in tokens)
        source = winner.anchored.source
        score = source.conf if source.conf is not None else 1.0
        evidence = Evidence(DETECTOR[0], DETECTOR[1], _rule(source.label), EvidenceStrength.STRONG, score)
        return Mention(
            mention_id=content_id("M", self.doc_key, ids[0], ids[-1], kind.value),
            entity_type=kind,
            token_ids=ids,
            evidence=(evidence,),
            value=" ".join(t.text for t in tokens),
        )

    def _first_clusters(self, anchored: Mapping[str, _Anchored]) -> dict[str, str]:
        """Each injected mention's cluster: the first, in id order, of the clusters that list it."""
        first: dict[str, str] = {}
        several: set[str] = set()
        for cluster in sorted(self.injection.clusters or (), key=lambda c: c.id):
            for mention_id in dict.fromkeys(cluster.mentions):
                a = anchored.get(mention_id)
                if a is None:
                    self.counts["unknown_members"] += 1
                    continue
                if mention_id in first:
                    several.add(mention_id)
                    continue
                first[mention_id] = cluster.id
                self.counts["non_person_members"] += a.family.default is not EntityType.PERSON
        self.counts["in_several"] = len(several)
        return first

    def _report(self, a: _Anchored, slots: Mapping[int, tuple[str, int]], owners: AbstractSet[str]) -> ReportedMention:
        tokens = [self.view.tokens[i] for i in a.tokens]
        return ReportedMention(
            mention_id=a.source.id,
            label=a.kind.value if a.kind is not None else a.family.reported,
            text=" ".join(t.text for t in tokens),
            token_ids=tuple(t.token_id for t in tokens),
            boxes=(*((t.page, t.bbox) for t in tokens), *((r.page, r.bbox) for r in a.regions)),
            slots=tuple(slots[i] for i in a.tokens) if a.kind is not None else (),
            owner=a.kind is None or a.source.id in owners,
            conf=a.source.conf,
        )

    # ---------------------------------------------------------------------------------------- notes
    def _mention_note(self) -> str:
        c, n = self.counts, self.injection.counts
        return (
            f"injected mentions: {n['mentions']} read ({n['mentions_duplicate']} with a repeated id, "
            f"{n['page_texts_differ']} page texts unlike the injected text's), {c['by_tokens']} placed by token ids, "
            f"{c['by_span']} by span, {c['by_regions']} by region anchors, {c['unanchored']} not placed "
            f"({c['token_ids_not_found']} token ids not found, {c['anchors_beyond']} anchors beyond the last page), "
            f"{c['nested']} sharing tokens another one owns, {c['split']} split into runs, "
            f"{c['location_split']} locations split into locality and postcode, {c['kept_acted']} of a kept type "
            f"redacted, {c['fallback']} labels taken as other_id, {c['extra_regions']} anchors not explained by "
            f"their tokens, {c['anchors_off_page']} anchors off their page, {c['unfakeable']} runs without a letter "
            "or digit removed"
        )

    def _cluster_note(self) -> str:
        c, n = self.counts, self.injection.counts
        return (
            f"injected clusters: {n['clusters']} read ({n['clusters_merged']} merged by id), {c['in_several']} "
            f"mentions in several clusters, {c['non_person_members']} non-person members, {c['unknown_members']} "
            "members that are not mentions"
        )


def _text_note(injection: Injection, beyond: int) -> str:
    n = injection.counts
    return (
        f"injected text: {n['tokens']} tokens ({n['tokens_empty']} empty, {n['tokens_duplicate']} with a repeated id, "
        f"{beyond} beyond the last page, {n['pages_merged']} page entries merged), {n['lines']} lines "
        f"({n['lines_split']} split at a wide gap)"
    )


def _rule(label: str) -> str:
    """The manifest's evidence rule for an injected label: never the label itself unless it is a known one."""
    return f"inject.{label}" if label in _RULE_LABELS else "inject.other"


def _fakeable(value: str) -> bool:
    """A fake can stand in for the value: it has a letter or a digit (fakes change those, and keep every other
    character as it is)."""
    return any(c.isalpha() or c.isdigit() for c in value)


def _region(page: int, kind: RegionKind, box: BBox, rule: str, score: float) -> Region:
    """An injected region, identified by its position like tool 2's own (so its stand-in is drawn the same way)."""
    return Region(content_id("G", page, kind.value, *box.as_tuple()), page, kind, box, score, REGION_DETECTOR, rule)


def _covered(box: BBox, others: Sequence[BBox]) -> float:
    """The fraction of ``box`` that ``others`` cover (counted as verification counts coverage)."""
    covered = sum(i.area for o in others if (i := box.intersection(o)) is not None)
    return min(1.0, covered / box.area) if box.area else 0.0


def _groups(texts: Sequence[_Text]) -> list[list[_Text]]:
    """Text mentions that share a token, transitively (union-find)."""
    parent = list(range(len(texts)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first: dict[int, int] = {}
    for k, t in enumerate(texts):
        for i in t.tokens:
            if i in first:
                parent[root(k)] = root(first[i])
            else:
                first[i] = k
    out: dict[int, list[_Text]] = defaultdict(list)
    for k, t in enumerate(texts):
        out[root(k)].append(t)
    return sorted(out.values(), key=lambda g: min(t.tokens[0] for t in g))


def _runs(indices: Sequence[int], view: TextView) -> list[list[int]]:
    """Maximal runs of tokens that follow each other in reading order on one page (a line break may fall inside)."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and runs[-1][-1] == i - 1 and view.tokens[i - 1].page == view.tokens[i].page:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _person_clusters(
    mentions: Sequence[Mention], groups: Mapping[str, Sequence[_Text]], first: Mapping[str, str]
) -> dict[str, str]:
    """Each person mention's injected cluster: its owner's, else the first, in id order, of the clusters of the
    other injected persons it shares tokens with."""
    out: dict[str, str] = {}
    for m in mentions:
        if m.entity_type is not EntityType.PERSON:
            continue
        owner, *others = groups[m.mention_id]
        cluster = first.get(owner.id)
        if cluster is None:
            cluster = min((first[t.id] for t in others if t.kind is EntityType.PERSON and t.id in first), default=None)
        if cluster is not None:
            out[m.mention_id] = cluster
    return out


def _with_regions(pages: Sequence[PageAnalysis], regions: Iterable[Region]) -> tuple[PageAnalysis, ...]:
    """The pages with the injected regions added, in reading order."""
    by_page: dict[int, list[Region]] = defaultdict(list)
    for r in regions:
        by_page[r.page].append(r)
    out = []
    for p in pages:
        added = sorted(
            by_page.get(p.geometry.index, ()),
            key=lambda r: (r.bbox.y0, r.bbox.x0, r.bbox.y1, r.bbox.x1, r.kind.value, r.region_id),
        )
        out.append(replace(p, regions=(*p.regions, *added)) if added else p)
    return tuple(out)
