"""The benchmark's injected stage inputs (``mlredact bench`` oracle modes): for the evaluator, never production.

In an oracle mode the benchmark puts perfect upstream input into ``--inject-dir``: ``text.json`` (``oracle_text``),
plus ``mentions.json`` (``oracle_mentions``), plus ``entities.json`` (``oracle_entities``), plus ``plan.json``
(``oracle_regions``).  They follow its stage-file schemas (``rb.text/0.1``, ``rb.mentions/0.1``, ``rb.entities/0.1``,
``rb.plan/0.1``), key for key, and are checked as the benchmark checks them (an unknown key is an error) and more
strictly: display points only, finite coordinates no further than a million points from the page.

The files hold document text.  Nothing here logs, and a file that cannot be read or is not valid raises
``EnvironmentErrorMl(E_CONFIG_INVALID)`` naming the file and counting the errors, nothing more: validation messages
repeat their input.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, model_validator

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.model import token_id
from mlredact.pipeline.injection import InjectedPage, InjectedToken

# A coordinate in display points: finite, and within a million points (some 350 m) of the page's origin, far beyond
# any page; scaled to pixels, it stays a number.
_FAR_PT = 1e6
Coordinate = Annotated[FiniteFloat, Field(ge=-_FAR_PT, le=_FAR_PT)]
Point = tuple[Coordinate, Coordinate]
Polygon = Annotated[list[Point], Field(min_length=3)]


class _Strict(BaseModel):
    """Unknown keys are errors, as in the benchmark's own schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class _Framed(_Strict):
    """Display points only: the benchmark converts every other frame before it injects a file.  A dpi may come with
    them, as the benchmark's schemas allow; display points do not need one, so it is not used."""

    frame: Literal["display_pt"] = "display_pt"
    dpi: float | None = Field(None, gt=0)


# -- rb.text/0.1 --------------------------------------------------------------------------------------------------
class TextToken(_Strict):
    id: str = Field(min_length=1)
    text: str = Field(repr=False)
    poly: Polygon
    conf: float | None = Field(None, ge=0, le=1)
    line: str | None = None
    order: int | None = Field(None, ge=0)


class TextPage(_Strict):
    index: int = Field(ge=0)
    tokens: list[TextToken] = Field(default=[], repr=False)
    detected_rotation: Literal[0, 90, 180, 270] | None = None
    skew_deg: float | None = None


class TextArtifact(_Framed):
    schema_id: Literal["rb.text/0.1"] = Field("rb.text/0.1", alias="schema")
    source: Literal["ocr", "text_layer", "hybrid"]
    pages: list[TextPage]


# -- rb.mentions/0.1 ----------------------------------------------------------------------------------------------
class RegionAnchor(_Strict):
    page: int = Field(ge=0)
    poly: Polygon


class TextSpanAnchor(_Strict):
    page: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class PredMention(_Strict):
    id: str = Field(min_length=1)
    label: str
    text: str | None = Field(None, repr=False)
    tokens: list[str] | None = None
    regions: list[RegionAnchor] | None = None
    span: TextSpanAnchor | None = None
    conf: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def _anchored(self) -> PredMention:
        if not (self.tokens or self.regions or self.span):
            raise ValueError("a mention needs tokens, regions or span")
        return self


class MentionsArtifact(_Framed):
    schema_id: Literal["rb.mentions/0.1"] = Field("rb.mentions/0.1", alias="schema")
    page_texts: dict[int, str] | None = Field(None, repr=False)
    mentions: list[PredMention]


# -- rb.entities/0.1 ----------------------------------------------------------------------------------------------
class Cluster(_Strict):
    id: str = Field(min_length=1)
    mentions: list[str] = Field(min_length=1)
    attributes: dict[str, str] = Field(default={}, repr=False)


class EntitiesArtifact(_Strict):
    schema_id: Literal["rb.entities/0.1"] = Field("rb.entities/0.1", alias="schema")
    clusters: list[Cluster]


# -- rb.plan/0.1 --------------------------------------------------------------------------------------------------
class PlanRegion(_Strict):
    id: str = Field(min_length=1)
    page: int = Field(ge=0)
    poly: Polygon
    action: Literal["remove", "replace"]
    replacement: str | None = Field(None, repr=False)
    mentions: list[str] = []


class PlanArtifact(_Framed):
    schema_id: Literal["rb.plan/0.1"] = Field("rb.plan/0.1", alias="schema")
    regions: list[PlanRegion]


# -- the inject folder --------------------------------------------------------------------------------------------
class InjectedFile(StrEnum):
    """An injected stage input, by its file name in the inject folder."""

    TEXT = "text.json"
    MENTIONS = "mentions.json"
    ENTITIES = "entities.json"
    PLAN = "plan.json"

    @property
    def kind(self) -> str:
        """The benchmark's name for this injection (its candidates' ``accepts_injection``)."""
        return _KINDS[self]


_KINDS = {
    InjectedFile.TEXT: "text",
    InjectedFile.MENTIONS: "mentions",
    InjectedFile.ENTITIES: "entities",
    InjectedFile.PLAN: "regions",
}
# The files each benchmark mode injects (the benchmark's MODE_INJECTS, benchmark/schemas/run.py).
MODE_FILES: dict[str, tuple[InjectedFile, ...]] = {
    "e2e": (),
    "oracle_text": (InjectedFile.TEXT,),
    "oracle_mentions": (InjectedFile.TEXT, InjectedFile.MENTIONS),
    "oracle_entities": (InjectedFile.TEXT, InjectedFile.MENTIONS, InjectedFile.ENTITIES),
    "oracle_regions": (InjectedFile.TEXT, InjectedFile.MENTIONS, InjectedFile.ENTITIES, InjectedFile.PLAN),
}
# What reading counts (every name is present): page entries merged into an earlier entry of the same page; tokens
# read, and dropped for empty text or an id seen before; lines formed, and split at a wide gap; pages whose given
# page_texts differ from the text's own; mentions, clusters and plan regions read, and dropped (merged for clusters)
# for an id seen before.
COUNTS = (
    "pages_merged",
    "tokens",
    "tokens_empty",
    "tokens_duplicate",
    "lines",
    "lines_split",
    "page_texts_differ",
    "mentions",
    "mentions_duplicate",
    "clusters",
    "clusters_merged",
    "plan_regions",
    "plan_regions_duplicate",
)
# A token without a line value joins the previous token's line when their heights overlap this much (relative to the
# smaller one), as rows are compared elsewhere (``BBox.vertical_overlap``).
_SAME_ROW = 0.5
# Two tokens of a row further apart than this many token heights are on different lines (table columns, a date
# across the page): a line's tokens are unioned when they are redacted together.
_WIDE_GAP = 3.0

_Box = tuple[float, float, float, float]
# A page's tokens in file order: (start, end) of the text in the page text, and the tool 2 id (None: dropped).
_Offsets = tuple[tuple[int, int, str | None], ...]


class _Identified(Protocol):
    @property
    def id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Injection:
    """The injected stage inputs of one document, as :func:`load_injection` read them.

    ``pages``: every page ``text.json`` lists, by index, with its tokens in reading order, numbered into lines as
    tool 2 numbers tokens; ``token_ids``: injected token id -> tool 2 token id (the first token with the id in page
    and reading order; a token dropped for empty text has none); ``page_texts``: each listed page's token texts joined
    by single spaces in file order, what a mention's ``span`` counts in; ``spans``: per page, each token of the file
    as (start, end) in its page text and its tool 2 id, if any; ``mentions``, ``clusters`` and ``plan``: the injected
    mentions, clusters and plan regions in file order, ids unique (None when the mode does not inject them);
    ``counts``: see :data:`COUNTS`."""

    mode: str
    text_source: str
    pages: Mapping[int, InjectedPage] = field(repr=False)
    token_ids: Mapping[str, str] = field(repr=False)
    page_texts: Mapping[int, str] = field(repr=False)
    spans: Mapping[int, _Offsets] = field(repr=False)
    mentions: tuple[PredMention, ...] | None = field(repr=False)
    clusters: tuple[Cluster, ...] | None = field(repr=False)
    plan: tuple[PlanRegion, ...] | None = field(repr=False)
    counts: Mapping[str, int]

    def page(self, index: int) -> InjectedPage:
        """Page ``index``'s injected text (no token when ``text.json`` does not list the page)."""
        page = self.pages.get(index)
        return page if page is not None else InjectedPage(index)

    def span_token_ids(self, page: int, start: int, end: int) -> tuple[str, ...]:
        """Tool 2 ids of the tokens a span ``[start, end)`` of ``page_texts[page]`` overlaps, in file order (the
        benchmark's own rule for spans)."""
        return tuple(tid for s, e, tid in self.spans.get(page, ()) if tid is not None and s < end and e > start)


def load_injection(mode: str, inject_dir: Path) -> Injection:
    """Read and check the files ``mode`` injects from ``inject_dir`` (no other file is opened).  A file that cannot
    be read or is not valid raises ``EnvironmentErrorMl(E_CONFIG_INVALID)`` with the file and an error count."""
    files = MODE_FILES.get(mode, ())
    if not files:
        raise ValueError("this mode injects no stage inputs")
    text = _read(inject_dir, InjectedFile.TEXT, TextArtifact)
    found = _read(inject_dir, InjectedFile.MENTIONS, MentionsArtifact) if InjectedFile.MENTIONS in files else None
    entities = _read(inject_dir, InjectedFile.ENTITIES, EntitiesArtifact) if InjectedFile.ENTITIES in files else None
    plan = _read(inject_dir, InjectedFile.PLAN, PlanArtifact) if InjectedFile.PLAN in files else None
    counts: Counter[str] = Counter()
    pages, token_ids, page_texts, spans = _read_text(text, counts)
    if found is not None and found.page_texts:
        counts["page_texts_differ"] = sum(
            given != page_texts.get(page, "") for page, given in sorted(found.page_texts.items())
        )
    return Injection(
        mode=mode,
        text_source=text.source,
        pages=pages,
        token_ids=token_ids,
        page_texts=page_texts,
        spans=spans,
        mentions=_first_by_id(found.mentions, "mentions", counts) if found is not None else None,
        clusters=_merged(entities.clusters, counts) if entities is not None else None,
        plan=_first_by_id(plan.regions, "plan_regions", counts) if plan is not None else None,
        counts={name: counts[name] for name in COUNTS},
    )


def _read[M: BaseModel](folder: Path, name: InjectedFile, model: type[M]) -> M:
    try:
        return model.model_validate(json.loads((folder / name.value).read_text(encoding="utf-8")))
    except ValidationError as exc:
        # Validation messages repeat their input: the count only.
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, inject=True, file=name, errors=exc.error_count()) from None
    except (OSError, ValueError, RecursionError):
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, inject=True, file=name, errors=1) from None


def _read_text(
    text: TextArtifact, counts: Counter[str]
) -> tuple[dict[int, InjectedPage], dict[str, str], dict[int, str], dict[int, _Offsets]]:
    """Pages in index order, tokens in reading order: by ``order`` (tokens without one after the others), ties in
    file order.  A token with an id seen before, or with empty text, is dropped."""
    listed: dict[int, list[TextToken]] = {}
    for p in text.pages:
        if p.index in listed:
            counts["pages_merged"] += 1
        listed.setdefault(p.index, []).extend(p.tokens)
    pages: dict[int, InjectedPage] = {}
    token_ids: dict[str, str] = {}
    page_texts: dict[int, str] = {}
    spans: dict[int, _Offsets] = {}
    seen: set[str] = set()
    for index in sorted(listed):
        file_order = listed[index]
        counts["tokens"] += len(file_order)
        kept: list[int] = []
        for i in sorted(range(len(file_order)), key=lambda i: _reading_key(file_order[i], i)):
            tok = file_order[i]
            if tok.id in seen:
                counts["tokens_duplicate"] += 1
            elif not tok.text.strip():
                counts["tokens_empty"] += 1
            else:
                kept.append(i)
            seen.add(tok.id)
        page = _lines(index, [file_order[i] for i in kept], counts)
        ids = {i: token_id(index, t.line_no, t.word_no) for i, t in zip(kept, page.tokens, strict=True)}
        for i in kept:
            token_ids[file_order[i].id] = ids[i]
        pages[index] = page
        page_texts[index] = " ".join(t.text for t in file_order)
        offsets: list[tuple[int, int, str | None]] = []
        pos = 0
        for i, t in enumerate(file_order):
            offsets.append((pos, pos + len(t.text), ids.get(i)))
            pos += len(t.text) + 1
        spans[index] = tuple(offsets)
    return pages, token_ids, page_texts, spans


def _reading_key(tok: TextToken, position: int) -> tuple[bool, int, int]:
    return (tok.order is None, tok.order if tok.order is not None else 0, position)


def _lines(index: int, tokens: Sequence[TextToken], counts: Counter[str]) -> InjectedPage:
    """Tokens (in reading order) numbered into lines.  A line is a run of tokens with the same line value; a value
    that comes back later starts a new line.  A token without one joins the previous token's line when it is on the
    same row and does not start left of it.  Tokens of one row further apart than :data:`_WIDE_GAP` heights are on
    different lines."""
    out: list[InjectedToken] = []
    line_no = word_no = -1
    value: str | None = None  # the line value of the current line
    prev: _Box | None = None
    for tok in tokens:
        box = _box(tok.poly)
        joins = False
        if prev is not None:
            row = _row_overlap(prev, box) >= _SAME_ROW
            joins = tok.line == value if tok.line is not None else row and box[0] >= prev[0]
            if joins and row and box[0] - prev[2] > _WIDE_GAP * max(prev[3] - prev[1], box[3] - box[1]):
                joins = False
                counts["lines_split"] += 1
        if joins:
            word_no += 1
        else:
            line_no, word_no, value = line_no + 1, 0, tok.line
            counts["lines"] += 1
        conf = tok.conf if tok.conf is not None else 1.0
        out.append(InjectedToken(line_no, word_no, tuple(tok.poly), conf, tok.text))
        prev = box
    return InjectedPage(index, tuple(out))


def _box(poly: Sequence[tuple[float, float]]) -> _Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _row_overlap(a: _Box, b: _Box) -> float:
    """Overlap of the y-extents relative to the smaller height (``BBox.vertical_overlap``, in points)."""
    lo, hi = max(a[1], b[1]), min(a[3], b[3])
    denom = min(a[3] - a[1], b[3] - b[1])
    return max(0.0, hi - lo) / denom if denom > 0 else 0.0


def _first_by_id[I: _Identified](items: Sequence[I], name: str, counts: Counter[str]) -> tuple[I, ...]:
    """The first item with each id, in file order."""
    seen: set[str] = set()
    out: list[I] = []
    for item in items:
        if item.id in seen:
            counts[f"{name}_duplicate"] += 1
            continue
        seen.add(item.id)
        out.append(item)
    counts[name] += len(items)
    return tuple(out)


def _merged(clusters: Sequence[Cluster], counts: Counter[str]) -> tuple[Cluster, ...]:
    """One cluster per id, in file order: the members of every entry with that id, each once."""
    members: dict[str, list[str]] = {}
    first: dict[str, Cluster] = {}
    for c in clusters:
        if c.id in first:
            counts["clusters_merged"] += 1
        first.setdefault(c.id, c)
        members.setdefault(c.id, []).extend(c.mentions)
    counts["clusters"] += len(clusters)
    return tuple(
        Cluster(id=cid, mentions=list(dict.fromkeys(members[cid])), attributes=c.attributes) for cid, c in first.items()
    )
