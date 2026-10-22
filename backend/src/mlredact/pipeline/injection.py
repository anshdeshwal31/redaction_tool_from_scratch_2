"""Stage inputs injected by the benchmark (``mlredact bench`` oracle modes): for the evaluator, never production.

The benchmark can give the tool perfect upstream input, from its ground truth or from another tool's output: the
text (``oracle_text``), the mentions to redact (``oracle_mentions``), their clusters (``oracle_entities``) and the
regions to apply (``oracle_regions``).  ``Redactor.run(inject=...)`` takes them through :class:`StageInputs`, whose
hooks replace stages of the pipeline; without it every stage is tool 2's own.  Only ``mlredact bench`` builds one.

Injected coordinates are display points (``display_pt``: the page as displayed, origin top-left, y down).  The
canonical raster is the displayed page, so a point converts to a pixel with the page's own dpi alone:
px = pt * dpi / 72.

Pure, no I/O.  At runtime this module imports only ``mlredact.core``: the pipeline names its types without loading
the benchmark adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from mlredact.core.geometry import BBox, Quad, union_all
from mlredact.core.model import Line, PageAnalysis, PageGeometry, RenderOp, Token, token_id
from mlredact.core.types import ActionKind

if TYPE_CHECKING:
    from mlredact.config.schema import AppConfig
    from mlredact.detect.engine import DetectionResult
    from mlredact.policy.engine import Decision
    from mlredact.surrogate.assign import MentionSurrogate
    from mlredact.verify.residual import ResidualHit

INJECTED_ENGINE = "inject"
# A pixel coordinate this close to a whole pixel is that pixel: display points written with 3 decimals (the
# benchmark's files, ``mlredact.bench.artifacts.display_poly``) then convert back to the box they came from.
_SNAP_PX = 0.01


@dataclass(frozen=True, slots=True)
class InjectedToken:
    """An injected token, numbered as tool 2 numbers its tokens.  ``points``: its polygon in display points (at least
    3 corners, in the benchmark's order: the top edge first in reading direction); ``confidence``: the injected one,
    or 1.0."""

    line_no: int
    word_no: int
    points: tuple[tuple[float, float], ...]
    confidence: float
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class InjectedPage:
    """One page's injected tokens in reading order, line after line (each line's tokens one after another)."""

    index: int
    tokens: tuple[InjectedToken, ...] = field(default=(), repr=False)


def _px(value: float, dpi: int) -> float:
    px = value * dpi / 72.0
    nearest = round(px)
    return float(nearest) if abs(px - nearest) <= _SNAP_PX else px


def _within(lo: int, hi: int, size: int) -> tuple[int, int]:
    """A half-open pixel extent clipped to ``[0, size)``, at least 1 px."""
    lo = min(max(lo, 0), size - 1)
    return lo, min(max(hi, lo + 1), size)


def pt_box(points: Sequence[tuple[float, float]], geometry: PageGeometry) -> BBox:
    """Display points -> the canonical-raster pixel box around them, rounded outward (a coordinate within 0.01 px of
    a whole pixel is that pixel), clipped to the page and at least 1 px wide and high."""
    box = BBox.from_points((_px(x, geometry.dpi), _px(y, geometry.dpi)) for x, y in points)
    x0, x1 = _within(box.x0, box.x1, geometry.width_px)
    y0, y1 = _within(box.y0, box.y1, geometry.height_px)
    return BBox(x0, y0, x1, y1)


def _quad(first: InjectedToken, last: InjectedToken, box: BBox, dpi: int) -> Quad:
    """A line's corners in pixels: the first token's left edge and the last token's right edge when both polygons
    are quadrilaterals (so a sideways line stays sideways for planning), otherwise the line's box."""
    if len(first.points) == 4 and len(last.points) == 4:
        (x0, y0), (x1, y1) = first.points[0], last.points[1]
        (x2, y2), (x3, y3) = last.points[2], first.points[3]
        s = dpi / 72.0
        return ((x0 * s, y0 * s), (x1 * s, y1 * s), (x2 * s, y2 * s), (x3 * s, y3 * s))
    return (
        (float(box.x0), float(box.y0)),
        (float(box.x1), float(box.y0)),
        (float(box.x1), float(box.y1)),
        (float(box.x0), float(box.y1)),
    )


def injected_lines(page: InjectedPage, geometry: PageGeometry) -> tuple[Line, ...]:
    """The page's injected tokens as tool 2's lines, built as OCR fusion builds new lines: one line per run of
    tokens with the same line number, pixel boxes from :func:`pt_box`, the injected confidence as both the mean and
    the weakest character confidence, no alternate readings."""
    if page.index != geometry.index:
        raise ValueError("the injected page is not the rendered page")
    runs: list[list[tuple[InjectedToken, BBox]]] = []
    for tok in page.tokens:
        box = pt_box(tok.points, geometry)
        if runs and runs[-1][0][0].line_no == tok.line_no:
            runs[-1].append((tok, box))
        else:
            runs.append([(tok, box)])
    words = {(t.line_no, t.word_no) for t in page.tokens}
    if len({run[0][0].line_no for run in runs}) != len(runs) or len(words) != len(page.tokens):
        raise ValueError("injected line and word numbers must be unique on a page")
    lines = []
    for run in runs:
        line_no = run[0][0].line_no
        tokens = tuple(
            Token(
                token_id(page.index, line_no, t.word_no),
                page.index,
                line_no,
                t.word_no,
                box,
                t.confidence,
                t.confidence,
                INJECTED_ENGINE,
                t.text,
            )
            for t, box in run
        )
        bbox = union_all([t.bbox for t in tokens])
        conf = round(sum(t.confidence for t in tokens) / len(tokens), 5)
        quad = _quad(run[0][0], run[-1][0], bbox, geometry.dpi)
        lines.append(
            Line(f"p{page.index:04d}.l{line_no:04d}", page.index, line_no, bbox, quad, conf, INJECTED_ENGINE, tokens)
        )
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class InjectedStage:
    """Detection and policy from injected mentions: the pages with the injected regions attached, the detection
    result over them and one decision per internal mention."""

    pages: tuple[PageAnalysis, ...] = field(repr=False)
    detection: DetectionResult = field(repr=False)
    decisions: tuple[Decision, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ReportedMention:
    """An injected mention as the stage files report it.  ``label``: the label reported for it (it maps back to the
    injected type); ``token_ids``: its own tool 2 tokens; ``boxes``: (page, box) of those tokens and of the regions it
    became; ``slots``: for each of its own tokens, the internal mention that redacts the token and the token's
    position in it; ``owner``: its tokens took its type (where injected mentions overlap, one of them owns the
    tokens); ``conf``: the injected confidence."""

    mention_id: str
    label: str
    text: str = field(repr=False)
    token_ids: tuple[str, ...]
    boxes: tuple[tuple[int, BBox], ...]
    slots: tuple[tuple[str, int], ...]
    owner: bool
    conf: float | None


@dataclass(frozen=True, slots=True)
class PlanEcho:
    """An injected plan region and how it was applied.  ``poly``: as injected; ``mentions``: the injected mentions it
    names that ``mentions.json`` lists; ``ops``: the (ref_id, box) of each operation that applied it (one per line
    of a region over several lines), which keys the box it erased."""

    region_id: str
    page: int
    poly: tuple[tuple[float, float], ...]
    mentions: tuple[str, ...]
    ops: tuple[tuple[str, BBox], ...]


@dataclass(frozen=True, slots=True)
class InjectReport:
    """What the stage files need to know about an injection (``JobTrace.injected``).

    ``mode``: the benchmark mode; ``text_source``: the injected text's ``source``; ``mentions``: the injected
    mentions (None while the mentions are tool 2's own); ``aliases``: per internal mention id, the injected mentions
    it stands for, owner first; ``clusters``: the injected clusters as (id, member ids), None while they are tool 2's
    own; ``plan``: the injected plan regions that were applied, by page and id (``oracle_regions`` only); ``notes``:
    count-only notes for ``run_info.json``, never document text."""

    mode: str
    text_source: str
    mentions: tuple[ReportedMention, ...] | None = field(default=None, repr=False)
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    clusters: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    plan: tuple[PlanEcho, ...] | None = None
    notes: tuple[str, ...] = ()


class StageInputs(Protocol):
    """Injected stage inputs for one job (``Redactor.run(inject=...)``).  Each hook replaces one stage, or returns
    None where tool 2's own stage runs.

    * ``detect_regions``: page analysis looks for non-text regions (signatures, stamps, barcodes, ...) around the
      injected text;
    * ``pages(page_count)``: the injected text of every page of the document, in page order (it replaces OCR);
    * ``stage(pages, config, doc_key)``: detection and policy from the injected mentions;
    * ``plan(...)``: the render operations of the injected plan; ``region_actions`` is the action of every region by
      region id, ``rotated`` the tokens on sideways lines;
    * ``replan_hits(hits)``: the residual hits verification may re-plan (all of them still count for the verdict);
    * ``report()``: what the stage files need to know (the job trace's ``injected``)."""

    @property
    def detect_regions(self) -> bool: ...

    def pages(self, page_count: int) -> list[InjectedPage]: ...

    def stage(self, pages: Sequence[PageAnalysis], config: AppConfig, doc_key: str) -> InjectedStage | None: ...

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
    ) -> dict[int, tuple[RenderOp, ...]] | None: ...

    def replan_hits(self, hits: Sequence[ResidualHit]) -> list[ResidualHit]: ...

    def report(self) -> InjectReport: ...
