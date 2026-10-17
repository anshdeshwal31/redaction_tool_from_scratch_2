"""Document model shared by all stages.

Text-bearing fields are declared with ``repr=False`` so that accidentally printing, logging or
raising with one of these objects never reveals document content.  Identifiers are derived from
*positions* (page/line/word indices, input hash), never from values.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from mlredact.core.geometry import BBox, Quad
from mlredact.core.types import (
    ActionKind,
    EntityType,
    EvidenceStrength,
    PageKind,
    RegionKind,
    ViewKind,
)


@dataclass(frozen=True, slots=True)
class PageGeometry:
    index: int
    width_pt: float  # displayed size (CropBox, after /Rotate)
    height_pt: float
    dpi: int
    width_px: int
    height_px: int

    @property
    def bounds(self) -> BBox:
        return BBox(0, 0, self.width_px, self.height_px)

    @property
    def px_per_pt(self) -> float:
        return self.dpi / 72.0

    def box_to_pt(self, box: BBox) -> tuple[float, float, float, float]:
        """Pixel box -> PDF user-space points (origin bottom-left), rounded to 1/1000 pt."""
        s = 72.0 / self.dpi
        x0, x1 = box.x0 * s, box.x1 * s
        y0, y1 = self.height_pt - box.y1 * s, self.height_pt - box.y0 * s
        return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))


def token_id(page: int, line_no: int, word_no: int) -> str:
    return f"p{page:04d}.l{line_no:04d}.w{word_no:03d}"


@dataclass(frozen=True, slots=True)
class Token:
    token_id: str
    page: int
    line_no: int
    word_no: int
    bbox: BBox
    confidence: float  # mean character confidence
    min_confidence: float  # weakest character confidence
    engine: str
    text: str = field(repr=False)
    # Alternative readings of the whole token (from per-character top-k), most likely first.
    alternates: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class Line:
    line_id: str
    page: int
    line_no: int
    bbox: BBox
    quad: Quad
    confidence: float
    engine: str
    tokens: tuple[Token, ...]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    def __repr__(self) -> str:  # text property must not leak through the default repr
        return f"Line(line_id={self.line_id!r}, tokens={len(self.tokens)})"


@dataclass(frozen=True, slots=True)
class PageAnalysis:
    geometry: PageGeometry
    kind: PageKind
    lines: tuple[Line, ...]
    embedded_char_count: int = 0
    regions: tuple[Region, ...] = ()
    # Fraction of the page's ink (ruling lines excluded) that no readable token explains and that
    # ended up in handwriting / graphic regions: the legibility signal (plan §6, §24).
    unexplained_ink: float = 0.0

    def tokens(self) -> Iterator[Token]:
        for line in self.lines:
            yield from line.tokens


@dataclass(frozen=True, slots=True, order=True)
class Evidence:
    detector_id: str
    detector_version: str
    rule_id: str
    strength: EvidenceStrength
    score: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """A detector hit expressed in the coordinates of one text view."""

    entity_type: EntityType
    view: ViewKind
    start: int
    end: int
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class Mention:
    """A resolved PII occurrence expressed as a set of canonical tokens."""

    mention_id: str
    entity_type: EntityType
    token_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    value: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Region:
    """A non-text (or unreadable) region removed as a whole."""

    region_id: str
    page: int
    kind: RegionKind
    bbox: BBox
    score: float
    detector: str = "ink"  # "zxing" | "yunet" | "ink" | "legibility"
    rule: str = ""  # which classification rule fired (manifest evidence)


@dataclass(frozen=True, slots=True)
class RenderOp:
    """One drawing operation on a page raster.  ``sort_key`` defines the canonical draw order."""

    page: int
    kind: ActionKind
    box: BBox  # region that is erased / filled (padded)
    reason: str  # EntityType / RegionKind value
    ref_id: str  # mention_id or region_id
    text: str | None = field(default=None, repr=False)  # surrogate / label text, if any
    ink_box: BBox | None = None  # unpadded token union: where the original glyphs were
    has_descender: bool = False  # original segment had descenders (font-size calibration only)
    # Original segment text: used in the worker ONLY to calibrate font size/baseline of the surrogate
    # (in-memory IPC; never logged, never written).
    source_text: str | None = field(default=None, repr=False)
    # RETYPESET_REGION ops re-typeset a whole line: the mentions whose text they carry.
    covers: tuple[str, ...] = ()

    @property
    def sort_key(self) -> tuple[int, int, int, int, int, str, str]:
        return (self.page, self.box.y0, self.box.x0, self.box.y1, self.box.x1, self.kind.value, self.ref_id)
