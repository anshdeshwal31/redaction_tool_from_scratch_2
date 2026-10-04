"""Ink accounting and region classification (plan §6 "ink accounting", §7.2 S4).

Every mark on the page must be explained: by an OCR token, a ruling line, or a classified region.
What OCR did not read is where PII escapes a text-only pipeline (signatures, handwriting, stamps,
barcodes, photos), so the classification is recall-first:

==================================  ===========================================  =====================
residual ink                        rule                                         region
==================================  ===========================================  =====================
symbols located by zxing            any barcode / QR / DataMatrix / PDF417       ``barcode``
faces found by YuNet                any face                                     ``face``
coloured ink not explained by       stamps, coloured pen                         ``stamp``
confident tokens
large mid-tone areas                photographs, scanned cards                   ``photo``
below a sign-off / near a           "Yours sincerely", "Signed", "Signature"     ``signature``
signature label
text-like, re-read fails            handwriting, text OCR missed                 ``handwriting``
graphics in header/footer bands     letterhead logos, fax headers                ``logo``
anything in the side margins        margin annotations, punch holes              ``handwriting``
faint marks                         show-through from the reverse, pencil        ``faint_ink``
large framed graphics in the body    figures, diagrams, charts                    ``graphic``
other unexplained marks             handwriting in form cells, scribbles         ``handwriting``
==================================  ===========================================  =====================

Figures cannot be told apart from handwriting in a form cell with certainty, so ``graphic`` is a
label for the manifest, not a licence to keep: every profile removes it by default.

Text-like residual ink is first re-read (upscaled crop through the same OCR engine).  It becomes
ordinary tokens - and goes through detection like any other text - only if every re-read line is
confident and the tokens explain most of its ink; otherwise the region is removed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from mlredact.config.schema import RegionsConfig
from mlredact.core.canonical import content_id
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, Region, Token, token_id
from mlredact.core.types import RegionKind
from mlredact.imaging.ink import U8, Blob, Mask, blobs, box_mask, drop_specks, ink_maps, ruling_lines, scaled

Reread = Callable[[U8], Sequence[Line]]

_SIGN_OFF = re.compile(
    r"(?i)\b(yours[ \t]+(?:sincerely|faithfully|truly)|(?:kind|best|warm)(?:est)?[ \t]+regards|regards|sincerely|"
    r"with[ \t]+thanks)\b"
)
_SIGN_LABEL = re.compile(r"(?i)\b(signed|signature|sign[ \t]+here|signatory|witnessed[ \t]+by|authori[sz]ed[ \t]+by)\b")
_REREAD_SCALE = 2
_DEFAULT_TOKEN_HEIGHT_AT_300 = 36  # 11 pt text at 300 dpi


@dataclass(frozen=True, slots=True)
class PageRegions:
    regions: tuple[Region, ...]
    extra_lines: tuple[Line, ...]  # confidently re-read text the primary OCR pass missed
    # OCR "words" read from show-through (faint ink only, inside a faint-ink region): ghosts of the
    # reverse side.  The pixels are suppressed; the tokens must not reach detection or rendering.
    ghost_tokens: frozenset[str] = frozenset()
    unexplained_ink: float = 0.0
    illegible: bool = False


def _median_line_height(lines: Sequence[Line], dpi: int) -> int:
    heights = sorted(t.bbox.height for line in lines for t in line.tokens)
    return heights[len(heights) // 2] if heights else scaled(_DEFAULT_TOKEN_HEIGHT_AT_300, dpi)


def _zones(lines: Sequence[Line], width: int, lh: int) -> list[BBox]:
    """Where signatures go: below sign-offs, and around signature labels."""
    zones: list[BBox] = []
    for line in lines:
        text = " ".join(t.text for t in line.tokens)
        b = line.bbox
        if _SIGN_OFF.search(text):
            zones.append(BBox(0, b.y0, width, b.y1 + 6 * lh))
        if _SIGN_LABEL.search(text):
            zones.append(BBox(max(0, b.x0 - lh), b.y0 - 2 * lh, width, b.y1 + 3 * lh))
    return zones


def _overlap_frac(a: BBox, others: Sequence[BBox]) -> float:
    if a.area == 0:
        return 0.0
    return min(1.0, sum(i.area for o in others if (i := a.intersection(o)) is not None) / a.area)


def _region(page: int, kind: RegionKind, box: BBox, rule: str, score: float = 1.0) -> Region:
    return Region(
        region_id=content_id("G", page, kind.value, *box.as_tuple()),
        page=page,
        kind=kind,
        bbox=box,
        score=score,
        detector="ink",
        rule=rule,
    )


def _map_lines(lines: Sequence[Line], origin: BBox, page: int, first_line_no: int) -> list[Line]:
    """Re-read lines (in upscaled-crop pixels) -> page pixels, renumbered after the existing lines."""

    def box(b: BBox) -> BBox:
        return BBox(
            origin.x0 + b.x0 // _REREAD_SCALE,
            origin.y0 + b.y0 // _REREAD_SCALE,
            origin.x0 + -(-b.x1 // _REREAD_SCALE),
            origin.y0 + -(-b.y1 // _REREAD_SCALE),
        )

    out: list[Line] = []
    for k, line in enumerate(lines):
        line_no = first_line_no + k
        tokens = tuple(
            replace(
                t,
                token_id=token_id(page, line_no, w),
                page=page,
                line_no=line_no,
                word_no=w,
                bbox=box(t.bbox),
                engine=f"{t.engine}+reread",
            )
            for w, t in enumerate(line.tokens)
        )
        quad = tuple((origin.x0 + x / _REREAD_SCALE, origin.y0 + y / _REREAD_SCALE) for x, y in line.quad)
        out.append(
            replace(
                line,
                line_id=f"p{page:04d}.l{line_no:04d}",
                page=page,
                line_no=line_no,
                bbox=box(line.bbox),
                quad=quad,  # type: ignore[arg-type]
                engine=f"{line.engine}+reread",
                tokens=tokens,
            )
        )
    return out


class _Analysis:
    def __init__(self, rgb: U8, lines: Sequence[Line], page: int, dpi: int, cfg: RegionsConfig) -> None:
        self.rgb, self.lines, self.page, self.dpi, self.cfg = rgb, lines, page, dpi, cfg
        self.h, self.w = int(rgb.shape[0]), int(rgb.shape[1])
        self.lh = max(8, _median_line_height(lines, dpi))
        self.maps = ink_maps(rgb, dpi, cfg.ink_darkness, cfg.faint_darkness)
        tokens = [t for line in lines for t in line.tokens]
        self.tokens: list[Token] = tokens
        # Only tokens that look like real text explain ink.  OCR happily "reads" a signature or a
        # scribble as a tall, low-confidence word ("soooor"); such tokens must not hide the ink.
        readable = [t for t in tokens if self._plausible(t) and t.min_confidence >= cfg.explain_min_confidence]
        pad = max(2, round(0.15 * self.lh))
        self.explained: Mask = box_mask((self.h, self.w), [t.bbox.pad(pad, pad) for t in readable])
        # Faint marks are explained only by tokens printed in real ink, or read near-perfectly:
        # show-through from the reverse gets "read" as confident garbage ("888" mirrors to "888").
        self.explained_faint: Mask = box_mask(
            (self.h, self.w),
            [
                t.bbox.pad(pad, pad)
                for t in readable
                if t.min_confidence >= 0.95
                or self.maps.ink[t.bbox.y0 : t.bbox.y1, t.bbox.x0 : t.bbox.x1].mean() >= 0.02
            ],
        )
        self.rules: Mask = ruling_lines(self.maps.ink, dpi)
        xs0 = sorted(t.bbox.x0 for t in readable)
        xs1 = sorted(t.bbox.x1 for t in readable)
        # The text column (robust to a few stray tokens); margins are measured against it.
        self.column = (
            (xs0[len(xs0) // 20], xs1[len(xs1) - 1 - len(xs1) // 20])
            if len(readable) >= 5
            else (round(cfg.side_margin * self.w), round((1 - cfg.side_margin) * self.w))
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        self.coloured: Mask = (hsv[:, :, 1] >= cfg.colour_saturation) & self.maps.ink
        self.min_area = scaled(cfg.min_component_px, dpi) * max(1, dpi // 300)
        self.zones = _zones(lines, self.w, self.lh)

    # ------------------------------------------------------------------------------- helpers
    def _plausible(self, t: Token) -> bool:
        """Shaped like text: not taller than a line, not wider than its characters can be."""
        chars = max(1, len(t.text.strip()))
        return t.bbox.height <= 1.8 * self.lh and t.bbox.width <= max(1.5 * self.lh, 1.5 * self.lh * chars)

    def in_band(self, b: BBox) -> bool:
        band = self.cfg.header_footer_band * self.h
        cy = (b.y0 + b.y1) / 2
        return cy < band or cy > self.h - band

    def in_margin(self, b: BBox) -> bool:
        """Centre outside the text column (or inside the fixed side strips)."""
        cx = (b.x0 + b.x1) / 2
        left, right = self.column
        m = self.cfg.side_margin * self.w
        return (
            cx < min(left, self.w - m) - 0.5 * self.lh
            or cx > max(right, m) + 0.5 * self.lh
            or b.x1 <= m
            or b.x0 >= self.w - m
        )

    def figure_like(self, b: BBox) -> bool:
        """A large mark with straight-line structure (axes, frames) around it: probably a figure."""
        if b.width < 3 * self.lh or b.height < 3 * self.lh:
            return False
        e = b.pad(self.lh, self.lh, BBox(0, 0, self.w, self.h))
        rule_px = int(self.rules[e.y0 : e.y1, e.x0 : e.x1].sum())
        return rule_px >= 0.5 * (b.width + b.height) * max(1, self.lh // 15)

    def midtone_fraction(self, b: BBox) -> float:
        crop = self.maps.gray[b.y0 : b.y1, b.x0 : b.x1]
        dark = self.maps.background[b.y0 : b.y1, b.x0 : b.x1].astype(np.int16) - crop.astype(np.int16)
        return float(((dark >= 25) & (crop >= 40)).mean()) if crop.size else 0.0

    def text_like(self, b: Blob) -> bool:
        return bool(b.box.height <= 3.5 * self.lh and b.box.width >= 0.8 * self.lh)

    def significant(self, b: Blob) -> bool:
        return bool(max(b.box.width, b.box.height) >= 0.5 * self.lh and b.pixels >= 4 * self.min_area)

    # --------------------------------------------------------------------------------- passes
    def colour_regions(self, skip: Sequence[BBox]) -> list[Region]:
        """Coloured ink (stamps, pen) unless it is confidently read printed text."""
        mask = drop_specks(self.coloured, self.min_area * 2, scaled(12, self.dpi))
        out: list[Region] = []
        for b in blobs(mask, scaled(30, self.dpi), scaled(30, self.dpi)):
            if not self.significant(b) or _overlap_frac(b.box, skip) >= 0.5:
                continue
            crop = mask[b.box.y0 : b.box.y1, b.box.x0 : b.box.x1]
            explained = self.explained[b.box.y0 : b.box.y1, b.box.x0 : b.box.x1]
            covered = float((crop & explained).sum()) / max(1, int(crop.sum()))
            confident = all(t.min_confidence >= 0.9 for t in self.tokens if t.bbox.intersection(b.box) is not None)
            if covered >= 0.6 and confident and not self.in_band(b.box) and _overlap_frac(b.box, self.zones) == 0:
                continue  # coloured printed text, read with confidence: ordinary tokens
            box = b.box.pad(scaled(10, self.dpi), scaled(10, self.dpi), BBox(0, 0, self.w, self.h))
            out.append(_region(self.page, RegionKind.STAMP, box, "ink.colour"))
        return out

    def residual_regions(self, skip: Sequence[BBox], reread: Reread | None) -> tuple[list[Region], list[Line]]:
        maps, cfg = self.maps, self.cfg
        residual = maps.ink & ~self.explained & ~self.rules & ~box_mask((self.h, self.w), skip)
        residual = drop_specks(residual, self.min_area, scaled(12, self.dpi))
        regions: list[Region] = []
        extra: list[Line] = []
        next_line = (max((line.line_no for line in self.lines), default=-1)) + 1
        bounds = BBox(0, 0, self.w, self.h)
        pad = scaled(8, self.dpi)
        for b in blobs(residual, round(0.6 * self.lh), max(1, round(0.25 * self.lh))):
            if not self.significant(b):
                continue
            box = b.box.pad(pad, pad, bounds)
            if b.box.area >= scaled(150, self.dpi) ** 2 and self.midtone_fraction(b.box) >= 0.3:
                regions.append(_region(self.page, RegionKind.PHOTO, box, "ink.midtone_area"))
            elif _overlap_frac(b.box, self.zones) >= 0.3:
                regions.append(_region(self.page, RegionKind.SIGNATURE, box, "ink.signature_zone"))
            elif self.text_like(b):
                read = self._reread(b, reread, next_line) if cfg.reread and reread is not None else None
                if read is not None:
                    extra.extend(read)
                    next_line += len(read)
                elif self.in_band(b.box):
                    regions.append(_region(self.page, RegionKind.LOGO, box, "ink.band_unreadable"))
                else:
                    regions.append(_region(self.page, RegionKind.HANDWRITING, box, "ink.unreadable_text"))
            elif self.in_band(b.box):
                regions.append(_region(self.page, RegionKind.LOGO, box, "ink.band_graphic"))
            elif self.in_margin(b.box):
                regions.append(_region(self.page, RegionKind.HANDWRITING, box, "ink.margin_mark"))
            elif self.figure_like(b.box):
                regions.append(_region(self.page, RegionKind.GRAPHIC, box, "ink.body_figure"))
            else:  # recall first: unexplained marks without figure structure are treated as handwriting
                regions.append(_region(self.page, RegionKind.HANDWRITING, box, "ink.unexplained_marks"))
        return regions, extra

    def _reread(self, b: Blob, reread: Reread, first_line_no: int) -> list[Line] | None:
        margin = max(4, round(0.3 * self.lh))
        origin = b.box.pad(margin, margin, BBox(0, 0, self.w, self.h))
        crop = self.rgb[origin.y0 : origin.y1, origin.x0 : origin.x1]
        big = cv2.resize(
            crop,
            (crop.shape[1] * _REREAD_SCALE, crop.shape[0] * _REREAD_SCALE),
            interpolation=cv2.INTER_CUBIC,
        )
        lines = [line for line in reread(np.ascontiguousarray(big)) if line.tokens]
        if not lines or any(line.confidence < self.cfg.reread_min_confidence for line in lines):
            return None
        mapped = _map_lines(lines, origin, self.page, first_line_no)
        token_boxes = box_mask((self.h, self.w), [t.bbox for line in mapped for t in line.tokens])
        ink = self.maps.ink[b.box.y0 : b.box.y1, b.box.x0 : b.box.x1]
        covered = float((ink & token_boxes[b.box.y0 : b.box.y1, b.box.x0 : b.box.x1]).sum()) / max(1, int(ink.sum()))
        return mapped if covered >= 0.8 else None

    def faint_regions(self, skip: Sequence[BBox]) -> list[Region]:
        mask = self.maps.faint & ~self.explained_faint & ~box_mask((self.h, self.w), skip)
        mask = drop_specks(mask, self.min_area * 4, scaled(24, self.dpi))
        bounds = BBox(0, 0, self.w, self.h)
        out: list[Region] = []
        for b in blobs(mask, round(0.6 * self.lh), max(1, round(0.25 * self.lh))):
            if max(b.box.width, b.box.height) < 0.5 * self.lh or b.pixels < 8 * self.min_area:
                continue
            box = b.box.pad(scaled(4, self.dpi), scaled(4, self.dpi), bounds)
            out.append(_region(self.page, RegionKind.FAINT_INK, box, "ink.faint"))
        return out

    def unexplained_fraction(self, unreadable: Sequence[BBox]) -> tuple[float, int]:
        """Share of the page's ink (ruling lines excluded) inside unreadable regions; total ink."""
        ink = self.maps.ink & ~self.rules
        total = int(ink.sum())
        if total == 0:
            return 0.0, 0
        inside = int((ink & box_mask((self.h, self.w), unreadable)).sum())
        return inside / total, total

    def content_box(self) -> BBox:
        ys, xs = np.nonzero(self.maps.ink)
        pad = scaled(10, self.dpi)
        return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1).pad(
            pad, pad, BBox(0, 0, self.w, self.h)
        )

    def grow_to_strokes(self, region: Region) -> Region:
        """Extend a pen-mark region over every stroke connected to it.

        Pieces of a signature that cross typed text lie inside readable tokens' boxes, so they are
        not residual ink and the region's box can stop short of them.  Every ink component touching
        the region is taken in (typed letters it crosses included); components far larger than the
        region (ruling lines, frames) are not.
        """
        if not hasattr(self, "_components"):
            _n, labels, stats, _c = cv2.connectedComponentsWithStats(self.maps.ink.astype(np.uint8), connectivity=8)
            self._components = (labels, stats)
        labels, stats = self._components
        b = region.bbox
        grown = b
        for label in np.unique(labels[b.y0 : b.y1, b.x0 : b.x1]).tolist():
            if label == 0:
                continue
            x, y, w, h = (int(v) for v in stats[label, :4])
            comp = BBox(x, y, x + w, y + h)
            if comp.area > 4 * max(1, b.area) or w > 3 * b.width + self.lh or h > 3 * b.height + self.lh:
                continue
            grown = grown.union(comp)
        if grown == b:
            return region
        pad = scaled(4, self.dpi)
        return _region(self.page, region.kind, grown.pad(pad, pad, BBox(0, 0, self.w, self.h)), region.rule)

    def ghosts(self, faint: Sequence[BBox]) -> frozenset[str]:
        """Tokens lying on faint-only ink inside faint-ink regions (never tokens on strong ink)."""
        out: set[str] = set()
        for t in self.tokens:
            b = t.bbox
            if t.min_confidence >= 0.95 or _overlap_frac(b, faint) < 0.6:
                continue
            if self.maps.ink[b.y0 : b.y1, b.x0 : b.x1].mean() < 0.02:
                out.add(t.token_id)
        return frozenset(out)


# Pen marks: their strokes may cross typed text, so their regions grow along connected ink.
_PEN_MARKS = frozenset({RegionKind.SIGNATURE, RegionKind.HANDWRITING})

# Regions whose ink is unreadable *text-like* content (not known objects such as signatures or codes).
_UNREADABLE = frozenset({RegionKind.HANDWRITING, RegionKind.GRAPHIC})


# Fragments of one mark (a signature cut by typed text, show-through split into words) are merged,
# so no stroke survives between two removal boxes.
_MERGE_GAP = {RegionKind.SIGNATURE: 2.0, RegionKind.HANDWRITING: 1.0, RegionKind.FAINT_INK: 2.0, RegionKind.STAMP: 0.5}


def _merge(regions: list[Region], lh: int) -> list[Region]:
    out = [r for r in regions if r.kind not in _MERGE_GAP]
    for kind, factor in sorted(_MERGE_GAP.items(), key=lambda kv: kv[0].value):
        gap = round(factor * lh)
        boxes = sorted((r.bbox for r in regions if r.kind is kind), key=lambda b: b.as_tuple())
        merged: list[BBox] = []
        for b in boxes:
            for i, m in enumerate(merged):
                if m.pad(gap, gap).intersection(b) is not None:
                    merged[i] = m.union(b)
                    break
            else:
                merged.append(b)
        changed = True
        while changed:  # unions can bring earlier boxes into reach
            changed = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if merged[i].pad(gap, gap).intersection(merged[j]) is not None:
                        merged[i] = merged[i].union(merged.pop(j))
                        changed = True
                        break
                if changed:
                    break
        rules = sorted({r.rule for r in regions if r.kind is kind})
        page = regions[0].page if regions else 0
        out.extend(_region(page, kind, b, "+".join(rules)) for b in sorted(merged, key=lambda b: b.as_tuple()))
    return out


def analyse_regions(
    rgb: U8,
    lines: Sequence[Line],
    page: int,
    dpi: int,
    cfg: RegionsConfig,
    *,
    detected: Sequence[Region] = (),
    reread: Reread | None = None,
) -> PageRegions:
    """Classify everything on the page that OCR did not explain.  ``detected``: barcode/face regions."""
    regions = list(detected)
    extra: list[Line] = []
    ghosts: frozenset[str] = frozenset()
    unexplained, illegible = 0.0, False
    if cfg.ink_accounting:
        a = _Analysis(rgb, lines, page, dpi, cfg)
        regions.extend(a.colour_regions([r.bbox for r in regions]))
        residual, extra = a.residual_regions([r.bbox for r in regions], reread)
        regions.extend(residual)
        if cfg.suppress_faint_ink:
            regions.extend(a.faint_regions([r.bbox for r in regions]))
    if cfg.ink_accounting:
        regions = _merge(regions, a.lh)
        regions = [a.grow_to_strokes(r) if r.kind in _PEN_MARKS else r for r in regions]
        ghosts = a.ghosts([r.bbox for r in regions if r.kind is RegionKind.FAINT_INK])
        unreadable = [r.bbox for r in regions if r.kind in _UNREADABLE]
        unexplained, total_ink = a.unexplained_fraction(unreadable)
        # Legibility gate: a page that is mostly unreadable is not trusted at all.
        if total_ink >= 50 * a.min_area and unexplained >= cfg.illegible_fraction:
            illegible = True
            regions.append(
                _region(page, RegionKind.ILLEGIBLE, a.content_box(), "legibility.page", round(unexplained, 4))
            )
    unique = {r.region_id: r for r in regions}
    ordered = sorted(unique.values(), key=lambda r: (r.bbox.y0, r.bbox.x0, r.bbox.y1, r.bbox.x1, r.kind.value))
    return PageRegions(tuple(ordered), tuple(extra), ghosts, round(unexplained, 4), illegible)
