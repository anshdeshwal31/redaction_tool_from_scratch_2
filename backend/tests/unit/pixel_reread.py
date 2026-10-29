"""A verification re-read that takes word POSITIONS from the rendered output image, as production OCR does, and the
words' TEXT from what the build drew (fictional documents only).

The in-process harness of test_bench_runner re-reads the output's text layer, where a re-typeset line's words are
spread over the line's whole box; OCR reads them where the renderer drew them (left-aligned, at natural width), so a
word after a fake of another width sits elsewhere.  For each line of the first reading, the output page is rendered and
the line's band is cut into ink runs separated by blank columns (a word gap); runs that are solid black (a black box)
are dropped.  The words expected on the line are the kept tokens no operation covers plus the words each operation
draws there, in x order.  When the runs and the words agree in number, each word gets its run's box; otherwise the
line is left out of the reading (counted in ``fallbacks``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pypdfium2 as pdfium
from test_bench_runner import InProcessPool

from mlredact.core.geometry import BBox
from mlredact.core.model import Line, PageAnalysis, Token, token_id


class PixelPool(InProcessPool):
    """InProcessPool whose verification re-read takes word positions from the rendered page."""

    def __init__(self) -> None:
        super().__init__()
        self.first: dict[int, PageAnalysis] = {}
        self.matched = 0
        self.fallbacks = 0

    def _analyse(self, ref: Any, page: int, profile: str = "primary") -> Any:
        got = super()._analyse(ref, page, profile)
        if profile != "verify":
            self.first[page] = got
            return got
        build = self.builds[-1]
        assert build.pdf is not None
        doc = pdfium.PdfDocument(build.pdf)
        try:
            pg = doc[page]
            w_px = got.geometry.width_px
            img = pg.render(scale=w_px / pg.get_width()).to_pil().convert("L")
        finally:
            doc.close()
        gray = np.asarray(img)
        src: PageAnalysis = self.first[page]
        ops = build.ops.get(page, ())

        def cov(box: BBox) -> bool:
            area = 0
            for op in ops:
                inter = box.intersection(op.box)
                if inter is not None:
                    area += inter.area
            return box.area > 0 and area / box.area >= 0.25

        lines: list[Line] = []
        for ln in src.lines:
            y0, y1 = ln.bbox.y0, ln.bbox.y1
            words: list[tuple[int, str]] = [(t.bbox.x0, t.text) for t in ln.tokens if not cov(t.bbox)]
            for op in ops:
                if not op.text:
                    continue
                box = op.ink_box or op.box
                ov = min(box.y1, y1) - max(box.y0, y0)
                if ov <= 0.5 * min(box.height, y1 - y0):
                    continue
                for k, w in enumerate(op.text.split()):
                    words.append((box.x0 + k, w))
            words.sort()
            band = gray[max(0, y0 - 2) : min(gray.shape[0], y1 + 2), :]
            dark = band < 140
            cols = dark.any(axis=0)
            runs: list[list[int]] = []
            gap_thr = max(6, round(0.3 * (y1 - y0)))
            x = 0
            n = cols.shape[0]
            while x < n:
                if cols[x]:
                    s = x
                    while x < n and cols[x]:
                        x += 1
                    if runs and s - runs[-1][1] < gap_thr:
                        runs[-1][1] = x
                    else:
                        runs.append([s, x])
                else:
                    x += 1
            segs = []
            for s, e in runs:
                block = dark[:, s:e]
                if block.mean() > 0.8:  # a black box
                    continue
                ys = np.nonzero(block.any(axis=1))[0]
                segs.append(BBox(s, max(0, y0 - 2) + int(ys[0]), e, max(0, y0 - 2) + int(ys[-1]) + 1))
            if not words:
                continue
            if len(segs) == len(words):
                self.matched += 1
                items = [(b, w) for b, (_x, w) in zip(segs, words, strict=True)]
            else:
                self.fallbacks += 1
                continue
            ln_no = len(lines)
            toks = tuple(
                Token(token_id(page, ln_no, i), page, ln_no, i, b, 0.99, 0.99, "pixel", w)
                for i, (b, w) in enumerate(items)
            )
            bb = toks[0].bbox
            for t in toks[1:]:
                bb = bb.union(t.bbox)
            quad = (
                (float(bb.x0), float(bb.y0)),
                (float(bb.x1), float(bb.y0)),
                (float(bb.x1), float(bb.y1)),
                (float(bb.x0), float(bb.y1)),
            )
            lines.append(Line(f"p{page:04d}.l{ln_no:04d}", page, ln_no, bb, quad, 0.99, "pixel", toks))
        return PageAnalysis(got.geometry, got.kind, tuple(lines), got.embedded_char_count, got.regions)
