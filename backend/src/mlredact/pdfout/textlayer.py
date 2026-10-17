"""Invisible text layer (plan §11) built ONLY from post-redaction content.

Kept OCR tokens and surrogate segments are written with text render mode 3 (invisible) over their
page positions, so the output is searchable/extractable and consistent with what is visible.
Nothing removed can enter this layer: it is generated from the plan, never copied from the input.
Base-14 Helvetica with WinAnsiEncoding is used (no embedded font program); characters outside
WinAnsi are replaced with '?'.
"""

from __future__ import annotations

from collections.abc import Sequence

from mlredact.core.geometry import BBox
from mlredact.core.model import PageGeometry, RenderOp, Token
from mlredact.core.types import ActionKind

FONT_NAME = "F1"
_COVERED_FRACTION = 0.25


def _escape(raw: bytes) -> str:
    out = []
    for b in raw:
        if b in (0x28, 0x29, 0x5C):  # ( ) \
            out.append("\\" + chr(b))
        elif 32 <= b <= 126:
            out.append(chr(b))
        else:
            out.append(f"\\{b:03o}")
    return "".join(out)


def _covered(box: BBox, ops: Sequence[RenderOp]) -> bool:
    if box.area == 0:
        return True
    covered = 0
    for op in ops:
        inter = box.intersection(op.box)
        if inter is not None:
            covered += inter.area
    return covered / box.area >= _COVERED_FRACTION


def text_layer_items(tokens: Sequence[Token], ops: Sequence[RenderOp]) -> list[tuple[BBox, str]]:
    """Kept tokens (not under any op) + surrogate / label segments, in reading order."""
    items: list[tuple[BBox, str]] = [(t.bbox, t.text) for t in tokens if not _covered(t.bbox, ops)]
    for op in ops:
        if op.kind in (ActionKind.SURROGATE, ActionKind.LABEL, ActionKind.RETYPESET_REGION) and op.text:
            items.append((op.ink_box or op.box, op.text))
    items.sort(key=lambda it: (it[0].y0, it[0].x0, it[0].y1, it[0].x1, it[1]))
    return items


def page_text_layer(geometry: PageGeometry, items: Sequence[tuple[BBox, str]]) -> bytes | None:
    if not items:
        return None
    lines = ["BT", "3 Tr"]
    for box, text in items:
        text = text.strip()
        if not text:
            continue
        x0, y0, x1, y1 = geometry.box_to_pt(box)
        height = max(0.5, y1 - y0)
        size = max(1.0, round(height * 0.72, 2))
        baseline = round(y0 + height * 0.22, 2)
        est_width = 0.5 * size * len(text)
        tz = max(10.0, min(1000.0, round(100.0 * (x1 - x0) / est_width, 2))) if est_width else 100.0
        encoded = text.encode("cp1252", errors="replace")
        lines.append(
            f"/{FONT_NAME} {size:.2f} Tf {tz:.2f} Tz 1 0 0 1 {x0:.2f} {baseline:.2f} Tm ({_escape(encoded)}) Tj"
        )
    lines.append("ET")
    return ("\n".join(lines) + "\n").encode("ascii")
