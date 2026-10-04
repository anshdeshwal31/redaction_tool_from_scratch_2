"""Raster editing.  Ops are applied in canonical order; each is verified right after it is drawn.

Later ops can only overwrite earlier ones with *new* content (fill or paper + glyphs), never bring
original pixels back, so a per-op check at draw time is sufficient for V3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mlredact.config.schema import RedactionRenderConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import RenderOp
from mlredact.core.types import ActionKind
from mlredact.render.label import draw_label, place_notice
from mlredact.render.typeset import TypesetResult, erased_cleanly, typeset


@dataclass(frozen=True, slots=True)
class Applied:
    ref_id: str
    planned: BBox
    erased: BBox


@dataclass(frozen=True, slots=True)
class PageEdit:
    raster: npt.NDArray[np.uint8]
    verified: bool
    applied: tuple[Applied, ...]
    notice_box: BBox | None


def fill_colour(op: RenderOp, cfg: RedactionRenderConfig) -> tuple[int, int, int]:
    return cfg.blackout_rgb if op.kind is ActionKind.BLACKOUT else cfg.remove_rgb


def _luma(px: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
    p = px.astype(np.int32)
    return np.asarray((299 * p[..., 0] + 587 * p[..., 1] + 114 * p[..., 2]) // 1000, dtype=np.int32)


def _suppress_ink(out: npt.NDArray[np.uint8], b: BBox, darkness: int) -> bool:
    """Faint marks (show-through, pencil) -> paper colour; strong ink is left alone.  Returns the
    pixel check: every pixel is now paper colour or strong ink."""
    region = out[b.y0 : b.y1, b.x0 : b.x1]
    if region.size == 0:
        return True
    luma = _luma(region)
    paper_level = int(np.percentile(luma, 95))
    paper = np.median(region[luma >= paper_level], axis=0).round().astype(np.uint8)
    faint = luma > paper_level - darkness
    region[faint] = paper
    strong = _luma(region) <= paper_level - darkness
    return bool(np.all(np.all(region == paper, axis=-1) | strong))


def _mark(drawn: npt.NDArray[np.bool_], result: TypesetResult) -> None:
    b = result.box
    drawn[b.y0 : b.y1, b.x0 : b.x1] |= result.glyph_mask


def apply_ops(
    rgb: npt.NDArray[np.uint8],
    ops: Sequence[RenderOp],
    cfg: RedactionRenderConfig,
    family: str = "Sans",
    *,
    notice: str | None = None,
    dpi: int = 300,
) -> PageEdit:
    """Apply every operation (and optionally the margin notice) to a copy of the page raster."""
    out = rgb.copy()
    drawn = np.zeros(out.shape[:2], dtype=bool)  # glyphs drawn so far: never erased again
    ok = True
    applied: list[Applied] = []
    ordered = sorted(ops, key=lambda o: o.sort_key)
    for i, op in enumerate(ordered):
        b = op.box
        if op.kind in (ActionKind.BLACKOUT, ActionKind.REMOVE):
            colour = np.array(fill_colour(op, cfg), dtype=np.uint8)
            out[b.y0 : b.y1, b.x0 : b.x1] = colour
            region = out[b.y0 : b.y1, b.x0 : b.x1]
            ok = ok and bool(np.all(region == colour))
            applied.append(Applied(op.ref_id, b, b))
        elif op.kind in (ActionKind.SURROGATE, ActionKind.RETYPESET_REGION):
            blockers = [o.box for j, o in enumerate(ordered) if j != i]
            result = typeset(out, op, family, blockers, protect=drawn)
            ok = ok and erased_cleanly(out, result)
            _mark(drawn, result)
            applied.append(Applied(op.ref_id, b, result.box))
        elif op.kind is ActionKind.SUPPRESS_INK:
            ok = _suppress_ink(out, b, cfg.suppress_ink_darkness) and ok
            applied.append(Applied(op.ref_id, b, b))
        elif op.kind is ActionKind.LABEL:
            blockers = [o.box for j, o in enumerate(ordered) if j != i]
            result = draw_label(out, op, cfg, family, blockers, protect=drawn)
            ok = ok and erased_cleanly(out, result)
            _mark(drawn, result)
            applied.append(Applied(op.ref_id, b, result.box))
        else:  # planning refuses unsupported kinds; this is a second guard
            raise ValueError("unsupported raster operation")
    notice_box = None
    if notice:
        placed = place_notice(out, notice, dpi, family)
        if placed is not None:
            notice_box = placed[0]
    return PageEdit(out, ok, tuple(applied), notice_box)
