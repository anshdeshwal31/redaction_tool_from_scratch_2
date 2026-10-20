"""Synthetic stand-ins for signatures, faces and photos: pixels, determinism, draw order."""

from __future__ import annotations

import numpy as np

from mlredact.config.loader import load_config
from mlredact.core.geometry import BBox
from mlredact.core.model import RenderOp
from mlredact.core.types import ActionKind, RegionKind
from mlredact.render.raster import apply_ops
from mlredact.render.synth import PEN_INK, PHOTO_BACKGROUND, PHOTO_FIGURE, draw_stand_in

RENDER = load_config("broad").redaction
A = ActionKind


def _noisy(h: int = 200, w: int = 600) -> np.ndarray:
    rng = np.random.default_rng(1)
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    img[60:140, 80:520] = rng.integers(0, 120, (80, 440, 3), dtype=np.uint8)  # "original" pen strokes
    return img


class TestStandIns:
    BOX = BBox(70, 50, 530, 150)

    def test_signature_replaces_every_original_pixel(self) -> None:
        img = _noisy()
        assert draw_stand_in(img, self.BOX, RegionKind.SIGNATURE.value, "G-1")
        region = img[50:150, 70:530].reshape(-1, 3)
        colours = {tuple(int(c) for c in px) for px in np.unique(region, axis=0)}
        assert PEN_INK in colours and len(colours) == 2  # paper + pen, nothing else
        ink = np.all(region == PEN_INK, axis=-1).mean()
        assert 0.02 < ink < 0.4  # visible strokes, not a filled box

    def test_signature_is_deterministic_and_differs_by_region(self) -> None:
        a, b, c = _noisy(), _noisy(), _noisy()
        draw_stand_in(a, self.BOX, "signature", "G-1")
        draw_stand_in(b, self.BOX, "signature", "G-1")
        draw_stand_in(c, self.BOX, "signature", "G-2")
        assert np.array_equal(a, b) and not np.array_equal(a, c)

    def test_face_and_photo_become_a_silhouette(self) -> None:
        img = _noisy()
        assert draw_stand_in(img, self.BOX, RegionKind.FACE.value, "G-3")
        colours = {tuple(int(c) for c in px) for px in np.unique(img[50:150, 70:530].reshape(-1, 3), axis=0)}
        assert colours == {PHOTO_BACKGROUND, PHOTO_FIGURE}

    def test_raster_editor_applies_and_verifies_stand_ins(self) -> None:
        op = RenderOp(0, A.SYNTHESISE, self.BOX, RegionKind.SIGNATURE.value, "G-1")
        edit = apply_ops(_noisy(), [op], RENDER)
        assert edit.verified and [a.ref_id for a in edit.applied] == ["G-1"]

    def test_stand_in_is_drawn_last_and_off_other_boxes(self) -> None:
        """A name line inside the signature region keeps what its own operation drew there."""
        name = BBox(70, 120, 400, 150)
        ops = [
            RenderOp(0, A.SYNTHESISE, self.BOX, RegionKind.SIGNATURE.value, "G-1"),
            RenderOp(0, A.BLACKOUT, name, "person", "M-1"),
        ]
        edit = apply_ops(_noisy(), ops, RENDER)
        assert edit.verified and [a.ref_id for a in edit.applied] == ["M-1", "G-1"]
        assert np.all(edit.raster[120:150, 70:400] == 0)  # the blackout survives
        assert np.any(np.all(edit.raster[50:120, 70:530] == PEN_INK, axis=-1))  # strokes above it
        assert not np.any(np.all(edit.raster[120:150, 70:530] == PEN_INK, axis=-1))

    def test_faint_ink_overlap_is_painted_over(self) -> None:
        """Faint-ink suppression leaves strong ink alone, so the stand-in must still cover it."""
        faint = BBox(70, 50, 300, 150)
        ops = [
            RenderOp(0, A.SYNTHESISE, self.BOX, RegionKind.SIGNATURE.value, "G-1"),
            RenderOp(0, A.SUPPRESS_INK, faint, RegionKind.FAINT_INK.value, "G-2"),
        ]
        edit = apply_ops(_noisy(), ops, RENDER)
        assert edit.verified
        region = edit.raster[50:150, 70:530].reshape(-1, 3)
        assert len(np.unique(region, axis=0)) == 2  # paper + pen only: no original stroke survives
