"""Ink accounting and region classification on synthetic pages with known token boxes (no OCR)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageDraw, ImageFont
from synthdocs.generator import _add_artefacts, _content

from mlredact.config.schema import RedactionRenderConfig, RegionsConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, Region, RenderOp, Token, token_id
from mlredact.core.types import ActionKind, RegionKind
from mlredact.layout.barcodes import find_barcodes
from mlredact.layout.regions import analyse_regions
from mlredact.render.raster import apply_ops
from mlredact.resources.loader import font_path

DPI = 200
CFG = RegionsConfig()


def typeset(lines: list[str], dpi: int = DPI) -> tuple[Image.Image, list[Line]]:
    """The synthdocs letter layout, with a token per word (what a perfect OCR pass would return)."""
    scale = dpi / 72.0
    img = Image.new("L", (round(595.276 * scale), round(841.89 * scale)), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path("Sans", "Regular")), size=round(11 * scale))
    out: list[Line] = []
    y = 60 * scale
    for line_text in lines:
        draw.text((60 * scale, y), line_text, fill=20, font=font)
        tokens: list[Token] = []
        pos = 0
        n = len(out)
        for w_no, word in enumerate(line_text.split()):
            start = line_text.index(word, pos)
            pos = start + len(word)
            x0 = 60 * scale + draw.textlength(line_text[:start], font=font)
            left, top, right, bottom = draw.textbbox((x0, y), word, font=font)
            box = BBox(int(left), int(top), int(right) + 1, int(bottom) + 1)
            tokens.append(Token(token_id(0, n, w_no), 0, n, w_no, box, 0.99, 0.99, "test", word))
        if tokens:
            lb = tokens[0].bbox
            for t in tokens[1:]:
                lb = lb.union(t.bbox)
            quad = ((lb.x0, lb.y0), (lb.x1, lb.y0), (lb.x1, lb.y1), (lb.x0, lb.y1))
            out.append(Line(f"p0000.l{n:04d}", 0, n, lb, quad, 0.99, "test", tuple(tokens)))
        y += 13 * scale
    return img, out


def quad_of(b: BBox) -> tuple[tuple[float, float], ...]:
    return ((b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1))


def rgb(img: Image.Image) -> npt.NDArray[np.uint8]:
    return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def covers(regions: tuple[Region, ...], kind: str, truth: tuple[int, int, int, int], frac: float = 0.95) -> bool:
    t = BBox(*truth)
    covered = sum(i.area for r in regions if r.kind.value == kind and (i := r.bbox.intersection(t)) is not None)
    return covered >= frac * t.area


@pytest.fixture(scope="module")
def letter() -> tuple[list[str], Image.Image, list[Line]]:
    text, _planted = _content()
    img, lines = typeset(text)
    return text, img, lines


def test_a_clean_page_is_fully_explained(letter: tuple[list[str], Image.Image, list[Line]]) -> None:
    _text, img, lines = letter
    found = analyse_regions(rgb(img), lines, 0, DPI, CFG)
    assert found.regions == () and found.extra_lines == () and found.ghost_tokens == frozenset()


def test_every_artefact_is_classified_and_covered(letter: tuple[list[str], Image.Image, list[Line]]) -> None:
    text, img, lines = letter
    page, truth = _add_artefacts(img, text, DPI, seed=7)
    raster = rgb(page)
    found = analyse_regions(raster, lines, 0, DPI, CFG, detected=find_barcodes(raster, 0, DPI))
    for kind, box in truth:
        assert covers(found.regions, kind, box), (kind, [(r.kind.value, r.bbox.as_tuple()) for r in found.regions])
    assert {r.kind for r in found.regions} == {
        RegionKind.SIGNATURE,
        RegionKind.BARCODE,
        RegionKind.STAMP,
        RegionKind.HANDWRITING,
        RegionKind.FAINT_INK,
    }


def test_garbage_tokens_do_not_explain_a_signature(letter: tuple[list[str], Image.Image, list[Line]]) -> None:
    text, img, lines = letter
    page, truth = _add_artefacts(img, text, DPI, seed=7)
    sig = BBox(*dict(truth)["signature"])
    # OCR "reads" the scribble as a tall, low-confidence word: it must not hide the signature.
    garbage = Token(token_id(0, 99, 0), 0, 99, 0, sig, 0.84, 0.42, "test", "soooor")
    fake = Line("p0000.l0099", 0, 99, sig, quad_of(sig), 0.84, "test", (garbage,))
    found = analyse_regions(rgb(page), [*lines, fake], 0, DPI, CFG)
    assert covers(found.regions, "signature", dict(truth)["signature"])


def test_show_through_tokens_are_ghosts_but_real_text_is_not(letter: tuple[list[str], Image.Image, list[Line]]) -> None:
    text, img, lines = letter
    page, truth = _add_artefacts(img, text, DPI, seed=7)
    faint = BBox(*dict(truth)["faint_ink"])
    ghost = Token(token_id(0, 98, 0), 0, 98, 0, faint, 0.97, 0.6, "test", "9160b9M7")
    fake = Line("p0000.l0098", 0, 98, faint, quad_of(faint), 0.97, "test", (ghost,))
    found = analyse_regions(rgb(page), [*lines, fake], 0, DPI, CFG)
    assert found.ghost_tokens == frozenset({ghost.token_id})


def test_unexplained_body_marks_are_removed_whether_figure_or_handwriting(
    letter: tuple[list[str], Image.Image, list[Line]],
) -> None:
    from mlredact.config.loader import load_config

    _text, img, lines = letter
    page = img.convert("RGB")
    draw = ImageDraw.Draw(page)
    # A framed chart with a plotted curve (a figure) ...
    x0, y0 = 300, 1300
    draw.rectangle([x0, y0, x0 + 450, y0 + 300], outline=(20, 20, 20), width=4)
    t = np.linspace(0, 1, 300)
    curve = list(zip((x0 + 20 + 410 * t).tolist(), (y0 + 150 + 110 * np.sin(9 * t)).tolist(), strict=True))
    draw.line(curve, fill=(20, 20, 20), width=4)
    # ... and a looping scribble elsewhere in the body.
    scribble = list(
        zip((1000 + 300 * t + 20 * np.sin(60 * t)).tolist(), (1400 + 60 * np.sin(40 * t)).tolist(), strict=True)
    )
    draw.line(scribble, fill=(20, 20, 20), width=4)
    found = analyse_regions(rgb(page), lines, 0, DPI, CFG)
    by_kind: dict[RegionKind, list[BBox]] = {}
    for r in found.regions:
        by_kind.setdefault(r.kind, []).append(r.bbox)
    assert any(b.intersection(BBox(x0 + 20, y0 + 40, x0 + 430, y0 + 260)) for b in by_kind[RegionKind.GRAPHIC])
    assert any(b.intersection(BBox(1000, 1340, 1300, 1460)) for b in by_kind[RegionKind.HANDWRITING])
    for profile in ("broad", "claimant_focused", "maximal"):  # a figure is never kept by default
        assert load_config(profile).policy.region_actions[RegionKind.GRAPHIC] is ActionKind.REMOVE


def test_regions_are_deterministic(letter: tuple[list[str], Image.Image, list[Line]]) -> None:
    text, img, lines = letter
    page, _truth = _add_artefacts(img, text, DPI, seed=3)
    a = analyse_regions(rgb(page), lines, 0, DPI, CFG)
    b = analyse_regions(rgb(page), lines, 0, DPI, CFG)
    assert a == b


def test_suppress_ink_removes_faint_marks_and_keeps_strong_ink() -> None:
    img = np.full((200, 400, 3), 245, dtype=np.uint8)
    img[50:60, 20:380] = 215  # faint show-through stroke
    img[100:110, 20:380] = 20  # real ink
    op = RenderOp(0, ActionKind.SUPPRESS_INK, BBox(0, 0, 400, 200), "faint_ink", "G-1")
    edit = apply_ops(img, [op], RedactionRenderConfig())
    assert edit.verified
    assert np.all(edit.raster[50:60, 20:380] == 245) and np.all(edit.raster[100:110, 20:380] == 20)


def test_legibility_gate_flags_a_mostly_unreadable_page(letter: tuple[list[str], Image.Image, list[Line]]) -> None:
    _text, img, lines = letter
    clean = analyse_regions(rgb(img), lines, 0, DPI, CFG)
    assert clean.unexplained_ink == 0.0 and not clean.illegible
    # Handwritten notes everywhere, with only the first two printed lines read by OCR.
    page = Image.new("RGB", img.size, (250, 250, 250))
    draw = ImageDraw.Draw(page)
    rng = np.random.default_rng(0)
    for row in range(14):
        t = np.linspace(0, 1, 400)
        k = rng.uniform(25, 40)
        xs = 170 + 1300 * t
        ys = 300 + 110 * row + 18 * np.sin(2 * np.pi * k * t)
        draw.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(30, 30, 30), width=3)
    found = analyse_regions(rgb(page), lines[:2], 0, DPI, CFG)
    assert found.illegible and found.unexplained_ink >= CFG.illegible_fraction
    gate = [r for r in found.regions if r.kind is RegionKind.ILLEGIBLE]
    assert len(gate) == 1 and gate[0].bbox.height > 1300  # covers the page's content
