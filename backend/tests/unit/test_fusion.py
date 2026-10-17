"""Engine D (the PDF's own text layer) and its fusion with the OCR reading - no OCR models needed."""

from __future__ import annotations

import io

import numpy as np
import pikepdf
from synthdocs import make_ime_letter

from mlredact.config.loader import load_config
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, PageAnalysis, Token, token_id
from mlredact.core.types import EntityType, PageKind
from mlredact.detect.engine import detect_document
from mlredact.ingest.embedded import EmbeddedWord
from mlredact.ingest.render import open_document, render_page
from mlredact.ocr.fusion import ENGINE_ID, fuse_embedded, plausible_text

CFG = load_config("broad")


def _render(pdf: bytes):  # type: ignore[no-untyped-def]
    return render_page(open_document(pdf, CFG.render), 0, CFG.render, CFG.limits, with_words=True)


def _pdf(lines: list[tuple[float, float, float, str, int]]) -> bytes:
    """(x, y, size, text, grey 0-255) text runs on an A4 page, base-14 Helvetica."""
    pdf = pikepdf.new()
    font = pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica)
    ops = []
    for x, y, size, text, grey in lines:
        g = grey / 255
        ops.append(f"BT {g:.3f} g /F1 {size} Tf {x} {y} Td ({text}) Tj ET")
    page = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=[0, 0, 595.276, 841.89],
        Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)),
        Contents=pikepdf.Stream(pdf, "\n".join(ops).encode("latin-1")),
    )
    pdf.pages.append(pikepdf.Page(page))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


def _token(line_no: int, word_no: int, box: BBox, text: str, conf: float) -> Token:
    return Token(token_id(0, line_no, word_no), 0, line_no, word_no, box, conf, conf, "ocr", text)


def _line(line_no: int, tokens: tuple[Token, ...]) -> Line:
    b = tokens[0].bbox
    for t in tokens[1:]:
        b = b.union(t.bbox)
    return Line(
        f"p0000.l{line_no:04d}",
        0,
        line_no,
        b,
        ((b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)),
        0.9,
        "ocr",
        tokens,
    )


def test_embedded_words_land_on_their_ink() -> None:
    page = _render(make_ime_letter(scanned=False).pdf)
    words = {w.text: w for w in page.words}
    assert {"SMITH", "WC1234567", "Phalen's"} <= set(words)
    for w in page.words:
        crop = page.rgb[w.bbox.y0 : w.bbox.y1, w.bbox.x0 : w.bbox.x1].mean(axis=2)
        assert (crop < 128).mean() > 0.05, w.text  # every word box sits on dark glyph pixels


def test_uncertain_ocr_is_corrected_confident_ocr_is_kept() -> None:
    rgb = np.full((400, 1200, 3), 250, dtype=np.uint8)
    rgb[100:140, 100:300] = 20
    rgb[100:140, 400:600] = 20
    unsure = _token(0, 0, BBox(100, 100, 300, 140), "Srnith", 0.55)
    sure = _token(0, 1, BBox(400, 100, 600, 140), "Smith", 0.99)
    words = [EmbeddedWord("Smith", BBox(102, 101, 298, 139), 0), EmbeddedWord("Vplwk", BBox(402, 101, 598, 139), 0)]
    (line,) = fuse_embedded([_line(0, (unsure, sure))], words, rgb, 0)
    a, b = line.tokens
    assert (a.text, a.alternates) == ("Smith", ("Srnith",))  # the exact text layer wins over a shaky read
    assert (b.text, b.alternates) == ("Smith", ("Vplwk",))  # a garbled font cannot overwrite a good read


def test_words_ocr_missed_become_tokens_and_invisible_words_are_ignored() -> None:
    rgb = np.full((400, 1200, 3), 250, dtype=np.uint8)
    rgb[200:212, 100:300] = 30  # tiny print OCR did not detect
    words = [EmbeddedWord("QUIMBY", BBox(100, 200, 300, 212), 3), EmbeddedWord("hidden", BBox(600, 200, 800, 212), 3)]
    lines = fuse_embedded([], words, rgb, 0)
    assert [(t.text, t.engine) for line in lines for t in line.tokens] == [("QUIMBY", ENGINE_ID)]


def test_tiny_print_reaches_detection_through_the_text_layer() -> None:
    pdf = _pdf([(60, 760, 11, "Medical report", 0), (60, 740, 3, "Patient name: Zelda QUIMBY  DOB: 02/11/1980", 0)])
    page = _render(pdf)
    lines = fuse_embedded([], page.words, page.rgb, 0)  # as if OCR had read nothing at all
    analysis = PageAnalysis(page.geometry, PageKind.BORN_DIGITAL, lines, page.embedded_char_count)
    tokens = {t.token_id: t for t in analysis.tokens()}
    found = {
        (m.entity_type, " ".join(tokens[t].text for t in m.token_ids))
        for m in detect_document([analysis], CFG, "doc").mentions
    }
    assert (EntityType.PERSON, "Zelda QUIMBY") in found
    assert (EntityType.DATE_OF_BIRTH, "02/11/1980") in found


def test_white_text_is_never_trusted() -> None:
    page = _render(_pdf([(60, 760, 11, "Visible heading", 0), (60, 700, 11, "Hidden Zelda QUIMBY", 255)]))
    lines = fuse_embedded([], page.words, page.rgb, 0)
    assert "QUIMBY" not in {t.text for line in lines for t in line.tokens}


def test_broken_encodings_are_rejected() -> None:
    assert plausible_text("Smith") and plausible_text("WC1234567") and plausible_text("O'Brien,")
    assert not plausible_text("") and not plausible_text("Sm�th") and not plausible_text("\x07\x08")


# ------------------------------------------------------------------------------- engine B (Tesseract)
TSV = "\n".join(
    [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "1\t1\t0\t0\t0\t0\t0\t0\t2481\t3508\t-1\t",
        "4\t1\t1\t1\t1\t0\t250\t250\t600\t45\t-1\t",
        "5\t1\t1\t1\t1\t1\t250\t250\t120\t45\t96.5\tMr",
        "5\t1\t1\t1\t1\t2\t390\t250\t200\t45\t91.0\tSmith",
        "5\t1\t1\t1\t2\t1\t250\t320\t300\t45\t41.0\tsoooor",
        "5\t1\t2\t1\t1\t1\t900\t900\t200\t40\t88.0\tQUIMBY",
    ]
)


def test_tesseract_tsv_is_parsed_into_lines_of_words() -> None:
    from mlredact.ocr.tesseract import parse_tsv

    words = parse_tsv(TSV)
    assert [(w.text, w.line, w.confidence) for w in words] == [
        ("Mr", 0, 0.965),
        ("Smith", 0, 0.91),
        ("soooor", 1, 0.41),
        ("QUIMBY", 2, 0.88),
    ]
    assert words[1].bbox == BBox(390, 250, 590, 295)


def test_tesseract_words_fuse_conservatively() -> None:
    from mlredact.ocr.fusion import fuse_tesseract
    from mlredact.ocr.tesseract import parse_tsv

    unsure = _token(0, 0, BBox(392, 251, 588, 294), "Srnith", 0.5)
    lines = fuse_tesseract([_line(0, (unsure,))], parse_tsv(TSV), 0, replace_below=0.9, min_new_confidence=0.7)
    texts = [(t.text, t.engine) for line in lines for t in line.tokens]
    assert texts[0] == ("Smith", "ocr")  # confident engine B fixes an uncertain engine-A read
    assert ("Mr", "tesseract") in texts and ("QUIMBY", "tesseract") in texts  # words A missed are kept
    assert not any(text == "soooor" for text, _e in texts)  # low-confidence garbage never becomes a token
