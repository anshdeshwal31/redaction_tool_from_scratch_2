"""Verifier mutation tests (plan §19, Appendix B ``make verify-mutations``).

Each test takes a clean, verifiable output and applies one realistic leak or execution error.  The
corresponding check (V1-V6) must fail.  A verifier that passes any of these is broken.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import pikepdf
import pytest

from mlredact.config.loader import load_config
from mlredact.config.schema import OutputConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import PageGeometry, RenderOp
from mlredact.core.types import ActionKind, EntityType
from mlredact.detect.engine import detect_document
from mlredact.pdfout.builder import PageSpec, build_pdf
from mlredact.pdfout.encode import encode_raster
from mlredact.pdfout.textlayer import page_text_layer
from mlredact.verify.checks import check_integrity, check_structure, check_text_leak
from mlredact.verify.residual import find_residuals
from support import pages_from_lines

CFG = load_config("broad")
REMOVED = ["John Andrew Smith", "2953 70150 1"]


def clean_pdf(text: str = "Patient Walsh reviewed (kept text)") -> bytes:
    geom = PageGeometry(0, 200.0, 100.0, 72, 200, 100)
    raster = np.full((100, 200, 3), 255, dtype=np.uint8)
    layer = page_text_layer(geom, [(BBox(10, 10, 150, 30), text)])
    return build_pdf([PageSpec(200.0, 100.0, encode_raster(raster, OutputConfig()), layer)], OutputConfig())


def mutate(fn: Callable[[pikepdf.Pdf], None]) -> bytes:
    with pikepdf.open(io.BytesIO(clean_pdf())) as pdf:
        fn(pdf)
        out = io.BytesIO()
        pdf.save(out, deterministic_id=True)
        return out.getvalue()


def test_clean_output_passes_every_static_check() -> None:
    pdf = clean_pdf()
    assert check_structure(pdf, 1, allow_fonts=True).passed
    assert check_text_leak(pdf, REMOVED).passed
    integrity, pixels = check_integrity([(200.0, 100.0)], pdf, 3, 3, [True])
    assert integrity.passed and pixels.passed


def _annotation(pdf: pikepdf.Pdf) -> None:
    pdf.pages[0].obj.Annots = pdf.make_indirect(
        pikepdf.Array([pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Text, Contents="J Smith")])
    )


def _metadata(pdf: pikepdf.Pdf) -> None:
    pdf.trailer.Info = pdf.make_indirect(pikepdf.Dictionary(Author="John Smith"))


def _javascript(pdf: pikepdf.Pdf) -> None:
    pdf.Root.OpenAction = pikepdf.Dictionary(S=pikepdf.Name.JavaScript, JS="app.alert(1)")


def _attachment(pdf: pikepdf.Pdf) -> None:
    pdf.attachments["original.pdf"] = pikepdf.AttachedFileSpec(pdf, b"%PDF-1.7 original")


def _form_xobject(pdf: pikepdf.Pdf) -> None:
    form = pikepdf.Stream(pdf, b"BT /F1 9 Tf (hidden) Tj ET")
    form.Type, form.Subtype, form.BBox = pikepdf.Name.XObject, pikepdf.Name.Form, [0, 0, 10, 10]
    pdf.pages[0].obj.Resources.XObject.Fm0 = form


def _image_mask(pdf: pikepdf.Pdf) -> None:
    for _name, image in pdf.pages[0].obj.Resources.XObject.items():
        image.SMask = pikepdf.Stream(pdf, b"\x00" * 4)


def _second_font(pdf: pikepdf.Pdf) -> None:
    pdf.pages[0].obj.Resources.Font.F2 = pikepdf.Dictionary(
        Type=pikepdf.Name.Font, Subtype=pikepdf.Name.TrueType, BaseFont=pikepdf.Name("/ABCDEF+Arial")
    )


def _optional_content(pdf: pikepdf.Pdf) -> None:
    pdf.Root.OCProperties = pikepdf.Dictionary(OCGs=pikepdf.Array())


@pytest.mark.parametrize(
    "mutation",
    [_annotation, _metadata, _javascript, _attachment, _form_xobject, _image_mask, _second_font, _optional_content],
)
def test_v1_rejects_objects_the_builder_never_emits(mutation: Callable[[pikepdf.Pdf], None]) -> None:
    assert not check_structure(mutate(mutation), 1, allow_fonts=True).passed


def test_v1_rejects_incremental_updates_and_wrong_page_counts() -> None:
    pdf = clean_pdf()
    appended = pdf + b"\n1 0 obj\n<< /Author (John Smith) >>\nendobj\nstartxref\n0\n%%EOF\n"
    assert not check_structure(appended, 1, allow_fonts=True).passed
    assert not check_structure(pdf, 2, allow_fonts=True).passed
    assert not check_structure(pdf, 1, allow_fonts=False).passed  # text layer present but not allowed


def test_v2_finds_removed_values_in_text_layer_in_any_layout() -> None:
    for leaked in ("Seen: John Andrew SMITH", "Medicare 2953-70150-1", "john andrew smith"):
        assert not check_text_leak(clean_pdf(leaked), REMOVED).passed, leaked


def test_v2_finds_hex_encoded_and_raw_stream_text() -> None:
    def hex_text(pdf: pikepdf.Pdf) -> None:
        page = pdf.pages[0]
        hexed = b"John Andrew Smith".hex().encode()
        extra = b"\nBT /F1 1 Tf <" + hexed + b"> Tj ET"
        page.obj.Contents = pikepdf.Stream(pdf, page.obj.Contents.read_bytes() + extra)

    assert not check_text_leak(mutate(hex_text), REMOVED).passed


def test_v3_and_v6_reject_execution_errors() -> None:
    pdf = clean_pdf()
    integrity, _ = check_integrity([(200.0, 100.0)], pdf, 3, 2, [True])  # an op was skipped
    assert not integrity.passed
    integrity, _ = check_integrity([(210.0, 100.0)], pdf, 3, 3, [True])  # page size changed
    assert not integrity.passed
    integrity, _ = check_integrity([(200.0, 100.0), (200.0, 100.0)], pdf, 3, 3, [True, True])  # page lost
    assert not integrity.passed
    _, pixels = check_integrity([(200.0, 100.0)], pdf, 3, 3, [False])  # a fill was not opaque
    assert not pixels.passed


def test_v4_flags_values_still_readable_in_the_output() -> None:
    source = ["Re: Mr John Andrew SMITH DOB: 14/03/1978"]
    removed = [m for m in detect_document(pages_from_lines(source), CFG, "doc").mentions]
    assert any(m.entity_type is EntityType.PERSON for m in removed)
    # The output still shows the name (e.g. an op was planned on the wrong page) - OCR'd as "SMlTH".
    output = pages_from_lines(["Re: Mr John Andrew SMlTH DOB: 14/03/1978"])
    hits = find_residuals(output, removed, {}, CFG)
    assert {h.reason for h in hits} >= {EntityType.PERSON.value}
    # Every token covered by an erased box: nothing residual.
    everything = RenderOp(0, ActionKind.BLACKOUT, BBox(0, 0, 3000, 4000), "test", "R-1")
    assert find_residuals(output, removed, {0: [everything]}, CFG) == []


def test_v4_kept_regions_exempt_redetection_but_never_removed_values() -> None:
    page = pages_from_lines(["Dr Peter Brown FRACS examined Mr John Andrew SMITH today"])
    tokens = list(page[0].tokens())
    brown = [t.bbox for t in tokens if t.text in ("Peter", "Brown")]
    # Without a kept region, a strong re-detection of "Dr Peter Brown" is residual.
    assert any(h.source == "redetect" for h in find_residuals(page, [], {}, CFG))
    # Deliberately kept (claimant-focused professional): exempt from re-detection ...
    assert not [h for h in find_residuals(page, [], {}, CFG, kept={0: brown}) if h.box in brown]
    # ... but a removed value inside a kept region is still reported.
    removed = detect_document(page, CFG, "doc").mentions
    hits = find_residuals(page, list(removed), {}, CFG, kept={0: [t.bbox for t in tokens]})
    assert any(h.source == "removed_value" for h in hits)


def test_v4_flags_a_barcode_that_survives_in_the_output() -> None:
    from dataclasses import replace

    from mlredact.core.model import Region
    from mlredact.core.types import RegionKind

    page = pages_from_lines(["Medical report"])[0]
    qr = Region("G-1", 0, RegionKind.BARCODE, BBox(1900, 100, 2200, 400), 1.0, "zxing", "barcode.decoded")
    output = [replace(page, regions=(qr,))]
    hits = find_residuals(output, [], {}, CFG)
    assert [(h.reason, h.source) for h in hits] == [("barcode", "region")]
    covered = {0: [RenderOp(0, ActionKind.REMOVE, BBox(1850, 50, 2250, 450), "barcode", "G-1")]}
    assert find_residuals(output, [], covered, CFG) == []
