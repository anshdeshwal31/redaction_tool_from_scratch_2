"""End-to-end pipeline tests on synthetic documents (slow: loads OCR models in worker processes)."""

from __future__ import annotations

import datetime as dt
import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pytest
from jsonschema import Draft202012Validator
from synthdocs import combine, encrypted, make_ime_letter

from mlredact.core.canonical import canonical_json
from mlredact.core.types import JobStatus
from mlredact.ingest.render import open_document, render_page
from mlredact.ocr.ppocr import PPOCREngine
from mlredact.pipeline.runner import JobResult, Redactor
from mlredact.registry.models import models_dir
from mlredact.surrogate.generators import canonical_date
from mlredact.text.normalize import match_key
from mlredact.verify.checks import extract_all_text
from support import make_config

pytestmark = [pytest.mark.slow, pytest.mark.models]

# Planted values with phase <= this (the implemented plan phase) must be unrecoverable.
CURRENT_PHASE = 3

CFG = make_config("broad", overrides={"redaction": {"style": "blackout"}, "runtime": {"workers": 2}})


@pytest.fixture(scope="module")
def redactor() -> Iterator[Redactor]:
    with Redactor(CFG) as r:
        yield r


@pytest.fixture(scope="module")
def reader() -> PPOCREngine:
    """An independent in-process OCR reader used to attack the outputs."""
    return PPOCREngine(CFG.ocr.engine_a, CFG.runtime.threads, models_dir(CFG.runtime.models_dir))


def _ocr_text(pdf: bytes, reader: PPOCREngine) -> str:
    doc = open_document(pdf, CFG.render)
    try:
        lines = []
        for i in range(len(doc)):
            page = render_page(doc, i, CFG.render, CFG.limits)
            lines.extend(line.text for line in reader.ocr_page(page.rgb, i))
        return "\n".join(lines)
    finally:
        doc.close()


MANIFEST_SCHEMA = Draft202012Validator(
    json.loads((Path(__file__).resolve().parents[2] / "schemas" / "manifest.v1.json").read_text("utf-8"))
)


def _assert_valid_manifest(manifest: dict[str, Any]) -> None:
    """Every manifest - released or quarantined - honours the published evaluator contract."""
    errors = [f"{list(e.absolute_path)}: {e.message}" for e in MANIFEST_SCHEMA.iter_errors(manifest)]
    assert errors == []


def _assert_released(res: JobResult) -> bytes:
    _assert_valid_manifest(res.manifest)
    assert res.status is JobStatus.RELEASED, res.manifest["reasons"]
    assert res.pdf is not None
    return res.pdf


def test_scanned_letter_is_released_and_planted_pii_unrecoverable(redactor: Redactor, reader: PPOCREngine) -> None:
    doc = make_ime_letter(scanned=True, degrade=True, seed=7)
    pdf = _assert_released(redactor.run(doc.pdf))
    text = _ocr_text(pdf, reader)
    key = match_key(text)
    leaked = [p.kind for p in doc.planted if p.phase <= CURRENT_PHASE and match_key(p.text) in key]
    assert leaked == []
    for kept in ("Tinel", "Phalen", "Panadeine", "February 2023", "12 June 2024"):
        assert kept in text, kept


def test_digital_letter_text_layer_is_rebuilt_without_pii(redactor: Redactor) -> None:
    doc = make_ime_letter(scanned=False)
    pdf = _assert_released(redactor.run(doc.pdf))
    extracted = extract_all_text(pdf)
    # The original text layer is gone; the new one holds only kept text (searchable) ...
    assert "Tinel" in extracted and "Panadeine" in extracted
    # ... and nothing that was removed.
    assert "Smith" not in extracted
    assert all(match_key(p.text) not in match_key(extracted) for p in doc.planted if p.phase <= CURRENT_PHASE)


def test_output_is_byte_identical_across_runs(redactor: Redactor) -> None:
    doc = make_ime_letter(scanned=True, degrade=True, seed=3)
    a, b = redactor.run(doc.pdf), redactor.run(doc.pdf)
    assert _assert_released(a) == _assert_released(b)
    assert canonical_json(a.manifest) == canonical_json(b.manifest)


def test_output_is_independent_of_worker_count(redactor: Redactor) -> None:
    pdf = combine(
        [
            make_ime_letter(scanned=True, degrade=True, seed=1).pdf,
            make_ime_letter(scanned=False).pdf,
            make_ime_letter(scanned=True, seed=2).pdf,
        ]
    )
    with Redactor(
        make_config("broad", overrides={"redaction": {"style": "blackout"}, "runtime": {"workers": 1}})
    ) as single:
        one = single.run(pdf)
    many = redactor.run(pdf)
    assert _assert_released(one) == _assert_released(many)
    assert canonical_json(one.manifest) == canonical_json(many.manifest)
    assert [p["index"] for p in many.manifest["pages"]] == [0, 1, 2]


def test_encrypted_input_is_quarantined(redactor: Redactor) -> None:
    res = redactor.run(encrypted(make_ime_letter(scanned=False).pdf))
    assert res.status is JobStatus.QUARANTINED and res.pdf is None
    assert res.manifest["reasons"] == ["E_INPUT_ENCRYPTED"]
    _assert_valid_manifest(res.manifest)


def test_non_pdf_input_is_quarantined(redactor: Redactor) -> None:
    res = redactor.run(b"definitely not a pdf, Zelphinia Quarrington")
    assert res.status is JobStatus.QUARANTINED and res.pdf is None
    assert res.manifest["reasons"] == ["E_INPUT_UNSUPPORTED"]
    _assert_valid_manifest(res.manifest)
    assert "Quarrington" not in canonical_json(res.manifest).decode()


def test_manifest_contains_no_document_text(redactor: Redactor) -> None:
    doc = make_ime_letter(scanned=False)
    res = redactor.run(doc.pdf)
    blob = canonical_json(res.manifest).decode() + canonical_json(res.run_record).decode()
    for p in doc.planted:
        assert p.text not in blob
    assert "Smith" not in blob and "HARBOURSIDE" not in blob


def test_output_pdf_opens_in_independent_engine(redactor: Redactor) -> None:
    pdf = _assert_released(redactor.run(make_ime_letter(scanned=False).pdf))
    doc = pdfium.PdfDocument(pdf)
    try:
        assert len(doc) == 1
    finally:
        doc.close()


# ------------------------------------------------------------------------------- surrogate mode
SURROGATE_CFG = make_config(
    "broad", overrides={"surrogate": {"allow_development_key": True}, "runtime": {"workers": 2}}
)


@pytest.fixture(scope="module")
def surrogate_redactor() -> Iterator[Redactor]:
    with Redactor(SURROGATE_CFG) as r:
        yield r


def _person_surrogates(res: JobResult) -> list[str]:
    return [m["surrogate"] for m in res.manifest["mentions"] if m["type"] == "person" and "surrogate" in m]


def test_surrogate_mode_releases_consistent_realistic_output(surrogate_redactor: Redactor, reader: PPOCREngine) -> None:
    doc = make_ime_letter(scanned=True, degrade=True, seed=7)
    res = surrogate_redactor.run(doc.pdf)
    pdf = _assert_released(res)
    text = _ocr_text(pdf, reader)
    assert [p.kind for p in doc.planted if p.phase <= CURRENT_PHASE and match_key(p.text) in match_key(text)] == []
    checks = {c["check"]: c for c in res.manifest["verification"]["checks"]}
    assert checks["V5_surrogate"]["passed"] and checks["V5_surrogate"]["metrics"]["surrogates"] > 10
    # The claimant ("Mr John Andrew SMITH") keeps one surrogate surname everywhere it appears.
    claimant = next(s for s in _person_surrogates(res) if len(s.split()) == 3 and s.split()[-1].isupper())
    surname = claimant.split()[-1].title()
    assert match_key(text).count(match_key(surname)) >= 4
    assert "De-identified copy" in text  # margin notice


def test_surrogate_mode_is_deterministic(surrogate_redactor: Redactor) -> None:
    doc = make_ime_letter(scanned=True, seed=5)
    a, b = surrogate_redactor.run(doc.pdf), surrogate_redactor.run(doc.pdf)
    assert _assert_released(a) == _assert_released(b)
    assert canonical_json(a.manifest) == canonical_json(b.manifest)


def test_scope_makes_surrogates_consistent_across_documents(surrogate_redactor: Redactor) -> None:
    scanned, digital = make_ime_letter(scanned=True, seed=9).pdf, make_ime_letter(scanned=False).pdf
    a = surrogate_redactor.run(scanned, scope_id="matter-001")
    b = surrogate_redactor.run(digital, scope_id="matter-001")
    c = surrogate_redactor.run(digital, scope_id="matter-002")
    for res in (a, b, c):
        _assert_released(res)

    def claimant(res: JobResult) -> str:
        return next(s for s in _person_surrogates(res) if len(s.split()) == 3 and s.split()[-1].isupper())

    assert claimant(a) == claimant(b)
    assert claimant(b) != claimant(c)
    assert a.manifest["provenance"]["surrogate"]["scope"] == "caller"


# ----------------------------------------------------------------------------------- label mode
def test_label_mode_releases_tagged_output(reader: PPOCREngine) -> None:
    cfg = make_config("broad", overrides={"redaction": {"style": "label"}, "runtime": {"workers": 1}})
    doc = make_ime_letter(scanned=True, degrade=True, seed=7)
    with Redactor(cfg) as r:
        res = r.run(doc.pdf)
    pdf = _assert_released(res)
    text = _ocr_text(pdf, reader)
    assert "[PERSON" in text and "[MEDICARE 1]" in text
    assert [p.kind for p in doc.planted if p.phase <= CURRENT_PHASE and match_key(p.text) in match_key(text)] == []


# ------------------------------------------------------------------------------ other profiles
def test_maximal_profile_shifts_dates_preserving_intervals(reader: PPOCREngine) -> None:
    cfg = make_config("maximal", overrides={"surrogate": {"allow_development_key": True}, "runtime": {"workers": 2}})
    doc = make_ime_letter(scanned=True, degrade=True, seed=7)
    with Redactor(cfg) as r:
        res = r.run(doc.pdf)
    pdf = _assert_released(res)
    text = _ocr_text(pdf, reader)
    for original in ("12 June 2024", "2 May 2024", "3 February 2023"):
        assert original not in text, original
    shifted = [m["surrogate"] for m in res.manifest["mentions"] if m["action"] == "date_shift"]
    dates = [dt.date.fromisoformat(iso) for v in shifted if (iso := canonical_date(v.rstrip(".,")))]
    assert len(dates) >= 3
    # Letter order: 12 June 2024, 2 May 2024, 3 February 2023 -> intervals of 41 and 454 days survive.
    assert [(a - b).days for a, b in itertools.pairwise(dates)][:2] == [41, 454]
    checks = {c["check"]: c for c in res.manifest["verification"]["checks"]}
    assert checks["V5_surrogate"]["metrics"]["shifted_dates"] >= 3


def test_claimant_focused_profile_keeps_the_treating_professional(reader: PPOCREngine) -> None:
    cfg = make_config("claimant_focused", overrides={"redaction": {"style": "blackout"}, "runtime": {"workers": 2}})
    doc = make_ime_letter(scanned=True, degrade=True, seed=7)
    with Redactor(cfg) as r:
        res = r.run(doc.pdf)
    text = _ocr_text(_assert_released(res), reader)
    assert "Peter Brown" in text  # the examining surgeon stays visible
    kept_by_policy = {"Peter Brown", "Smith & Partners Lawyers"}
    leaked = [
        p.text
        for p in doc.planted
        if p.phase <= CURRENT_PHASE
        and p.kind not in ("provider_number", "organisation")
        and p.text not in kept_by_policy
        and match_key(p.text) in match_key(text)
    ]
    assert leaked == []


# ------------------------------------------------------------------- regions and ink accounting
def test_signatures_barcodes_stamps_handwriting_and_show_through_are_removed(redactor: Redactor) -> None:
    import zxingcpp

    doc = make_ime_letter(scanned=True, degrade=True, seed=7, artefacts=True)
    res = redactor.run(doc.pdf)
    pdf = _assert_released(res)
    kinds = {g["kind"] for g in res.manifest["regions"]}
    assert {"signature", "barcode", "stamp", "handwriting", "faint_ink"} <= kinds
    out = render_page(open_document(pdf, CFG.render), 0, CFG.render, CFG.limits).rgb
    assert zxingcpp.read_barcodes(out, return_errors=True) == []
    regions = [g["box"].get("erased_px", g["box"]["px"]) for g in res.manifest["regions"]]
    for kind, (x0, y0, x1, y1) in doc.artefacts:  # every artefact lies inside a removed region
        assert any(r[0] <= x0 + 3 and r[1] <= y0 + 3 and r[2] >= x1 - 3 and r[3] >= y1 - 3 for r in regions), kind
    # The show-through area is blank paper: no ghost words were typeset into it.
    x0, y0, x1, y1 = dict(doc.artefacts)["faint_ink"]
    assert out[y0:y1, x0:x1].min() >= 150


def test_tiff_scans_are_accepted(redactor: Redactor, reader: PPOCREngine) -> None:
    import io as _io

    from PIL import Image

    doc = make_ime_letter(scanned=True, degrade=True, seed=7)
    page = render_page(open_document(doc.pdf, CFG.render), 0, CFG.render, CFG.limits).rgb
    buf = _io.BytesIO()
    Image.fromarray(page).convert("L").save(buf, format="TIFF", dpi=(300, 300), compression="tiff_lzw")
    res = redactor.run(buf.getvalue())
    pdf = _assert_released(res)
    assert res.manifest["job"]["source_format"] == "tiff"
    text = _ocr_text(pdf, reader)
    assert [p.kind for p in doc.planted if p.phase <= CURRENT_PHASE and match_key(p.text) in match_key(text)] == []


def test_sealed_sensitive_manifest_opens_only_for_the_evaluator(redactor: Redactor) -> None:
    from mlredact.manifest.sensitive import keypair, open_sealed

    private, public = keypair()
    doc = make_ime_letter(scanned=False)
    res = redactor.run(doc.pdf, sensitive_to=public)
    _assert_released(res)
    assert res.sensitive is not None and b"SMITH" not in res.sensitive
    opened = open_sealed(res.sensitive, private)
    assert any("SMITH" in m["text"] for m in opened["mentions"])
    assert "SMITH" not in canonical_json(res.manifest).decode()  # the safe manifest is unchanged
