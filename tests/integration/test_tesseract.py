"""Engine B on the real binary (skipped where Tesseract is not installed, e.g. the dev laptop)."""

from __future__ import annotations

import shutil

import pytest
from synthdocs import make_ime_letter

from mlredact.config.loader import load_config
from mlredact.ingest.render import open_document, render_page
from mlredact.ocr.tesseract import TesseractEngine
from mlredact.registry.models import models_dir

pytestmark = [
    pytest.mark.slow,
    pytest.mark.models,
    pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed"),
]


def test_tesseract_reads_the_scanned_letter_deterministically() -> None:
    cfg = load_config("broad")
    engine = TesseractEngine(cfg.ocr.engine_b, models_dir(cfg.runtime.models_dir))
    page = render_page(
        open_document(make_ime_letter(scanned=True, degrade=True, seed=7).pdf, cfg.render), 0, cfg.render, cfg.limits
    )
    words = engine.words(page.rgb, page.geometry.dpi)
    texts = {w.text.strip(",.") for w in words}
    assert {"SMITH", "Medicare", "Phalen's"} <= texts
    assert engine.words(page.rgb, page.geometry.dpi) == words
