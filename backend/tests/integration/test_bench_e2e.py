"""``mlredact bench`` end to end on a synthetic scan, started the way the benchmark starts it: as a
subprocess from a run folder that is not the backend folder (slow: loads the models)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pytest
from synthdocs import SynthDoc, make_ime_letter

from mlredact.bench.artifacts import LABELS
from mlredact.text.normalize import match_key
from support import EXTRA_CONFIG

pytestmark = [pytest.mark.slow, pytest.mark.models]

KEPT_BY_CLIENT = {"date", "date_of_birth", "date_of_death", "age", "locality", "postcode"}
SCAN_DPI = 300


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _bounds(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


@pytest.fixture(scope="module")
def bench_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SynthDoc, Path]]:
    doc = make_ime_letter(scanned=True, seed=7, artefacts=True)
    run = tmp_path_factory.mktemp("run")
    (run / "input.pdf").write_bytes(doc.pdf)
    env = dict(os.environ)
    if EXTRA_CONFIG:  # e.g. the laptop arrangement without NER and Tesseract
        env["MLREDACT_CONFIG"] = os.pathsep.join(str(p.resolve()) for p in EXTRA_CONFIG)
    args = ["--input", str(run / "input.pdf"), "--workdir", str(run / "out"), "--mode", "e2e", "--inject-dir", ""]
    proc = subprocess.run(
        [sys.executable, "-m", "mlredact.cli.main", "bench", *args, "--scope-id", "", "--seed", "0"],
        cwd=run,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    yield doc, run / "out"


def test_output_and_verdict_are_handed_over(bench_run: tuple[SynthDoc, Path]) -> None:
    _doc, out = bench_run
    pdf = pdfium.PdfDocument((out / "output.pdf").read_bytes())
    try:
        assert len(pdf) == 1
    finally:
        pdf.close()
    info = _json(out / "run_info.json")
    assert info["release"] == {"released": True, "checks_failed": []}
    assert re.fullmatch(r"\d+\.\d+\.\d+(\+g[0-9a-f]+(\.dirty)?)?", info["candidate_version"])
    assert info["config"]["profile"] == "client" and info["seed"] == 0
    assert info["models"] and all(m["name"] and m["version"] for m in info["models"])
    if "MLREDACT_MODELS_DIR" not in os.environ:
        assert "models from the backend's own .models folder" in info["notes"]


def test_mentions_are_what_the_client_policy_replaces(bench_run: tuple[SynthDoc, Path]) -> None:
    doc, out = bench_run
    mentions = _json(out / "mentions.json")["mentions"]
    labels = {m["label"] for m in mentions}
    assert labels <= set(LABELS)
    assert not labels & KEPT_BY_CLIENT
    assert {"signature", "barcode"} <= labels
    found = match_key(" ".join(m.get("text", "") for m in mentions))
    for p in doc.planted:
        if p.kind in ("person", "phone", "email", "medicare"):
            assert match_key(p.text) in found, p.kind


def test_stage_files_name_each_other_consistently(bench_run: tuple[SynthDoc, Path]) -> None:
    _doc, out = bench_run
    tokens = {t["id"] for p in _json(out / "text.json")["pages"] for t in p["tokens"]}
    mentions = _json(out / "mentions.json")["mentions"]
    ids = {m["id"] for m in mentions}
    assert len(ids) == len(mentions)
    assert all(set(m.get("tokens", [])) <= tokens for m in mentions)
    plan = _json(out / "plan.json")["regions"]
    assert len({r["id"] for r in plan}) == len(plan)
    assert all(set(r["mentions"]) <= ids for r in plan)
    assert all(set(c["mentions"]) <= ids for c in _json(out / "entities.json")["clusters"])
    pseudonyms = _json(out / "pseudonyms.json")
    assert pseudonyms["mentions"] and all(m["mention"] in ids for m in pseudonyms["mentions"])


def test_polygons_are_points_of_the_displayed_page(bench_run: tuple[SynthDoc, Path]) -> None:
    doc, out = bench_run
    (page,) = _json(out / "manifest.json")["pages"]
    width, height = page["width_pt"], page["height_pt"]
    text = _json(out / "text.json")
    assert text["source"] == "ocr"
    polys = [t["poly"] for t in text["pages"][0]["tokens"]]
    polys += [r["poly"] for m in _json(out / "mentions.json")["mentions"] for r in m["regions"]]
    polys += [r["poly"] for r in _json(out / "plan.json")["regions"]]
    for poly in polys:
        x0, y0, x1, y1 = _bounds(poly)
        assert -0.01 <= x0 <= x1 <= width + 0.01 and -0.01 <= y0 <= y1 <= height + 0.01
    # y grows downwards: the letterhead is the first line of the page.
    head = next(t for t in text["pages"][0]["tokens"] if "HARBOURSIDE" in t["text"])
    assert _bounds(head["poly"])[1] < 0.15 * height
    # The planted signature (pixels of the 300 dpi scan) lies inside the signature mentions.
    sx0, sy0, sx1, sy1 = (v * 72 / SCAN_DPI for v in dict(doc.artefacts)["signature"])
    covered = 0.0
    for m in _json(out / "mentions.json")["mentions"]:
        if m["label"] == "signature":
            mx0, my0, mx1, my1 = _bounds(m["regions"][0]["poly"])
            covered += max(0.0, min(sx1, mx1) - max(sx0, mx0)) * max(0.0, min(sy1, my1) - max(sy0, my0))
    assert covered >= 0.8 * (sx1 - sx0) * (sy1 - sy0)
