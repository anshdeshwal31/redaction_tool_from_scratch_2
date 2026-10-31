"""Sideways and upside-down pages are held for review, not released (``mlredact.layout.orientation``).

The pipeline runs in this process as in test_bench_runner (page analysis reads the PDF's text layer, no models).  The
letter is synthetic (``tools/synthdocs``); its sideways forms are made here: turned by its /Rotate setting, or drawn
turned a quarter turn on a page the setting leaves upright.  Upside-down text is OCR's judgement (its line classifier),
which the text layer cannot give: the upside-down runs mark the lines the way OCR does.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pikepdf
import pytest
import yaml
from test_bench_runner import LETTER, NO_MODELS, InProcessPool
from typer.testing import CliRunner

import mlredact.bench.environment as bench_env
from mlredact.bench.artifacts import unreleased_note
from mlredact.cli.main import app
from mlredact.config.loader import load_config
from mlredact.config.schema import LimitsConfig
from mlredact.core.errors import ReasonCode
from mlredact.core.geometry import BBox, Quad
from mlredact.core.model import Line, PageAnalysis, PageGeometry, Token, token_id
from mlredact.core.types import JobStatus, PageKind
from mlredact.ingest.inspect import inspect_pdf
from mlredact.layout.orientation import MIN_SIDEWAYS_CHARS, line_runs_sideways, sideways_pages
from mlredact.ocr import ppocr
from mlredact.ocr.ctc import DecodedChar
from mlredact.pipeline.runner import JobResult, Redactor
from mlredact.pipeline.shared import BlobRef

ROTATED = ReasonCode.PAGE_ROTATED_SIDEWAYS
CONTENT = ReasonCode.PAGE_CONTENT_SIDEWAYS
UPSIDE_ROTATED = ReasonCode.PAGE_ROTATED_UPSIDE_DOWN
UPSIDE_CONTENT = ReasonCode.PAGE_CONTENT_UPSIDE_DOWN


# ------------------------------------------------------------------------------ the documents
def _saved(pdf: pikepdf.Pdf) -> bytes:
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


def turned_by_rotate(data: bytes, degrees: int = 90) -> bytes:
    """Every page's /Rotate set to ``degrees``: a viewer shows the page turned clockwise by that much."""
    with pikepdf.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page.obj.Rotate = degrees
        return _saved(pdf)


def drawn_sideways(data: bytes) -> bytes:
    """Every page's content drawn turned a quarter turn counter-clockwise (its text reads bottom to top) and scaled
    into the top of a page of the same size; /Rotate stays 0, so the page itself is upright."""
    with pikepdf.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            x0, y0, x1, y1 = (float(v) for v in page.mediabox)
            w, h = x1 - x0, y1 - y0
            s = w / h
            cm = f"q 0 {s:.6f} {-s:.6f} 0 {x0 + s * y1:.6f} {y1 - s * (w + x0):.6f} cm\n"
            page.contents_add(pdf.make_stream(cm.encode("ascii")), prepend=True)
            page.contents_add(pdf.make_stream(b"\nQ\n"), prepend=False)
        return _saved(pdf)


class Pool(InProcessPool):
    """test_bench_runner's pool, without its stand-in signature where the page has no room for it: the signature sits
    below the letter's text on a portrait page, which a page turned landscape by its /Rotate is too short to hold."""

    def _analyse(self, ref: BlobRef, page: int, profile: str = "primary") -> PageAnalysis:
        found = super()._analyse(ref, page, profile)
        bounds = found.geometry.bounds
        return replace(found, regions=tuple(r for r in found.regions if bounds.intersection(r.bbox) == r.bbox))


class UpsideDownPool(Pool):
    """Every line of the first reading marked as read upside down, as OCR's line classifier marks the lines of a page
    displayed upside down (the text layer this pool reads cannot tell)."""

    def _analyse(self, ref: BlobRef, page: int, profile: str = "primary") -> PageAnalysis:
        found = super()._analyse(ref, page, profile)
        if profile != "primary":
            return found
        return replace(found, lines=tuple(replace(line, upside_down=True) for line in found.lines))


def redact(monkeypatch: pytest.MonkeyPatch, data: bytes, pool: Pool | None = None) -> JobResult:
    (pool or Pool()).install(monkeypatch)
    config = load_config("client", (), {**NO_MODELS, "surrogate": {"allow_development_key": True}})
    with Redactor(config) as redactor:
        return redactor.run(data, keep_held_back=True)


# ------------------------------------------------------------------------------ the pipeline
def test_an_upright_letter_is_still_released(monkeypatch: pytest.MonkeyPatch) -> None:
    result = redact(monkeypatch, LETTER)
    assert result.status is JobStatus.RELEASED
    assert result.manifest["reasons"] == [] and result.pdf is not None


@pytest.mark.parametrize(
    ("name", "make", "reason"),
    [
        ("turned by its /Rotate setting", lambda: turned_by_rotate(LETTER, 90), ROTATED),
        ("turned the other way", lambda: turned_by_rotate(LETTER, 270), ROTATED),
        ("drawn sideways on an upright page", lambda: drawn_sideways(LETTER), CONTENT),
    ],
)
def test_a_sideways_page_holds_the_document_for_review(
    monkeypatch: pytest.MonkeyPatch, name: str, make: Callable[[], bytes], reason: ReasonCode
) -> None:
    result = redact(monkeypatch, make())
    # The hold is the only reason: every verification check passed on the output.
    assert (result.status, result.manifest["reasons"]) == (JobStatus.QUARANTINED, [reason.value]), name
    assert (result.run_record["status"], result.run_record["reasons"]) == ("quarantined", [reason.value])
    # Nothing is released, but the output is still built for the person who reviews it.
    assert result.pdf is None and result.held_back_pdf is not None


@pytest.mark.parametrize(
    ("name", "make", "reason"),
    [
        ("turned upside down by its /Rotate setting", lambda: turned_by_rotate(LETTER, 180), UPSIDE_ROTATED),
        ("scanned upside down on an upright page", lambda: LETTER, UPSIDE_CONTENT),
    ],
)
def test_an_upside_down_page_holds_the_document_for_review(
    monkeypatch: pytest.MonkeyPatch, name: str, make: Callable[[], bytes], reason: ReasonCode
) -> None:
    """Upside-down text runs across the page, so no box shows it: OCR's line classifier does (the pool marks the lines
    as it would).  Before the hold, such a document was released with fakes typeset upright over upside-down lines,
    a locality left unreplaced and an ID number only partly removed."""
    result = redact(monkeypatch, make(), UpsideDownPool())
    assert result.status is JobStatus.QUARANTINED, name
    assert reason.value in result.manifest["reasons"], name
    assert result.pdf is None and result.held_back_pdf is not None


def test_a_rotate_setting_that_turns_sideways_content_upright_is_processed_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The setting corrects a page stored on its side (a landscape scan, or a page turned in a PDF editor): the page
    displays upright, so it is not held."""
    result = redact(monkeypatch, turned_by_rotate(drawn_sideways(LETTER), 90))
    assert (result.status, result.manifest["reasons"]) == (JobStatus.RELEASED, [])


def test_bench_hands_over_a_sideways_document_as_held_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Pool().install(monkeypatch)
    monkeypatch.setattr(bench_env, "candidate_version", lambda root=None: "0.1.0+gtest")
    monkeypatch.delenv("MLREDACT_CONFIG", raising=False)
    monkeypatch.setenv("MLREDACT_MODELS_DIR", str(tmp_path))  # never read: model files are not resolved
    monkeypatch.chdir(tmp_path)
    config, src, out = tmp_path / "no-models.yaml", tmp_path / "input.pdf", tmp_path / "out"
    config.write_text(yaml.safe_dump(NO_MODELS), "utf-8")
    src.write_bytes(drawn_sideways(LETTER))
    res = CliRunner().invoke(app, ["bench", "--input", str(src), "--workdir", str(out), "--config", str(config)])
    assert res.exit_code == 0, res.output
    info = json.loads((out / "run_info.json").read_text("utf-8"))
    assert info["release"] == {"released": False, "checks_failed": [CONTENT.value]}
    assert "held for review" in info["notes"] and "sideways" in info["notes"]
    assert (out / "output.pdf").is_file()
    assert json.loads((out / "quarantine.json").read_text("utf-8"))["reasons"] == [CONTENT.value]
    *logged, last = res.output.splitlines()
    assert last == f"not released ({CONTENT.value}): output.pdf written"
    # The log names the page and the reason (no text): what a reviewer needs to find the page.
    held = [json.loads(r) for r in logged if '"job.page_sideways"' in r]
    assert [(r["page"], r["reason"], r["level"]) for r in held] == [(0, CONTENT.value, "WARNING")]


def test_bench_hands_over_an_upside_down_document_as_held_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    UpsideDownPool().install(monkeypatch)
    monkeypatch.setattr(bench_env, "candidate_version", lambda root=None: "0.1.0+gtest")
    monkeypatch.delenv("MLREDACT_CONFIG", raising=False)
    monkeypatch.setenv("MLREDACT_MODELS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    config, src, out = tmp_path / "no-models.yaml", tmp_path / "input.pdf", tmp_path / "out"
    config.write_text(yaml.safe_dump(NO_MODELS), "utf-8")
    src.write_bytes(turned_by_rotate(LETTER, 180))
    res = CliRunner().invoke(app, ["bench", "--input", str(src), "--workdir", str(out), "--config", str(config)])
    assert res.exit_code == 0, res.output
    info = json.loads((out / "run_info.json").read_text("utf-8"))
    assert info["release"]["released"] is False
    assert UPSIDE_ROTATED.value in info["release"]["checks_failed"]
    assert "held for review" in info["notes"] and "upside down" in info["notes"]
    assert (out / "output.pdf").is_file()
    held = [json.loads(r) for r in res.output.splitlines() if '"job.page_upside_down"' in r]
    assert [(r["page"], r["reason"], r["level"]) for r in held] == [(0, UPSIDE_ROTATED.value, "WARNING")]


def test_the_note_on_unreleased_output_says_a_sideways_page_held_it() -> None:
    for reasons in ([CONTENT.value], [ROTATED.value, "V4_RESIDUAL"], [UPSIDE_ROTATED.value], [UPSIDE_CONTENT.value]):
        note = unreleased_note(reasons)
        assert note.startswith("held for review (") and "sideways or upside down" in note
        assert all(r in note for r in reasons)
    # A step that failed after the build is still reported as such.
    assert unreleased_note([CONTENT.value, "E_OCR_FAILED"]).startswith("a step failed after the output was built")


# ------------------------------------------------------------------------------ the rotation setting
def test_inspection_records_each_pages_rotation_as_the_renderer_applies_it() -> None:
    pdf = pikepdf.new()
    for _ in range(6):
        pdf.add_blank_page()
    pdf.Root.Pages.Rotate = 270  # inherited by every page that sets none
    for i, value in [(1, 90), (2, -90), (3, 450), (4, 45), (5, pikepdf.Name.Ninety)]:
        pdf.pages[i].obj.Rotate = value
    report = inspect_pdf(_saved(pdf), LimitsConfig())
    assert report.rotations == (270, 90, 270, 90, 0, 0)
    assert report.as_manifest().keys().isdisjoint({"rotations"})  # the manifest is unchanged
    assert inspect_pdf(LETTER, LimitsConfig()).rotations == (0,)


# ------------------------------------------------------------------------------ the judgement
def _line(n: int, words: list[tuple[str, BBox]], quad: Quad | None = None) -> Line:
    tokens = tuple(
        Token(token_id(0, n, i), 0, n, i, box, 0.99, 0.99, "test", text) for i, (text, box) in enumerate(words)
    )
    box = tokens[0].bbox
    for t in tokens[1:]:
        box = box.union(t.bbox)
    if quad is None:  # a plain box, as the text layer's lines have
        quad = ((box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1))
    return Line(f"p0000.l{n:04d}", 0, n, box, quad, 0.99, "test", tokens)


def _across(n: int, text: str, y: int = 100) -> Line:
    """A line of ``text`` written left to right, one 30 px tall box per word."""
    words, x = [], 100
    for word in text.split():
        words.append((word, BBox(x, y, x + 18 * len(word), y + 30)))
        x += 18 * len(word) + 12
    return _line(n, words)


def _down(n: int, text: str, x: int = 100) -> Line:
    """The same line turned on its side: its words stacked top to bottom in 30 px wide boxes."""
    words, y = [], 100
    for word in text.split():
        words.append((word, BBox(x, y, x + 30, y + 18 * len(word))))
        y += 18 * len(word) + 12
    return _line(n, words)


def _page(lines: list[Line]) -> PageAnalysis:
    return PageAnalysis(PageGeometry(0, 595.0, 842.0, 150, 1240, 1754), PageKind.BORN_DIGITAL, tuple(lines))


TEXT = "The claimant attended the clinic on the morning of the assessment"  # 56 characters


def test_a_line_runs_along_the_long_side_of_its_box() -> None:
    assert not line_runs_sideways(_across(0, TEXT))
    assert line_runs_sideways(_down(0, TEXT))  # the text layer's words, stacked top to bottom
    # OCR reads a detected box much taller than wide as vertical text, and gives each of its words the whole box: its
    # first edge runs across the text, its long side down it.
    box = BBox(100, 100, 130, 900)
    down: Quad = ((100.0, 100.0), (130.0, 100.0), (130.0, 900.0), (100.0, 900.0))
    words = [(w, box) for w in TEXT.split()]
    assert line_runs_sideways(_line(0, words, down))
    # A box along the text, turned a quarter turn: sideways too.
    turned: Quad = ((100.0, 900.0), (100.0, 100.0), (130.0, 100.0), (130.0, 900.0))
    assert line_runs_sideways(_line(0, words, turned))
    # Tilted less than 45 degrees (a skewed scan): upright.
    tilted: Quad = ((100.0, 300.0), (900.0, 100.0), (907.0, 130.0), (107.0, 330.0))
    assert not line_runs_sideways(_line(0, [("attended", BBox(100, 100, 907, 330))], tilted))
    assert line_runs_sideways(_line(0, [("attended", BBox(100, 100, 130, 400))]))  # one word, on its side
    assert not line_runs_sideways(_line(0, [("Dr", BBox(100, 100, 136, 140))]))  # too square to tell


def test_a_page_is_held_when_most_of_its_text_runs_sideways() -> None:
    [held] = sideways_pages([_page([_down(0, TEXT), _across(1, "Page 1 of 2")])], [0])
    assert (held.page, held.reason) == (0, CONTENT)
    [held] = sideways_pages([_page([_down(0, TEXT)])], [90])
    assert held.reason is ROTATED
    assert sideways_pages([_page([_down(0, TEXT)])], [270])[0].reason is ROTATED
    assert sideways_pages([_page([_down(0, TEXT)])], [180])[0].reason is CONTENT
    assert sideways_pages([_page([_down(0, TEXT)])], [])[0].reason is CONTENT  # no rotation known: upright


def test_an_upright_page_with_a_little_sideways_text_is_not_held() -> None:
    upright = [_across(i, TEXT, y=100 + 50 * i) for i in range(3)]
    assert sideways_pages([_page([*upright, _down(9, TEXT, x=1100)])], [0]) == []  # a margin note
    assert sideways_pages([_page(upright)], [90]) == []  # the setting turns stored-sideways text upright
    short = "Copy for file"  # fewer sideways characters than a line of text
    assert len(short.replace(" ", "")) < MIN_SIDEWAYS_CHARS
    assert sideways_pages([_page([_down(0, short)])], [90]) == []
    assert sideways_pages([_page([])], [90]) == []


def _upside_down(line: Line) -> Line:
    """The line as OCR marks it when its line classifier turned it half a turn before reading it."""
    return replace(line, upside_down=True)


def test_a_page_is_held_when_most_of_its_text_is_upside_down() -> None:
    page = _page([_upside_down(_across(0, TEXT)), _across(1, "Page 1 of 2", y=200)])
    [held] = sideways_pages([page], [0])
    assert (held.page, held.reason, held.upside_down) == (0, UPSIDE_CONTENT, True)
    assert sideways_pages([page], [180])[0].reason is UPSIDE_ROTATED  # the setting turns it upside down
    assert sideways_pages([page], [90])[0].reason is UPSIDE_CONTENT
    assert sideways_pages([page], [])[0].reason is UPSIDE_CONTENT  # no rotation known: upright
    # A sideways line that OCR also turned half a turn (text reading top to bottom) counts as sideways only.
    [held] = sideways_pages([_page([_upside_down(_down(0, TEXT))])], [270])
    assert (held.reason, held.upside_down) == (ROTATED, False)


def test_an_upright_page_with_a_little_upside_down_text_is_not_held() -> None:
    upright = [_across(i, TEXT, y=100 + 50 * i) for i in range(3)]
    assert sideways_pages([_page([*upright, _upside_down(_across(9, TEXT, y=1000))])], [0]) == []  # a stamp
    short = "Copy for file"
    assert sideways_pages([_page([_upside_down(_across(0, short))])], [180]) == []
    assert sideways_pages([_page(upright)], [180]) == []  # the setting turns stored-upside-down text upright


def test_ocr_marks_a_line_its_classifier_read_upside_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mark comes from OCR's own line classifier: a line it turns half a turn before reading is marked, the others
    are not.  The detector, classifier and recogniser are stand-ins; the engine's own code builds the lines."""

    class Found:
        def __init__(self, labels: list[str]) -> None:
            self.labels = labels

        def __call__(self, imgs: list[object]) -> object:
            label = self.labels.pop(0)
            return type("Cls", (), {"cls_res": [(label, 0.99)], "img_list": imgs})()

    boxes = [[[10, 10], [210, 10], [210, 40], [10, 40]], [[10, 60], [210, 60], [210, 90], [10, 90]]]
    engine = object.__new__(ppocr.PPOCREngine)
    engine._cfg = load_config("client", (), NO_MODELS).ocr.engine_a
    engine._det = lambda _img: type("Det", (), {"boxes": ppocr.np.asarray(boxes, dtype="float32")})()
    engine._cls = Found(["180", "0"])
    chars = [DecodedChar(c, 10.0 * i, 0.99) for i, c in enumerate("Sample")]
    monkeypatch.setattr(engine, "_recognise", lambda _img: (chars, 1.0), raising=False)
    lines = engine.ocr_page(ppocr.np.full((120, 240, 3), 255, dtype=ppocr.np.uint8), 0)
    assert [line.upside_down for line in lines] == [True, False]
