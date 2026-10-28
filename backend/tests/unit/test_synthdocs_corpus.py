"""The synthetic corpus's answer keys against the pipeline, without models.

The corpus is nine scanned documents, scored with OCR (``synthdocs.score``, which needs the models).  Here each
document is typeset as usual (the page images are drawn and dropped), the text it draws is taken as a perfect reader
would hand it over and written as a born-digital PDF with the same positions, and the real runner redacts that in this
process (page analysis reads the text layer, no OCR; a detector stands in for the NER models and finds the people and
organisations the key lists).  The scorer's own value scoring then compares the text of the output with the key.

What the key says about places: a suburb or postcode inside an address takes the street_address action, so it must
not be readable in the output; a suburb on its own is kept, so it must be.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pikepdf
import pytest
from synthdocs import corpus
from synthdocs.corpus_docs import DOCS
from synthdocs.score import _score_values, _Text
from test_bench_runner import NO_MODELS, InProcessPool

import mlredact.pipeline.runner as runner_module
from mlredact.config.loader import load_config
from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate
from mlredact.pipeline.runner import JobResult, Redactor
from mlredact.text.views import TextView

NER_KINDS = {"person": EntityType.PERSON, "organisation": EntityType.ORGANISATION}
SPECS = {spec.slug: spec for spec in DOCS}


@dataclass(frozen=True)
class Drawn:
    """A piece of text a document draws: its page, position and size in points."""

    page: int
    x: float
    y: float
    size: float
    text: str


class KeyedNer:
    """Stands in for the NER models: finds the people and organisations the answer key lists, wherever they are."""

    detector_id = "ner.key"
    version = "1"

    def __init__(self, values: list[tuple[EntityType, str]]) -> None:
        self.patterns = [(t, re.compile(r"\s+".join(map(re.escape, v.split())))) for t, v in values]

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        return [
            candidate(self, kind, ViewKind.LINE, m.start(), m.end(), "key", EvidenceStrength.STRONG, 0.9)
            for kind, pattern in self.patterns
            for m in pattern.finditer(text)
        ]


def typeset(slug: str) -> tuple[dict[str, Any], list[Drawn]]:
    """The answer key of a document and the text it draws."""
    spec = SPECS[slug]
    key = corpus._Key()
    drawn: list[Drawn] = []
    scale = spec.scan.dpi / 72.0
    put, hand, init = corpus._Typesetter.put, corpus._hand, corpus._Typesetter.__init__
    page = [0]

    def recording_put(self: Any, x: float, y: float, text: str, size: float, *args: Any, **kwargs: Any) -> None:
        drawn.append(Drawn(self.index, x, y, size, text))
        put(self, x, y, text, size, *args, **kwargs)

    def recording_hand(draw: Any, xy: tuple[float, float], text: str, px: float, *args: Any) -> Any:
        drawn.append(Drawn(page[0], xy[0] / scale, xy[1] / scale, px / scale, text))
        return hand(draw, xy, text, px, *args)

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        page[0] = self.index

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(corpus._Typesetter, "put", recording_put)
        mp.setattr(corpus._Typesetter, "__init__", recording_init)
        mp.setattr(corpus, "_hand", recording_hand)
        corpus._typeset(spec, key)
    built = corpus.Built(spec, b"", key.items, key.regions)
    return json.loads(corpus.answer_json(built)), drawn


def text_pdf(slug: str, drawn: list[Drawn]) -> bytes:
    """A born-digital PDF with the drawn text at its positions (pages of the typeset size)."""
    pdf = pikepdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
            Encoding=pikepdf.Name.WinAnsiEncoding,
        )
    )
    for index, spec_page in enumerate(SPECS[slug].pages):
        width, height = corpus.LANDSCAPE if spec_page.landscape else corpus.PORTRAIT
        ops = ["BT"]
        for d in sorted((d for d in drawn if d.page == index), key=lambda d: (d.y, d.x)):
            escaped = d.text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"/F1 {d.size:.2f} Tf 1 0 0 1 {d.x:.2f} {height - d.y - 0.8 * d.size:.2f} Tm ({escaped}) Tj")
        ops.append("ET")
        pdf.pages.append(
            pikepdf.Page(
                pikepdf.Dictionary(
                    Type=pikepdf.Name.Page,
                    MediaBox=[0, 0, width, height],
                    Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)),
                    Contents=pikepdf.Stream(pdf, "\n".join(ops).encode("cp1252", "replace")),
                )
            )
        )
    out = io.BytesIO()
    pdf.save(out, deterministic_id=True)
    return out.getvalue()


ROW_TOLERANCE_PT = 4.0
TEXT_OPERATOR = re.compile(rb"1 0 0 1 (-?[\d.]+) (-?[\d.]+) Tm \(((?:[^()\\]|\\.)*)\) Tj")
ESCAPED = re.compile(rb"\\(.)")


def layer_text(pdf: bytes) -> str:
    """The text layer as a reader sees it: the strings the builder wrote, a row at a time in reading order.  (Read back
    by a viewer the layer merges words and repeats some; the strings themselves are what the tool wrote.)"""
    rows: list[str] = []
    with pikepdf.open(io.BytesIO(pdf)) as doc:
        for page in doc.pages:
            found = [
                (-float(y), float(x), ESCAPED.sub(rb"\1", raw).decode("cp1252"))
                for x, y, raw in TEXT_OPERATOR.findall(page.obj.Contents.read_bytes())
            ]
            row: list[tuple[float, str]] = []
            top = 0.0
            for y, x, text in sorted(found):
                if row and y - top > ROW_TOLERANCE_PT:
                    rows.append(" ".join(t for _x, t in sorted(row)))
                    row = []
                if not row:
                    top = y
                row.append((x, text))
            if row:
                rows.append(" ".join(t for _x, t in sorted(row)))
    return "\n".join(rows)


@dataclass(frozen=True)
class Scored:
    result: JobResult
    values: list[dict[str, Any]]


def run_document(slug: str) -> Scored:
    key, drawn = typeset(slug)
    source = text_pdf(slug, drawn)
    ner = KeyedNer(
        [(NER_KINDS[v["kind"]], v["text"]) for v in key["values"] if v["kind"] in NER_KINDS and v["action"] != "keep"]
    )
    pool = InProcessPool()
    config = load_config("client", (), {**NO_MODELS, "surrogate": {"allow_development_key": True}})
    default_detectors = runner_module.default_detectors
    with pytest.MonkeyPatch.context() as mp:
        pool.install(mp)
        mp.setattr(runner_module, "default_detectors", lambda: (*default_detectors(), ner))
        with Redactor(config) as redactor:
            result = redactor.run(source, keep_held_back=True)
    output = result.pdf or result.held_back_pdf
    assert output is not None
    original, redacted = _Text.of(layer_text(source)), _Text.of(layer_text(output))
    return Scored(result, _score_values(key, {"text": original}, {"text": redacted}, redacted))


_RUNS: dict[str, Scored] = {}


def scored(slug: str) -> Scored:
    if slug not in _RUNS:
        _RUNS[slug] = run_document(slug)
    return _RUNS[slug]


# ------------------------------------------------------------------------------------------------ the keys
@pytest.mark.parametrize("slug", sorted(SPECS))
def test_a_place_inside_an_address_takes_the_street_address_action_and_one_on_its_own_stays_kept(slug: str) -> None:
    key, _drawn = typeset(slug)
    address = [v for v in key["values"] if v["in_address"]]
    alone = [v for v in key["values"] if v["kind"] in ("locality", "postcode") and not v["in_address"]]
    assert address
    assert {v["action"] for v in address} == {"surrogate"}
    assert {v["kind"] for v in address} <= {"locality", "postcode"}
    assert {v["action"] for v in alone} <= {"keep"}


# Values of an address that the pipeline still leaves readable, by document.  A numbered street name split over a
# wrapped line of running text ("She lives at 23 Lomandra" / "Street, Chermside QLD 4032.") is read as a place alone.
KNOWN_LEAKS = {"low-03-solicitor-instruction-letter": ["23 Lomandra Street"]}


@pytest.mark.parametrize("slug", sorted(SPECS))
def test_no_address_part_is_readable_and_no_place_on_its_own_is_damaged(slug: str) -> None:
    run = scored(slug)
    address = [v for v in run.values if v["in_address"] or v["kind"] == "street_address"]
    alone = [v for v in run.values if v["kind"] in ("locality", "postcode") and not v["in_address"]]
    assert [v["text"] for v in address if v["status"] == "leaked"] == KNOWN_LEAKS.get(slug, [])
    assert {v["status"] for v in address} <= {"replaced", "leaked", "not readable in the original"}
    assert [v["text"] for v in alone if v["status"] in ("removed", "partly removed")] == []
    assert {v["status"] for v in alone} <= {"kept", "not readable in the original"}


@pytest.mark.parametrize("slug", sorted(SPECS))
def test_the_address_checks_of_the_run_pass(slug: str) -> None:
    """The run may be held back for other reasons (the stand-in for the NER models is not the models); the checks on
    addresses must pass."""
    checks = {c["check"]: c for c in scored(slug).result.manifest["verification"]["checks"]}
    assert checks["V2_text"]["passed"] and checks["V2_text"]["metrics"]["address_hits"] == 0
    assert checks["V4_residual"]["passed"]
    assert {k: v for k, v in checks["V5_surrogate"]["metrics"].items() if k.startswith("address_") and v} == {}
