"""Injected stage inputs of the benchmark's oracle modes: reading the files, reading order and lines, points to
pixels, and tool 2's lines from injected tokens.

The files are written here in the benchmark's own format (no benchmark import).  All text is fictitious.
"""

from __future__ import annotations

import json
import pickle
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlredact.bench.artifacts import display_poly
from mlredact.bench.inject import MODE_FILES, InjectedFile, Injection, TextArtifact, load_injection
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import PageGeometry
from mlredact.pipeline.injection import INJECTED_ENGINE, InjectedPage, InjectedToken, injected_lines, pt_box
from mlredact.render.plan import rotated_token_ids
from mlredact.security.exceptions import sanitize

CANARY = "Zelphinia Quarrington"
# A4 at 150 dpi.
GEOM = PageGeometry(0, 595.276, 841.89, 150, 1240, 1754)
Box = tuple[float, float, float, float]


# ------------------------------------------------------------------------------- injected files
def poly(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def token(
    tid: str, text: str, box: Box, *, line: str | None = None, order: int | None = None, conf: float | None = None
) -> dict[str, Any]:
    return {"id": tid, "text": text, "poly": poly(*box), "conf": conf, "line": line, "order": order}


def text_doc(*pages: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """``rb.text/0.1`` with every key, as the benchmark writes it."""
    return {
        "schema": "rb.text/0.1",
        "frame": "display_pt",
        "dpi": None,
        "source": "text_layer",
        "pages": [{"index": i, "tokens": t, "detected_rotation": None, "skew_deg": None} for i, t in pages],
    }


def full_docs() -> dict[str, Any]:
    """All four injected files, every key of every schema set."""
    text = text_doc(
        (
            0,
            [
                token("w0", "Ms", (60, 100, 75, 111), line="l0", order=0, conf=0.97),
                token("w1", "Jane", (78, 100, 100, 111), line="l0", order=1),
                token("w2", "Citizen", (103, 100, 140, 111), line="l0", order=2),
            ],
        )
    )
    text["pages"][0].update(detected_rotation=0, skew_deg=0.4)
    mentions = {
        "schema": "rb.mentions/0.1",
        "frame": "display_pt",
        "dpi": None,
        "page_texts": {"0": "Ms Jane Citizen"},
        "mentions": [
            {
                "id": "m1",
                "label": "PERSON_NAME",
                "text": "Jane Citizen",
                "tokens": ["w1", "w2"],
                "regions": [{"page": 0, "poly": poly(78, 100, 140, 111)}],
                "span": {"page": 0, "start": 3, "end": 15},
                "conf": 0.9,
            },
            {
                "id": "m2",
                "label": "SIGNATURE",
                "text": None,
                "tokens": None,
                "regions": [{"page": 0, "poly": poly(60, 300, 200, 340)}],
                "span": None,
                "conf": None,
            },
        ],
    }
    entities = {
        "schema": "rb.entities/0.1",
        "clusters": [{"id": "e1", "mentions": ["m1", "m2"], "attributes": {"role": "claimant"}}],
    }
    plan = {
        "schema": "rb.plan/0.1",
        "frame": "display_pt",
        "dpi": None,
        "regions": [
            {
                "id": "m1-0",
                "page": 0,
                "poly": poly(78, 100, 140, 111),
                "action": "replace",
                "replacement": "Ann Example",
                "mentions": ["m1"],
            },
            {"id": "m2-0", "page": 0, "poly": poly(60, 300, 200, 340), "action": "remove", "replacement": None},
        ],
    }
    return {"text.json": text, "mentions.json": mentions, "entities.json": entities, "plan.json": plan}


def write(folder: Path, files: dict[str, Any]) -> Path:
    """Each file as JSON (``bytes`` as they are)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        data = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
        (folder / name).write_bytes(data)
    return folder


def read_text(folder: Path, *pages: tuple[int, list[dict[str, Any]]]) -> Injection:
    return load_injection("oracle_text", write(folder, {"text.json": text_doc(*pages)}))


def numbered(injection: Injection, index: int = 0) -> list[tuple[int, int, str]]:
    """(line, word, text) of a page's tokens, in reading order."""
    return [(t.line_no, t.word_no, t.text) for t in injection.page(index).tokens]


# ------------------------------------------------------------------------------------- parsing
def test_every_key_of_the_benchmark_schemas_is_read(tmp_path: Path) -> None:
    injection = load_injection("oracle_regions", write(tmp_path, full_docs()))
    assert (injection.mode, injection.text_source) == ("oracle_regions", "text_layer")
    assert [(t.text, t.confidence) for t in injection.page(0).tokens] == [("Ms", 0.97), ("Jane", 1.0), ("Citizen", 1.0)]
    assert injection.mentions is not None and injection.clusters is not None and injection.plan is not None
    person, signature = injection.mentions
    assert (person.id, person.label, person.text, person.tokens, person.conf) == (
        "m1",
        "PERSON_NAME",
        "Jane Citizen",
        ["w1", "w2"],
        0.9,
    )
    assert person.span is not None and (person.span.page, person.span.start, person.span.end) == (0, 3, 15)
    assert signature.regions is not None and signature.regions[0].poly == [(60, 300), (200, 300), (200, 340), (60, 340)]
    assert [(c.id, c.mentions, c.attributes) for c in injection.clusters] == [
        ("e1", ["m1", "m2"], {"role": "claimant"})
    ]
    assert [(r.id, r.action, r.replacement, r.mentions) for r in injection.plan] == [
        ("m1-0", "replace", "Ann Example", ["m1"]),
        ("m2-0", "remove", None, []),
    ]
    assert injection.counts["page_texts_differ"] == 0


def _token_of(docs: dict[str, Any], i: int) -> dict[str, Any]:
    token: dict[str, Any] = docs["text.json"]["pages"][0]["tokens"][i]
    return token


INVALID: dict[str, tuple[InjectedFile, Callable[[dict[str, Any]], object]]] = {
    "an unknown key": (InjectedFile.TEXT, lambda d: _token_of(d, 0).update(font="Helvetica")),
    "another frame": (InjectedFile.MENTIONS, lambda d: d["mentions.json"].update(frame="pdf_user")),
    "a dpi of 0": (InjectedFile.PLAN, lambda d: d["plan.json"].update(dpi=0)),
    "a wrong schema id": (InjectedFile.TEXT, lambda d: d["text.json"].update(schema="rb.text/0.2")),
    "a frame in entities.json": (InjectedFile.ENTITIES, lambda d: d["entities.json"].update(frame="display_pt")),
    "a NaN coordinate": (InjectedFile.TEXT, lambda d: _token_of(d, 1)["poly"][0].__setitem__(0, float("nan"))),
    "an infinite coordinate": (
        InjectedFile.PLAN,
        lambda d: d["plan.json"]["regions"][0]["poly"][2].__setitem__(1, 1e400),
    ),
    # Finite, but no page is that large: scaled to pixels, it would not be a number any more.
    "a coordinate too large to scale": (InjectedFile.TEXT, lambda d: _token_of(d, 0)["poly"][1].__setitem__(0, 1e308)),
    "a coordinate far off the page": (
        InjectedFile.MENTIONS,
        lambda d: d["mentions.json"]["mentions"][1]["regions"][0]["poly"][0].__setitem__(1, -2e6),
    ),
    "a 2-point polygon": (
        InjectedFile.MENTIONS,
        lambda d: d["mentions.json"]["mentions"][1]["regions"][0].update(poly=[[0, 0], [9, 9]]),
    ),
    "a mention without an anchor": (
        InjectedFile.MENTIONS,
        lambda d: d["mentions.json"]["mentions"][0].update(tokens=[], regions=None, span=None),
    ),
    "a cluster without members": (
        InjectedFile.ENTITIES,
        lambda d: d["entities.json"]["clusters"][0].update(mentions=[]),
    ),
    "a confidence above 1": (InjectedFile.TEXT, lambda d: _token_of(d, 2).update(conf=1.5)),
    "a negative order": (InjectedFile.TEXT, lambda d: _token_of(d, 2).update(order=-1)),
    "an unknown action": (InjectedFile.PLAN, lambda d: d["plan.json"]["regions"][1].update(action="blur")),
    "a missing file": (InjectedFile.ENTITIES, lambda d: d.pop("entities.json")),
    "not JSON": (InjectedFile.MENTIONS, lambda d: d.update({"mentions.json": b"{'mentions': []}"})),
    "not UTF-8": (InjectedFile.TEXT, lambda d: d.update({"text.json": b'{"source": "\xff"}'})),
    "not an object": (InjectedFile.PLAN, lambda d: d.update({"plan.json": b"[]"})),
    "nested too deeply": (
        InjectedFile.ENTITIES,
        lambda d: d.update(
            {"entities.json": b'{"schema": "rb.entities/0.1", "clusters": ' + b"[" * 200_000 + b"]" * 200_000 + b"}"}
        ),
    ),
}


@pytest.mark.parametrize("case", INVALID)
def test_invalid_files_are_refused(tmp_path: Path, case: str) -> None:
    name, change = INVALID[case]
    docs = full_docs()
    change(docs)
    with pytest.raises(EnvironmentErrorMl) as caught:
        load_injection("oracle_regions", write(tmp_path, docs))
    err = caught.value
    assert err.code is ReasonCode.CONFIG_INVALID
    assert err.params["inject"] is True and err.params["file"] is name
    errors = err.params["errors"]
    assert isinstance(errors, int) and errors >= 1


def test_a_dpi_given_with_display_points_is_not_used(tmp_path: Path) -> None:
    """The benchmark's schemas allow a dpi in any frame (another candidate's stage files can carry one into a stage
    swap); display points do not need it."""
    docs = full_docs()
    for name in ("text.json", "mentions.json", "plan.json"):
        docs[name]["dpi"] = 300
    with_dpi = load_injection("oracle_regions", write(tmp_path / "dpi", docs))
    assert with_dpi == load_injection("oracle_regions", write(tmp_path / "none", full_docs()))


def test_errors_never_repeat_injected_content(tmp_path: Path) -> None:
    docs = full_docs()
    tok = _token_of(docs, 1)
    tok.update(text=CANARY, conf=CANARY, line=1)
    tok[CANARY] = CANARY
    docs["text.json"]["source"] = CANARY
    with pytest.raises(ValidationError) as raw:  # what the validator itself says: the input, repeated
        TextArtifact.model_validate(docs["text.json"])
    assert CANARY in str(raw.value)
    with pytest.raises(EnvironmentErrorMl) as caught:
        load_injection("oracle_text", write(tmp_path, docs))
    err, info = caught.value, sanitize(caught.value)
    assert err.params["errors"] == raw.value.error_count()
    for shown in (str(err), repr(err), repr(info), str(info.as_fields()), str(err.params)):
        assert CANARY not in shown


def test_each_mode_reads_only_its_own_files(tmp_path: Path) -> None:
    folder = write(tmp_path / "all", {**full_docs(), "plan.json": b"\xff not a plan"})
    for mode in ("oracle_text", "oracle_mentions", "oracle_entities"):
        injection = load_injection(mode, folder)
        assert (injection.mentions is None, injection.clusters is None, injection.plan) == (
            mode == "oracle_text",
            mode != "oracle_entities",
            None,
        )
    with pytest.raises(EnvironmentErrorMl) as caught:
        load_injection("oracle_regions", folder)
    assert caught.value.params["file"] is InjectedFile.PLAN
    only_text = write(tmp_path / "text", {"text.json": full_docs()["text.json"]})
    with pytest.raises(EnvironmentErrorMl) as caught:
        load_injection("oracle_mentions", only_text)
    assert caught.value.params["file"] is InjectedFile.MENTIONS
    with pytest.raises(ValueError, match="no stage inputs"):
        load_injection("e2e", folder)
    assert {mode: [f.kind for f in files] for mode, files in MODE_FILES.items()} == {
        "e2e": [],
        "oracle_text": ["text"],
        "oracle_mentions": ["text", "mentions"],
        "oracle_entities": ["text", "mentions", "entities"],
        "oracle_regions": ["text", "mentions", "entities", "regions"],
    }


def test_what_was_read_shows_no_text(tmp_path: Path) -> None:
    docs = full_docs()
    _token_of(docs, 1)["text"] = CANARY
    docs["mentions.json"]["mentions"][0]["text"] = CANARY
    docs["mentions.json"]["page_texts"] = {"0": CANARY}
    docs["entities.json"]["clusters"][0]["attributes"] = {"name": CANARY}
    docs["plan.json"]["regions"][0]["replacement"] = CANARY
    injection = load_injection("oracle_regions", write(tmp_path, docs))
    shown = repr(injection) + repr(injection.page(0)) + repr(injection.page(0).tokens)
    shown += repr(injection.mentions) + repr(injection.clusters) + repr(injection.plan)
    assert CANARY not in shown
    assert CANARY in injection.page_texts[0]  # it was read


def test_mentions_clusters_and_plan_regions_are_unique(tmp_path: Path) -> None:
    docs = full_docs()
    mentions = docs["mentions.json"]["mentions"]
    mentions.append({**mentions[0], "label": "ORGANISATION"})
    docs["mentions.json"]["page_texts"] = {"0": "Ms  Jane Citizen", "3": ""}
    docs["entities.json"]["clusters"] += [{"id": "e2", "mentions": ["m2"]}, {"id": "e1", "mentions": ["m2", "m3"]}]
    docs["plan.json"]["regions"].append({**docs["plan.json"]["regions"][1], "action": "replace"})
    injection = load_injection("oracle_regions", write(tmp_path, docs))
    assert injection.mentions is not None and injection.clusters is not None and injection.plan is not None
    assert [(m.id, m.label) for m in injection.mentions] == [("m1", "PERSON_NAME"), ("m2", "SIGNATURE")]
    assert [(c.id, c.mentions) for c in injection.clusters] == [("e1", ["m1", "m2", "m3"]), ("e2", ["m2"])]
    assert [(r.id, r.action) for r in injection.plan] == [("m1-0", "replace"), ("m2-0", "remove")]
    counts = injection.counts
    assert (counts["mentions"], counts["mentions_duplicate"]) == (3, 1)
    assert (counts["clusters"], counts["clusters_merged"]) == (3, 1)
    assert (counts["plan_regions"], counts["plan_regions_duplicate"]) == (3, 1)
    assert counts["page_texts_differ"] == 1  # page 0's differs; page 3 is not in text.json and is given empty


# ------------------------------------------------------------------------------ reading order
def at(x: float, y: float = 100, w: float = 20, h: float = 10) -> Box:
    return (x, y, x + w, y + h)


def test_reading_order_follows_order_then_file_position(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (
            0,
            [
                token("a", "A", at(50), line="l", order=2),
                token("b", "B", at(0), line="l", order=0),
                token("c", "C", at(75), line="l"),
                token("d", "D", at(25), line="l", order=0),
                token("e", "E", at(100), line="l"),
            ],
        ),
    )
    assert numbered(injection) == [(0, 0, "B"), (0, 1, "D"), (0, 2, "A"), (0, 3, "C"), (0, 4, "E")]


def test_a_line_is_a_run_of_one_line_value(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (
            0,
            [
                token("a", "A", at(0), line="l1", order=0),
                token("b", "B", at(25), line="l1", order=1),
                token("c", "C", at(50), line="l2", order=2),
                token("d", "D", at(75), line="l1", order=3),  # l1 comes back: a new line
                # A sideways line: one line value, the tokens one below the other.
                token("e", "E", (300, 100, 310, 130), line="v", order=4),
                token("f", "F", (300, 135, 310, 165), line="v", order=5),
            ],
        ),
    )
    assert numbered(injection) == [(0, 0, "A"), (0, 1, "B"), (1, 0, "C"), (2, 0, "D"), (3, 0, "E"), (3, 1, "F")]
    assert injection.counts["lines"] == 4


def test_a_token_without_a_line_joins_its_row_left_to_right(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (
            0,
            [
                token("a", "A", (0, 100, 20, 110), line="l1", order=0),
                token("b", "B", (25, 101, 45, 111), order=1),  # same row, to the right: joins
                token("c", "C", (10, 100, 30, 110), order=2),  # starts left of B: a new line
                token("d", "D", (40, 130, 60, 140), order=3),  # another row: a new line
                token("e", "E", (65, 131, 85, 141), order=4),  # joins D
                token("f", "F", (90, 131, 110, 141), line="l1", order=5),  # l1 comes back: a new line
            ],
        ),
    )
    assert numbered(injection) == [(0, 0, "A"), (0, 1, "B"), (1, 0, "C"), (2, 0, "D"), (2, 1, "E"), (3, 0, "F")]


def test_a_wide_gap_in_a_row_splits_the_line(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (
            0,
            [
                token("a", "A", (0, 100, 20, 110), line="l1"),
                token("b", "B", (51, 100, 70, 110), line="l1"),  # 31 apart: more than 3 heights
                token("c", "C", (75, 100, 95, 110), line="l1"),
                token("d", "D", (125, 100, 145, 110), line="l1"),  # 30 apart: still the line
            ],
        ),
    )
    assert numbered(injection) == [(0, 0, "A"), (1, 0, "B"), (1, 1, "C"), (1, 2, "D")]
    assert (injection.counts["lines"], injection.counts["lines_split"]) == (2, 1)


def test_empty_and_repeated_tokens_are_dropped_but_spans_still_count_them(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (
            0,
            [
                token("w0", "Mr", at(0), line="l"),
                token("w1", "", at(25), line="l"),
                token("w0", "Again", at(25), line="l"),
                token("w2", "  ", at(25), line="l"),
                token("w3", "Smith", at(25), line="l"),
            ],
        ),
    )
    assert numbered(injection) == [(0, 0, "Mr"), (0, 1, "Smith")]
    counts = injection.counts
    assert (counts["tokens"], counts["tokens_empty"], counts["tokens_duplicate"]) == (5, 2, 1)
    assert dict(injection.token_ids) == {"w0": "p0000.l0000.w000", "w3": "p0000.l0000.w001"}
    # page_texts: every token of the file, in file order, joined by single spaces.
    assert injection.page_texts[0] == "Mr  Again    Smith"
    assert injection.span_token_ids(0, 0, 2) == ("p0000.l0000.w000",)
    assert injection.span_token_ids(0, 4, 9) == ()  # the repeated id's text
    assert injection.span_token_ids(0, 13, 19) == ("p0000.l0000.w001",)
    assert injection.span_token_ids(0, 0, 99) == ("p0000.l0000.w000", "p0000.l0000.w001")
    assert injection.span_token_ids(1, 0, 99) == ()


def test_page_entries_are_merged_and_the_first_page_keeps_an_id(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (1, [token("x", "One", at(0), line="l")]),
        (0, [token("x", "Zero", at(0), line="l"), token("y", "Two", at(25), line="l")]),
        (1, [token("z", "Three", at(25), line="l")]),
    )
    assert sorted(injection.pages) == [0, 1] and injection.counts["pages_merged"] == 1
    assert numbered(injection, 0) == [(0, 0, "Zero"), (0, 1, "Two")]
    assert numbered(injection, 1) == [(0, 0, "Three")]
    assert injection.token_ids["x"] == "p0000.l0000.w000" and injection.counts["tokens_duplicate"] == 1
    assert injection.page_texts[1] == "One Three"
    assert injection.page(7) == InjectedPage(7)


# ------------------------------------------------------------------------------ points -> pixels
def test_points_round_outward_and_written_points_convert_back() -> None:
    box = pt_box([(72, 30.325), (121.005, 30.325), (121.005, 42.691), (72, 42.691)], GEOM)
    assert box == BBox(150, 63, 253, 89)
    written = display_poly(box, GEOM)
    assert written == [[72.0, 30.24], [121.44, 30.24], [121.44, 42.72], [72.0, 42.72]]
    assert pt_box([(x, y) for x, y in written], GEOM) == box


@pytest.mark.parametrize("dpi", [150, 200, 300, 207])
def test_every_written_box_converts_back_to_itself(dpi: int) -> None:
    """207 dpi: a large page rendered below the configured resolution (``effective_dpi``)."""
    geometry = PageGeometry(0, 595.276, 841.89, dpi, round(595.276 * dpi / 72), round(841.89 * dpi / 72))
    rng = random.Random(dpi)
    for _ in range(400):
        x0, y0 = rng.randrange(geometry.width_px - 1), rng.randrange(geometry.height_px - 1)
        x1, y1 = rng.randrange(x0 + 1, geometry.width_px + 1), rng.randrange(y0 + 1, geometry.height_px + 1)
        box = BBox(x0, y0, x1, y1)
        assert pt_box([(x, y) for x, y in display_poly(box, geometry)], geometry) == box


def test_boxes_are_clipped_to_the_page_and_never_empty() -> None:
    assert pt_box([(-10, -10), (20, -10), (20, 20)], GEOM) == BBox(0, 0, 42, 42)
    assert pt_box([(590, 830), (700, 830), (700, 900)], GEOM) == BBox(1229, 1729, 1240, 1754)
    assert pt_box([(72, 72)] * 3, GEOM) == BBox(150, 150, 151, 151)
    assert pt_box([(1000, 1000), (1100, 1000), (1100, 1100)], GEOM) == BBox(1239, 1753, 1240, 1754)


# --------------------------------------------------------------------------------- tool 2 lines
def quad(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def test_injected_tokens_become_tool_2_lines(tmp_path: Path) -> None:
    injection = read_text(
        tmp_path,
        (
            3,
            [
                token("a", "Ms", (72, 100, 96, 110), line="l", conf=0.8),
                token("b", "Jane", (100, 101, 130, 111), line="l"),
                token("c", "Citizen", (72, 130, 120, 140), line="m", conf=0.5),
            ],
        ),
    )
    geometry = PageGeometry(3, 595.276, 841.89, 150, 1240, 1754)
    first, second = injected_lines(injection.page(3), geometry)
    assert (first.line_id, first.page, first.line_no, second.line_id) == ("p0003.l0000", 3, 0, "p0003.l0001")
    assert [
        (t.token_id, t.text, t.confidence, t.min_confidence, t.engine, t.alternates)
        for t in (*first.tokens, *second.tokens)
    ] == [
        ("p0003.l0000.w000", "Ms", 0.8, 0.8, INJECTED_ENGINE, ()),
        ("p0003.l0000.w001", "Jane", 1.0, 1.0, INJECTED_ENGINE, ()),
        ("p0003.l0001.w000", "Citizen", 0.5, 0.5, INJECTED_ENGINE, ()),
    ]
    assert first.tokens[0].bbox == pt_box(quad(72, 100, 96, 110), geometry)
    assert first.bbox == first.tokens[0].bbox.union(first.tokens[1].bbox)
    assert (first.confidence, first.engine, second.confidence) == (0.9, INJECTED_ENGINE, 0.5)
    assert set(injection.token_ids.values()) == {t.token_id for line in (first, second) for t in line.tokens}
    # The first token's left edge and the last token's right edge, in pixels.
    corners = [72, 100, 130, 101, 130, 111, 72, 110]
    assert [v for point in first.quad for v in point] == pytest.approx([v * 150 / 72 for v in corners])


def test_a_polygon_that_is_not_a_quadrilateral_gives_the_box_corners() -> None:
    page = InjectedPage(
        0,
        (
            InjectedToken(0, 0, quad(72, 100, 96, 110), 1.0, "Ms"),
            InjectedToken(0, 1, ((100, 101), (130, 101), (115, 111)), 1.0, "Jane"),
        ),
    )
    (line,) = injected_lines(page, GEOM)
    b = line.bbox
    assert line.quad == ((b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1))


def test_sideways_lines_stay_sideways_for_planning() -> None:
    page = InjectedPage(
        0,
        (
            InjectedToken(0, 0, quad(60, 100, 100, 110), 1.0, "Upright"),
            # Rotated +90 degrees: the text runs down the page.  -90 degrees: it runs up.
            InjectedToken(1, 0, ((200, 100), (200, 140), (190, 140), (190, 100)), 1.0, "Down"),
            InjectedToken(2, 0, ((300, 140), (300, 100), (310, 100), (310, 140)), 1.0, "Up"),
        ),
    )
    upright, down, up = injected_lines(page, GEOM)
    assert rotated_token_ids([upright, down, up]) == {down.tokens[0].token_id, up.tokens[0].token_id}


def test_numbering_must_be_unique_and_the_page_must_match() -> None:
    box = quad(60, 100, 100, 110)
    with pytest.raises(ValueError, match="rendered page"):
        injected_lines(InjectedPage(1, (InjectedToken(0, 0, box, 1.0, "A"),)), GEOM)
    for tokens in (
        (InjectedToken(0, 0, box, 1.0, "A"), InjectedToken(1, 0, box, 1.0, "B"), InjectedToken(0, 1, box, 1.0, "C")),
        (InjectedToken(0, 0, box, 1.0, "A"), InjectedToken(0, 0, box, 1.0, "B")),
    ):
        with pytest.raises(ValueError, match="unique"):
            injected_lines(InjectedPage(0, tokens), GEOM)


def test_injected_pages_travel_to_workers_and_show_no_text() -> None:
    page = InjectedPage(0, (InjectedToken(0, 0, ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0)), 1.0, CANARY),))
    assert pickle.loads(pickle.dumps(page)) == page
    assert CANARY not in repr(page) + repr(page.tokens[0])
