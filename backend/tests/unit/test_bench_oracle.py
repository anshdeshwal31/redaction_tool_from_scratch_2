"""``mlredact bench`` in the oracle modes (``oracle_text``, ``oracle_mentions``, ``oracle_entities``,
``oracle_regions``): the real runner, fed the benchmark's injected stage inputs for the synthetic letter.

Jobs run as in tests/unit/test_bench_runner.py: its in-process worker pool and its configuration without models, so
the verification re-read reads the output's own text layer.  The injected files are written in the benchmark's own
format (tests/inject_support.py).  All text is fictitious.
"""

from __future__ import annotations

import functools
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result
from synthdocs import SynthDoc, make_ime_letter
from synthdocs.generator import medicare_number
from test_bench_runner import LETTER, NO_MODELS, RESIDUAL, InProcessPool
from typer.testing import CliRunner

import mlredact.bench.environment as bench_env
import mlredact.pipeline.runner as runner_module
import mlredact.pipeline.workers as workers_module
from inject_support import (
    APPLIED,
    LETTER_CLUSTERS,
    LETTER_MENTIONS,
    PARTIAL,
    SIGNATURE_BOX,
    Docs,
    add_line,
    changed_shares,
    keep_mentions,
    letter_docs,
    mention,
    poly,
    token_polys,
    without_mentions,
    without_regions,
    write_inject_dir,
)
from mlredact.bench.artifacts import display_poly
from mlredact.bench.inject import MODE_FILES
from mlredact.bench.labels import LABEL_MAP
from mlredact.cli.main import app
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import PageGeometry
from mlredact.pipeline.injection import pt_box
from mlredact.text.gazetteer import fold_name, load_gazetteer
from mlredact.text.normalize import digits_only, match_key
from mlredact.verify.checks import extract_all_text

DPI = 150  # NO_MODELS renders at 150 dpi
# The configuration of tests/unit/test_bench_runner.py, without face detection (its model is the only one region
# analysis needs) and with lossless page images.
CONFIG: dict[str, Any] = {**NO_MODELS, "regions": {"faces": False}, "output": {"image_encoding": "flate"}}
# The same with NER on, as the client profile has it: the modes that inject mentions switch it off (without models a
# Redactor with NER on could not even start).
WITH_NER = {k: v for k, v in CONFIG.items() if k != "detection"}
MODES = ("oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions")
STAGE_FILES = ("entities.json", "mentions.json", "plan.json", "pseudonyms.json", "text.json")
CHANGED = ("constant", "persistent")  # a word of the letter, injected as another word (oracle_text)
# A mention left out of the injection (oracle_mentions), or out of the plan only (oracle_regions).
LEFT_OUT = "phone-mobile"
CC = "person-cc"  # "J. Smith", injected on a line of its own (oracle_entities)
DASH = "claim-dash"  # "WC" and "-" of "WC 123 - 456", injected on a line of its own (oracle_entities)
LOGO = "org-harbourside-3"  # the plan region of the letterhead's logo


@dataclass
class Bench:
    """One ``bench`` run: what it was given, what it wrote, and what the stand-ins saw.  ``state``: the worker state
    after the run (the engines it made); ``regions``: each region analysis, as (token texts, keyword arguments);
    ``detections``: tool 2's detection runs; ``ner``: per Redactor, whether GLiNER and the privacy filter were on."""

    mode: str
    docs: Docs
    result: Result
    out: Path
    builds: int
    state: frozenset[str]
    regions: list[tuple[tuple[str, ...], dict[str, Any]]]
    detections: int
    ner: list[tuple[bool, bool]]

    def json(self, name: str) -> Any:
        return json.loads((self.out / name).read_text("utf-8"))

    def read(self, name: str) -> bytes:
        return (self.out / name).read_bytes()

    def mentions(self) -> dict[str, dict[str, Any]]:
        return {m["id"]: m for m in self.json("mentions.json")["mentions"]}

    def plan(self) -> list[dict[str, Any]]:
        regions: list[dict[str, Any]] = self.json("plan.json")["regions"]
        return regions

    def replacement(self, mention_id: str) -> str:
        (found,) = [m["replacement"] for m in self.json("pseudonyms.json")["mentions"] if m["mention"] == mention_id]
        return str(found)


def bench(
    folder: Path,
    mode: str,
    docs: Docs,
    *,
    pdf: bytes = LETTER,
    config: Mapping[str, Any] = CONFIG,
    pool: InProcessPool | None = None,
    ghost: bool = False,
    options: Sequence[str] = (),
) -> Bench:
    """``mlredact bench --mode <mode>`` with ``docs`` injected, as tests/unit/test_bench_runner.py runs bench.
    ``ghost``: region analysis also takes the page's first token for show-through; ``options``: more command-line
    options."""
    pool = pool or InProcessPool()
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    detections: list[int] = []
    ner: list[tuple[bool, bool]] = []
    analyse, detect, build_ner = (
        workers_module.analyse_regions,
        runner_module.detect_document,
        runner_module.build_ner_detectors,
    )

    def regions(rgb: Any, lines: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        calls.append((tuple(t.text for line in lines for t in line.tokens), kwargs))
        found = analyse(rgb, lines, *args, **kwargs)
        return replace(found, ghost_tokens=frozenset({lines[0].tokens[0].token_id})) if ghost else found

    def detection(*args: Any, **kwargs: Any) -> Any:
        detections.append(1)
        return detect(*args, **kwargs)

    def ner_detectors(cfg: Any, directory: Any) -> Any:
        ner.append((cfg.detection.ner.gliner.enabled, cfg.detection.ner.privacy_filter.enabled))
        return build_ner(cfg, directory)

    with pytest.MonkeyPatch.context() as mp:
        pool.install(mp)
        mp.setattr(workers_module, "analyse_regions", regions)
        mp.setattr(runner_module, "detect_document", detection)
        mp.setattr(runner_module, "build_ner_detectors", ner_detectors)
        mp.setattr(bench_env, "candidate_version", lambda root=None: "0.1.0+gtest")
        mp.delenv("MLREDACT_CONFIG", raising=False)
        mp.setenv("MLREDACT_MODELS_DIR", str(folder))  # never read: model files are not resolved
        mp.chdir(folder)
        cfg, src, out = folder / "config.yaml", folder / "input.pdf", folder / "out"
        cfg.write_text(yaml.safe_dump(dict(config)), "utf-8")
        src.write_bytes(pdf)
        inject = write_inject_dir(folder / "inject", mode, docs)
        args = ["bench", "--input", str(src), "--workdir", str(out), "--config", str(cfg)]
        injected = ["--inject-dir", str(inject)] if MODE_FILES[mode] else []
        result = CliRunner().invoke(app, [*args, "--mode", mode, *injected, *options])
        state = frozenset(workers_module._STATE)
    return Bench(mode, docs, result, out, len(pool.builds), state, calls, len(detections), ner)


# ------------------------------------------------------------------------------------- injected files
@functools.cache
def _letter() -> str:
    return json.dumps(letter_docs(LETTER, DPI))


def letter() -> Docs:
    """The letter's injected files (a fresh copy)."""
    docs: Docs = json.loads(_letter())
    return docs


def changed_word() -> Docs:
    docs = letter()
    (token,) = [t for t in docs["text.json"]["pages"][0]["tokens"] if t["text"] == CHANGED[0]]
    token["text"] = CHANGED[1]
    return docs


def with_cc(cluster: str) -> Docs:
    """The letter with "Copy to J. Smith" injected below it, "J. Smith" a mention in ``cluster``."""
    docs = letter()
    ids = add_line(docs, "lcc", "Copy to J. Smith", (60.0, 520.0))
    docs["mentions.json"]["mentions"].append(
        mention(CC, "PERSON_NAME", "J. Smith", ids[2:], token_polys(docs, ids[2:]))
    )
    (members,) = [c["mentions"] for c in docs["entities.json"]["clusters"] if c["id"] == cluster]
    members.append(CC)
    return docs


def with_dash() -> Docs:
    """The letter with "Other claim WC 123 - 456" injected below it, a claim number made of "WC" and the dash: two
    pieces, and the dash has no letter or digit for a fake to change."""
    docs = letter()
    ids = add_line(docs, "ldash", "Other claim WC 123 - 456", (60.0, 520.0))
    pieces = [ids[2], ids[4]]
    docs["mentions.json"]["mentions"].append(mention(DASH, "ID_CLAIM", None, pieces, token_polys(docs, pieces)))
    docs["entities.json"]["clusters"].append({"id": DASH, "mentions": [DASH], "attributes": {}})
    return docs


@functools.cache
def scanned_letter() -> SynthDoc:
    """The letter scanned, with a signature, a QR code, a stamp, margin handwriting and show-through."""
    return make_ime_letter(scanned=True, artefacts=True, dpi=DPI)


def scanned() -> Docs:
    """The scanned letter's words with perfect boxes (tests/unit/test_regions.py's typesetting of the letter)."""
    from test_regions import typeset

    _image, lines = typeset(list(scanned_letter().lines), DPI)
    geometry = PageGeometry(0, 595.276, 841.89, DPI, 1240, 1754)
    tokens = [
        {"id": t.token_id, "text": t.text, "poly": display_poly(t.bbox, geometry), "line": line.line_id}
        for line in lines
        for t in line.tokens
    ]
    for order, t in enumerate(tokens):
        t["order"] = order
    return {"text.json": {"schema": "rb.text/0.1", "source": "ocr", "pages": [{"index": 0, "tokens": tokens}]}}


RUNS: dict[str, Callable[[Path], Bench]] = {
    # Tool 2's own detection of the letter (end to end), for the fakes the oracle modes must equal.
    "e2e": lambda d: bench(d, "e2e", {}),
    # A residual barcode on the first re-read; region analysis takes a word for show-through.
    "oracle_text": lambda d: bench(
        d, "oracle_text", changed_word(), pool=InProcessPool(residuals=(RESIDUAL,)), ghost=True
    ),
    "oracle_text again": lambda d: bench(
        d, "oracle_text", changed_word(), pool=InProcessPool(residuals=(RESIDUAL,)), ghost=True
    ),
    "oracle_text, scanned": lambda d: bench(d, "oracle_text", scanned(), pdf=scanned_letter().pdf),
    "oracle_mentions": lambda d: bench(d, "oracle_mentions", without_mentions(letter(), {LEFT_OUT}), config=WITH_NER),
    "oracle_mentions again": lambda d: bench(
        d, "oracle_mentions", without_mentions(letter(), {LEFT_OUT}), config=WITH_NER
    ),
    "oracle_mentions, one Smith": lambda d: bench(
        d, "oracle_mentions", keep_mentions(letter(), {"person-smith-exam"}), config=WITH_NER
    ),
    "oracle_entities": lambda d: bench(d, "oracle_entities", with_cc("ent-john"), config=WITH_NER),
    "oracle_entities again": lambda d: bench(d, "oracle_entities", with_cc("ent-john"), config=WITH_NER),
    "oracle_entities, J. Smith is Jane": lambda d: bench(d, "oracle_entities", with_cc("ent-jane"), config=WITH_NER),
    "oracle_entities, a dash in a claim number": lambda d: bench(d, "oracle_entities", with_dash(), config=WITH_NER),
    # The mobile number's regions are left out of the plan; a residual barcode on the first re-read.
    "oracle_regions": lambda d: bench(
        d,
        "oracle_regions",
        without_regions(letter(), {LEFT_OUT}),
        config=WITH_NER,
        pool=InProcessPool(residuals=(RESIDUAL,)),
    ),
    "oracle_regions again": lambda d: bench(
        d,
        "oracle_regions",
        without_regions(letter(), {LEFT_OUT}),
        config=WITH_NER,
        pool=InProcessPool(residuals=(RESIDUAL,)),
    ),
}
RELEASE = {
    "oracle_text": {"released": True, "checks_failed": []},
    # The mobile number was not injected: verification finds it again, and does not re-plan it.  V2 matches the
    # digits of a whole line, and the line's digits (0412345678) hold the injected claim number's (WC1234567).
    "oracle_mentions": {"released": False, "checks_failed": ["V2_TEXT_LEAK", "V4_RESIDUAL"]},
    # Only the injected mentions are redacted: the letter's ground truth leaves out 'Level 4,' before '123 George
    # Street' and the postcode after 'Parramatta', which the grammar reads as parts of those addresses; they stay, and
    # verification finds them.
    "oracle_entities": {"released": False, "checks_failed": ["V4_RESIDUAL"]},
    # The mobile number is injected but not in the plan: it stays, which both checks see, and nothing is re-planned.
    "oracle_regions": {"released": False, "checks_failed": ["V2_TEXT_LEAK", "V4_RESIDUAL"]},
}


class Runs:
    """Each run made once, on first use (every run renders the letter)."""

    def __init__(self, factory: pytest.TempPathFactory) -> None:
        self._factory = factory
        self._done: dict[str, Bench] = {}

    def __getitem__(self, name: str) -> Bench:
        if name not in self._done:
            self._done[name] = RUNS[name](self._factory.mktemp("bench"))
        return self._done[name]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    return Runs(tmp_path_factory)


def _box(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ----------------------------------------------------------------------------------------- every mode
@pytest.mark.parametrize("mode", MODES)
def test_the_output_and_every_stage_file_are_handed_over(runs: Runs, mode: str) -> None:
    run = runs[mode]
    assert run.result.exit_code == 0, run.result.output
    info = run.json("run_info.json")
    assert info["release"] == RELEASE[mode]
    held = [] if info["release"]["released"] else ["quarantine.json"]
    assert sorted(p.name for p in run.out.iterdir()) == sorted(
        ["manifest.json", "output.pdf", "run.json", "run_info.json", *STAGE_FILES, *held]
    )
    assert {k: info["config"][k] for k in ("mode", "injected", "ner_enabled")} == {
        "mode": mode,
        "injected": sorted(f.kind for f in MODE_FILES[mode]),
        "ner_enabled": False,
    }
    # Count notes: what was injected, then verification's residual hits (before the note on a held-back output).
    notes = [n.split(":")[0] for n in info["notes"].split("; ")]
    injected = {
        "oracle_text": ["injected text"],
        "oracle_mentions": ["injected text", "injected mentions"],
        "oracle_entities": ["injected text", "injected mentions", "injected clusters"],
        "oracle_regions": ["injected text", "injected mentions", "injected clusters", "injected plan"],
    }[mode]
    assert [n for n in notes if n.startswith("injected ")] == injected
    assert notes[len(notes) - (1 if info["release"]["released"] else 2)] == "V4 residual hits"


@pytest.mark.parametrize("mode", MODES)
def test_text_json_is_the_injected_text(runs: Runs, mode: str) -> None:
    run = runs[mode]
    injected = sorted(run.docs["text.json"]["pages"][0]["tokens"], key=lambda t: t["order"])
    text = run.json("text.json")
    (page,) = text["pages"]
    assert text["source"] == "text_layer"
    assert [t["text"] for t in page["tokens"]] == [t["text"] for t in injected]
    assert [t["order"] for t in page["tokens"]] == list(range(len(injected)))
    # Tool 2's pixel boxes, rounded outward: at most one pixel (72/dpi points) larger on each side.
    for written, given in zip(page["tokens"], injected, strict=True):
        (x0, y0, x1, y1), (g0, h0, g1, h1) = _box(written["poly"]), _box(given["poly"])
        assert all(-1e-3 <= d <= 72 / DPI + 1e-3 for d in (g0 - x0, h0 - y0, x1 - g1, y1 - h1))


@pytest.mark.parametrize("mode", MODES)
def test_the_stage_files_name_only_what_they_list(runs: Runs, mode: str) -> None:
    run = runs[mode]
    tokens = {t["id"] for p in run.json("text.json")["pages"] for t in p["tokens"]}
    mentions = run.mentions()
    assert all(set(m.get("tokens") or ()) <= tokens for m in mentions.values())
    named = {i for r in run.plan() for i in r["mentions"]}
    named |= {i for c in run.json("entities.json")["clusters"] for i in c["mentions"]}
    named |= {m["mention"] for m in run.json("pseudonyms.json")["mentions"]}
    assert named and named <= set(mentions)


@pytest.mark.parametrize("mode", MODES)
def test_no_ocr_engine_is_made(runs: Runs, mode: str) -> None:
    assert not any(key.startswith("ocr:") for key in runs[mode].state)


@pytest.mark.parametrize("mode", MODES)
def test_two_runs_are_identical(runs: Runs, mode: str) -> None:
    first, again = runs[mode], runs[f"{mode} again"]
    for name in ("output.pdf", "manifest.json", *STAGE_FILES):
        assert first.read(name) == again.read(name), name


# ------------------------------------------------------------------------------------------ oracle_text
def test_regions_are_found_around_the_injected_text(runs: Runs) -> None:
    run = runs["oracle_text"]
    ((texts, kwargs),) = run.regions
    injected = sorted(run.docs["text.json"]["pages"][0]["tokens"], key=lambda t: t["order"])
    assert texts == tuple(t["text"] for t in injected) and kwargs["reread"] is None
    # The word region analysis took for show-through is still read.
    assert run.json("text.json")["pages"][0]["tokens"][0]["text"] == "HARBOURSIDE"
    assert run.detections == 1


def test_the_text_is_the_injection_s_not_the_pdf_s(runs: Runs) -> None:
    texts = [t["text"] for t in runs["oracle_text"].json("text.json")["pages"][0]["tokens"]]
    assert CHANGED[1] in texts and CHANGED[0] not in texts


def test_tool_2_detects_and_keeps_as_usual(runs: Runs) -> None:
    found = {(m["label"], m.get("text")) for m in runs["oracle_text"].mentions().values()}
    assert {
        ("person", "John Andrew SMITH"),
        ("phone", "0412 345 678"),
        ("email", "john.smith78@example.org"),
        ("medicare", medicare_number("29537015", 1)),
    } <= found
    # The client keeps dates and places on their own; the letter has none, and the places of its addresses are
    # the places of their units ('<unit>.place').
    assert not {label for label, _text in found} & {"date", "date_of_birth", "postcode"}
    places = [i for i, m in runs["oracle_text"].mentions().items() if m["label"] == "locality"]
    assert places and all(i.endswith(".place") for i in places)


def test_every_residual_hit_is_re_planned(runs: Runs) -> None:
    run = runs["oracle_text"]
    assert run.builds == 2 and any(r["id"].startswith("R-") for r in run.plan())
    assert (
        "V4 residual hits: 0 (0 removed values, 0 re-detection, 0 regions) after 1 re-plans"
        in run.json("run_info.json")["notes"]
    )


def test_the_regions_of_a_scanned_page_are_found_around_the_injected_words(runs: Runs) -> None:
    run = runs["oracle_text, scanned"]
    assert run.result.exit_code == 0, run.result.output
    kinds = {r["kind"] for r in run.json("manifest.json")["regions"]}
    assert {"signature", "barcode", "stamp"} <= kinds
    assert {"signature", "barcode", "stamp"} <= {m["label"] for m in run.mentions().values()}


# -------------------------------------------------------------------------------------- oracle_mentions
@pytest.mark.parametrize("mode", ["oracle_mentions", "oracle_entities", "oracle_regions"])
def test_the_injection_replaces_detection_and_region_analysis(runs: Runs, mode: str) -> None:
    run = runs[mode]
    assert (run.detections, run.regions) == (0, [])
    assert run.ner == [(False, False)]  # NER is on in the configuration file
    detectors = run.json("manifest.json")["provenance"]["detectors"]
    assert detectors == [{"id": "bench.inject", "version": "1"}, {"id": "resolve.persons", "version": "1"}]


@pytest.mark.parametrize("mode", ["oracle_mentions", "oracle_entities", "oracle_regions"])
def test_the_injected_mentions_are_reported_under_their_ids(runs: Runs, mode: str) -> None:
    run = runs[mode]
    injected = {m["id"]: m for m in run.docs["mentions.json"]["mentions"]}
    reported = run.mentions()
    assert set(reported) == set(injected)
    texts = {t["id"]: t["text"] for t in run.docs["text.json"]["pages"][0]["tokens"]}
    tool_2 = {t["id"]: t["text"] for t in run.json("text.json")["pages"][0]["tokens"]}
    for mention_id, m in reported.items():
        assert LABEL_MAP[m["label"]] == injected[mention_id]["label"], mention_id
        own = [texts[t] for t in injected[mention_id]["tokens"] or ()]
        assert [tool_2[t] for t in m.get("tokens", ())] == own and m.get("text", "") == " ".join(own)


@pytest.mark.parametrize("mode", ["oracle_mentions", "oracle_entities", "oracle_regions"])
def test_a_locality_is_replaced_with_its_address_as_one_unit(runs: Runs, mode: str) -> None:
    """The injected LOCATION joins the injected ADDRESS before it: the street line and the place are parts of one
    address, and the place's fake is a locality of the list with its own postcode, in the original's state."""
    run = runs[mode]
    output = extract_all_text(run.read("output.pdf"))
    rows = {(loc.folded, loc.postcode, loc.state) for loc in load_gazetteer().rows}
    for mention_id, place, postcode, state in (
        ("loc-penrith", "Penrith", "2750", "NSW"),
        ("loc-sydney", "Sydney", "2000", "NSW"),
    ):
        *name, fake_state, fake_postcode = run.replacement(mention_id).split()
        assert fake_state == state and (fold_name(" ".join(name)), fake_postcode, state) in rows, mention_id
        assert re.search(rf"\b({place}|{postcode})\b", output) is None


def test_the_oracle_draws_the_fakes_that_end_to_end_draws(runs: Runs) -> None:
    """Same document, same scope (the document itself), same key: the fake of an address is the same wherever its
    words come from.  Each injected part shows the words of the end-to-end part that holds its tokens."""
    theirs = {m["mention"]: m["replacement"] for m in runs["e2e"].json("pseudonyms.json")["mentions"]}
    parts = {i: m for i, m in runs["e2e"].mentions().items() if m["label"] in ("street_address", "locality")}
    for mode in ("oracle_mentions", "oracle_entities", "oracle_regions"):
        run = runs[mode]
        for mention_id in ("addr-george", "addr-pobox", "addr-wattle", "loc-sydney", "loc-parramatta", "loc-penrith"):
            own = run.mentions()[mention_id]["text"]
            (part,) = [i for i, m in parts.items() if own in m["text"]]
            assert f" {run.replacement(mention_id)}" in f" {theirs[part]}", (mode, mention_id)


@pytest.mark.parametrize("name", ["e2e", "oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions"])
def test_verification_knows_the_address_units_in_every_mode(runs: Runs, name: str) -> None:
    checks = {c["check"]: c["metrics"] for c in runs[name].json("manifest.json")["verification"]["checks"]}
    assert checks["V2_text"]["address_needles"] > 0 and "address_word_collision" in checks["V5_surrogate"]


def test_the_plan_s_replacement_for_an_address_is_the_units_fake(runs: Runs) -> None:
    """A region of an address draws the unit's fake over the words of the line it covers: the street line's and the
    place's together where they share a line."""
    run = runs["oracle_regions"]
    for street, place in (("addr-wattle", "loc-penrith"), ("addr-george", "loc-sydney")):
        texts = {r["replacement"] for r in run.plan() if r["mentions"] in ([street], [place])}
        assert len(texts) == 1
        (text,) = texts
        assert run.replacement(street) in text and run.replacement(place) in text


def test_a_name_inside_an_organisation_is_drawn_with_it(runs: Runs) -> None:
    run = runs["oracle_mentions"]
    assert run.mentions()["person-firm"]["label"] == "person"
    naming = [r["mentions"] for r in run.plan() if "person-firm" in r["mentions"]]
    assert naming and all(names[0] == "org-firm" for names in naming)
    assert "person-firm" not in {m["mention"] for m in run.json("pseudonyms.json")["mentions"]}
    assert "Smith & Partners" not in extract_all_text(run.read("output.pdf"))


def test_the_signature_becomes_a_stand_in(runs: Runs) -> None:
    run = runs["oracle_mentions"]
    (region,) = [r for r in run.json("manifest.json")["regions"] if r["kind"] == "signature"]
    assert (region["action"], region["detector"], region["rule"]) == ("synthesise", "inject", "inject.SIGNATURE")
    # A region's operation replaces only when it draws a stand-in (otherwise it removes).
    (drawn,) = [r for r in run.plan() if r["mentions"] == ["signature"]]
    assert drawn["action"] == "replace" and drawn["id"].startswith(region["region_id"])


def test_a_mention_left_out_stays_and_holds_the_document_back(runs: Runs) -> None:
    run = runs["oracle_mentions"]
    assert "0412345678" in digits_only(extract_all_text(run.read("output.pdf")))
    (phone,) = [m for m in LETTER_MENTIONS if m[0] == LEFT_OUT]
    words = [t for t in run.json("text.json")["pages"][0]["tokens"] if t["text"] in phone[3].split()]
    assert words and not any(_overlap(_box(w["poly"]), _box(r["poly"])) for w in words for r in run.plan())
    # Verification finds it again, which holds the document back, but re-plans only removed values.
    assert run.builds == 1 and not any(r["id"].startswith("R-") for r in run.plan())
    notes = run.json("run_info.json")["notes"]
    assert "0 removed values" in notes and "(re-plans from: removed values)" in notes


def test_a_removed_value_still_readable_is_re_planned(runs: Runs) -> None:
    run = runs["oracle_mentions, one Smith"]
    assert set(run.mentions()) == {"person-smith-exam"}
    replanned = [r for r in run.plan() if r["id"].startswith("R-")]
    assert run.builds == 2 and replanned and all(r["mentions"] == [] for r in replanned)
    assert "after 1 re-plans" in run.json("run_info.json")["notes"]


def test_of_what_verification_finds_only_removed_values_are_re_planned(runs: Runs) -> None:
    """Verification finds the injected "Smith" elsewhere, and PII that was not injected: it re-plans only the
    former.  The rest stays as it was and still counts, so the document is held back."""
    run = runs["oracle_mentions, one Smith"]
    info = run.json("run_info.json")
    assert info["release"] == {"released": False, "checks_failed": ["V2_TEXT_LEAK", "V4_RESIDUAL"]}
    found = re.search(r"V4 residual hits: \d+ \((\d+) removed values, (\d+) re-detection", info["notes"])
    assert found is not None and (int(found[1]), int(found[2]) > 0) == (0, True)
    # Each re-plan blacks out another "Smith", and no other word.
    geometry = _geometry(run)
    words = [(t["text"], pt_box(t["poly"], geometry)) for t in run.json("text.json")["pages"][0]["tokens"]]
    for r in run.plan():
        if r["id"].startswith("R-"):
            box = pt_box(r["poly"], geometry)
            under = [text for text, word in words if word.overlap_fraction(box) >= 0.5]
            assert under and {match_key(text) for text in under} == {"smith"}, r["id"]


# ------------------------------------------------------------------------------------- oracle_entities
def test_the_clusters_are_the_injected_ones(runs: Runs) -> None:
    run = runs["oracle_entities"]
    clusters = run.json("entities.json")["clusters"]
    assert [(c["id"], c["mentions"]) for c in clusters] == [
        ("ent-jane", ["person-citizen", "person-jane"]),
        ("ent-john", [CC, "person-email", "person-john", "person-smith-contact", "person-smith-exam"]),
        ("ent-peter", ["person-peter"]),  # not the signature: entities.json clusters people's names
        ("person-firm", ["person-firm"]),  # the firm's namesake, injected as an entity of its own
    ]
    assert {c for c, _members in LETTER_CLUSTERS} < {c["id"] for c in run.docs["entities.json"]["clusters"]}


def test_the_cluster_chooses_the_initial(runs: Runs) -> None:
    john, jane = runs["oracle_entities"], runs["oracle_entities, J. Smith is Jane"]
    with_john, with_jane = john.replacement(CC).split(), jane.replacement(CC).split()
    # The initial is drawn from the fake given name of the person the cluster says J. Smith is.
    assert with_john[0] == john.replacement("person-john")[0] + "."
    assert with_jane[0] == jane.replacement("person-jane")[0] + "."
    assert with_john[0] != with_jane[0] and with_john[1] == with_jane[1]


def test_a_piece_without_a_letter_or_digit_is_removed(runs: Runs) -> None:
    """No fake can be made for the dash (fakes change letters and digits only): it is removed, the rest of the
    claim number replaced, and the document goes through."""
    run = runs["oracle_entities, a dash in a claim number"]
    assert run.result.exit_code == 0, run.result.output
    assert (run.out / "output.pdf").is_file() and run.mentions()[DASH]["text"] == "WC -"
    assert sorted(r["action"] for r in run.plan() if DASH in r["mentions"]) == ["remove", "replace"]
    assert "1 runs without a letter or digit removed" in run.json("run_info.json")["notes"]


def test_pseudonyms_name_the_injected_clusters(runs: Runs) -> None:
    run = runs["oracle_entities"]
    entries = run.json("pseudonyms.json")["entries"]
    by_original = {e["original"]: e["cluster"] for e in entries}
    assert by_original["John Andrew SMITH"] == by_original["J. Smith"] == "ent-john"
    assert by_original["Jane Citizen"] == "ent-jane" and by_original["0412 345 678"] is None
    assert {e["cluster"] for e in entries} <= {c["id"] for c in run.docs["entities.json"]["clusters"]} | {None}


# -------------------------------------------------------------------------------------- oracle_regions
def _geometry(run: Bench) -> PageGeometry:
    page = run.json("manifest.json")["pages"][0]
    keys = ("index", "width_pt", "height_pt", "dpi", "width_px", "height_px")
    return PageGeometry(*(page[k] for k in keys))


def _px(points: Sequence[Sequence[float]], geometry: PageGeometry) -> list[int]:
    """A display-point polygon as tool 2's pixel box, as the manifest writes it."""
    return list(pt_box([(x, y) for x, y in points], geometry).as_tuple())


def test_each_mention_is_drawn_once_per_line_over_its_words(runs: Runs) -> None:
    run = runs["oracle_regions"]
    geometry, manifest = _geometry(run), run.json("manifest.json")
    words = {t["id"]: pt_box(t["poly"], geometry) for t in run.json("text.json")["pages"][0]["tokens"]}
    left_out = run.mentions()[LEFT_OUT]["tokens"]
    plan_boxes = [pt_box(r["poly"], geometry) for r in run.docs["plan.json"]["regions"]]
    for m in manifest["mentions"]:
        lines: dict[str, list[BBox]] = defaultdict(list)
        for token in m["token_ids"]:
            # A word of an address's unit that no injected region covers (the letter's "Level 4,") is not drawn.
            if any(words[token].overlap_fraction(box) >= 0.5 for box in plan_boxes):
                lines[token.rsplit(".", 1)[0]].append(words[token])
        if m["token_ids"] == left_out:
            assert (m["render"], m["boxes"]) == ([], [])  # none of its regions is in the plan
            continue
        # One operation per line, over the line's words and nothing more: the regions are the words' boxes, or
        # characters inside one (the e-mail address's name).
        assert m["render"] == ["surrogate"]
        assert sorted(b["px"] for b in m["boxes"]) == sorted(list(union_all(b).as_tuple()) for b in lines.values())
    regions = {r["kind"]: (r["action"], r["box"]["px"]) for r in manifest["regions"]}
    (logo,) = [r["poly"] for r in run.docs["plan.json"]["regions"] if r["id"] == LOGO]
    assert regions == {
        "logo": ("remove", _px(logo, geometry)),
        "signature": ("synthesise", _px(poly(SIGNATURE_BOX), geometry)),
    }
    assert manifest["residual_boxes"] == []  # nothing re-planned, nothing left unresolved


def test_plan_json_repeats_the_injected_plan_with_what_was_done(runs: Runs) -> None:
    run = runs["oracle_regions"]
    injected = {r["id"]: r for r in run.docs["plan.json"]["regions"]}
    plan = run.plan()
    assert [r["id"] for r in plan] == sorted(injected)  # every region (all on page 0), by id
    assert all((r["poly"], r["mentions"]) == (injected[r["id"]]["poly"], injected[r["id"]]["mentions"]) for r in plan)
    done = {r["id"]: (r["action"], "replacement" in r) for r in plan}
    # The letterhead's logo is removed, as the client removes logos; the signature gets a stand-in, which is no text.
    assert (done.pop(LOGO), done.pop("signature-0")) == (("remove", False), ("replace", False))
    assert set(done.values()) == {("replace", True)}
    # The organisation's words are joined: each of their regions names the one fake drawn over all of them.
    firm = {r["replacement"] for r in plan if r["mentions"] == ["org-firm"]}
    assert len(firm) == 1 and firm != {"Smith & Partners Lawyers"}
    n, notes = len(injected), run.json("run_info.json")["notes"]
    assert f"injected plan: {n} regions read (0 with a repeated id), {n} applied by " in notes


def test_every_region_in_the_plan_is_applied_to_the_page(runs: Runs) -> None:
    """A fake drawn where a word was changes the word's strokes, not the paper between them, so the benchmark rates
    a replaced word "partial" or "applied"; the removed logo turns grey all over; regions left out stay as they were."""
    run = runs["oracle_regions"]
    left_out = [r for r in letter()["plan.json"]["regions"] if r["mentions"] == [LEFT_OUT]]
    shares = changed_shares(LETTER, run.read("output.pdf"), [*run.plan(), *left_out])
    applied = [shares[r["id"]] for r in run.plan()]
    assert len(applied) == len(run.docs["plan.json"]["regions"]) and min(applied) >= PARTIAL
    assert shares[LOGO] >= APPLIED
    assert left_out and all(shares[r["id"]] < PARTIAL for r in left_out)


def test_a_mention_left_out_of_the_plan_stays_and_holds_the_document_back(runs: Runs) -> None:
    run = runs["oracle_regions"]
    assert "0412345678" in digits_only(extract_all_text(run.read("output.pdf")))
    # Still injected and reported: only its regions were left out, so no fake of it was drawn.
    assert LEFT_OUT in run.mentions() and not any(LEFT_OUT in r["mentions"] for r in run.plan())
    assert LEFT_OUT not in {m["mention"] for m in run.json("pseudonyms.json")["mentions"]}
    # Verification holds the document back (RELEASE) and counts the residual barcode, but re-plans nothing.
    assert run.builds == 1
    assert "1 regions) after 0 re-plans (re-plans from: none)" in run.json("run_info.json")["notes"]
