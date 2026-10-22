"""Benchmark adapter: stage files from a job trace, and the ``bench`` command (the pipeline is stubbed).

All text is fictitious.  The schemas themselves belong to the benchmark and are checked there.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

import mlredact.bench.artifacts as bench_artifacts
import mlredact.bench.environment as bench_env
import mlredact.pipeline.runner as runner_module
from mlredact import __version__
from mlredact.bench.artifacts import (
    LABELS,
    PII_REGION_KINDS,
    display_poly,
    mentions_artifact,
    plan_action,
    run_info,
    stage_files,
    text_artifact,
    unreleased_note,
)
from mlredact.bench.oracle import OracleInputs
from mlredact.cli.main import app
from mlredact.config.loader import load_config
from mlredact.config.schema import AppConfig
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.geometry import BBox
from mlredact.core.model import Evidence, Line, Mention, PageAnalysis, PageGeometry, Region, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, JobStatus, PageKind, RegionKind
from mlredact.ocr.fusion import ENGINE_ID as TEXT_LAYER_ENGINE
from mlredact.pipeline.runner import JobResult, JobTrace, Redactor, RegionDecision
from mlredact.policy.engine import Decision
from mlredact.render.plan import plan_ops, rotated_token_ids
from mlredact.resolve.persons import Gender, ParsedMention, PersonResolution
from mlredact.surrogate.assign import MentionSurrogate

# Page 0: portrait A4 at 300 dpi; page 1: landscape A4 rendered at 200 dpi.
GEOM0 = PageGeometry(0, 595.276, 841.89, 300, 2481, 3508)
GEOM1 = PageGeometry(1, 841.89, 595.276, 200, 2339, 1654)
PDF = b"%PDF-1.7 stub output"


def _tok(page: int, line: int, word: int, box: BBox, text: str, conf: float = 0.9, engine: str = "ppocrv6") -> Token:
    return Token(f"p{page:04d}.l{line:04d}.w{word:03d}", page, line, word, box, conf, conf, engine, text)


def _line(tokens: list[Token]) -> Line:
    t0 = tokens[0]
    box = BBox(t0.bbox.x0, t0.bbox.y0, tokens[-1].bbox.x1, t0.bbox.y1)
    quad = ((box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1))
    return Line(f"p{t0.page:04d}.l{t0.line_no:04d}", t0.page, t0.line_no, box, quad, 0.9, t0.engine, tuple(tokens))


def _mention(mid: str, etype: EntityType, tokens: list[Token], score: float = 0.97) -> Mention:
    ev = Evidence("test", "1", "rule", EvidenceStrength.STRONG, score)
    return Mention(mid, etype, tuple(t.token_id for t in tokens), (ev,), " ".join(t.text for t in tokens))


def _region(rid: str, kind: RegionKind, box: BBox, score: float = 1.0) -> Region:
    return Region(rid, 0, kind, box, score)


def _op(
    page: int, kind: ActionKind, box: BBox, ref: str, text: str | None = None, covers: tuple[str, ...] = ()
) -> RenderOp:
    return RenderOp(page, kind, box, "test", ref, text=text, covers=covers)


# Page 0
RE, ANN, EXAMPLE = (
    _tok(0, 0, 0, BBox(300, 600, 400, 660), "Re:"),
    _tok(0, 0, 1, BBox(420, 600, 600, 660), "Ann"),
    _tok(0, 0, 2, BBox(620, 600, 900, 660), "Example"),
)
SEEN, DATE = _tok(0, 1, 0, BBox(300, 700, 450, 760), "Seen"), _tok(0, 1, 1, BBox(470, 700, 760, 760), "12/03/2024")
PH, PHONE = _tok(0, 2, 0, BBox(300, 800, 380, 860), "Ph", 1.2), _tok(0, 2, 1, BBox(400, 800, 800, 860), "0491570006")
# Page 1 (200 dpi): the same person again, and a kept professional.
ANN1, EXAMPLE1 = _tok(1, 0, 0, BBox(200, 400, 300, 450), "Ann"), _tok(1, 0, 1, BBox(310, 400, 400, 450), "Example")
DR, KEPT = _tok(1, 1, 0, BBox(200, 500, 260, 550), "Dr"), _tok(1, 1, 1, BBox(270, 500, 400, 550), "Kept")

PERSON_A = _mention("M-person-a", EntityType.PERSON, [ANN, EXAMPLE])
SEEN_DATE = _mention("M-date", EntityType.DATE, [DATE], 1.0)
PHONE_M = _mention("M-phone", EntityType.PHONE, [PHONE], 1.0)
PERSON_B = _mention("M-person-b", EntityType.PERSON, [ANN1, EXAMPLE1], 3.0)  # a score that is not a probability
PROFESSIONAL = _mention("M-prof", EntityType.PERSON, [KEPT])

SIGNATURE = _region("G-sig", RegionKind.SIGNATURE, BBox(300, 3000, 900, 3200), 0.8)
BARCODE = _region("G-bar", RegionKind.BARCODE, BBox(2000, 300, 2300, 600))
FAINT = _region("G-faint", RegionKind.FAINT_INK, BBox(300, 1000, 2000, 1500))
FIGURE = _region("G-fig", RegionKind.GRAPHIC, BBox(300, 1600, 1200, 2400))
FIGURE_REMOVED = _region("G-fig2", RegionKind.GRAPHIC, BBox(1300, 1600, 2200, 2400))

PERSON_A_BOX = BBox(416, 596, 904, 664)
PERSON_A_ERASED = BBox(410, 590, 930, 668)


def make_trace(pdf: bytes | None = PDF, *, text_layer_page1: bool = False) -> JobTrace:
    pages = (
        PageAnalysis(
            GEOM0,
            PageKind.IMAGE_ONLY,
            (_line([RE, ANN, EXAMPLE]), _line([SEEN, DATE]), _line([PH, PHONE])),
            regions=(SIGNATURE, BARCODE, FAINT, FIGURE, FIGURE_REMOVED),
        ),
        PageAnalysis(
            GEOM1,
            PageKind.BORN_DIGITAL if text_layer_page1 else PageKind.IMAGE_ONLY,
            (_line([ANN1, EXAMPLE1]), _line([DR, KEPT])),
            embedded_char_count=40 if text_layer_page1 else 0,
        ),
    )
    decisions = (
        Decision(PERSON_A, ActionKind.SURROGATE),
        Decision(SEEN_DATE, ActionKind.KEEP),
        Decision(PHONE_M, ActionKind.SURROGATE),
        Decision(PERSON_B, ActionKind.SURROGATE),
        Decision(PROFESSIONAL, ActionKind.KEEP),
    )
    parsed = {
        m.mention_id: ParsedMention(m.mention_id, (), None, Gender.UNKNOWN, cluster)
        for m, cluster in ((PERSON_A, "P000"), (PERSON_B, "P000"), (PROFESSIONAL, "P001"))
    }
    persons = PersonResolution(parsed, {}, {"P000": ("M-person-a", "M-person-b"), "P001": ("M-prof",)})
    surrogates = {
        "M-person-a": MentionSurrogate("M-person-a", ("Bea", "Sample")),
        "M-person-b": MentionSurrogate("M-person-b", ("Bea", "Sample")),
        "M-phone": MentionSurrogate("M-phone", ("0491570110",)),
    }
    ops = {
        0: (
            _op(0, ActionKind.SURROGATE, PERSON_A_BOX, "M-person-a", "Bea Sample"),
            _op(0, ActionKind.RETYPESET_REGION, BBox(296, 796, 804, 864), "L-line", "Ph 0491570110", ("M-phone",)),
            _op(0, ActionKind.SYNTHESISE, SIGNATURE.bbox, "G-sig"),
            _op(0, ActionKind.REMOVE, BARCODE.bbox, "G-bar"),
            _op(0, ActionKind.SUPPRESS_INK, FAINT.bbox, "G-faint"),
            _op(0, ActionKind.REMOVE, FIGURE_REMOVED.bbox, "G-fig2"),
            _op(0, ActionKind.BLACKOUT, BBox(1000, 2600, 1100, 2650), "R-residual"),
        ),
        # A mention drawn as two segments shares one ref id.
        1: (
            _op(1, ActionKind.SURROGATE, BBox(196, 396, 304, 454), "M-person-b", "Bea"),
            _op(1, ActionKind.SURROGATE, BBox(306, 396, 404, 454), "M-person-b", "Sample"),
        ),
    }
    regions = (
        RegionDecision(SIGNATURE, ActionKind.SYNTHESISE),
        RegionDecision(BARCODE, ActionKind.REMOVE),
        RegionDecision(FAINT, ActionKind.SUPPRESS_INK),
        RegionDecision(FIGURE, ActionKind.KEEP),
        RegionDecision(FIGURE_REMOVED, ActionKind.REMOVE),
    )
    erased = {("M-person-a", PERSON_A_BOX): PERSON_A_ERASED}
    return JobTrace(pdf, pages, decisions, persons, surrogates, ops, erased, regions)


def _files(trace: JobTrace | None = None) -> dict[str, dict[str, Any]]:
    return stage_files(trace or make_trace(), text_layer=True)


def test_trace_repr_shows_no_document_text() -> None:
    trace = make_trace()
    shown = repr(trace) + repr(JobResult(JobStatus.RELEASED, PDF, {}, {}, trace=trace))
    for value in ("Ann", "Example", "12/03/2024", "0491570006", "Bea", "Sample", "Kept"):
        assert value not in shown


# ------------------------------------------------------------------------------- conversions
def test_pixel_boxes_become_display_points_with_each_pages_own_dpi() -> None:
    assert display_poly(BBox(300, 600, 900, 660), GEOM0) == [[72, 144], [216, 144], [216, 158.4], [72, 158.4]]
    assert display_poly(BBox(200, 400, 400, 450), GEOM1) == [[72, 144], [144, 144], [144, 162], [72, 162]]
    tokens = {t["id"]: t for p in _files()["text.json"]["pages"] for t in p["tokens"]}
    assert tokens[ANN1.token_id]["poly"] == display_poly(ANN1.bbox, GEOM1)
    assert tokens[ANN1.token_id]["poly"][0] == [72, 144]


def test_text_lists_every_token_in_reading_order_per_page() -> None:
    text = _files()["text.json"]
    assert text["frame"] == "display_pt"
    assert [p["index"] for p in text["pages"]] == [0, 1]
    page0 = text["pages"][0]["tokens"]
    assert [t["order"] for t in page0] == list(range(7))
    assert [t["text"] for t in page0[:3]] == ["Re:", "Ann", "Example"]
    assert {t["line"] for t in page0} == {"p0000.l0000", "p0000.l0001", "p0000.l0002"}
    assert page0[5]["conf"] == 1.0  # clamped to a probability
    assert text["pages"][1]["tokens"][0]["order"] == 0


def test_text_source_tells_ocr_text_layer_and_hybrid_apart() -> None:
    assert text_artifact(make_trace(), text_layer=True)["source"] == "ocr"
    mixed = make_trace(text_layer_page1=True)
    assert text_artifact(mixed, text_layer=True)["source"] == "hybrid"
    assert text_artifact(mixed, text_layer=False)["source"] == "ocr"  # the text layer is not fused
    digital = PageAnalysis(
        GEOM0, PageKind.BORN_DIGITAL, (_line([_tok(0, 0, 0, BBox(0, 0, 9, 9), "Hello", 0.99, TEXT_LAYER_ENGINE)]),), 5
    )
    only = JobTrace(PDF, (digital,), (), None, {}, {}, {}, ())
    assert text_artifact(only, text_layer=True)["source"] == "text_layer"


def test_kept_values_are_not_mentions() -> None:
    ids = [m["id"] for m in _files()["mentions.json"]["mentions"]]
    assert "M-date" not in ids and "M-prof" not in ids
    assert ids[:3] == ["M-person-a", "M-phone", "M-person-b"]


def test_mentions_carry_text_tokens_token_boxes_and_probability_scores() -> None:
    mentions = {m["id"]: m for m in mentions_artifact(make_trace())["mentions"]}
    person = mentions["M-person-a"]
    assert person["label"] == "person" and person["text"] == "Ann Example"
    assert person["tokens"] == [ANN.token_id, EXAMPLE.token_id]
    assert person["regions"] == [
        {"page": 0, "poly": display_poly(ANN.bbox, GEOM0)},
        {"page": 0, "poly": display_poly(EXAMPLE.bbox, GEOM0)},
    ]
    assert person["conf"] == 0.97
    assert mentions["M-person-b"]["regions"][0] == {"page": 1, "poly": display_poly(ANN1.bbox, GEOM1)}
    assert "conf" not in mentions["M-person-b"]  # 3.0 is not a probability


def test_only_pii_regions_that_are_removed_or_replaced_become_mentions() -> None:
    mentions = {m["id"]: m for m in _files()["mentions.json"]["mentions"]}
    assert mentions["G-sig"] == {
        "id": "G-sig",
        "label": "signature",
        "regions": [{"page": 0, "poly": display_poly(SIGNATURE.bbox, GEOM0)}],
        "conf": 0.8,
    }
    assert mentions["G-bar"]["label"] == "barcode"
    # Faint ink is only lightened; figures are kept or, when removed, are layout rather than PII.
    assert not {"G-faint", "G-fig", "G-fig2"} & set(mentions)


def test_every_label_bench_can_emit_is_published() -> None:
    assert set(LABELS) == {t.value for t in EntityType} | {k.value for k in PII_REGION_KINDS}
    assert {RegionKind.FAINT_INK, RegionKind.GRAPHIC}.isdisjoint(PII_REGION_KINDS)
    assert {m["label"] for m in _files()["mentions.json"]["mentions"]} <= set(LABELS)


@pytest.mark.parametrize(
    ("kind", "action"),
    [
        (ActionKind.SURROGATE, "replace"),
        (ActionKind.RETYPESET_REGION, "replace"),
        (ActionKind.LABEL, "replace"),
        (ActionKind.SYNTHESISE, "replace"),
        (ActionKind.GENERALISE, "replace"),
        (ActionKind.DATE_SHIFT, "replace"),
        (ActionKind.REMOVE, "remove"),
        (ActionKind.BLACKOUT, "remove"),
        (ActionKind.SUPPRESS_INK, None),
        (ActionKind.KEEP, None),
    ],
)
def test_plan_actions(kind: ActionKind, action: str | None) -> None:
    assert plan_action(kind) == action


def test_plan_has_one_region_per_operation_with_the_box_it_erased() -> None:
    regions = _files()["plan.json"]["regions"]
    assert [r["id"] for r in regions] == [
        "M-person-a.0",
        "L-line.0",
        "G-sig.0",
        "G-bar.0",
        "G-fig2.0",
        "R-residual.0",
        "M-person-b.0",
        "M-person-b.1",
    ]  # faint-ink suppression is not a removal: no region
    first = regions[0]
    assert first["poly"] == display_poly(PERSON_A_ERASED, GEOM0)
    assert (first["action"], first["replacement"]) == ("replace", "Bea Sample")
    assert regions[6]["poly"] == display_poly(BBox(196, 396, 304, 454), GEOM1)
    by_id = {r["id"]: r for r in regions}
    assert by_id["G-sig.0"]["action"] == "replace" and "replacement" not in by_id["G-sig.0"]
    assert by_id["R-residual.0"]["action"] == "remove"


def test_plan_names_only_mentions_listed_in_mentions_json() -> None:
    by_id = {r["id"]: r["mentions"] for r in _files()["plan.json"]["regions"]}
    assert by_id["M-person-a.0"] == ["M-person-a"]
    assert by_id["L-line.0"] == ["M-phone"]  # the line id is not a mention; what it carries is
    assert by_id["G-sig.0"] == ["G-sig"]
    assert by_id["G-fig2.0"] == [] and by_id["R-residual.0"] == []


def test_clusters_list_resolved_people_with_their_listed_mentions() -> None:
    clusters = _files()["entities.json"]["clusters"]
    # P001 holds only a kept professional, so it is not reported.
    assert clusters == [{"id": "P000", "mentions": ["M-person-a", "M-person-b"]}]


def test_pseudonyms_map_mentions_and_originals_to_replacements() -> None:
    pseudonyms = _files()["pseudonyms.json"]
    assert pseudonyms["mentions"] == [
        {"mention": "M-person-a", "replacement": "Bea Sample"},
        {"mention": "M-phone", "replacement": "0491570110"},
        {"mention": "M-person-b", "replacement": "Bea Sample"},
    ]
    assert pseudonyms["entries"] == [
        {"cluster": "P000", "original": "Ann Example", "replacement": "Bea Sample"},
        {"cluster": None, "original": "0491570006", "replacement": "0491570110"},
    ]


def test_pseudonyms_leave_out_surrogates_that_were_never_drawn() -> None:
    # On a sideways line (a vertical table header) the planner blacks a mention out instead of typesetting
    # the surrogate it was given: that surrogate appears nowhere in the output.
    tok = _tok(0, 0, 0, BBox(2200, 1000, 2260, 1400), "Zed")
    quad = ((2260.0, 1400.0), (2260.0, 1000.0), (2200.0, 1000.0), (2200.0, 1400.0))
    sideways = Line("p0000.l0000", 0, 0, tok.bbox, quad, 0.9, tok.engine, (tok,))
    decision = Decision(_mention("M-sideways", EntityType.PERSON, [tok]), ActionKind.SURROGATE)
    surrogates = {"M-sideways": MentionSurrogate("M-sideways", ("Yan",))}
    redaction = load_config("client").redaction
    ops = plan_ops([decision], {tok.token_id: tok}, {0: GEOM0}, redaction, surrogates, rotated_token_ids([sideways]))
    page = PageAnalysis(GEOM0, PageKind.IMAGE_ONLY, (sideways,))
    files = _files(JobTrace(PDF, (page,), (decision,), None, surrogates, ops, {}, ()))
    assert [m["id"] for m in files["mentions.json"]["mentions"]] == ["M-sideways"]
    assert [(r["action"], r.get("replacement")) for r in files["plan.json"]["regions"]] == [("remove", None)]
    assert files["pseudonyms.json"] == {"schema": "rb.pseudonyms/0.1", "entries": [], "mentions": []}


def test_run_info_records_the_release_verdict() -> None:
    info = run_info(
        version="0.1.0+gabc1234",
        config={"profile": "client"},
        seed=None,
        models=[{"id": "ppocrv6_det_small", "version": "v6", "license": "Apache-2.0", "files": []}],
        started_at="2026-10-21T00:00:00Z",
        finished_at="2026-10-21T00:01:00Z",
        notes=["a", "b"],
        released=False,
        checks_failed=["V4_RESIDUAL", "V2_TEXT_LEAK"],
    )
    assert info["models"] == [{"name": "ppocrv6_det_small", "version": "v6"}]
    assert info["release"] == {"released": False, "checks_failed": ["V2_TEXT_LEAK", "V4_RESIDUAL"]}
    assert info["notes"] == "a; b" and info["candidate"] == "mlredact"


def test_the_note_on_unreleased_output_says_whether_verification_held_it_back() -> None:
    assert unreleased_note(["V4_RESIDUAL", "V2_TEXT_LEAK"]).startswith("held back by the tool's own verification")
    # A step that fails after the build stops the checks before they finish: nothing was held back.
    for reasons in (["E_OCR_FAILED"], ["V4_RESIDUAL", "E_INTERNAL"], []):
        note = unreleased_note(reasons)
        assert note.startswith("a step failed after the output was built") and "verification" not in note
    assert "(E_OCR_FAILED)" in unreleased_note(["E_OCR_FAILED"])


def test_region_decisions_follow_the_policy_and_the_render_style() -> None:
    page = PageAnalysis(GEOM0, PageKind.IMAGE_ONLY, (), regions=(SIGNATURE, BARCODE, FAINT, FIGURE))

    def decided(style: str) -> dict[str, ActionKind]:
        redactor = object.__new__(Redactor)  # the decisions need only the configuration
        redactor.config = load_config("client", (), {"redaction": {"style": style}})
        return {d.region.region_id: d.action for d in redactor._region_decisions(page)}

    expected = {"G-bar": ActionKind.REMOVE, "G-faint": ActionKind.SUPPRESS_INK, "G-fig": ActionKind.KEEP}
    assert decided("surrogate") == {"G-sig": ActionKind.SYNTHESISE, **expected}
    assert decided("blackout") == {"G-sig": ActionKind.REMOVE, **expected}  # stand-ins only with surrogates


# ------------------------------------------------------------------------------ environment
def test_version_names_the_commit_and_marks_uncommitted_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "pyproject.toml").write_text("", "utf-8")
    status = {"out": ""}

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        out = "abc1234\n" if "rev-parse" in cmd else status["out"]
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(bench_env.shutil, "which", lambda _name: "git")
    monkeypatch.setattr(bench_env.subprocess, "run", fake_run)
    assert bench_env.candidate_version(tmp_path) == f"{__version__}+gabc1234"
    status["out"] = " M src/mlredact/cli/main.py\n"
    assert bench_env.candidate_version(tmp_path) == f"{__version__}+gabc1234.dirty"


def test_version_is_the_plain_version_without_git_or_a_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert bench_env.candidate_version(tmp_path) == __version__  # no pyproject.toml: not a checkout
    (tmp_path / "pyproject.toml").write_text("", "utf-8")
    monkeypatch.setattr(bench_env.shutil, "which", lambda _name: None)
    assert bench_env.candidate_version(tmp_path) == __version__


# ------------------------------------------------------------------------------ the command
MODELS = [{"id": "ppocrv6_det_small", "version": "v6", "license": "Apache-2.0", "files": []}]


def _manifest(reasons: list[str]) -> dict[str, Any]:
    return {"reasons": reasons, "job": {"input_sha256": "0" * 64}, "provenance": {"models": MODELS}}


class _StubRedactor:
    configs: ClassVar[list[AppConfig]] = []
    calls: ClassVar[list[dict[str, Any]]] = []
    outcome: ClassVar[str] = "released"
    failure: ClassVar[Exception | None] = None

    def __init__(self, config: AppConfig, **_: Any) -> None:
        if self.failure is not None:
            raise self.failure
        self.configs.append(config)

    def __enter__(self) -> _StubRedactor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, data: bytes, **kwargs: Any) -> JobResult:
        self.calls.append(kwargs)
        if self.outcome == "released":
            return JobResult(JobStatus.RELEASED, PDF, _manifest([]), {"workers": 1}, trace=make_trace(PDF))
        if self.outcome == "held":
            held = b"%PDF-1.7 held back"
            return JobResult(JobStatus.QUARANTINED, None, _manifest(["V4_RESIDUAL"]), {}, trace=make_trace(held))
        return JobResult(JobStatus.QUARANTINED, None, _manifest(["E_RENDER_FAILED"]), {}, trace=make_trace(None))


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_StubRedactor]:
    _StubRedactor.configs, _StubRedactor.calls = [], []
    _StubRedactor.outcome, _StubRedactor.failure = "released", None
    monkeypatch.setattr(runner_module, "Redactor", _StubRedactor)
    monkeypatch.setattr(bench_env, "candidate_version", lambda root=None: "0.1.0+gtest")
    for name in ("MLREDACT_CONFIG", "MLREDACT_MODELS_DIR", "MLREDACT_SURROGATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return _StubRedactor


def _invoke(tmp_path: Path, *extra: str) -> Result:
    src = tmp_path / "input.pdf"
    src.write_bytes(b"%PDF-1.7 stub input")
    return CliRunner().invoke(app, ["bench", "--input", str(src), "--workdir", str(tmp_path / "out"), *extra])


def _bench(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    return _invoke(tmp_path, *extra).exit_code, tmp_path / "out"


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


STAGE_FILES = ["entities.json", "mentions.json", "plan.json", "pseudonyms.json", "text.json"]


def test_released_job_writes_output_and_every_stage_file(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    code, out = _bench(tmp_path, "--seed", "4")
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["manifest.json", "output.pdf", "run.json", "run_info.json", *STAGE_FILES]
    )
    assert (out / "output.pdf").read_bytes() == PDF
    info = _json(out / "run_info.json")
    assert info["release"] == {"released": True, "checks_failed": []}
    assert info["seed"] == 4 and info["candidate_version"] == "0.1.0+gtest"
    assert info["models"] == [{"name": "ppocrv6_det_small", "version": "v6"}]
    assert info["config"]["profile"] == "client" and info["config"]["scope"] == "document"
    assert len(info["config"]["config_hash"]) == 64
    assert _json(out / "mentions.json") == mentions_artifact(make_trace())
    assert stub.calls[0]["trace"] is True and stub.calls[0]["scope_id"] is None


def test_held_back_output_is_handed_over_with_the_tools_verdict(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    stub.outcome = "held"
    code, out = _bench(tmp_path)
    assert code == 0
    assert (out / "output.pdf").read_bytes() == b"%PDF-1.7 held back"
    info = _json(out / "run_info.json")
    assert info["release"] == {"released": False, "checks_failed": ["V4_RESIDUAL"]}
    assert "held back by the tool's own verification" in info["notes"]
    assert _json(out / "quarantine.json")["reasons"] == ["V4_RESIDUAL"]
    assert all((out / name).is_file() for name in STAGE_FILES)


def test_a_failure_while_writing_the_files_leaves_no_output_and_says_why(
    tmp_path: Path, stub: type[_StubRedactor], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise KeyError("Ann Example")  # an exception's text never reaches the console

    monkeypatch.setattr(bench_artifacts, "stage_files", broken)
    stub.outcome = "held"
    res = _invoke(tmp_path)
    out = tmp_path / "out"
    assert res.exit_code == 1 and "Ann" not in res.output
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "quarantine.json", "run.json", "run_info.json"]
    info = _json(out / "run_info.json")
    assert info["release"] == {"released": False, "checks_failed": ["E_INTERNAL", "V4_RESIDUAL"]}
    assert "no output.pdf" in info["notes"] and "held back" not in info["notes"]


def test_no_output_exits_1_with_run_info_only(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    stub.outcome = "failed"
    code, out = _bench(tmp_path)
    assert code == 1
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "quarantine.json", "run.json", "run_info.json"]
    assert _json(out / "run_info.json")["release"] == {"released": False, "checks_failed": ["E_RENDER_FAILED"]}


def test_a_job_that_cannot_start_still_records_why(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    stub.failure = EnvironmentErrorMl(ReasonCode.MODEL_MISSING)
    code, out = _bench(tmp_path)
    assert code == 1
    assert sorted(p.name for p in out.iterdir()) == ["run_info.json"]
    assert _json(out / "run_info.json")["release"] == {"released": False, "checks_failed": ["E_MODEL_MISSING"]}


CANARY = "Zelphinia Quarrington"


def _inject_dir(folder: Path, **replaced: Any) -> Path:
    """A folder of injected stage inputs (one fictitious name), every file any mode reads; ``replaced``: files by
    stem, as JSON values or raw ``bytes`` (None leaves the file out)."""
    box = [[60, 100], [90, 100], [90, 111], [60, 111]]
    files: dict[str, Any] = {
        "text": {"schema": "rb.text/0.1", "source": "text_layer", "pages": [{"index": 0, "tokens": []}]},
        "mentions": {"schema": "rb.mentions/0.1", "mentions": [{"id": "m1", "label": "PERSON_NAME", "tokens": ["w0"]}]},
        "entities": {"schema": "rb.entities/0.1", "clusters": [{"id": "e1", "mentions": ["m1"]}]},
        "plan": {
            "schema": "rb.plan/0.1",
            "regions": [{"id": "m1-0", "page": 0, "poly": box, "action": "replace", "mentions": ["m1"]}],
        },
    }
    files["text"]["pages"][0]["tokens"] = [{"id": "w0", "text": "Ann", "poly": box, "line": "l0", "order": 0}]
    files.update(replaced)
    folder.mkdir(parents=True, exist_ok=True)
    for stem, content in files.items():
        if content is not None:
            data = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
            (folder / f"{stem}.json").write_bytes(data)
    return folder


@pytest.mark.parametrize(
    ("mode", "injected"),
    [
        ("oracle_text", ["text"]),
        ("oracle_mentions", ["mentions", "text"]),
        ("oracle_entities", ["entities", "mentions", "text"]),
        ("oracle_regions", ["entities", "mentions", "regions", "text"]),
    ],
)
def test_oracle_modes_run_with_the_injected_inputs(
    tmp_path: Path, stub: type[_StubRedactor], mode: str, injected: list[str]
) -> None:
    code, out = _bench(tmp_path, "--mode", mode, "--inject-dir", str(_inject_dir(tmp_path / "inject")))
    assert code == 0 and (out / "output.pdf").read_bytes() == PDF
    (call,) = stub.calls
    assert call["trace"] is True and isinstance(call["inject"], OracleInputs) and call["inject"].mode == mode
    # NER finds mentions: where they are injected it is switched off (the configuration hash records it).
    ner = stub.configs[0].detection.ner
    ner_on = mode == "oracle_text"
    assert (ner.gliner.enabled, ner.privacy_filter.enabled) == (ner_on, ner_on)
    info = _json(out / "run_info.json")
    assert {k: info["config"][k] for k in ("mode", "injected", "ner_enabled")} == {
        "mode": mode,
        "injected": injected,
        "ner_enabled": ner_on,
    }
    assert "injected text: 1 tokens" in info["notes"] and "V4 residual check not run" in info["notes"]


def test_e2e_has_no_injection_and_records_no_mode(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    code, out = _bench(tmp_path)
    assert code == 0 and stub.calls[0]["inject"] is None
    assert not {"mode", "injected", "ner_enabled"} & set(_json(out / "run_info.json")["config"])
    assert stub.configs[0].detection.ner.gliner.enabled  # NER is not touched


@pytest.mark.parametrize("inject_dir", [None, "", "missing", "a file"])
def test_an_oracle_mode_needs_a_folder_of_injected_inputs(
    tmp_path: Path, stub: type[_StubRedactor], inject_dir: str | None
) -> None:
    (tmp_path / "a file").write_text("{}", "utf-8")
    extra = [] if inject_dir is None else ["--inject-dir", str(tmp_path / inject_dir) if inject_dir else ""]
    code, out = _bench(tmp_path, "--mode", "oracle_mentions", *extra)
    assert code == 2 and not out.exists() and stub.calls == []


@pytest.mark.parametrize(
    ("mode", "replaced"),
    [
        ("oracle_text", {"text": {"schema": "rb.text/0.1", "source": CANARY, "pages": []}}),
        ("oracle_mentions", {"mentions": None}),
        ("oracle_entities", {"entities": b"{not json " + CANARY.encode()}),
        ("oracle_regions", {"plan": {"schema": "rb.plan/0.1", "regions": [{"id": "r", "page": 0, CANARY: 1}]}}),
        ("oracle_regions", {"plan": None}),
    ],
)
def test_unreadable_injected_inputs_leave_only_run_info(
    tmp_path: Path, stub: type[_StubRedactor], mode: str, replaced: dict[str, Any]
) -> None:
    res = _invoke(tmp_path, "--mode", mode, "--inject-dir", str(_inject_dir(tmp_path / "inject", **replaced)))
    out = tmp_path / "out"
    assert res.exit_code == 1 and CANARY not in res.output
    assert sorted(p.name for p in out.iterdir()) == ["run_info.json"] and stub.calls == []
    info = _json(out / "run_info.json")
    assert info["release"] == {"released": False, "checks_failed": ["E_CONFIG_INVALID"]}
    assert info["config"]["mode"] == mode and CANARY not in json.dumps(info)


@pytest.mark.parametrize("mode", ["e2e2", "oracle_plan", "ORACLE_REGIONS"])
def test_unsupported_modes_exit_3(tmp_path: Path, stub: type[_StubRedactor], mode: str) -> None:
    code, out = _bench(tmp_path, "--mode", mode, "--inject-dir", str(_inject_dir(tmp_path / "inject")))
    assert code == 3
    assert not out.exists() and stub.calls == []


def test_the_help_names_every_mode_and_what_it_injects(stub: type[_StubRedactor]) -> None:
    res = CliRunner().invoke(app, ["bench", "--help"], terminal_width=200)
    assert res.exit_code == 0
    text = " ".join(res.output.split())
    for mode in ("e2e", "oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions"):
        assert mode in text
    assert all(name in text for name in ("text.json", "mentions.json", "entities.json", "plan.json"))
    assert "3 unsupported mode" in text


@pytest.mark.parametrize("mode", ["oracle_mentions", "oracle_entities", "oracle_regions"])
def test_a_profile_that_keeps_injected_regions_is_refused(tmp_path: Path, stub: type[_StubRedactor], mode: str) -> None:
    keeps = tmp_path / "keeps.yaml"
    keeps.write_text("policy:\n  region_actions:\n    signature: keep\n", "utf-8")
    inject = str(_inject_dir(tmp_path / "inject"))
    code, out = _bench(tmp_path, "--mode", mode, "--inject-dir", inject, "--config", str(keeps))
    assert code == 1 and stub.calls == []
    assert _json(out / "run_info.json")["release"]["checks_failed"] == ["E_CONFIG_INVALID"]
    # Tool 2 detects the regions itself in oracle_text: the profile is its own business there.
    code, _out = _bench(tmp_path, "--mode", "oracle_text", "--inject-dir", inject, "--config", str(keeps))
    assert code == 0


def test_e2e_takes_no_injected_inputs(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    code, _out = _bench(tmp_path, "--inject-dir", str(tmp_path))
    assert code == 2 and stub.calls == []


def test_scope_id_is_passed_only_when_given(tmp_path: Path, stub: type[_StubRedactor]) -> None:
    _bench(tmp_path, "--scope-id", "")
    code, out = _bench(tmp_path, "--scope-id", "MATTER-7")
    assert code == 0
    assert [c["scope_id"] for c in stub.calls] == [None, "MATTER-7"]
    assert _json(out / "run_info.json")["config"]["scope"] == "caller"


def test_env_configs_apply_before_config_options(
    tmp_path: Path, stub: type[_StubRedactor], monkeypatch: pytest.MonkeyPatch
) -> None:
    env_a, env_b, cli = tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "cli.yaml"
    env_a.write_text("ocr:\n  engine_b:\n    enabled: false\n", "utf-8")
    env_b.write_text("runtime:\n  workers: 2\n", "utf-8")
    cli.write_text("runtime:\n  workers: 3\n", "utf-8")
    monkeypatch.setenv("MLREDACT_CONFIG", f"{env_a}{bench_env.os.pathsep}{env_b}")
    code, out = _bench(tmp_path, "--config", str(cli))
    assert code == 0
    cfg = stub.configs[0]
    assert cfg.ocr.engine_b.enabled is False and cfg.runtime.workers == 3
    assert _json(out / "run_info.json")["config"] | {"config_hash": ""} == {
        "profile": "client",
        "render_style": "surrogate",
        "config_hash": "",
        "scope": "document",
        "workers": 3,
        "engine_b_enabled": False,
    }


def test_a_missing_env_config_is_a_usage_error(
    tmp_path: Path, stub: type[_StubRedactor], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLREDACT_CONFIG", str(tmp_path / "missing.yaml"))
    code, _out = _bench(tmp_path)
    assert code == 2 and stub.calls == []


def test_models_fall_back_to_the_backends_own_folder(
    tmp_path: Path, stub: type[_StubRedactor], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = tmp_path / "backend"
    (backend / ".models").mkdir(parents=True)
    monkeypatch.setattr(bench_env, "backend_root", lambda: backend)
    code, out = _bench(tmp_path)  # the current folder (tmp_path) has no .models
    assert code == 0
    assert stub.configs[-1].runtime.models_dir == str(backend / ".models")
    assert "models from the backend's own .models folder" in _json(out / "run_info.json")["notes"]
    monkeypatch.setenv("MLREDACT_MODELS_DIR", str(tmp_path))  # an explicit folder always wins
    _bench(tmp_path)
    monkeypatch.delenv("MLREDACT_MODELS_DIR")
    (tmp_path / ".models").mkdir()  # so does a configured folder that exists here
    _bench(tmp_path)
    assert [c.runtime.models_dir for c in stub.configs[1:]] == [".models", ".models"]


def test_run_info_says_which_surrogate_key_was_used(
    tmp_path: Path, stub: type[_StubRedactor], monkeypatch: pytest.MonkeyPatch
) -> None:
    _code, out = _bench(tmp_path)
    assert "public development surrogate key" in _json(out / "run_info.json")["notes"]
    assert stub.configs[0].surrogate.allow_development_key is True
    monkeypatch.setenv("MLREDACT_SURROGATE_KEY", "ab" * 32)
    _code, out = _bench(tmp_path)
    notes = _json(out / "run_info.json")["notes"]
    assert "configured surrogate key (env)" in notes and "development" not in notes


def test_bench_version_prints_the_candidate_version(stub: type[_StubRedactor]) -> None:
    res = CliRunner().invoke(app, ["bench-version"])
    assert res.exit_code == 0 and res.stdout.strip() == "0.1.0+gtest"
