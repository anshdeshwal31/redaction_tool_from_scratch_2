"""Benchmark adapter: stage files from a job trace.

All text is fictitious.  The schemas themselves belong to the benchmark and are checked there.
"""

from __future__ import annotations

from typing import Any

import pytest

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
from mlredact.config.loader import load_config
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
