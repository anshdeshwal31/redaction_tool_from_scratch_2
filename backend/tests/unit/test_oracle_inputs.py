"""Injected mentions, clusters and plans as tool 2's stages (``mlredact.bench.oracle``, the benchmark's oracle modes):
where the mentions land on tool 2's tokens, what they become (mentions, regions), how shared tokens are redacted, the
types, the clusters, the decisions, the operations that apply an injected plan, and what the stage files are told.

Pages are built from the injected tokens on a hand-made A4 page at 150 dpi, without rendering anything.  The files are
written in the benchmark's own format (no benchmark import).  All text is fictitious.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from synthdocs.generator import medicare_number, provider_number

from inject_support import busy_docs, oracle_outcome, shuffled
from mlredact.bench.artifacts import display_poly, stage_files
from mlredact.bench.inject import load_injection
from mlredact.bench.oracle import OracleInputs, check_region_actions
from mlredact.config.loader import load_config
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import PageAnalysis, PageGeometry, Region, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType, PageKind, RegionKind
from mlredact.pipeline.injection import (
    InjectedPage,
    InjectedStage,
    InjectReport,
    PlanEcho,
    ReportedMention,
    injected_lines,
    pt_box,
)
from mlredact.pipeline.runner import JobTrace, Redactor
from mlredact.render.plan import rotated_token_ids
from mlredact.surrogate.assign import MentionSurrogate
from mlredact.verify.residual import ResidualHit

E = EntityType
GEOM = PageGeometry(0, 595.276, 841.89, 150, 1240, 1754)
CONFIG = load_config("client")
CANARY = "Zelphinia Quarrington"


# ------------------------------------------------------------------------------------------ helpers
def poly(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def layout(*lines: str) -> list[dict[str, Any]]:
    """Tokens ``t<line>.<word>`` for lines of words, in reading order: 11 pt high, lines 20 pt apart."""
    tokens: list[dict[str, Any]] = []
    for row, line in enumerate(lines):
        x, y = 60.0, 100.0 + 20.0 * row
        for k, word in enumerate(line.split()):
            width = 5.5 * len(word)
            box = poly(x, y, x + width, y + 11.0)
            tokens.append({"id": f"t{row}.{k}", "text": word, "poly": box, "line": f"l{row}", "order": len(tokens)})
            x += width + 3.0
    return tokens


def polys(tokens: list[dict[str, Any]], *ids: str) -> list[dict[str, Any]]:
    """Region anchors on the tokens ``ids`` (their word boxes)."""
    by_id = {t["id"]: t["poly"] for t in tokens}
    return [{"page": 0, "poly": by_id[i]} for i in ids]


def mention(mention_id: str, label: str, **anchors: Any) -> dict[str, Any]:
    return {"id": mention_id, "label": label, **anchors}


@dataclass
class Staged:
    inputs: OracleInputs
    stage: InjectedStage
    report: InjectReport

    @property
    def mentions(self) -> list[tuple[str, str]]:
        """(type, value) of tool 2's mentions."""
        return [(m.entity_type.value, m.value) for m in self.stage.detection.mentions]

    @property
    def reported(self) -> dict[str, ReportedMention]:
        assert self.report.mentions is not None
        return {m.mention_id: m for m in self.report.mentions}

    @property
    def regions(self) -> tuple[Region, ...]:
        return self.stage.pages[0].regions

    def internal(self, value: str) -> str:
        """The id of tool 2's mention with ``value``."""
        (found,) = [m.mention_id for m in self.stage.detection.mentions if m.value == value]
        return found

    def note(self, prefix: str) -> str:
        (found,) = [n for n in self.report.notes if n.startswith(prefix)]
        return found


# Injected tokens: of page 0, or by page.
Tokens = list[dict[str, Any]] | dict[int, list[dict[str, Any]]]


def docs(tokens: Tokens, mentions: list[dict[str, Any]], clusters: Any = None) -> dict[str, Any]:
    by_page = tokens if isinstance(tokens, dict) else {0: tokens}
    pages = [{"index": i, "tokens": t} for i, t in sorted(by_page.items())]
    out: dict[str, Any] = {
        "text.json": {"schema": "rb.text/0.1", "source": "text_layer", "pages": pages},
        "mentions.json": {"schema": "rb.mentions/0.1", "mentions": mentions},
    }
    if clusters is not None:
        out["entities.json"] = {
            "schema": "rb.entities/0.1",
            "clusters": [{"id": c, "mentions": list(m)} for c, m in clusters],
        }
    return out


def write(folder: Path, files: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content), "utf-8")
    return folder


def analysed(page: InjectedPage) -> PageAnalysis:
    geometry = replace(GEOM, index=page.index)
    return PageAnalysis(geometry, PageKind.BORN_DIGITAL, injected_lines(page, geometry))


def stage(
    folder: Path,
    tokens: Tokens,
    mentions: list[dict[str, Any]],
    clusters: Any = None,
    config: Any = CONFIG,
) -> Staged:
    """OracleInputs.stage on pages made of the injected tokens (``oracle_entities`` when clusters are given)."""
    mode = "oracle_mentions" if clusters is None else "oracle_entities"
    inputs = OracleInputs(load_injection(mode, write(folder, docs(tokens, mentions, clusters))))
    count = max(tokens) + 1 if isinstance(tokens, dict) else 1
    staged = inputs.stage([analysed(p) for p in inputs.pages(count)], config, "doc")
    assert staged is not None
    return Staged(inputs, staged, inputs.report())


# ------------------------------------------------------------------------------------------ anchors
def test_mentions_are_placed_by_token_ids_by_span_or_by_their_boxes(tmp_path: Path) -> None:
    tokens = layout("Ms Jane Citizen", "Ring 0412 345 678 today", "Seen by Dr Peter Brown")
    page_text = " ".join(t["text"] for t in tokens)
    start = page_text.index("0412 345 678")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("by-ids", "PERSON_NAME", tokens=["t0.1", "t0.2"]),
            mention("by-span", "PHONE", span={"page": 0, "start": start, "end": start + 12}),
            mention("by-boxes", "PERSON_NAME", regions=polys(tokens, "t2.3", "t2.4")),
        ],
    )
    assert s.mentions == [("person", "Jane Citizen"), ("phone", "0412 345 678"), ("person", "Peter Brown")]
    reported = s.reported
    assert [reported[i].token_ids for i in ("by-ids", "by-span", "by-boxes")] == [
        ("p0000.l0000.w001", "p0000.l0000.w002"),
        ("p0000.l0001.w001", "p0000.l0001.w002", "p0000.l0001.w003"),
        ("p0000.l0002.w003", "p0000.l0002.w004"),
    ]
    assert s.regions == ()  # word boxes are their words
    assert "1 placed by token ids, 1 by span, 1 by region anchors, 0 not placed" in s.note("injected mentions")


def test_region_labels_become_regions(tmp_path: Path) -> None:
    tokens = layout("Initialled JS", "and again KL", "Ms Jane Citizen")
    signature = poly(60, 400, 200, 440)
    s = stage(
        tmp_path,
        tokens,
        [
            mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": signature}], conf=0.8),
            mention("marks", "HANDWRITTEN_MARK", tokens=["t0.1", "t1.2"]),
            mention("person", "PERSON_NAME", tokens=["t2.1", "t2.2"]),
        ],
    )
    assert s.mentions == [("person", "Jane Citizen")]  # the marks are regions, not text
    lines = s.stage.pages[0].lines
    by_kind = {(r.kind, r.bbox) for r in s.regions}
    assert by_kind == {
        (RegionKind.SIGNATURE, pt_box([tuple(p) for p in signature], GEOM)),
        (RegionKind.HANDWRITING, lines[0].tokens[1].bbox),  # one region per line of its words
        (RegionKind.HANDWRITING, lines[1].tokens[2].bbox),
    }
    sig = next(r for r in s.regions if r.kind is RegionKind.SIGNATURE)
    assert (sig.detector, sig.rule, sig.score) == ("inject", "inject.SIGNATURE", 0.8)
    assert s.report.aliases[sig.region_id] == ("sig",)
    reported = s.reported
    assert (reported["sig"].label, reported["sig"].token_ids, reported["sig"].conf) == ("signature", (), 0.8)
    assert (reported["marks"].label, len(reported["marks"].token_ids), reported["marks"].slots) == (
        "handwriting",
        2,
        (),
    )
    assert len(reported["marks"].boxes) == 4  # its two words, and the two regions they became


def test_anchors_its_words_do_not_explain_become_regions(tmp_path: Path) -> None:
    tokens = layout("HARBOURSIDE MEDICOLEGAL", "Email john.smith78@example.org today")
    logo = poly(50, 90, 200, 120)  # a letterhead logo around the name
    email = tokens[3]["poly"]
    characters = poly(email[0][0] + 27.5, email[0][1], email[0][0] + 55.0, email[2][1])  # "smith" inside the word
    s = stage(
        tmp_path,
        tokens,
        [
            mention(
                "org",
                "ORGANISATION",
                tokens=["t0.0", "t0.1"],
                regions=[*polys(tokens, "t0.0", "t0.1"), {"page": 0, "poly": logo}],
            ),
            mention("person", "PERSON_NAME", tokens=["t1.1"], regions=[{"page": 0, "poly": characters}]),
            mention("code", "ID_OTHER", regions=[{"page": 0, "poly": poly(400, 600, 460, 660)}]),
        ],
    )
    assert {(r.kind, r.rule) for r in s.regions} == {
        (RegionKind.LOGO, "inject.ORGANISATION"),
        (RegionKind.BARCODE, "inject.ID_OTHER"),  # a text label with no word at all
    }
    assert s.mentions == [("organisation", "HARBOURSIDE MEDICOLEGAL"), ("person", "john.smith78@example.org")]
    reported = s.reported
    assert len(reported["org"].boxes) == 3 and reported["org"].label == "organisation"
    assert (reported["code"].label, reported["code"].token_ids) == ("other_id", ())
    assert "2 anchors not explained by their tokens" in s.note("injected mentions")


def test_mentions_that_land_nowhere_are_counted(tmp_path: Path) -> None:
    tokens = layout("Ms Jane Citizen")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("unknown-ids", "PERSON_NAME", tokens=["nope", "t9.9"]),
            mention("page-2", "SIGNATURE", regions=[{"page": 2, "poly": poly(60, 400, 200, 440)}]),
            mention("person", "PERSON_NAME", tokens=["t0.1", "nope"]),
        ],
    )
    assert set(s.reported) == {"person"} and s.regions == ()
    note = s.note("injected mentions")
    assert "1 placed by token ids, 0 by span, 0 by region anchors, 2 not placed" in note
    assert "(3 token ids not found, 1 anchors beyond the last page)" in note


def test_a_box_inside_a_word_is_the_word(tmp_path: Path) -> None:
    """A mention anchored only by the box of some of a word's characters (part of an e-mail address) takes the whole
    word, as its token id or a span would give it, and no removal region."""
    tokens = layout("Email john.smith78@example.org today")
    email = tokens[1]["poly"]
    characters = poly(email[0][0] + 27.5, email[0][1], email[0][0] + 55.0, email[2][1])  # "smith", a fifth of it
    person = mention("person", "PERSON_NAME", regions=[{"page": 0, "poly": characters}])
    s = stage(tmp_path / "alone", tokens, [person])
    assert s.mentions == [("person", "john.smith78@example.org")] and s.regions == ()
    assert s.reported["person"].token_ids == ("p0000.l0000.w001",)
    note = s.note("injected mentions")
    assert "0 placed by token ids, 0 by span, 1 by region anchors, 0 not placed" in note
    assert "0 anchors not explained by their tokens" in note
    # With the address injected too, the name is inside it.
    both = stage(tmp_path / "both", tokens, [person, mention("email", "EMAIL", tokens=["t0.1"])])
    assert both.mentions == [("email", "john.smith78@example.org")] and both.regions == ()
    assert both.report.aliases[both.internal("john.smith78@example.org")] == ("email", "person")
    # Its plan region, the characters' box, has the word's fake drawn over the whole word.
    p = planned(tmp_path / "plan", tokens, [person], [plan_region("person-0", characters, "person")])
    (op,) = p.page_ops
    assert (op.kind, op.ref_id, op.box) == (ActionKind.SURROGATE, p.staged.internal(p.words[1].text), p.words[1].bbox)


def test_anchors_and_plan_regions_off_their_page_are_not_placed(tmp_path: Path) -> None:
    tokens = layout("Ms Jane Citizen")
    off = poly(600, 100, 660, 120)  # right of the page (A4: 595 points wide)
    mentions = [
        mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": off}]),
        mention("person", "PERSON_NAME", tokens=["t0.1"], regions=[*polys(tokens, "t0.1"), {"page": 0, "poly": off}]),
    ]
    s = stage(tmp_path / "stage", tokens, mentions)
    # Not a sliver at the page's edge: the signature is not placed, and the name is its word.
    assert set(s.reported) == {"person"} and s.regions == ()
    note = s.note("injected mentions")
    assert "1 placed by token ids, 0 by span, 0 by region anchors, 1 not placed" in note
    assert "2 anchors off their page" in note
    # Nor is a plan region: nothing is applied there, and plan.json does not say it was.
    p = planned(tmp_path / "plan", tokens, mentions, [*word_regions(tokens, "person", "t0.1"), plan_region("off", off)])
    assert set(p.echoes) == {"person-0"} and [op.ref_id for op in p.page_ops] == [p.staged.internal("Jane")]
    assert "1 applied by 1 operations" in p.note and "1 off their page" in p.note


# ----------------------------------------------------------------------------------- shared tokens
def test_a_name_inside_an_organisation_is_redacted_with_it(tmp_path: Path) -> None:
    tokens = layout("Smith & Partners Lawyers")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("person", "PERSON_NAME", tokens=["t0.0"]),
            mention("org", "ORGANISATION", tokens=["t0.0", "t0.1", "t0.2", "t0.3"]),
        ],
    )
    assert s.mentions == [("organisation", "Smith & Partners Lawyers")]
    org = s.internal("Smith & Partners Lawyers")
    assert s.report.aliases[org] == ("org", "person")  # the owner first
    reported = s.reported
    assert (reported["org"].owner, reported["person"].owner) == (True, False)
    assert (reported["person"].label, reported["person"].slots) == ("person", ((org, 0),))
    assert "1 sharing tokens another one owns" in s.note("injected mentions")


@pytest.mark.parametrize(
    ("line", "outer", "inner", "kind"),
    [
        # The e-mail address over the name inside it (its text is the longer).
        ("john.smith78@example.org", "EMAIL", ("PERSON_NAME", "smith"), E.EMAIL),
        # The file path over the court reference inside it: type priority alone would choose the court file.
        (r"C:\Matters\NSWSC-2024-1234.pdf", "FILE_PATH", ("ID_LEGAL_REF", "NSWSC-2024-1234"), E.FILE_PATH),
    ],
)
def test_one_token_two_mentions_takes_the_longer_text(
    tmp_path: Path, line: str, outer: str, inner: tuple[str, str], kind: EntityType
) -> None:
    tokens = layout(line)
    same = polys(tokens, "t0.0")  # equal boxes, as anchors of part of a word look without character boxes
    s = stage(
        tmp_path,
        tokens,
        [
            mention("a-inner", inner[0], text=inner[1], tokens=["t0.0"], regions=same),
            mention("b-outer", outer, tokens=["t0.0"], regions=same),
        ],
    )
    assert s.mentions == [(kind.value, line)]
    assert s.report.aliases[s.internal(line)] == ("b-outer", "a-inner")  # the owner first, whatever the ids
    assert s.reported["b-outer"].owner and not s.reported["a-inner"].owner
    assert s.reported["a-inner"].label == {"PERSON_NAME": "person", "ID_LEGAL_REF": "court_file"}[inner[0]]


def test_overlapping_mentions_are_redacted_as_their_union(tmp_path: Path) -> None:
    tokens = layout("Seen at Dr Peter Brown Medical Group today")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("person", "PERSON_NAME", tokens=["t0.3", "t0.4"]),
            mention("org", "ORGANISATION", tokens=["t0.4", "t0.5", "t0.6"]),
        ],
    )
    assert s.mentions == [("organisation", "Peter Brown Medical Group")]
    assert s.report.aliases[s.internal("Peter Brown Medical Group")] == ("org", "person")


def test_a_mention_in_pieces_is_redacted_piece_by_piece(tmp_path: Path) -> None:
    tokens = layout("Jane (nee Jones) Citizen", "Dr Peter", "Brown")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("pieces", "PERSON_NAME", tokens=["t0.0", "t0.3"]),
            # Consecutive in reading order across a line break: one piece.
            mention("broken", "PERSON_NAME", tokens=["t1.1", "t2.0"]),
        ],
    )
    assert s.mentions == [("person", "Jane"), ("person", "Citizen"), ("person", "Peter Brown")]
    pieces = s.reported["pieces"]
    assert pieces.slots == ((s.internal("Jane"), 0), (s.internal("Citizen"), 0))
    assert "1 split into runs" in s.note("injected mentions")


def test_a_mention_across_a_page_break_is_redacted_page_by_page(tmp_path: Path) -> None:
    second = [{**t, "id": f"next.{t['id']}"} for t in layout("Citizen and family")]
    pages = {0: layout("Kind regards, Jane"), 1: second}
    s = stage(tmp_path, pages, [mention("p", "PERSON_NAME", tokens=["t0.2", "next.t0.0"])])
    assert s.mentions == [("person", "Jane"), ("person", "Citizen")]
    assert [page for page, _box in s.reported["p"].boxes] == [0, 1]


# ------------------------------------------------------------------------------------------- types
def test_a_location_s_postcode_is_a_mention_of_its_own(tmp_path: Path) -> None:
    tokens = layout("Penrith NSW 2750", "Sydney NSW 2000,", "2150", "Parramatta")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("penrith", "LOCATION", tokens=["t0.0", "t0.1", "t0.2"]),
            mention("sydney", "LOCATION", tokens=["t1.0", "t1.1", "t1.2"]),  # a capital city
            mention("postcode", "LOCATION", tokens=["t2.0"]),
            mention("parramatta", "LOCATION", tokens=["t3.0"]),
        ],
    )
    assert s.mentions == [
        ("locality", "Penrith NSW"),
        ("postcode", "2750"),
        ("locality", "Sydney NSW"),
        ("postcode", "2000,"),
        ("postcode", "2150"),
        ("locality", "Parramatta"),
    ]
    assert {k: m.label for k, m in s.reported.items()} == {
        "penrith": "locality",
        "sydney": "locality",
        "postcode": "postcode",
        "parramatta": "locality",
    }
    assert s.report.aliases[s.internal("2750")] == ("penrith",)
    # The client keeps places: every injected one is replaced all the same.
    assert {d.action for d in s.stage.decisions} == {ActionKind.SURROGATE}
    note = s.note("injected mentions")
    assert "2 locations split into locality and postcode, 6 of a kept type redacted" in note


def test_identifiers_take_the_type_tool_2_s_rules_recognise(tmp_path: Path) -> None:
    tokens = layout(
        f"Medicare {medicare_number('29537015', 1)}",
        "Licence XK 1234",
        f"Provider {provider_number('212345', 'A')}",
        "ACN 004 085 616",
        "DOB 14/03/1978",
        "Height 182cm",
    )
    s = stage(
        tmp_path,
        tokens,
        [
            mention("medicare", "ID_GOVERNMENT", tokens=["t0.1", "t0.2", "t0.3"]),
            mention("licence", "ID_GOVERNMENT", tokens=["t1.1", "t1.2"]),
            mention("provider", "ID_PROVIDER", tokens=["t2.1"]),
            mention("acn", "ID_BUSINESS", tokens=["t3.1", "t3.2", "t3.3"]),
            mention("dob", "DATE", tokens=["t4.1"]),
            mention("height", "QUASI_IDENTIFIER", tokens=["t5.1"]),
            mention("odd", f"LABEL {CANARY}", tokens=["t5.0"]),
        ],
    )
    types = {k: m.label for k, m in s.reported.items()}
    assert types == {
        "medicare": "medicare",
        "licence": "driver_licence",  # nothing recognised: the family's default
        "provider": "provider_number",
        "acn": "acn",  # the family's default is abn
        "dob": "date_of_birth",  # the date-of-birth fake keeps the year
        "height": "other_id",
        "odd": "other_id",
    }
    rules = {m.value: m.evidence[0].rule_id for m in s.stage.detection.mentions}
    assert rules["182cm"] == "inject.QUASI_IDENTIFIER" and rules["Height"] == "inject.other"  # never the label's text
    assert "2 labels taken as other_id" in s.note("injected mentions")


def test_every_injected_mention_is_redacted(tmp_path: Path) -> None:
    tokens = layout("Jane Citizen 14/03/1978 Penrith")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("person", "PERSON_NAME", tokens=["t0.0", "t0.1"]),
            mention("dob", "DATE", tokens=["t0.2"]),
            mention("place", "LOCATION", tokens=["t0.3"]),
        ],
    )
    assert [(d.mention.entity_type, d.action) for d in s.stage.decisions] == [
        (E.PERSON, ActionKind.SURROGATE),
        (E.DATE_OF_BIRTH, ActionKind.SURROGATE),  # kept by the client profile
        (E.LOCALITY, ActionKind.SURROGATE),  # kept by the client profile
    ]
    assert "2 of a kept type redacted" in s.note("injected mentions")
    detection = s.stage.detection
    assert detection.detectors == (("bench.inject", "1"), ("resolve.persons", "1"))
    assert (detection.low_evidence, detection.propagation_rounds) == ((), 0)


# ---------------------------------------------------------------------------------------- clusters
PEOPLE = ("John Andrew Smith", "Mr Smith", "J. Smith", "Jane Smith")


def people() -> list[dict[str, Any]]:
    return [
        mention("p-john", "PERSON_NAME", tokens=["t0.0", "t0.1", "t0.2"]),
        mention("p-mr", "PERSON_NAME", tokens=["t1.1"]),
        mention("p-j", "PERSON_NAME", tokens=["t2.0", "t2.1"]),
        mention("p-jane", "PERSON_NAME", tokens=["t3.0", "t3.1"]),
    ]


def clusters_of(s: Staged) -> dict[str, str | None]:
    """Each mention's cluster, by value (None for a mention that is not a person's)."""
    persons = s.stage.detection.persons.mentions
    return {m.value: p.cluster if (p := persons.get(m.mention_id)) else None for m in s.stage.detection.mentions}


def test_injected_clusters_replace_tool_2_s(tmp_path: Path) -> None:
    own = stage(tmp_path / "own", layout(*PEOPLE), people())
    assert own.report.clusters is None
    # Tool 2's own: two people named Smith, so it cannot tell whom "Mr Smith" and "J. Smith" are.
    own_clusters = clusters_of(own)
    assert own_clusters == {"John Andrew Smith": "P001", "Smith": None, "J. Smith": None, "Jane Smith": "P000"}
    # "J. Smith" is Jane here; Jane is listed twice and takes the first cluster in id order.
    injected = [("c2", ("p-j", "p-jane")), ("c1", ("p-john", "p-mr", "nobody")), ("c0", ("p-jane",))]
    s = stage(tmp_path / "injected", layout(*PEOPLE), people(), injected)
    assert clusters_of(s) == {"John Andrew Smith": "c1", "Smith": "c1", "J. Smith": "c2", "Jane Smith": "c0"}
    assert s.stage.detection.persons.clusters == {
        "c0": (s.internal("Jane Smith"),),
        "c1": tuple(sorted((s.internal("John Andrew Smith"), s.internal("Smith")))),
        "c2": (s.internal("J. Smith"),),
    }
    assert s.report.clusters == (("c0", ("p-jane",)), ("c1", ("p-john", "p-mr")), ("c2", ("p-j",)))
    note = s.note("injected clusters")
    assert "3 read (0 merged by id), 1 mentions in several clusters" in note
    assert "1 members that are not mentions" in note


def test_a_person_takes_the_cluster_of_the_names_it_absorbed(tmp_path: Path) -> None:
    tokens = layout("Dr Peter Brown", "Brown & Co")
    mentions = [
        mention("full", "PERSON_NAME", tokens=["t0.1", "t0.2"]),
        mention("surname", "PERSON_NAME", tokens=["t0.2"]),
        mention("firm", "ORGANISATION", tokens=["t1.0", "t1.1", "t1.2"]),
        mention("firm-name", "PERSON_NAME", tokens=["t1.0"]),
    ]
    s = stage(tmp_path, tokens, mentions, [("c-brown", ("surname",)), ("c-firm", ("firm-name", "firm"))])
    assert clusters_of(s) == {"Peter Brown": "c-brown", "Brown & Co": None}  # an organisation has no cluster
    assert "1 non-person members" in s.note("injected clusters")


# ------------------------------------------------------------------------------------- the inputs
def test_each_mode_re_plans_its_own_residual_hits(tmp_path: Path) -> None:
    hits = [ResidualHit(0, BBox(0, 0, 9, 9), "phone", source) for source in ("removed_value", "redetect", "region")]
    files = docs(layout("Jane"), [mention("p", "PERSON_NAME", tokens=["t0.0"])], [("c", ("p",))])
    files["plan.json"] = {"schema": "rb.plan/0.1", "regions": []}
    folder = write(tmp_path, files)
    replanned = {
        mode: [h.source for h in OracleInputs(load_injection(mode, folder)).replan_hits(hits)]
        for mode in ("oracle_text", "oracle_mentions", "oracle_entities", "oracle_regions")
    }
    assert replanned == {
        "oracle_text": ["removed_value", "redetect", "region"],
        "oracle_mentions": ["removed_value"],
        "oracle_entities": ["removed_value"],
        "oracle_regions": [],  # exactly the injected regions are applied
    }


def test_text_mode_stages_nothing(tmp_path: Path) -> None:
    inputs = OracleInputs(load_injection("oracle_text", write(tmp_path, docs(layout("Jane Citizen"), []))))
    assert inputs.detect_regions and inputs.plan(**_plan_args()) is None
    (page,) = inputs.pages(1)
    assert inputs.stage([PageAnalysis(GEOM, PageKind.BORN_DIGITAL, injected_lines(page, GEOM))], CONFIG, "doc") is None
    report = inputs.report()
    assert (report.mode, report.text_source, report.mentions, report.clusters, report.plan) == (
        "oracle_text",
        "text_layer",
        None,
        None,
        None,
    )
    assert [n.split(":")[0] for n in report.notes] == ["injected text"]


def _plan_args() -> dict[str, Any]:
    return {
        "decisions": (),
        "surrogates": {},
        "tokens": {},
        "pages": (),
        "region_actions": {},
        "rotated": frozenset(),
        "config": CONFIG,
    }


def test_tokens_beyond_the_document_are_counted(tmp_path: Path) -> None:
    files = docs(layout("Jane Citizen"), [mention("p", "PERSON_NAME", tokens=["t0.0"])])
    beyond = [{**t, "id": f"x{t['id']}"} for t in layout("Elsewhere entirely")]
    files["text.json"]["pages"].append({"index": 3, "tokens": beyond})
    inputs = OracleInputs(load_injection("oracle_mentions", write(tmp_path, files)))
    pages = inputs.pages(2)
    assert [(p.index, len(p.tokens)) for p in pages] == [(0, 2), (1, 0)]
    assert "2 beyond the last page" in inputs.report().notes[0]


def test_only_an_injected_plan_replaces_tool_2_s_planning(tmp_path: Path) -> None:
    files = {
        **docs(layout("Jane"), [mention("p", "PERSON_NAME", tokens=["t0.0"])], [("c", ("p",))]),
        "plan.json": {"schema": "rb.plan/0.1", "regions": []},
    }
    folder = write(tmp_path, files)
    for mode, planned_by_tool_2 in (("oracle_entities", True), ("oracle_regions", False)):
        inputs = OracleInputs(load_injection(mode, folder))
        (page,) = inputs.pages(1)
        pages = [PageAnalysis(GEOM, PageKind.BORN_DIGITAL, injected_lines(page, GEOM))]
        staged = inputs.stage(pages, CONFIG, "doc")
        assert staged is not None and not inputs.detect_regions
        args = {**_plan_args(), "decisions": staged.decisions, "pages": staged.pages}
        assert (inputs.plan(**args) is None) is planned_by_tool_2, mode
        assert (inputs.report().plan is None) is planned_by_tool_2, mode
        # A repeated run replaces the report: the next job's plan is not known before it is planned.
        inputs.stage(pages, CONFIG, "doc")
        assert inputs.report().plan is None


def test_a_profile_that_keeps_injected_regions_is_refused() -> None:
    check_region_actions(CONFIG)
    keeps = load_config("client", (), {"policy": {"region_actions": {"signature": "keep"}}})
    with pytest.raises(EnvironmentErrorMl) as caught:
        check_region_actions(keeps)
    assert caught.value.code is ReasonCode.CONFIG_INVALID
    assert caught.value.params == {"inject": True, "kept_region": RegionKind.SIGNATURE}


def test_the_report_shows_no_injected_text_outside_the_mentions(tmp_path: Path) -> None:
    tokens = layout(f"Dear {CANARY}")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("p", "PERSON_NAME", text=CANARY, tokens=["t0.1", "t0.2"]),
            mention("x", CANARY, regions=[{"page": 0, "poly": poly(60, 400, 200, 440)}]),
        ],
    )
    assert s.reported["p"].text == CANARY  # what mentions.json lists
    shown = " ".join(s.report.notes) + repr(s.report) + repr(s.report.mentions) + repr(s.stage)
    shown += " ".join(r.rule for r in s.regions) + " ".join(
        e.rule_id for m in s.stage.detection.mentions for e in m.evidence
    )
    assert CANARY.split()[0] not in shown


def test_the_file_order_does_not_matter(tmp_path: Path) -> None:
    tokens = layout("Smith & Partners Lawyers", "Mr John Smith of Penrith NSW 2750", "Medicare 2123 45670 1")
    mentions = [
        mention("org", "ORGANISATION", tokens=["t0.0", "t0.1", "t0.2", "t0.3"]),
        mention("org-name", "PERSON_NAME", tokens=["t0.0"]),
        mention("john", "PERSON_NAME", tokens=["t1.1", "t1.2"]),
        mention("place", "LOCATION", tokens=["t1.4", "t1.5", "t1.6"], regions=polys(tokens, "t1.4")),
        mention("medicare", "ID_GOVERNMENT", tokens=["t2.1", "t2.2", "t2.3"]),
        mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": poly(60, 400, 200, 440)}]),
    ]
    clusters = [("c-john", ("john", "sig")), ("c-firm", ("org-name",)), ("c-org", ("org",))]

    def outcome(s: Staged) -> Any:
        persons = s.stage.detection.persons
        return (
            s.stage.detection.mentions,
            s.stage.decisions,
            persons,
            s.regions,
            s.report,
        )

    first = outcome(stage(tmp_path / "a", tokens, mentions, clusters))
    rng = random.Random(7)
    for k in range(3):
        shuffled_tokens, shuffled_mentions = list(tokens), [dict(m) for m in mentions]
        rng.shuffle(shuffled_tokens)
        rng.shuffle(shuffled_mentions)
        for m in shuffled_mentions:
            if "tokens" in m:
                m["tokens"] = rng.sample(m["tokens"], len(m["tokens"]))
        shuffled_clusters = [(c, tuple(rng.sample(list(ms), len(ms)))) for c, ms in rng.sample(clusters, len(clusters))]
        again = stage(tmp_path / f"b{k}", shuffled_tokens, shuffled_mentions, shuffled_clusters)
        assert outcome(again) == first


def test_a_repeated_stage_replaces_the_report(tmp_path: Path) -> None:
    s = stage(tmp_path, layout("Jane Citizen"), [mention("p", "PERSON_NAME", tokens=["t0.0", "t0.1"])])
    assert s.report.mentions is not None and len(s.report.mentions) == 1
    blank = PageAnalysis(GEOM, PageKind.BORN_DIGITAL, ())
    again = s.inputs.stage([blank], CONFIG, "doc")
    assert again is not None and again.detection.mentions == ()
    assert s.inputs.report().mentions == ()


# ------------------------------------------------------------------------------------------- stage files
def files_of(s: Staged, ops: list[RenderOp], surrogates: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """The stage files of a job that drew ``ops`` with ``surrogates``."""
    trace = JobTrace(
        pdf=b"%PDF-1.7 stub",
        pages=s.stage.pages,
        decisions=s.stage.decisions,
        persons=s.stage.detection.persons,
        surrogates={k: MentionSurrogate(k, v) for k, v in surrogates.items()},
        ops={0: tuple(ops)},
        erased={},
        regions=(),
        injected=s.report,
    )
    return stage_files(trace, text_layer=False)


def test_the_stage_files_name_the_injected_mentions(tmp_path: Path) -> None:
    tokens = layout("Smith & Partners Lawyers")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("org", "ORGANISATION", tokens=["t0.0", "t0.1", "t0.2", "t0.3"]),
            mention("person", "PERSON_NAME", tokens=["t0.0"]),
            mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": poly(60, 400, 200, 440)}], conf=0.75),
        ],
    )
    org, (signature,) = s.internal("Smith & Partners Lawyers"), s.regions
    line = s.stage.pages[0].lines[0].bbox
    drawn = RenderOp(0, ActionKind.SURROGATE, line, "organisation", org, text="Lakeview & Partners Lawyers")
    ops = [
        drawn,
        RenderOp(0, ActionKind.SYNTHESISE, signature.bbox, "signature", signature.region_id),
        RenderOp(0, ActionKind.BLACKOUT, BBox(900, 900, 950, 950), "person", "R-replanned"),
    ]
    fake = {org: ("Lakeview", "&", "Partners", "Lawyers")}
    files = files_of(s, ops, fake)
    assert files["text.json"]["source"] == "text_layer"
    mentions = files["mentions.json"]["mentions"]
    assert [(m["id"], m["label"], m.get("text"), m.get("conf")) for m in mentions] == [
        ("org", "organisation", "Smith & Partners Lawyers", None),
        ("person", "person", "Smith", None),
        ("sig", "signature", None, 0.75),  # a confidence only where one was injected
    ]
    smith = s.stage.pages[0].lines[0].tokens[0]
    assert (mentions[1]["tokens"], mentions[1]["regions"]) == (
        [smith.token_id],
        [{"page": 0, "poly": display_poly(smith.bbox, GEOM)}],
    )
    assert mentions[2]["regions"] == [{"page": 0, "poly": display_poly(signature.bbox, GEOM)}]
    assert "tokens" not in mentions[2]
    assert {r["id"]: r["mentions"] for r in files["plan.json"]["regions"]} == {
        f"{org}.0": ["org", "person"],  # the owner first
        f"{signature.region_id}.0": ["sig"],
        "R-replanned.0": [],
    }
    # The name inside the organisation shows the organisation's fake: it has none of its own.
    assert files["pseudonyms.json"]["mentions"] == [{"mention": "org", "replacement": "Lakeview & Partners Lawyers"}]
    assert files["pseudonyms.json"]["entries"] == [
        {"cluster": None, "original": "Smith & Partners Lawyers", "replacement": "Lakeview & Partners Lawyers"}
    ]
    # Blacked out instead (on a sideways line, say), the fake was never drawn.
    blacked = files_of(s, [replace(drawn, kind=ActionKind.BLACKOUT, text=None)], fake)
    assert blacked["pseudonyms.json"]["mentions"] == []


def test_tool_2_s_clusters_name_each_injected_person_once(tmp_path: Path) -> None:
    # The injected "Jane ... Smith" is two pieces, which tool 2 clusters with Jane Brown and with Tom Smith.
    tokens = layout("Jane and Smith", "Jane Brown", "Tom Smith")
    s = stage(
        tmp_path,
        tokens,
        [
            mention("pieces", "PERSON_NAME", tokens=["t0.0", "t0.2"]),
            mention("brown", "PERSON_NAME", tokens=["t1.0", "t1.1"]),
            mention("tom", "PERSON_NAME", tokens=["t2.0", "t2.1"]),
        ],
    )
    assert files_of(s, [], {})["entities.json"]["clusters"] == [
        {"id": "P000", "mentions": ["brown", "pieces"]},
        {"id": "P001", "mentions": ["tom"]},
    ]
    jane, smith = s.internal("Jane"), s.internal("Smith")
    boxes = [t.bbox for t in s.stage.pages[0].lines[0].tokens]
    ops = [
        RenderOp(0, ActionKind.SURROGATE, boxes[0], "person", jane, text="Mei"),
        RenderOp(0, ActionKind.SURROGATE, boxes[2], "person", smith, text="Rossi"),
    ]
    fakes = {jane: ("Mei",), smith: ("Rossi",)}
    drawn = files_of(s, ops, fakes)["pseudonyms.json"]["mentions"]
    assert drawn == [{"mention": "pieces", "replacement": "Mei Rossi"}]
    # A piece blacked out: the mention's fake was not drawn whole.
    assert files_of(s, ops[:1], fakes)["pseudonyms.json"]["mentions"] == []


# ----------------------------------------------------------------------------------------------- the plan
def plan_region(
    region_id: str,
    points: list[list[float]],
    *mentions: str,
    action: str = "replace",
    page: int = 0,
    replacement: str | None = None,
) -> dict[str, Any]:
    """An ``rb.plan/0.1`` region with every key, as the benchmark writes it."""
    return {
        "id": region_id,
        "page": page,
        "poly": points,
        "action": action,
        "replacement": replacement,
        "mentions": list(mentions),
    }


def word_regions(tokens: list[dict[str, Any]], mention_id: str, *ids: str) -> list[dict[str, Any]]:
    """One region per word ``ids`` of a mention, its word's box: how the benchmark plans a mention's words."""
    by_id = {t["id"]: t["poly"] for t in tokens}
    return [plan_region(f"{mention_id}-{k}", by_id[i], mention_id) for k, i in enumerate(ids)]


@dataclass
class Planned:
    """OracleInputs.plan after its stage: the operations, and what the stage files are told."""

    staged: Staged
    ops: dict[int, tuple[RenderOp, ...]]

    @property
    def report(self) -> InjectReport:
        return self.staged.report

    @property
    def page_ops(self) -> list[RenderOp]:
        return list(self.ops.get(0, ()))

    @property
    def words(self) -> list[Token]:
        """Page 0's tokens, in reading order."""
        return list(self.staged.stage.pages[0].tokens())

    @property
    def echoes(self) -> dict[str, PlanEcho]:
        assert self.report.plan is not None
        return {e.region_id: e for e in self.report.plan}

    @property
    def note(self) -> str:
        return self.staged.note("injected plan")

    def of(self, value: str) -> list[RenderOp]:
        """The operations of tool 2's mention with ``value``."""
        ref = self.staged.internal(value)
        return [op for op in self.page_ops if op.ref_id == ref]

    def applied(self) -> dict[str, tuple[tuple[ActionKind, str], ...]]:
        """Each applied region's operations, as (kind, reason)."""
        ops = {(op.ref_id, op.box): op for op in self.page_ops}
        return {k: tuple((ops[key].kind, ops[key].reason) for key in e.ops) for k, e in self.echoes.items()}


def region_actions(pages: Sequence[PageAnalysis], config: Any) -> dict[str, ActionKind]:
    """The runner's action on each region (``Redactor._region_decisions`` reads nothing but the configuration)."""
    redactor: Any = SimpleNamespace(config=config)
    return {d.region.region_id: d.action for p in pages for d in Redactor._region_decisions(redactor, p)}


def planned(
    folder: Path,
    tokens: Tokens,
    mentions: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    *,
    config: Any = CONFIG,
    fakes: Mapping[str, tuple[str, ...] | None] | None = None,
) -> Planned:
    """OracleInputs.stage, then OracleInputs.plan as the runner calls it, in ``oracle_regions`` on pages made of the
    injected tokens.  Each mention's fake is its words with their case swapped, unless ``fakes`` gives another one
    (by the mention's value; None: no fake)."""
    files = {**docs(tokens, mentions, []), "plan.json": {"schema": "rb.plan/0.1", "regions": regions}}
    inputs = OracleInputs(load_injection("oracle_regions", write(folder, files)))
    count = max(tokens) + 1 if isinstance(tokens, dict) else 1
    result = inputs.stage([analysed(p) for p in inputs.pages(count)], config, "doc")
    assert result is not None
    pages = result.pages
    by_id = {t.token_id: t for p in pages for t in p.tokens()}
    surrogates = {}
    for d in result.decisions:
        m = d.mention
        texts = (fakes or {}).get(m.value, tuple(by_id[t].text.swapcase() for t in m.token_ids))
        if texts is not None:
            surrogates[m.mention_id] = MentionSurrogate(m.mention_id, texts)
    ops = inputs.plan(
        decisions=result.decisions,
        surrogates=surrogates,
        tokens=by_id,
        pages=pages,
        region_actions=region_actions(pages, config),
        rotated=rotated_token_ids([line for p in pages for line in p.lines]),
        config=config,
    )
    assert ops is not None
    return Planned(Staged(inputs, result, inputs.report()), ops)


def test_a_mention_s_regions_on_a_line_are_applied_by_one_operation(tmp_path: Path) -> None:
    tokens = layout("Dear Jane Andrew Citizen today")
    mentions = [mention("jane", "PERSON_NAME", tokens=["t0.1", "t0.2", "t0.3"])]
    regions = word_regions(tokens, "jane", "t0.1", "t0.2", "t0.3")
    regions[0]["replacement"] = "Somebody Else"  # tool 2 draws its own fake
    p = planned(tmp_path / "words", tokens, mentions, regions)
    (op,) = p.page_ops
    words = p.words[1:4]
    assert (op.kind, op.ref_id, op.reason) == (ActionKind.SURROGATE, p.staged.internal("Jane Andrew Citizen"), "person")
    # The regions and their words, without padding (the regions are the words' own boxes).
    assert op.box == op.ink_box == union_all([w.bbox for w in words])
    assert (op.text, op.source_text) == ("jANE aNDREW cITIZEN", "Jane Andrew Citizen")
    assert {k: e.ops for k, e in p.echoes.items()} == {f"jane-{k}": ((op.ref_id, op.box),) for k in range(3)}
    assert "3 applied by 1 operations (1 joining regions of a mention on a line" in p.note
    assert "1 replacement texts not used" in p.note
    # A fake of another length is put whole in the first word's place: joined, it is drawn in one box.
    whole = planned(tmp_path / "whole", tokens, mentions, regions, fakes={"Jane Andrew Citizen": ("Mei Lee", "", "")})
    (op,) = whole.page_ops
    assert (op.text, op.box) == ("Mei Lee", union_all([w.bbox for w in words]))


def test_a_word_left_out_of_the_plan_splits_the_operation_and_stays(tmp_path: Path) -> None:
    tokens = layout("Dear Jane Andrew Citizen today")
    mentions = [mention("jane", "PERSON_NAME", tokens=["t0.1", "t0.2", "t0.3"])]
    p = planned(tmp_path, tokens, mentions, word_regions(tokens, "jane", "t0.1", "t0.3"))
    words = p.words
    assert [(op.box, op.text) for op in p.page_ops] == [(words[1].bbox, "jANE"), (words[3].bbox, "cITIZEN")]
    assert all(op.box.intersection(words[2].bbox) is None for op in p.page_ops)  # "Andrew" is not applied
    assert "2 applied by 2 operations (0 joining" in p.note


def test_the_mention_s_words_between_a_region_s_words_are_drawn_with_them(tmp_path: Path) -> None:
    tokens = layout("Jane Andrew Citizen")
    tokens[1]["poly"] = poly(85, 104, 118, 126)  # set low on the line: less than half of it is under the region
    mentions = [mention("jane", "PERSON_NAME", tokens=["t0.0", "t0.1", "t0.2"])]
    over = poly(58, 98, 162, 113)
    p = planned(tmp_path, tokens, mentions, [plan_region("jane-0", over, "jane")])
    # "Andrew" lies between the region's words, mostly inside its box: it is erased, so its piece of the fake is drawn.
    (op,) = p.page_ops
    box = union_all([pt_box([(x, y) for x, y in over], GEOM), *(w.bbox for w in p.words)])
    assert (op.text, op.box) == ("jANE aNDREW cITIZEN", box)


def test_an_operation_covers_its_regions_and_their_whole_words_without_padding(tmp_path: Path) -> None:
    tokens = layout("Email john.smith78@example.org today", "Ring Jane today")
    email = tokens[1]["poly"]
    characters = poly(email[0][0] + 27.5, email[0][1], email[0][0] + 55.0, email[2][1])  # "smith" inside the word
    wide = poly(83, 117, 109, 134)  # around "Jane", with room to spare
    mentions = [mention("email", "EMAIL", tokens=["t0.1"]), mention("jane", "PERSON_NAME", tokens=["t1.1"])]
    regions = [plan_region("email-0", characters, "email"), plan_region("jane-0", wide, "jane")]
    p = planned(tmp_path, tokens, mentions, regions)
    words = p.words
    ((email_op,), (jane_op,)) = p.of("john.smith78@example.org"), p.of("Jane")
    assert email_op.box == email_op.ink_box == words[1].bbox  # the characters' region grows to the whole word
    assert (jane_op.box, jane_op.ink_box) == (pt_box([(x, y) for x, y in wide], GEOM), words[4].bbox)


def test_sideways_text_and_a_mention_without_a_fake_are_blacked_out(tmp_path: Path) -> None:
    tokens = layout("Ms Jane Citizen")
    # Words running up the page, each on a line of its own: a name, and a word just under "Citizen".
    up = [[300, 340], [300, 300], [310, 300], [310, 340]]
    tokens.append({"id": "up", "text": "Brown", "poly": up, "line": "l9", "order": 3})
    tokens.append({"id": "up2", "text": "Note", "poly": [[100, 140], [100, 112], [108, 112], [108, 140]], "line": "l8"})
    tokens[-1]["order"] = 4
    mentions = [
        mention("jane", "PERSON_NAME", tokens=["t0.1"]),
        mention("citizen", "PERSON_NAME", tokens=["t0.2"]),
        mention("brown", "PERSON_NAME", tokens=["up"]),
    ]
    regions = [*word_regions(tokens, "jane", "t0.1"), *word_regions(tokens, "brown", "up")]
    regions.append(plan_region("citizen-0", poly(98, 98, 140, 142), "citizen"))  # "Citizen" and the word under it
    p = planned(tmp_path, tokens, mentions, regions, fakes={"Citizen": None})
    kinds = {value: [(op.kind, op.text) for op in p.of(value)] for value in ("Jane", "Citizen", "Brown")}
    assert kinds == {
        "Jane": [(ActionKind.SURROGATE, "jANE")],
        "Citizen": [(ActionKind.BLACKOUT, None)],  # no fake could be made
        "Brown": [(ActionKind.BLACKOUT, None)],  # tool 2 typesets only across the page
    }
    # The sideways word under "Citizen" is in the region too: it cannot be drawn again across the page.
    (note,) = [op for op in p.page_ops if op.ref_id.startswith("I")]
    assert (note.kind, note.box, note.covers) == (ActionKind.BLACKOUT, p.words[4].bbox, (p.staged.internal("Citizen"),))


def test_non_text_regions_take_the_profile_s_action_on_their_kind(tmp_path: Path) -> None:
    tokens = layout("Initialled JS", "Ms Jane Citizen")
    signature, other = poly(60, 400, 200, 440), poly(300, 400, 360, 440)
    mentions = [
        mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": signature}]),
        mention("marks", "HANDWRITTEN_MARK", tokens=["t0.1"]),
        mention("sig-2", "SIGNATURE", regions=[{"page": 0, "poly": other}]),
        mention("jane", "PERSON_NAME", tokens=["t1.1", "t1.2"]),
    ]
    regions = [
        plan_region("sig-0", signature, "sig"),
        plan_region("marks-0", tokens[1]["poly"], "marks"),
        plan_region("sig-2-0", other, "sig-2", action="remove"),
        plan_region("jane-0", tokens[3]["poly"], "jane"),
        plan_region("jane-1", tokens[4]["poly"], "jane", action="remove"),
    ]
    client = planned(tmp_path / "client", tokens, mentions, regions)
    assert client.applied() == {
        "sig-0": ((ActionKind.SYNTHESISE, "signature"),),
        "marks-0": ((ActionKind.REMOVE, "handwriting"),),
        "sig-2-0": ((ActionKind.REMOVE, "signature"),),  # removal is all the plan asks for
        "jane-0": ((ActionKind.SURROGATE, "person"),),
        "jane-1": ((ActionKind.REMOVE, "person"),),
    }
    # A region's operation is the region's own: its id (its stand-in is drawn from it) and the box it was planned with.
    ops = {op.ref_id: op for op in client.page_ops}
    assert len(client.staged.regions) == 3 and all(ops[r.region_id].box == r.bbox for r in client.staged.regions)
    label = load_config("client", (), {"redaction": {"style": "label"}})
    labelled = planned(tmp_path / "label", tokens, mentions, regions, config=label).applied()
    # Stand-ins come only with realistic fakes.
    assert (labelled["sig-0"], labelled["jane-0"]) == (
        ((ActionKind.REMOVE, "signature"),),
        ((ActionKind.LABEL, "person"),),
    )
    # A region the profile would keep is applied all the same (bench refuses such a profile before it gets here).
    keeps = load_config("client", (), {"policy": {"region_actions": {"signature": "keep"}}})
    kept = planned(tmp_path / "keeps", tokens, mentions, regions, config=keeps).applied()
    assert kept["sig-0"] == ((ActionKind.REMOVE, "signature"),)


def test_a_logo_around_a_word_is_told_from_the_word(tmp_path: Path) -> None:
    tokens = layout("HARBOURSIDE Medicolegal")
    word = tokens[0]["poly"]
    logo = poly(57, 97, 124, 114)  # around the word, which fills most of it: not the word's box, a logo
    anchors = [{"page": 0, "poly": word}, {"page": 0, "poly": logo}]
    mentions = [mention("org", "ORGANISATION", tokens=["t0.0"], regions=anchors)]
    below = poly(57, 108, 124, 130)  # over the logo's lower edge only, and under no word
    regions = [plan_region("org-0", word, "org"), plan_region("org-1", logo, "org"), plan_region("org-2", below, "org")]
    p = planned(tmp_path, tokens, mentions, regions)
    (region,) = p.staged.regions
    # The logo overlaps the word's region enough to pass for it, but the word matches that region better.
    assert region.kind is RegionKind.LOGO and 0.5 < region.bbox.iou(p.words[0].bbox) < 0.9
    assert p.applied() == {
        "org-0": ((ActionKind.SURROGATE, "organisation"),),
        "org-1": ((ActionKind.REMOVE, "logo"),),
        "org-2": ((ActionKind.REMOVE, "illegible"),),  # too little of the logo to be it, and no word: nothing there
    }
    assert (p.echoes["org-0"].ops, p.echoes["org-1"].ops) == (
        ((p.staged.internal("HARBOURSIDE"), p.words[0].bbox),),
        ((region.region_id, region.bbox),),
    )
    assert "1 applied as non-text regions" in p.note


def test_a_region_over_no_mention_s_word_is_removed_as_illegible(tmp_path: Path) -> None:
    tokens = layout("Dear Jane Citizen")
    blank = poly(300, 500, 340, 520)
    regions = [
        plan_region("x-0", poly(60, 100, 72, 111), "nobody"),  # most of "Dear", for a mention that was not injected
        plan_region("y-0", blank),
        plan_region("y-1", blank),  # the same again: one operation
    ]
    p = planned(tmp_path, tokens, [mention("jane", "PERSON_NAME", tokens=["t0.1", "t0.2"])], regions)
    dear, removed = p.page_ops
    assert (dear.kind, dear.reason, dear.box, dear.covers) == (ActionKind.REMOVE, "illegible", p.words[0].bbox, ())
    assert (removed.kind, removed.reason, removed.box) == (
        ActionKind.REMOVE,
        "illegible",
        pt_box([(x, y) for x, y in blank], GEOM),
    )
    assert dear.ref_id.startswith("I") and removed.ref_id.startswith("I")
    assert p.echoes["y-0"].ops == p.echoes["y-1"].ops == ((removed.ref_id, removed.box),)
    assert p.echoes["x-0"].mentions == ()  # it names only mentions that mentions.json lists
    assert "3 applied by 2 operations (0 joining regions of a mention on a line, 1 the same as another)" in p.note
    assert "3 over no mention's word removed as illegible" in p.note


def test_a_name_inside_an_organisation_is_drawn_with_it(tmp_path: Path) -> None:
    tokens = layout("Smith & Partners Lawyers")
    mentions = [
        mention("org", "ORGANISATION", tokens=["t0.0", "t0.1", "t0.2", "t0.3"]),
        mention("person", "PERSON_NAME", tokens=["t0.0"]),
    ]
    regions = [*word_regions(tokens, "org", "t0.0", "t0.1", "t0.2", "t0.3"), *word_regions(tokens, "person", "t0.0")]
    p = planned(tmp_path, tokens, mentions, regions)
    (op,) = p.page_ops
    assert (op.ref_id, op.text) == (p.staged.internal("Smith & Partners Lawyers"), "sMITH & pARTNERS lAWYERS")
    assert p.echoes["person-0"].ops == p.echoes["org-0"].ops == ((op.ref_id, op.box),)
    assert p.echoes["person-0"].mentions == ("person",)


def test_a_region_over_two_mentions_and_other_words_re_typesets_its_line(tmp_path: Path) -> None:
    tokens = layout("Jane Citizen and Peter Brown met")
    mentions = [
        mention("jane", "PERSON_NAME", tokens=["t0.0", "t0.1"]),
        mention("peter", "PERSON_NAME", tokens=["t0.3", "t0.4"]),
    ]
    over = poly(58, 98, 205, 113)  # every word but the last
    box = pt_box([(x, y) for x, y in over], GEOM)
    regions = [plan_region("both-0", over, "jane", "peter")]
    p = planned(tmp_path / "replace", tokens, mentions, regions)
    (op,) = p.page_ops
    covers = tuple(sorted((p.staged.internal("Jane Citizen"), p.staged.internal("Peter Brown"))))
    assert (op.kind, op.reason, op.box, op.covers) == (ActionKind.RETYPESET_REGION, "retypeset", box, covers)
    # The mentions' words become their fakes; "and" is drawn again as it reads.
    assert (op.text, op.source_text) == ("jANE cITIZEN and pETER bROWN", "Jane Citizen and Peter Brown")
    assert op.ref_id.startswith("I") and "1 over several mentions or other words" in p.note
    # One mention and another word: the same.
    one = planned(tmp_path / "one", tokens, mentions, [plan_region("peter-0", poly(125, 98, 205, 113), "peter")])
    (op,) = one.page_ops
    peter = (one.staged.internal("Peter Brown"),)
    assert (op.kind, op.text, op.covers) == (ActionKind.RETYPESET_REGION, "and pETER bROWN", peter)
    # Removed, or where a mention's words cannot be drawn as text, the line is not drawn again.
    removed = planned(tmp_path / "remove", tokens, mentions, [{**regions[0], "action": "remove"}])
    blackout = load_config("client", (), {"redaction": {"style": "blackout"}})
    blacked = planned(tmp_path / "blackout", tokens, mentions, regions, config=blackout)
    removes = load_config("client", (), {"policy": {"entity_actions": {"person": "remove"}}})
    unfaked = planned(tmp_path / "unfaked", tokens, mentions, regions, config=removes)
    assert [(op.kind, op.text, op.covers) for op in (*removed.page_ops, *blacked.page_ops, *unfaked.page_ops)] == [
        (ActionKind.REMOVE, None, covers),
        (ActionKind.BLACKOUT, None, covers),
        (ActionKind.BLACKOUT, None, covers),
    ]


def test_a_region_s_line_without_a_mention_s_word_is_drawn_again_as_it_reads(tmp_path: Path) -> None:
    tokens = layout("Seen by", "Peter Brown")
    mentions = [mention("peter", "PERSON_NAME", tokens=["t1.0", "t1.1"])]
    p = planned(tmp_path, tokens, mentions, [plan_region("peter-0", poly(58, 98, 125, 133), "peter")])
    ref, words = p.staged.internal("Peter Brown"), p.words
    seen, peter = p.page_ops
    # Over several lines, each line is applied in its own band; the words the region holds are not dropped.
    assert (seen.kind, seen.text, seen.box, seen.covers) == (
        ActionKind.RETYPESET_REGION,
        "Seen by",
        union_all([w.bbox for w in words[:2]]),
        (ref,),
    )
    assert (peter.kind, peter.ref_id, peter.text, peter.box) == (
        ActionKind.SURROGATE,
        ref,
        "pETER bROWN",
        union_all([w.bbox for w in words[2:]]),
    )


def test_a_region_over_two_lines_is_applied_line_by_line(tmp_path: Path) -> None:
    tokens = layout("Dr Peter", "Brown")
    mentions = [mention("peter", "PERSON_NAME", tokens=["t0.1", "t1.0"])]
    p = planned(tmp_path, tokens, mentions, [plan_region("peter-0", poly(72, 98, 110, 133), "peter")])
    ref, words = p.staged.internal("Peter Brown"), p.words
    # One operation per line, each in its own line's band: the region's words there ("Dr" is not under it).
    assert [(op.ref_id, op.box, op.text) for op in p.page_ops] == [
        (ref, words[1].bbox, "pETER"),
        (ref, words[2].bbox, "bROWN"),
    ]
    assert p.echoes["peter-0"].ops == tuple((op.ref_id, op.box) for op in p.page_ops)


def test_plan_json_repeats_each_applied_region_with_what_was_done(tmp_path: Path) -> None:
    tokens = layout("Dr Peter", "Brown", "Ms Jane Citizen")
    signature = poly(60, 400, 200, 440)
    mentions = [
        mention("peter", "PERSON_NAME", tokens=["t0.1", "t1.0"]),
        mention("jane", "PERSON_NAME", tokens=["t2.1", "t2.2"]),
        mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": signature}]),
    ]
    regions = [
        plan_region("peter-0", poly(72, 98, 110, 133), "peter", "ghost"),  # "ghost" is not a mention
        plan_region("jane-0", tokens[4]["poly"], "jane", replacement=CANARY),
        plan_region("jane-1", tokens[5]["poly"], "jane", action="remove"),
        plan_region("sig-0", signature, "sig"),
        plan_region("sig-1", signature, "sig", page=3),  # beyond the last page
    ]
    p = planned(tmp_path, tokens, mentions, regions)
    plan = files_of(p.staged, p.page_ops, {})["plan.json"]
    assert (plan["schema"], plan["frame"]) == ("rb.plan/0.1", "display_pt")
    assert [(r["id"], r["page"], r["action"], r["mentions"], r.get("replacement")) for r in plan["regions"]] == [
        ("jane-0", 0, "replace", ["jane"], "jANE"),  # the fake drawn, never the injected replacement
        ("jane-1", 0, "remove", ["jane"], None),
        ("peter-0", 0, "replace", ["peter"], "pETER bROWN"),  # both of its lines' operations
        ("sig-0", 0, "replace", ["sig"], None),  # a stand-in is drawn, no text
    ]
    by_id = {r["id"]: r["poly"] for r in regions}
    assert all(r["poly"] == by_id[r["id"]] for r in plan["regions"])  # as injected
    assert "1 beyond the last page" in p.note and CANARY not in " ".join(p.report.notes)


def test_the_plan_does_not_depend_on_the_file_order(tmp_path: Path) -> None:
    tokens = layout("Smith & Partners Lawyers", "Jane Citizen and Peter Brown met", "Dr Peter", "Brown")
    mentions = [
        mention("org", "ORGANISATION", tokens=["t0.0", "t0.1", "t0.2", "t0.3"]),
        mention("person", "PERSON_NAME", tokens=["t0.0"]),
        mention("jane", "PERSON_NAME", tokens=["t1.0", "t1.1"]),
        mention("peter", "PERSON_NAME", tokens=["t1.3", "t1.4"]),
        mention("dr", "PERSON_NAME", tokens=["t2.1", "t3.0"]),
        mention("sig", "SIGNATURE", regions=[{"page": 0, "poly": poly(60, 400, 200, 440)}]),
    ]
    regions = [
        *word_regions(tokens, "org", "t0.0", "t0.1", "t0.2", "t0.3"),
        *word_regions(tokens, "person", "t0.0"),
        plan_region("both-0", poly(58, 118, 205, 133), "jane", "peter", "org"),
        plan_region("dr-0", poly(72, 138, 110, 173), "dr"),
        plan_region("x-0", poly(300, 500, 340, 520)),
        plan_region("x-1", poly(300, 500, 340, 520)),
        # The same box twice, to be replaced and to be removed: the first by id is applied.
        plan_region("sig-0", poly(60, 400, 200, 440), "sig"),
        plan_region("sig-1", poly(60, 400, 200, 440), "sig", action="remove"),
    ]
    first = planned(tmp_path / "a", tokens, mentions, regions)
    assert first.applied()["sig-1"] == ((ActionKind.SYNTHESISE, "signature"),)
    rng = random.Random(11)
    for k in range(3):
        shuffled = [{**r, "mentions": rng.sample(r["mentions"], len(r["mentions"]))} for r in regions]
        again = planned(
            tmp_path / f"b{k}",
            rng.sample(tokens, len(tokens)),
            rng.sample(mentions, len(mentions)),
            rng.sample(shuffled, len(shuffled)),
        )
        assert (again.ops, again.report.plan, again.note) == (first.ops, first.report.plan, first.note)


# -------------------------------------------------------------------------------------------- every stage
# What the busy document of tests/inject_support.py meets: every case the stages handle, as the notes count them.
BUSY_CASES = (
    "injected text: 58 tokens (1 empty, 1 with a repeated id",
    "18 placed by token ids, 0 by span, 2 by region anchors, 1 not placed",
    "2 sharing tokens another one owns",
    "2 locations split into locality and postcode, 4 of a kept type redacted, 1 labels taken as other_id, 1 anchors "
    "not explained by their tokens",
    "injected clusters: 5 read (1 merged by id), 1 mentions in several clusters, 2 non-person members, 1 members",
    "47 applied by 25 operations (13 joining regions of a mention on a line, 2 the same as another), 4 applied as "
    "non-text regions, 2 over several mentions or other words, 2 over no mention's word removed as illegible, 1 beyond "
    "the last page, 1 replacement texts not used",
)


@pytest.mark.parametrize("mode", ["oracle_mentions", "oracle_entities", "oracle_regions"])
def test_no_stage_depends_on_the_order_of_the_injected_files(tmp_path: Path, mode: str) -> None:
    """Every list of the injected files whose order means nothing shuffled (tokens with their reading order set): the
    mentions, decisions, clusters, regions, surrogates, operations, report and stage files stay the same."""
    first = oracle_outcome(tmp_path / "a", mode, busy_docs(), CONFIG)
    if mode == "oracle_regions":
        notes = " ".join(first["report"]["notes"])
        assert [case for case in BUSY_CASES if case not in notes] == []
    rng = random.Random(29)
    for k in range(3):
        again = oracle_outcome(tmp_path / f"b{k}", mode, shuffled(busy_docs(), rng), CONFIG)
        assert [part for part, made in first.items() if again[part] != made] == [], k
