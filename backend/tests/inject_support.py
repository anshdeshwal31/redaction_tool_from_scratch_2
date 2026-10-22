"""The benchmark's injected stage inputs for tests: its oracle files for a synthetic letter (``rb.text/0.1``,
``rb.mentions/0.1``, ``rb.entities/0.1``, ``rb.plan/0.1``), written in its own format (no benchmark import) the way it
builds them from ground truth (its ``benchmark/runner/oracle.py``): the words with their line and reading order, the
required mentions anchored by their words' ids and boxes, the clusters, and one plan region per mention anchor.

Also a busy two-page document of the same kind (:func:`busy_docs`), what every stage of an oracle mode makes of
injected files without rendering anything (:func:`oracle_outcome`), for the determinism tests, and how the benchmark
measures that a plan region was applied (:func:`changed_shares`).

All text is fictitious.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pypdfium2 as pdfium
from synthdocs.generator import medicare_number, provider_number

from mlredact.bench.artifacts import display_poly, stage_files
from mlredact.bench.inject import MODE_FILES, load_injection
from mlredact.bench.oracle import OracleInputs
from mlredact.config.loader import load_config
from mlredact.config.schema import AppConfig
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import PageAnalysis, PageGeometry
from mlredact.core.types import ActionKind, PageKind
from mlredact.ingest.render import open_document, render_page
from mlredact.pipeline.injection import injected_lines
from mlredact.pipeline.runner import JobTrace, Redactor
from mlredact.render.plan import plan_ops, rotated_token_ids
from mlredact.surrogate.assign import SurrogateAssigner, forbidden_keys
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory

# The injected files by name.
Docs = dict[str, dict[str, Any]]
PointBox = tuple[float, float, float, float]

# The letter's required mentions as its ground truth would list them: (id, benchmark type, text-layer line, the words'
# text).  These are the values synthdocs planted (``SynthDoc.planted``), but for the date of birth (the benchmark's
# policy does not require dates) and with whole places for the planted localities and postcode, plus the letterhead's
# organisation, a capital city and every occurrence of "Smith".
LETTER_MENTIONS = (
    ("org-harbourside", "ORGANISATION", 0, "HARBOURSIDE MEDICOLEGAL SERVICES"),
    ("addr-george", "ADDRESS", 1, "123 George Street"),
    ("loc-sydney", "LOCATION", 1, "Sydney NSW 2000"),  # a capital city
    ("phone-office", "PHONE", 2, "(02) 9876 5432"),
    ("email-office", "EMAIL", 2, "admin@harbourside-medicolegal.example"),
    ("person-jane", "PERSON_NAME", 4, "Jane Citizen"),
    ("org-firm", "ORGANISATION", 5, "Smith & Partners Lawyers"),
    ("person-firm", "PERSON_NAME", 5, "Smith"),  # a whole word of the firm's name
    ("addr-pobox", "ADDRESS", 6, "PO Box 123"),
    ("loc-parramatta", "LOCATION", 6, "Parramatta"),
    ("ref-ours", "ID_CLAIM", 7, "HMS-2024-0187"),
    ("ref-yours", "ID_CLAIM", 7, "SP/4471/JC"),
    ("person-john", "PERSON_NAME", 8, "John Andrew SMITH"),
    ("ref-claim", "ID_CLAIM", 9, "WC1234567"),
    ("medicare", "ID_GOVERNMENT", 9, medicare_number("29537015", 1)),
    ("addr-wattle", "ADDRESS", 10, "42 Wattle Grove Road"),
    ("loc-penrith", "LOCATION", 10, "Penrith NSW 2750"),
    ("person-citizen", "PERSON_NAME", 11, "Citizen"),
    ("person-smith-exam", "PERSON_NAME", 12, "Smith"),
    ("person-smith-contact", "PERSON_NAME", 16, "Smith"),
    ("phone-mobile", "PHONE", 16, "0412 345 678"),
    ("email-john", "EMAIL", 17, "john.smith78@example.org"),
    ("person-peter", "PERSON_NAME", 19, "Peter Brown"),
    ("provider", "ID_PROVIDER", 20, provider_number("212345", "A")),
)
# A surname inside the e-mail address: anchored by its characters' box inside the word, as ground truth anchors part
# of a word.
EMAIL_PERSON = "person-email"
# A signature with no words, in the blank area below the letter (display points).
SIGNATURE = "signature"
SIGNATURE_BOX: PointBox = (60.0, 450.0, 200.0, 480.0)
# The letterhead's logo: a box around the organisation's words, which they do not explain.
LOGO_MARGIN_PT = 4.0
LETTER_CLUSTERS = (
    ("ent-john", ("person-john", "person-smith-exam", "person-smith-contact", EMAIL_PERSON)),
    ("ent-jane", ("person-jane", "person-citizen")),
    ("ent-peter", ("person-peter", SIGNATURE)),
)


@dataclass(frozen=True)
class Word:
    """A word of the letter's text layer: its injected id, text, pixel box and text-layer line."""

    id: str
    text: str
    box: BBox
    line: int


def letter_words(pdf: bytes, dpi: int) -> tuple[PageGeometry, list[Word]]:
    """The letter's words from its own text layer, rendered at ``dpi``: what a perfect reader hands over."""
    cfg = load_config("client", (), {"render": {"dpi": dpi}})
    doc = open_document(pdf, cfg.render)
    try:
        page = render_page(doc, 0, cfg.render, cfg.limits, with_words=True)
    finally:
        doc.close()
    return page.geometry, [Word(f"w{i:04d}", w.text, w.bbox, w.line) for i, w in enumerate(page.words)]


def locate(words: Sequence[Word], line: int, value: str) -> list[Word]:
    """The run of words on ``line`` that reads ``value`` (punctuation at its ends aside)."""
    on_line = [w for w in words if w.line == line]
    n = len(value.split())
    for k in range(len(on_line) - n + 1):
        run = on_line[k : k + n]
        if " ".join(w.text for w in run).strip(",.:;") == value:
            return run
    raise ValueError(f"no run of words reads the value on line {line}")


def poly(box: PointBox) -> list[list[float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def mention(
    mention_id: str,
    label: str,
    text: str | None,
    tokens: Sequence[str],
    regions: Sequence[list[list[float]]],
    *,
    page: int = 0,
) -> dict[str, Any]:
    """An ``rb.mentions/0.1`` mention with every key, as the benchmark writes it (its region anchors on ``page``)."""
    return {
        "id": mention_id,
        "label": label,
        "text": text,
        "tokens": list(tokens) or None,
        "regions": [{"page": page, "poly": p} for p in regions] or None,
        "span": None,
        "conf": None,
    }


def letter_docs(pdf: bytes, dpi: int) -> Docs:
    """The four injected files for the one-page synthetic letter ``pdf``, rendered at ``dpi``."""
    geometry, words = letter_words(pdf, dpi)
    tokens = [
        {
            "id": w.id,
            "text": w.text,
            "poly": display_poly(w.box, geometry),
            "conf": None,
            "line": f"l{w.line}",
            "order": i,
        }
        for i, w in enumerate(words)
    ]
    page = {"index": 0, "tokens": tokens, "detected_rotation": None, "skew_deg": None}
    text = {"schema": "rb.text/0.1", "frame": "display_pt", "dpi": None, "source": "text_layer", "pages": [page]}
    mentions = []
    for mention_id, label, line, value in LETTER_MENTIONS:
        run = locate(words, line, value)
        regions = [display_poly(w.box, geometry) for w in run]
        if mention_id == "org-harbourside":
            x0, y0, x1, y1 = (v * 72 / geometry.dpi for v in union_all([w.box for w in run]).as_tuple())
            m = LOGO_MARGIN_PT
            regions.append(poly((x0 - m, y0 - m, x1 + m, y1 + m)))
        mentions.append(mention(mention_id, label, value, [w.id for w in run], regions))
    (email,) = locate(words, 17, "john.smith78@example.org")
    mentions.append(mention(EMAIL_PERSON, "PERSON_NAME", "smith", [email.id], [_characters(email, 5, 10, geometry)]))
    mentions.append(mention(SIGNATURE, "SIGNATURE", None, [], [poly(SIGNATURE_BOX)]))
    clustered = {m for _cluster, members in LETTER_CLUSTERS for m in members}
    clusters = [{"id": c, "mentions": list(members), "attributes": {}} for c, members in LETTER_CLUSTERS]
    # Every other mention is an entity of its own, as the benchmark injects it.
    clusters += [{"id": m["id"], "mentions": [m["id"]], "attributes": {}} for m in mentions if m["id"] not in clustered]
    plan = [
        {"id": f"{m['id']}-{k}", "page": r["page"], "poly": r["poly"], "action": "replace", "mentions": [m["id"]]}
        for m in mentions
        for k, r in enumerate(m["regions"] or ())
    ]
    return {
        "text.json": text,
        "mentions.json": {
            "schema": "rb.mentions/0.1",
            "frame": "display_pt",
            "dpi": None,
            "page_texts": None,
            "mentions": mentions,
        },
        "entities.json": {"schema": "rb.entities/0.1", "clusters": clusters},
        "plan.json": {"schema": "rb.plan/0.1", "frame": "display_pt", "dpi": None, "regions": plan},
    }


def _characters(word: Word, start: int, end: int, geometry: PageGeometry) -> list[list[float]]:
    """The box of characters ``start:end`` of a word, at an even advance per character."""
    advance = word.box.width / len(word.text)
    x0, x1 = word.box.x0 + start * advance, word.box.x0 + end * advance
    s = 72 / geometry.dpi
    return poly((round(x0 * s, 3), round(word.box.y0 * s, 3), round(x1 * s, 3), round(word.box.y1 * s, 3)))


def keep_mentions(docs: Docs, ids: Collection[str]) -> Docs:
    """A copy injecting only the mentions ``ids``: the others are left out of every file, as a policy that does not
    require them would leave them out."""
    out = copy.deepcopy(docs)
    out["mentions.json"]["mentions"] = [m for m in out["mentions.json"]["mentions"] if m["id"] in ids]
    clusters = []
    for c in out["entities.json"]["clusters"]:
        members = [m for m in c["mentions"] if m in ids]
        if members:
            clusters.append({**c, "mentions": members})
    out["entities.json"]["clusters"] = clusters
    out["plan.json"]["regions"] = [r for r in out["plan.json"]["regions"] if set(r["mentions"]) <= set(ids)]
    return out


def without_mentions(docs: Docs, ids: Collection[str]) -> Docs:
    """A copy with the mentions ``ids`` left out of every file."""
    every = {m["id"] for m in docs["mentions.json"]["mentions"]}
    return keep_mentions(docs, every - set(ids))


def without_regions(docs: Docs, ids: Collection[str]) -> Docs:
    """A copy whose plan leaves out the regions of the mentions ``ids``: they are still injected, but not to be
    applied."""
    out = copy.deepcopy(docs)
    out["plan.json"]["regions"] = [r for r in out["plan.json"]["regions"] if not set(r["mentions"]) & set(ids)]
    return out


def words_poly(docs: Docs, line: int, *texts: str) -> list[list[float]]:
    """The box around the words ``texts`` of the letter's text-layer line ``line`` (display points)."""
    tokens = [t for t in docs["text.json"]["pages"][0]["tokens"] if t["line"] == f"l{line}" and t["text"] in texts]
    if len(tokens) != len(texts):
        raise ValueError("the words are not on the line")
    xs = [x for t in tokens for x, _y in t["poly"]]
    ys = [y for t in tokens for _x, y in t["poly"]]
    return poly((min(xs), min(ys), max(xs), max(ys)))


def add_line(docs: Docs, line_id: str, text: str, origin: tuple[float, float]) -> list[str]:
    """Inject a line of words where the page is blank (``origin``: its top-left corner in display points), 11 pt
    high; the words' ids, in order."""
    page = docs["text.json"]["pages"][0]["tokens"]
    x, y = origin
    ids = []
    for word in text.split():
        width = 5.5 * len(word)
        token_id = f"{line_id}.{len(ids)}"
        page.append({"id": token_id, "text": word, "poly": poly((x, y, x + width, y + 11.0)), "line": line_id})
        page[-1].update(conf=None, order=len(page) - 1)
        ids.append(token_id)
        x += width + 3.0
    return ids


def token_polys(docs: Docs, ids: Sequence[str]) -> list[list[list[float]]]:
    """The injected polygons of tokens ``ids``."""
    by_id = {t["id"]: t["poly"] for t in docs["text.json"]["pages"][0]["tokens"]}
    return [by_id[i] for i in ids]


def write_inject_dir(folder: Path, mode: str, docs: Mapping[str, Any]) -> Path:
    """The files ``mode`` injects, written into ``folder`` (created)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODE_FILES[mode]:
        (folder / name.value).write_text(json.dumps(docs[name.value]), "utf-8")
    return folder


# ------------------------------------------------------------------------------------------- plan fidelity
# How the benchmark measures that a plan region was applied (its ``plan_fidelity``): pixels of a 100 dpi grey
# rendering that change by more than 40 after a 3x3 box blur, as a share of the region.  A region is "applied" from
# 50%, "partial" from 10%, otherwise "not applied".
FIDELITY_DPI = 100
APPLIED, PARTIAL = 0.5, 0.1


def changed_shares(before: bytes, after: bytes, regions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The share of each ``rb.plan/0.1`` region, by id, whose pixels changed from the PDF ``before`` to the PDF
    ``after`` (those whose centre lies inside the region), as the benchmark counts them."""
    pages = sorted({r["page"] for r in regions})
    changed = {p: np.abs(_blurred(before, p) - _blurred(after, p)) > 40 for p in pages}
    return {r["id"]: _changed_share(changed[r["page"]], r["poly"]) for r in regions}


def _blurred(pdf: bytes, page: int) -> npt.NDArray[np.int16]:
    """A page rendered in grey at the benchmark's resolution, and blurred."""
    doc = pdfium.PdfDocument(pdf)
    try:
        image = doc[page].render(scale=FIDELITY_DPI / 72, grayscale=True).to_pil().convert("L")
    finally:
        doc.close()
    return cv2.blur(np.asarray(image, dtype=np.uint8), (3, 3)).astype(np.int16)


def _changed_share(changed: npt.NDArray[np.bool_], points: Sequence[Sequence[float]]) -> float:
    s = FIDELITY_DPI / 72
    xs, ys = [p[0] * s for p in points], [p[1] * s for p in points]
    c0, c1 = (int(np.ceil(v - 0.5)) for v in (min(xs), max(xs)))
    r0, r1 = (int(np.ceil(v - 0.5)) for v in (min(ys), max(ys)))
    area = (c1 - c0) * (r1 - r0)
    return float(changed[r0:r1, c0:c1].sum()) / area if area > 0 else 0.0


# ------------------------------------------------------------------------------------------- a busy document
# Two pages where every case of the injected stages meets: an organisation with a name inside it, another inside a
# logo box, a government identifier tool 2's rules recognise, a location with its postcode and a capital city, people
# clustered across both pages (an initial among them), a name inside an e-mail address (its characters' box), a name
# across a line break, a handwritten mark on a word, a signature with no words, a mention found by its box only, a
# label nobody maps and a mention that lands nowhere.
BUSY_LINES = {
    0: (
        "HARBOURSIDE MEDICOLEGAL",
        "Smith & Partners Lawyers",
        "Re: Mr John Andrew SMITH Medicare 2123 45670 1",
        "Address: 42 Wattle Grove Road, Penrith NSW 2750",
        "Mr Smith met Ms Jane Smith and J. Smith today",
        "Email john.smith78@example.org or ring 0412 345 678",
        "Dr Peter",
        "Brown FRACS",
        "Initialled JS",
    ),
    1: ("Seen again by Dr Peter Brown", "Sydney NSW 2000", "Claim WC1234567"),
}
# (id, benchmark type, page, its words as (line, word)): anchored by the words' ids and boxes, as ground truth is.
BUSY_MENTIONS = (
    ("org-letterhead", "ORGANISATION", 0, ((0, 0), (0, 1))),
    ("org-firm", "ORGANISATION", 0, ((1, 0), (1, 1), (1, 2), (1, 3))),
    ("person-firm", "PERSON_NAME", 0, ((1, 0),)),
    ("person-john", "PERSON_NAME", 0, ((2, 2), (2, 3), (2, 4))),
    ("medicare", "ID_GOVERNMENT", 0, ((2, 6), (2, 7), (2, 8))),
    ("address", "ADDRESS", 0, ((3, 1), (3, 2), (3, 3), (3, 4))),
    ("place-penrith", "LOCATION", 0, ((3, 5), (3, 6), (3, 7))),
    ("person-mr-smith", "PERSON_NAME", 0, ((4, 1),)),
    ("person-jane", "PERSON_NAME", 0, ((4, 4), (4, 5))),
    ("person-j", "PERSON_NAME", 0, ((4, 7), (4, 8))),
    ("email", "EMAIL", 0, ((5, 1),)),
    ("phone", "PHONE", 0, ((5, 4), (5, 5), (5, 6))),
    ("person-peter", "PERSON_NAME", 0, ((6, 1), (7, 0))),
    ("fellowship", "QUASI_IDENTIFIER", 0, ((7, 1),)),
    ("marks", "HANDWRITTEN_MARK", 0, ((8, 1),)),
    ("person-peter-again", "PERSON_NAME", 1, ((0, 4), (0, 5))),
    ("place-sydney", "LOCATION", 1, ((1, 0), (1, 1), (1, 2))),
)
BUSY_SIGNATURE: PointBox = (60.0, 400.0, 200.0, 440.0)
BUSY_BLANK: PointBox = (300.0, 500.0, 340.0, 520.0)
# The same cluster id twice is one cluster; "nobody" is not a mention; Mr Smith is in two clusters.
BUSY_CLUSTERS = (
    ("e-john", ("person-john", "person-mr-smith", "person-email")),
    ("e-jane", ("person-jane",)),
    ("e-peter", ("person-peter", "signature", "person-peter-again")),
    ("e-firm", ("org-firm", "person-firm")),
    ("e-jane", ("person-j", "nobody", "person-mr-smith")),
)
# A fixed surrogate key for :func:`oracle_outcome`.
SURROGATE_KEY = hashlib.sha256(b"inject_support").digest()
# A4 at 150 dpi, the resolution NO_MODELS renders at (tests/unit/test_bench_runner.py).
A4 = PageGeometry(0, 595.276, 841.89, 150, 1240, 1754)


def page_tokens(page: int, lines: Sequence[str]) -> list[dict[str, Any]]:
    """``rb.text/0.1`` tokens ``t<page>.<line>.<word>`` for lines of words, in reading order: 11 pt high, lines 20 pt
    apart from 100 pt down the page, 5.5 pt per character and 3 pt between words (display points)."""
    tokens: list[dict[str, Any]] = []
    for row, line in enumerate(lines):
        x, y = 60.0, 100.0 + 20.0 * row
        for k, word in enumerate(line.split()):
            width = 5.5 * len(word)
            box = poly((x, y, x + width, y + 11.0))
            tokens.append({"id": f"t{page}.{row}.{k}", "text": word, "poly": box, "conf": None, "line": f"l{row}"})
            tokens[-1]["order"] = len(tokens) - 1
            x += width + 3.0
    return tokens


def busy_docs() -> Docs:
    """The four injected files of the busy document (:data:`BUSY_LINES`), with one plan region per mention anchor (one
    replacement text given) and regions over a name and the words around it, over a name's two lines, over blank
    paper (twice), over the signature again to be removed, and on a page the document does not have.  The second
    page's first word repeats an id of the first page's, and a word of blanks follows its last."""
    pages = [{"index": page, "tokens": page_tokens(page, lines)} for page, lines in BUSY_LINES.items()]
    words = {t["id"]: t for p in pages for t in p["tokens"]}
    second = pages[1]["tokens"]
    second[0]["id"] = "t0.2.0"
    second.append({**second[-1], "id": "t1.blank", "text": "  ", "poly": poly(BUSY_BLANK), "order": len(second)})

    def bounds(*ids: str) -> PointBox:
        xs = [x for i in ids for x, _y in words[i]["poly"]]
        ys = [y for i in ids for _x, y in words[i]["poly"]]
        return min(xs), min(ys), max(xs), max(ys)

    mentions = []
    for mention_id, label, page, at in BUSY_MENTIONS:
        ids = [f"t{page}.{row}.{k}" for row, k in at]
        text = " ".join(words[i]["text"] for i in ids)
        mentions.append(mention(mention_id, label, text, ids, [words[i]["poly"] for i in ids], page=page))
    x0, y0, x1, y1 = bounds("t0.0.0", "t0.0.1")
    margin = LOGO_MARGIN_PT
    mentions[0]["regions"].append({"page": 0, "poly": poly((x0 - margin, y0 - margin, x1 + margin, y1 + margin))})
    email = words["t0.5.1"]["poly"]
    characters = poly((email[0][0] + 5 * 5.5, email[0][1], email[0][0] + 10 * 5.5, email[2][1]))  # "smith"
    mentions += [
        mention("person-email", "PERSON_NAME", "smith", ["t0.5.1"], [characters]),
        mention("signature", "SIGNATURE", None, [], [poly(BUSY_SIGNATURE)]),
        mention("claim", "ID_CLAIM", "WC1234567", [], [words["t1.2.1"]["poly"]], page=1),
        mention("nowhere", "PERSON_NAME", "Nobody", ["no-such-word"], []),
    ]
    plan = [
        {"id": f"{m['id']}-{k}", "page": r["page"], "poly": r["poly"], "action": "replace", "mentions": [m["id"]]}
        for m in mentions
        for k, r in enumerate(m["regions"] or ())
    ]
    for r in plan:
        if r["id"] == "person-john-0":
            r["replacement"] = "Somebody Else"  # tool 2 draws its own fakes
    extra = [
        ("mixed-0", 0, poly(bounds("t0.4.2", "t0.4.5")), "replace", ["person-jane"]),
        ("two-lines-0", 0, poly(bounds("t0.6.1", "t0.7.0")), "replace", ["person-peter"]),
        ("blank-0", 0, poly(BUSY_BLANK), "replace", []),
        ("blank-1", 0, poly(BUSY_BLANK), "replace", []),
        ("signature-again", 0, poly(BUSY_SIGNATURE), "remove", ["signature"]),
        ("beyond-0", 5, poly(BUSY_BLANK), "replace", ["person-john"]),
    ]
    plan += [{"id": i, "page": p, "poly": b, "action": a, "mentions": n} for i, p, b, a, n in extra]
    return {
        "text.json": {"schema": "rb.text/0.1", "frame": "display_pt", "source": "text_layer", "pages": pages},
        "mentions.json": {"schema": "rb.mentions/0.1", "frame": "display_pt", "mentions": mentions},
        "entities.json": {
            "schema": "rb.entities/0.1",
            "clusters": [{"id": c, "mentions": list(members)} for c, members in BUSY_CLUSTERS],
        },
        "plan.json": {"schema": "rb.plan/0.1", "frame": "display_pt", "regions": plan},
    }


def shuffled(docs: Docs, rng: random.Random) -> Docs:
    """A copy with every list whose order means nothing shuffled: the pages and each page's tokens (their reading order
    is set), the mentions with their token ids and region anchors, the clusters with their members, and the plan
    regions with the mentions they name.  (A span counts in the text's file order: no mention may have one.)"""
    out = copy.deepcopy(docs)
    text = out["text.json"]
    rng.shuffle(text["pages"])
    for page in text["pages"]:
        rng.shuffle(page["tokens"])
    for m in out["mentions.json"]["mentions"]:
        if m.get("span") is not None:
            raise ValueError("a span counts in the text's file order")
        for key in ("tokens", "regions"):
            if m.get(key):
                rng.shuffle(m[key])
    rng.shuffle(out["mentions.json"]["mentions"])
    for c in out["entities.json"]["clusters"]:
        rng.shuffle(c["mentions"])
    rng.shuffle(out["entities.json"]["clusters"])
    for r in out["plan.json"]["regions"]:
        rng.shuffle(r["mentions"])
    rng.shuffle(out["plan.json"]["regions"])
    return out


def oracle_outcome(folder: Path, mode: str, docs: Docs, config: AppConfig) -> dict[str, Any]:
    """What every stage of ``mode`` (one that injects mentions) makes of the injected files ``docs``, without rendering
    anything: the pages are the injected tokens on hand-made A4 pages at 150 dpi; then detection and policy from the
    injected mentions, surrogates as the runner draws them (with :data:`SURROGATE_KEY`), the injected plan's operations
    or tool 2's own planning, the report and the stage files.  Each part as plain data in the order its stage made it
    (dicts as (key, value) pairs), so an order that depends on anything else shows."""
    inputs = OracleInputs(load_injection(mode, write_inject_dir(folder, mode, docs)))
    analysed = []
    for page in inputs.pages(1 + max(p["index"] for p in docs["text.json"]["pages"])):
        geometry = replace(A4, index=page.index)
        analysed.append(PageAnalysis(geometry, PageKind.BORN_DIGITAL, injected_lines(page, geometry)))
    staged = inputs.stage(analysed, config, "doc")
    if staged is None:
        raise ValueError("the mode injects no mentions")
    pages, decisions, persons = staged.pages, staged.decisions, staged.detection.persons
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
    factory = SurrogateFactory(Chooser(SURROGATE_KEY), forbidden_keys(removed, persons))
    surrogates = SurrogateAssigner(factory, persons, tokens).assign(decisions)
    lines = [line for p in pages for line in p.lines]
    rotated = rotated_token_ids(lines)
    # The runner's region helpers read nothing but the configuration.
    redactor: Any = SimpleNamespace(config=config)
    redactor._region_decisions = functools.partial(Redactor._region_decisions, redactor)
    regions = tuple(d for p in pages for d in redactor._region_decisions(p))
    ops = inputs.plan(
        decisions=decisions,
        surrogates=surrogates,
        tokens=tokens,
        pages=pages,
        region_actions={d.region.region_id: d.action for d in regions},
        rotated=rotated,
        config=config,
    )
    if ops is None:  # tool 2 plans, as the runner does
        geoms = {p.geometry.index: p.geometry for p in pages}
        planned = plan_ops(decisions, tokens, geoms, config.redaction, surrogates, rotated, lines)
        ops = Redactor._with_region_ops(redactor, planned, pages)
    report = inputs.report()
    trace = JobTrace(None, pages, decisions, persons, surrogates, ops, {}, regions, report)
    return {
        "mentions": _plain(staged.detection.mentions),
        "decisions": _plain(decisions),
        "persons": _plain(persons),
        "regions": _plain(regions),
        "surrogates": _plain(surrogates),
        "ops": _plain(ops),
        "report": _plain(report),
        "files": _plain(stage_files(trace, text_layer=False)),
    }


def _plain(value: Any) -> Any:
    """``value`` as plain data in the order it holds it: dataclasses as their fields, mappings as (key, value) pairs,
    enums as their values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return [[_plain(k), _plain(v)] for k, v in value.items()]
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, set | frozenset):
        raise TypeError("a set has no order of its own")
    return value
