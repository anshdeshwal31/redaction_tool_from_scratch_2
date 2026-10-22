"""The benchmark's injected stage inputs for tests: its oracle files for a synthetic letter (``rb.text/0.1``,
``rb.mentions/0.1``, ``rb.entities/0.1``, ``rb.plan/0.1``), written in its own format (no benchmark import) the way it
builds them from ground truth (its ``benchmark/runner/oracle.py``): the words with their line and reading order, the
required mentions anchored by their words' ids and boxes, the clusters, and one plan region per mention anchor.

All text is fictitious.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthdocs.generator import medicare_number, provider_number

from mlredact.bench.artifacts import display_poly
from mlredact.bench.inject import MODE_FILES
from mlredact.config.loader import load_config
from mlredact.core.geometry import BBox, union_all
from mlredact.core.model import PageGeometry
from mlredact.ingest.render import open_document, render_page

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
) -> dict[str, Any]:
    """An ``rb.mentions/0.1`` mention with every key, as the benchmark writes it."""
    return {
        "id": mention_id,
        "label": label,
        "text": text,
        "tokens": list(tokens) or None,
        "regions": [{"page": 0, "poly": p} for p in regions] or None,
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
