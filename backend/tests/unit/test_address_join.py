"""The address join (``mlredact.policy.addresses``): the shared case table at the join level, the shapes that
PENDING_CASES lists, the explicit pairing, the verification targets and the exempt positions.  Pure: no models."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from mlredact.config.loader import load_config
from mlredact.core.model import Evidence, Line, Mention, PageAnalysis
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength
from mlredact.policy.addresses import (
    AddressJoin,
    JoinedUnit,
    UnitPart,
    fold_word,
    fold_words,
    join_addresses,
)
from mlredact.policy.engine import split_kept_places
from mlredact.text.address import STRUCTURE_ROLES, Band, BoxRole, KeptKind, Role, UnitKind
from support import GridLine, column, pages_from_grid

E = EntityType
Pos = tuple[int, int]
_TABLE = json.loads((Path(__file__).resolve().parents[1] / "data" / "address_cases.json").read_text("utf-8"))
_CASES: list[dict[str, Any]] = _TABLE["cases"]
_BY_ID = {c["id"]: c for c in _CASES}
_RULES = _TABLE["own_rules"]
_PROFILES = ("client", "broad", "claimant_focused", "maximal")
_POLICY = {name: load_config(name).policy for name in _PROFILES}
_PARTS = _POLICY["client"].model_copy(update={"addresses": "parts"})
_EVIDENCE = (Evidence("test", "1", "test", EvidenceStrength.STRONG, 1.0),)
_NEIGHBOUR_TYPES = {
    "person_name": E.PERSON,
    "title_and_name": E.PERSON,
    "organisation": E.ORGANISATION,
    "hospital": E.ORGANISATION,
    "court": E.ORGANISATION,
    "public_body": E.ORGANISATION,
    "phone": E.PHONE,
    "url": E.URL,
    "business_id": E.ABN,
    "document_ref": E.REFERENCE,
    "date": E.DATE,
    "date_before_street_word": E.DATE,
}
_STATE_WORDS = frozenset({"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"})


# ------------------------------------------------------------------------------------------------ helpers
class Doc:
    """Tool 2 pages for grid lines, with the case table's ``(line, word)`` positions mapped to token ids."""

    def __init__(self, lines: Sequence[GridLine]) -> None:
        self.pages: list[PageAnalysis] = pages_from_grid(lines)
        self.lines: list[Line] = [line for p in self.pages for line in p.lines]
        self.texts: dict[str, str] = {t.token_id: t.text for line in self.lines for t in line.tokens}
        self.order: list[str] = [t.token_id for line in self.lines for t in line.tokens]

    def tid(self, pos: Pos) -> str:
        return self.lines[pos[0]].tokens[pos[1]].token_id

    def tids(self, positions: Sequence[Pos]) -> tuple[str, ...]:
        return tuple(self.tid(p) for p in positions)

    def text_of(self, ids: Sequence[str]) -> str:
        return " ".join(self.texts[t] for t in ids)

    def mention(self, etype: EntityType, positions: Sequence[Pos], name: str = "") -> Mention:
        ids = self.tids(positions)
        return Mention(f"M-{etype.value}-{name}{ids[0]}", etype, ids, _EVIDENCE, self.text_of(ids))

    def span(self, etype: EntityType, line: int, first: int, last: int) -> Mention:
        """A mention over the words ``first``..``last`` of one line."""
        return self.mention(etype, [(line, w) for w in range(first, last + 1)])

    def ids_of(self, join: AddressJoin, etype: EntityType) -> list[str]:
        return [self.text_of(m.token_ids) for m in join.mentions if m.entity_type is etype]


def case_doc(case: Mapping[str, Any]) -> tuple[Doc, dict[str, Band], dict[str, BoxRole]]:
    """A case on tool 2 pages, with the band and box tags for the lines the position rule cannot tell."""
    grid = [
        GridLine(ln["text"], ln.get("page", 0), ln["row"], float(ln.get("x0", 72)), ln.get("x1"))
        for ln in case["lines"]
    ]
    doc = Doc(grid)
    bands = {line.line_id: Band(ln.get("band", "body")) for line, ln in zip(doc.lines, case["lines"], strict=True)}
    boxes = {
        line.line_id: BoxRole(ln["box"]) for line, ln in zip(doc.lines, case["lines"], strict=True) if ln.get("box")
    }
    return doc, bands, boxes


def runs(positions: Sequence[Pos]) -> list[list[Pos]]:
    out: list[list[Pos]] = []
    for p in sorted(positions):
        if out and out[-1][-1][0] == p[0] and out[-1][-1][1] + 1 == p[1]:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def detected_parts(doc: Doc, case: Mapping[str, Any]) -> list[Mention]:
    """The address parts as the detectors give them: the street line a STREET_ADDRESS, the suburb a LOCALITY and the
    postcode a POSTCODE; the state and the country belong to no mention."""
    out: list[Mention] = []
    for unit in case["expect"]["addresses"]:
        roles = {(li, wi): role for li, wi, _t, role in unit["words"]}
        street = {tuple(x) for x in unit["street_line"]}
        groups = (
            (E.STREET_ADDRESS, [p for p in roles if p in street]),
            (E.LOCALITY, [p for p, r in roles.items() if r == "suburb"]),
            (E.POSTCODE, [p for p, r in roles.items() if r in ("postcode", "state+postcode")]),
        )
        for etype, group in groups:
            out.extend(doc.mention(etype, run) for run in runs(group))
    return out


def detected_places(doc: Doc, case: Mapping[str, Any]) -> list[Mention]:
    """The places kept on their own as a model reads them: the place words a LOCALITY and a postcode a POSTCODE."""
    out: list[Mention] = []
    for item in case["expect"]["kept_places"]:
        if item["kind"] in ("street_in_prose", "state", "country"):
            continue
        words = [tuple(p) for p in item["words"]]
        bare = {p: doc.texts[doc.tid(p)].strip(".,") for p in words}
        postcode = [p for p in words if bare[p].isdigit() and len(bare[p]) == 4]
        place = [p for p in words if p not in postcode and bare[p] not in _STATE_WORDS]
        out.extend(doc.mention(E.LOCALITY, run) for run in runs(place))
        out.extend(doc.mention(E.POSTCODE, run) for run in runs(postcode))
    return out


def neighbours(doc: Doc, case: Mapping[str, Any]) -> list[Mention]:
    """The look-alikes and neighbours that another rule owns: names, organisations, phones, dates, ..."""
    out: list[Mention] = []
    for key in ("lookalikes", "not_address"):
        for item in case["expect"][key]:
            etype = _NEIGHBOUR_TYPES.get(item["kind"])
            if etype is not None:
                out.extend(doc.mention(etype, run, "n") for run in runs([tuple(p) for p in item["words"]]))
    return out


def join_case(case: Mapping[str, Any], profile: str, variant: str) -> tuple[Doc, AddressJoin, list[Mention]]:
    """Join a case.  The variants are the mentions the detectors plausibly give: none at all (``bare``), the address
    parts (``parts``), plus the places kept on their own (``places``), plus names, organisations, phones and dates
    (``neighbours``)."""
    doc, bands, boxes = case_doc(case)
    mentions: list[Mention] = []
    if variant in ("parts", "places", "neighbours"):
        mentions += detected_parts(doc, case)
    if variant in ("places", "neighbours"):
        mentions += detected_places(doc, case)
    if variant == "neighbours":
        mentions += neighbours(doc, case)
    mentions.sort(key=lambda m: m.token_ids[0])
    result = join_addresses(mentions, doc.pages, _POLICY[profile], band_fraction=0.0, bands=bands, boxes=boxes)
    return doc, result, mentions


def expected_unit_sets(doc: Doc, case: Mapping[str, Any]) -> list[frozenset[str]]:
    return sorted(
        (
            frozenset(doc.tids([tuple(p) for p in e["street_line"]] + [tuple(p) for p in e["locality"]]))
            for e in case["expect"]["addresses"]
        ),
        key=min,
    )


def unit_sets(result: AddressJoin) -> list[frozenset[str]]:
    return sorted((frozenset(u.roles) for u in result.units.values()), key=min)


def check_join(
    case: Mapping[str, Any], doc: Doc, result: AddressJoin, mentions: Sequence[Mention], profile: str
) -> list[str]:
    """README 'What to assert per case', item 1 (units, kept places, look-alikes), at the join level."""
    errs: list[str] = []
    expect = case["expect"]
    if unit_sets(result) != expected_unit_sets(doc, case):
        return [f"units differ: {len(result.units)} found, {len(expect['addresses'])} expected"]
    by_id = {m.mention_id: m for m in result.mentions}
    for unit in result.units.values():
        carried = {t for mid in unit.mention_ids for t in by_id[mid].token_ids}
        if carried != set(unit.roles) - set(unit.excluded):
            errs.append("unit tokens are not carried by its mentions")
        for mid in unit.mention_ids:
            if by_id[mid].entity_type is not E.STREET_ADDRESS or len(result.layouts[mid].roles) != len(
                by_id[mid].token_ids
            ):
                errs.append("a unit mention is not a STREET_ADDRESS with a role per token")
    in_units = {t for u in result.units.values() for t in u.roles}
    kept_tokens = {t for k in result.kept for t in k.token_ids}
    for item in expect["kept_places"]:
        ids = set(doc.tids([tuple(p) for p in item["words"]]))
        if ids & in_units:
            errs.append(f"kept place '{item['kind']}' inside a unit")
        if not ids <= kept_tokens:
            errs.append(f"kept place '{item['kind']}' not recorded")
    for key in ("lookalikes", "not_address"):
        for item in expect[key]:
            if set(doc.tids([tuple(p) for p in item["words"]])) & in_units:
                errs.append(f"{key[:-1]} '{item['kind']}' inside a unit")
    # a mention that no unit touches comes out as it went in
    for m in mentions:
        if not set(m.token_ids) & in_units and by_id.get(m.mention_id) != m:
            errs.append(f"untouched {m.entity_type.value} mention changed")
    # every unit word is replaced under every profile
    actions = _POLICY[profile].entity_actions
    replaced = {t for m in result.mentions if actions[m.entity_type] is not ActionKind.KEEP for t in m.token_ids}
    left = {t for u in result.units.values() for t in u.excluded}
    if not (in_units - left) <= replaced:
        errs.append("a unit word is not replaced")
    return errs


# ------------------------------------------------------------------------------------------------ the case table
@pytest.mark.parametrize("variant", ["bare", "parts", "places", "neighbours"])
@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_case_client(case: dict[str, Any], variant: str) -> None:
    """Under the client profile: each expected unit is exactly one unit, no other unit exists, kept places stay out of
    every unit and are recorded, look-alikes stay out, and every unit word is replaced, whatever the detectors found
    (nothing at all, the address parts, the places kept on their own, or names, phones and dates besides)."""
    doc, result, mentions = join_case(case, "client", variant)
    assert check_join(case, doc, result, mentions, "client") == []


@pytest.mark.parametrize("profile", [p for p in _PROFILES if p != "client"])
@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_case_other_profiles(case: dict[str, Any], profile: str) -> None:
    """Units, roles, structure words and look-alikes hold in every profile (owner ruling 5)."""
    doc, result, mentions = join_case(case, profile, "neighbours")
    assert check_join(case, doc, result, mentions, profile) == []


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_roles_keys_and_repeats_follow_the_grammar(case: dict[str, Any]) -> None:
    """The unit's roles, kind, key, overseas flag and repeats are the table's."""
    doc, result, _ = join_case(case, "client", "parts")
    by_set = {frozenset(u.roles): u for u in result.units.values()}
    ids: dict[str, JoinedUnit] = {}
    for e in case["expect"]["addresses"]:
        unit = by_set[frozenset(doc.tids([tuple(p) for p in e["street_line"]] + [tuple(p) for p in e["locality"]]))]
        ids[e["id"]] = unit
        assert unit.kind.value == e["kind"]
        assert unit.overseas == e.get("overseas", False)
        for li, wi, _text, role in e["words"]:
            assert unit.roles[doc.tid((li, wi))] == role
        assert {t for t, r in unit.roles.items() if r in STRUCTURE_ROLES} == set(
            doc.tids([tuple(p) for p in e["structure"]])
        )
        assert {p.token_id: (p.keep, p.change) for p in unit.partial} == {
            doc.tid(tuple(x["at"])): (x["keep"], x["change"]) for x in e.get("partial_structure", [])
        }
    for e in case["expect"]["addresses"]:
        if e.get("repeat_of"):
            assert ids[e["id"]].repeat_of == ids[e["repeat_of"]].unit_id


def test_matters_give_one_key_per_address() -> None:
    """The units' keys: nine spellings of one address share one, a different address in the suburb does not."""
    for m in _TABLE["matters"]:
        keys: dict[str, JoinedUnit] = {}
        for cid in m["cases"]:
            doc, result, _ = join_case(_BY_ID[cid], "client", "parts")
            by_set = {frozenset(u.roles): u for u in result.units.values()}
            for e in _BY_ID[cid]["expect"]["addresses"]:
                ids = doc.tids([tuple(p) for p in e["street_line"]] + [tuple(p) for p in e["locality"]])
                keys[f"{cid}/{e['id']}"] = by_set[frozenset(ids)]
        for group in m.get("same_fake", []):
            for ref in group[1:]:
                assert keys[group[0]].key.same_address(keys[ref].key), ref
        for a, b in m.get("different_fake", []):
            if a in keys and b in keys:
                assert not keys[a].key.same_address(keys[b].key), (a, b)


# ------------------------------------------------------------------------------------------------ item 4: targets
def simulated_output(
    doc: Doc, result: AddressJoin, profile: str, leak: Callable[[JoinedUnit, str, Role | None], bool] | None = None
) -> dict[str, str]:
    """The output of a correct tool: every unit word outside the structure words replaced (a glued word keeps its
    letters), every other replaced mention replaced by its own rule.  ``leak`` leaves chosen unit words unchanged."""
    actions = _POLICY[profile].entity_actions
    out = dict(doc.texts)
    n = 0
    in_units: set[str] = set()
    for u in result.units.values():
        partial = {p.token_id for p in u.partial}
        for t, role in u.roles.items():
            if t in u.excluded:
                continue
            in_units.add(t)
            if (leak is not None and leak(u, t, role)) or role in STRUCTURE_ROLES:
                continue
            n += 1
            out[t] = re.sub(r"\d", "9", doc.texts[t]) if t in partial else f"Fake{n}x"
    for m in result.mentions:
        if actions[m.entity_type] is not ActionKind.KEEP:
            for t in m.token_ids:
                if t not in in_units:
                    n += 1
                    out[t] = f"Other{n}x"
    return out


def target_hits(doc: Doc, result: AddressJoin, output: Mapping[str, str], exempt: set[str]) -> list[tuple[str, str]]:
    """Occurrences of a unit's targets in the output at positions that are not exempt (README item 4)."""
    flat = [(t, fold_word(output[t])) for t in doc.order]
    flat = [(t, w) for t, w in flat if w]
    found: list[tuple[str, str]] = []
    for uid, targets in result.targets.items():
        for t, w in flat:
            if w in targets.whole and t not in exempt:
                found.append((uid, w))
        for c in targets.compound:
            for i in range(len(flat) - len(c) + 1):
                window = flat[i : i + len(c)]
                if tuple(w for _, w in window) == c and not all(t in exempt for t, _ in window):
                    found.append((uid, " ".join(c)))
    return found


@pytest.mark.parametrize("profile", _PROFILES)
def test_a_correct_output_has_no_hit_outside_the_exempt_positions(profile: str) -> None:
    """The targets and the join's exempt positions (kept places and kept mentions) are enough for a correct tool: no
    hit in any of the 293 cases, under every profile.  Look-alikes without a mention (clinical phrases, citations,
    labels) need no exemption of their own."""
    bad: dict[str, list[tuple[str, str]]] = {}
    for case in _CASES:
        doc, result, _ = join_case(case, profile, "neighbours")
        hits = target_hits(doc, result, simulated_output(doc, result, profile), set(result.exempt.all))
        if hits:
            bad[case["id"]] = hits[:2]
    assert bad == {}


@pytest.mark.parametrize(
    ("name", "keep"),
    [
        ("keeps the locality", {Role.SUBURB, Role.POSTCODE, Role.STATE_AND_POSTCODE}),
        ("keeps the street name", {Role.STREET_NAME}),
    ],
)
def test_a_leaking_output_is_caught_on_every_affected_unit(name: str, keep: set[Role]) -> None:
    """A tool that leaves the suburb and postcode, or the street name, of a unit unchanged is caught by that unit's
    targets: all 256 units with a suburb or postcode, all 200 with a street name."""
    affected = caught = 0
    missed: list[str] = []
    for case in _CASES:
        doc, result, _ = join_case(case, "client", "neighbours")
        for uid, u in result.units.items():
            if not any(r in keep for r in u.roles.values()):
                continue
            affected += 1
            out = simulated_output(
                doc, result, "client", leak=lambda uu, _t, r, uid=uid: uu.unit_id == uid and r in keep
            )
            if any(h[0] == uid for h in target_hits(doc, result, out, set(result.exempt.all))):
                caught += 1
            else:
                missed.append(f"{case['id']}")
    assert (name, caught, missed) == (name, affected, [])
    assert affected >= (256 if Role.SUBURB in keep else 200)


# ------------------------------------------------------------------------------------------------ kept places
@pytest.mark.parametrize("profile", _PROFILES)
def test_kept_places_follow_each_profiles_own_rule(profile: str) -> None:
    """A place on its own stays where the profile keeps its type (the client's locality and postcode) and is replaced by
    its own rule elsewhere; a state, a country or a street in prose has no mention and stays everywhere."""
    actions = _POLICY[profile].entity_actions
    for case in _CASES:
        doc, result, _ = join_case(case, profile, "places")
        mention_of = {t: m for m in result.mentions for t in m.token_ids}
        exempt = set(result.exempt.kept_places)
        for item in case["expect"]["kept_places"]:
            for t in doc.tids([tuple(p) for p in item["words"]]):
                m = mention_of.get(t)
                replaced = m is not None and actions[m.entity_type] is not ActionKind.KEEP
                assert (t in exempt) == (not replaced), (case["id"], item["kind"], profile)
        # nothing is exempt unless it is a recorded kept place
        assert exempt <= {t for k in result.kept for t in k.token_ids}


def test_the_client_keeps_every_recorded_place() -> None:
    for case in _CASES:
        doc, result, _ = join_case(case, "client", "places")
        wanted = {t for item in case["expect"]["kept_places"] for t in doc.tids([tuple(p) for p in item["words"]])}
        assert wanted <= set(result.exempt.kept_places), case["id"]
        assert not wanted & {t for u in result.units.values() for t in u.roles}


def test_a_place_after_a_cue_keeps_its_kind() -> None:
    doc, result, _ = join_case(_BY_ID["date-and-place-of-birth-label"], "client", "places")
    (kept,) = [k for k in result.kept if k.kind is KeptKind.PLACE_AFTER_CUE]
    assert doc.text_of(kept.token_ids) == "Penrith NSW 2750" and kept.after_cue


# ------------------------------------------------------------------------------------------ the shapes of PENDING_CASES
def run_join(
    texts: Sequence[str],
    specs: Sequence[tuple[EntityType, int, int, int]] = (),
    profile: str = "client",
    **kw: Any,
) -> tuple[Doc, AddressJoin]:
    """Lines one under the other with mentions over word ranges ``(type, line, first word, last word)``."""
    doc = Doc(column(*texts))
    mentions = [doc.span(*s) for s in specs]
    kw.setdefault("band_fraction", 0.0)
    return doc, join_addresses(mentions, doc.pages, _POLICY[profile], **kw)


def unit_texts(doc: Doc, result: AddressJoin) -> list[str]:
    return [doc.text_of(sorted(u.roles, key=doc.order.index)) for u in result.units.values()]


class TestPendingCases:
    """The shapes the reference parser got wrong (PENDING_CASES.md), through the join, with and without the mentions the
    detectors give."""

    @pytest.mark.parametrize(
        ("line", "unit"),
        [
            ("Address: Smith Street, Nth Ryde 2113", "Smith Street, Nth Ryde 2113"),
            ("Smith Street, Nth Ryde NSW", "Smith Street, Nth Ryde NSW"),
            ("Address: Smith Street, Sth Hurstville NSW 2221", "Smith Street, Sth Hurstville NSW 2221"),
        ],
    )
    def test_a_numberless_street_in_an_address_position_is_an_address(self, line: str, unit: str) -> None:
        doc, result = run_join([line])
        assert unit_texts(doc, result) == [unit]
        assert [m.value for m in result.mentions] == [unit]

    def test_a_street_and_its_locality_pre_detected_become_one_mention(self) -> None:
        _doc, result = run_join(
            ["Address: Smith Street, Sth Hurstville NSW 2221"],
            [(E.LOCALITY, 0, 3, 4), (E.POSTCODE, 0, 6, 6)],
        )
        assert [(m.entity_type, m.value) for m in result.mentions] == [
            (E.STREET_ADDRESS, "Smith Street, Sth Hurstville NSW 2221")
        ]

    @pytest.mark.parametrize(
        ("texts", "place", "street"),
        [
            (
                ["She walked along Smith Street. Penrith NSW 2750 is where she lives."],
                "Penrith NSW 2750",
                "Smith Street.",
            ),
            (
                ["She was injured on Victoria Road. Ryde NSW 2112 is where she lives."],
                "Ryde NSW 2112",
                "Victoria Road.",
            ),
            (["She lives on Smith Street North; Ryde NSW 2112 is nearby."], "Ryde NSW 2112", "Smith Street North;"),
            (
                ["She walked along Smith Street.", "Penrith NSW 2750 is where she lives."],
                "Penrith NSW 2750",
                "Smith Street.",
            ),
            (
                ["She was injured on Victoria Road.", "Ryde NSW 2112 is where she lives."],
                "Ryde NSW 2112",
                "Victoria Road.",
            ),
            (["She lives on Smith Street North;", "Ryde NSW 2112 is nearby."], "Ryde NSW 2112", "Smith Street North;"),
        ],
    )
    def test_nothing_joins_across_a_sentence_break(self, texts: list[str], place: str, street: str) -> None:
        doc, result = run_join(texts)
        assert unit_texts(doc, result) == [place]
        (unit,) = result.units.values()
        assert unit.kind is UnitKind.PLACE_ONLY
        assert [(k.kind, doc.text_of(k.token_ids).rstrip(".;")) for k in result.kept] == [
            (KeptKind.STREET_IN_PROSE, street.rstrip(".;"))
        ]

    @pytest.mark.parametrize(
        ("texts", "unit"),
        [
            (["She was injured at 12 Victoria Road,", "Ryde NSW 2112 in May."], "12 Victoria Road, Ryde NSW 2112"),
            (["She was injured on Victoria Road,", "Ryde NSW 2112 in May."], "Victoria Road, Ryde NSW 2112"),
            (["She was injured on Victoria Road, Ryde", "NSW 2112 in May."], "Victoria Road, Ryde NSW 2112"),
        ],
    )
    def test_a_place_wrapped_onto_the_next_line_still_joins(self, texts: list[str], unit: str) -> None:
        doc, result = run_join(texts)
        assert unit_texts(doc, result) == [unit]
        (mention,) = result.mentions
        assert mention.entity_type is E.STREET_ADDRESS and mention.value == unit

    def test_a_numbered_street_before_a_later_place_is_two_units(self) -> None:
        doc, result = run_join(["She walked from 12 George Street to Parramatta NSW 2150"])
        assert sorted(unit_texts(doc, result)) == ["12 George Street", "Parramatta NSW 2150"]
        assert sorted(u.kind.value for u in result.units.values()) == ["place_only", "street"]
        assert "to" not in [m.value for m in result.mentions]

    def test_a_street_whose_suburb_starts_with_a_type_word(self) -> None:
        doc, result = run_join(["She visited Smith Lane Cove NSW 2066"])
        assert unit_texts(doc, result) == ["Lane Cove NSW 2066"]
        (unit,) = result.units.values()
        assert unit.kind is UnitKind.PLACE_ONLY


# ------------------------------------------------------------------------------------- shapes found in the review
class TestMoreShapes:
    """Address shapes the case table does not hold, each replaced whole (fictional lines)."""

    @pytest.mark.parametrize(
        ("line", "unit"),
        [
            ("L4/123 George St, Sydney NSW 2000", "L4/123 George St, Sydney NSW 2000"),
            ("U5/3 Smith Street, Penrith NSW 2750", "U5/3 Smith Street, Penrith NSW 2750"),
            ("#12 Smith Street, Penrith NSW 2750", "#12 Smith Street, Penrith NSW 2750"),
            ("12, Smith Street, Penrith NSW 2750", "12, Smith Street, Penrith NSW 2750"),
        ],
    )
    def test_a_glued_level_or_unit_code_a_hash_or_a_comma_after_the_number(self, line: str, unit: str) -> None:
        doc, result = run_join([line])
        assert unit_texts(doc, result) == [unit]

    @pytest.mark.parametrize(
        "line", ["She drives a 4 Wheel Drive.", "They rented a 3 Bedroom Place.", "It was a 2 Storey Terrace."]
    )
    def test_a_count_noun_after_a_number_is_no_street(self, line: str) -> None:
        doc, result = run_join([line])
        assert unit_texts(doc, result) == []

    @pytest.mark.parametrize(
        "line",
        [
            "Level 2 evidence supports L4 fusion.",
            "MRI shows an L4/5 disc protrusion at the L4/5 Level.",
            "See page 12, Smith said nothing.",
        ],
    )
    def test_look_alikes_of_those_shapes_are_no_address(self, line: str) -> None:
        doc, result = run_join([line])
        assert unit_texts(doc, result) == []


# ------------------------------------------------------------------------------------------------ mentions around units
class TestMentions:
    def test_a_fused_locality_becomes_the_address(self) -> None:
        """3a: a weak LOCALITY span over a whole address is the address."""
        _doc, result = run_join(["8 Broadway, Ultimo NSW 2007"], [(E.LOCALITY, 0, 0, 4)])
        (m,) = result.mentions
        assert (m.entity_type, m.value) == (E.STREET_ADDRESS, "8 Broadway, Ultimo NSW 2007")

    def test_a_phone_after_the_place_stays_out(self) -> None:
        texts = ["Address: 42 Wattle Grove Road, Penrith NSW 2750 T (02) 5550 6610"]
        _doc, result = run_join(texts, [(E.STREET_ADDRESS, 0, 1, 7), (E.PHONE, 0, 8, 11)])
        assert [(m.entity_type, m.value) for m in result.mentions] == [
            (E.STREET_ADDRESS, "42 Wattle Grove Road, Penrith NSW 2750"),
            (E.PHONE, "T (02) 5550 6610"),
        ]

    def test_an_address_span_that_runs_on_keeps_the_rest_as_an_address_mention(self) -> None:
        """Fail closed: words of a STREET_ADDRESS mention outside the unit stay in a mention of that type."""
        texts = ["Address: 42 Wattle Grove Road, Penrith NSW 2750 T (02) 5550 6610"]
        _doc, result = run_join(texts, [(E.STREET_ADDRESS, 0, 1, 11)])
        assert [(m.entity_type, m.value) for m in result.mentions] == [
            (E.STREET_ADDRESS, "42 Wattle Grove Road, Penrith NSW 2750"),
            (E.STREET_ADDRESS, "T (02) 5550 6610"),
        ]
        assert len(result.units) == 1

    def test_persons_are_never_absorbed(self) -> None:
        doc, result = run_join(["12 Smith Street, Penrith NSW 2750"], [(E.PERSON, 0, 1, 1)])
        (unit,) = result.units.values()
        assert doc.text_of(unit.excluded) == "Smith"
        assert [(m.entity_type, m.value) for m in result.mentions] == [
            (E.STREET_ADDRESS, "12"),
            (E.PERSON, "Smith"),
            (E.STREET_ADDRESS, "Street, Penrith NSW 2750"),
        ]
        assert [result.layouts[i].part for i in unit.mention_ids] == [UnitPart.STREET, UnitPart.JOINED]

    @pytest.mark.parametrize("etype", [E.ORGANISATION, E.EMAIL, E.ABN, E.REFERENCE, E.FILE_PATH])
    def test_organisations_contacts_and_identifiers_are_never_absorbed(self, etype: EntityType) -> None:
        doc, result = run_join(["Westpoint Tower 45 Smith Street, Penrith NSW 2750"], [(etype, 0, 0, 1)])
        assert [m.entity_type for m in result.mentions].count(etype) == 1
        assert unit_texts(doc, result) == ["45 Smith Street, Penrith NSW 2750"]

    def test_a_date_inside_a_unit_gives_way(self) -> None:
        """A street named after a month: a DATE mention over '12 March' loses the words the unit owns."""
        _doc, result = run_join(["12 March Road, Richmond NSW 2753"], [(E.DATE, 0, 0, 1)])
        assert [(m.entity_type, m.value) for m in result.mentions] == [
            (E.STREET_ADDRESS, "12 March Road, Richmond NSW 2753")
        ]

    def test_a_date_beside_a_unit_is_untouched(self) -> None:
        _doc, result = run_join(["Seen 12 June 2024 at 12 Smith Street, Penrith NSW 2750"], [(E.DATE, 0, 1, 3)])
        assert [(m.entity_type, m.value) for m in result.mentions] == [
            (E.DATE, "12 June 2024"),
            (E.STREET_ADDRESS, "12 Smith Street, Penrith NSW 2750"),
        ]

    def test_the_parts_of_a_block_are_one_mention_across_its_lines(self) -> None:
        _doc, result = run_join(
            ["Westpoint Tower", "45 Smith Street", "Penrith NSW 2750"],
            [(E.STREET_ADDRESS, 1, 0, 2), (E.LOCALITY, 2, 0, 0), (E.POSTCODE, 2, 2, 2)],
        )
        (m,) = result.mentions
        assert m.value == "Westpoint Tower 45 Smith Street Penrith NSW 2750"
        layout = result.layouts[m.mention_id]
        assert layout.part is UnitPart.JOINED and layout.place_start == 5
        assert layout.roles[0] is Role.BUILDING and layout.roles[-1] is Role.POSTCODE

    def test_a_separator_makes_two_mentions_of_one_unit(self) -> None:
        _doc, result = run_join(["12 Smith St | Penrith NSW 2750"])
        (unit,) = result.units.values()
        assert [result.layouts[i].part for i in unit.mention_ids] == [UnitPart.STREET, UnitPart.PLACE]
        assert [result.layouts[i].place_start for i in unit.mention_ids] == [None, 0]

    def test_an_ner_address_over_a_place_on_its_own_stays_a_kept_place(self) -> None:
        """A model that reads a place of birth as an address: the join retypes it as the LOCALITY it always was (client
        keeps it); another profile replaces it by its own locality rule."""
        texts = ["Born in Dubbo NSW 2830, she moved."]
        _, result = run_join(texts, [(E.STREET_ADDRESS, 0, 2, 4)], "client")
        assert [(m.entity_type, m.value) for m in result.mentions] == [(E.LOCALITY, "Dubbo NSW 2830,")]
        assert _POLICY["client"].entity_actions[E.LOCALITY] is ActionKind.KEEP and not result.units
        _doc, result = run_join(texts, [(E.STREET_ADDRESS, 0, 2, 4)], "broad")
        assert [(m.entity_type, m.value) for m in result.mentions] == [(E.LOCALITY, "Dubbo NSW"), (E.POSTCODE, "2830,")]
        assert all(_POLICY["broad"].entity_actions[m.entity_type] is ActionKind.SURROGATE for m in result.mentions)

    def test_an_ner_address_over_a_street_in_prose_stays_replaced(self) -> None:
        """Fail closed: a street named in prose is kept only when no detector made it a mention."""
        doc, result = run_join(["She walks along Smith Street every morning."], [(E.STREET_ADDRESS, 0, 3, 4)])
        assert [(m.entity_type, m.value) for m in result.mentions] == [(E.STREET_ADDRESS, "Smith Street")]
        assert KeptKind.STREET_IN_PROSE in [k.kind for k in result.kept]
        assert not {"Smith", "Street"} & {doc.texts[t] for t in result.exempt.kept_places}  # it is replaced

    def test_a_replaced_postcode_is_split_off_a_place_mention(self) -> None:
        policy = load_config("client", overrides={"policy": {"entity_actions": {"postcode": "surrogate"}}}).policy
        doc = Doc(column("Born in Dubbo NSW 2830, she moved."))
        result = join_addresses([doc.span(E.STREET_ADDRESS, 0, 2, 4)], doc.pages, policy, band_fraction=0.0)
        assert [(m.entity_type, m.value) for m in result.mentions] == [(E.LOCALITY, "Dubbo NSW"), (E.POSTCODE, "2830,")]

    def test_unrelated_mentions_come_out_unchanged_and_sorted(self) -> None:
        doc = Doc(column("Seen by Dr Jane Citizen", "12 Smith Street, Penrith NSW 2750", "Phone 0412 345 678"))
        person, phone = doc.span(E.PERSON, 0, 3, 4), doc.span(E.PHONE, 2, 1, 3)
        result = join_addresses([phone, person], doc.pages, _POLICY["client"], band_fraction=0.0)
        assert [m.entity_type for m in result.mentions] == [E.PERSON, E.STREET_ADDRESS, E.PHONE]
        assert result.mentions[0] is person and result.mentions[2] is phone

    def test_a_mention_the_pages_do_not_hold_is_passed_through(self) -> None:
        doc = Doc(column("12 Smith Street, Penrith NSW 2750"))
        stray = Mention("M-stray", E.PERSON, ("p0009.l0000.w000",), _EVIDENCE, "X")
        result = join_addresses([stray], doc.pages, _POLICY["client"], band_fraction=0.0)
        assert [m.entity_type for m in result.mentions] == [E.STREET_ADDRESS, E.PERSON]
        assert result.mentions[1] is stray

    def test_the_evidence_of_the_absorbed_mentions_is_kept_and_the_join_adds_its_own(self) -> None:
        doc = Doc(column("12 Smith Street, Penrith NSW 2750"))
        weak = Evidence("ner", "1", "ner.address", EvidenceStrength.WEAK, 0.4)
        m = Mention("M-1", E.LOCALITY, doc.tids([(0, i) for i in range(5)]), (weak,), "x")
        (out,) = join_addresses([m], doc.pages, _POLICY["client"], band_fraction=0.0).mentions
        assert weak in out.evidence
        join_ev = [e for e in out.evidence if e.detector_id == "policy.join"]
        assert len(join_ev) == 1 and join_ev[0].strength is EvidenceStrength.STRONG
        assert list(out.evidence) == sorted(out.evidence)


class TestFormUnits:
    def test_the_parts_of_a_form_stay_separate_mentions_each_with_its_role(self) -> None:
        doc, result, _ = join_case(_BY_ID["form-boxes-on-own-lines"], "client", "bare")
        (unit,) = result.units.values()
        assert unit.form
        parts = [
            (doc.text_of(next(m.token_ids for m in result.mentions if m.mention_id == i)), result.layouts[i])
            for i in unit.mention_ids
        ]
        assert [(text, lay.part) for text, lay in parts] == [
            ("5", UnitPart.STREET),
            ("12", UnitPart.STREET),
            ("Smith Street", UnitPart.STREET),
            ("Penrith", UnitPart.PLACE),
            ("NSW", UnitPart.PLACE),
            ("2750", UnitPart.PLACE),
        ]
        assert [lay.roles for _t, lay in parts] == [
            (Role.UNIT_NUMBER,),
            (Role.NUMBER,),
            (Role.STREET_NAME, Role.STREET_TYPE),
            (Role.SUBURB,),
            (Role.STATE,),
            (Role.POSTCODE,),
        ]
        assert [lay.place_start for _t, lay in parts] == [None, None, None, 0, 0, 0]

    def test_one_row_of_boxes_is_one_unit_with_the_box_tags(self) -> None:
        case = _BY_ID["form-one-row-colonless"]
        _doc, result, _ = join_case(case, "client", "bare")
        assert len(result.units) == 1 and len(next(iter(result.units.values())).mention_ids) == 4

    def test_unreadable_labels_need_the_box_tags(self) -> None:
        case = _BY_ID["form-unreadable-labels"]
        doc, bands, boxes = case_doc(case)
        tagged = join_addresses([], doc.pages, _POLICY["client"], band_fraction=0.0, bands=bands, boxes=boxes)
        assert len(tagged.units) == 1
        untagged = join_addresses([], doc.pages, _POLICY["client"], band_fraction=0.0, bands=bands)
        assert unit_sets(untagged) != expected_unit_sets(doc, case)
        assert unit_sets(tagged) == expected_unit_sets(doc, case)


# --------------------------------------------------------------------------------------------- settings and passthrough
class TestAddressesParts:
    def test_parts_is_the_old_split(self) -> None:
        """``addresses: parts`` calls split_kept_places and nothing else: no layouts, units or targets."""
        doc = Doc(column("12 Smith Street, Parramatta NSW 2150"))
        m = doc.span(E.STREET_ADDRESS, 0, 0, 5)
        tokens = {t.token_id: t for p in doc.pages for t in p.tokens()}
        result = join_addresses([m], doc.pages, _PARTS)
        assert result.mentions == tuple(split_kept_places([m], tokens, _PARTS))
        assert [(x.entity_type, x.value) for x in result.mentions] == [
            (E.STREET_ADDRESS, "12 Smith Street,"),
            (E.LOCALITY, "Parramatta NSW 2150"),
        ]
        assert not result.layouts and not result.units and not result.targets and not result.kept
        assert result.exempt.all == frozenset()


# ------------------------------------------------------------------------------------------------ pairs
class TestPairs:
    """The explicit pairing of the oracle modes: a place mention and the street-line mention of its address."""

    def test_a_pair_makes_one_unit_where_the_grammar_reads_none(self) -> None:
        """'Zarnok' is no locality, so the grammar reads no address and a client profile would keep the LOCALITY."""
        doc = Doc(column("Flurble Quux,", "Zarnok NSW 2750", "", "Nothing follows."))
        street, place = doc.span(E.STREET_ADDRESS, 0, 0, 1), doc.span(E.LOCALITY, 1, 0, 2)
        alone = join_addresses([street, place], doc.pages, _POLICY["client"], band_fraction=0.0)
        assert not alone.units and alone.mentions == (street, place)
        pairs = {place.mention_id: street.mention_id}
        paired = join_addresses([street, place], doc.pages, _POLICY["client"], pairs, band_fraction=0.0)
        (unit,) = paired.units.values()
        assert unit.forced and doc.text_of(unit.token_ids) == "Flurble Quux, Zarnok NSW 2750"
        assert doc.text_of(unit.locality) == "Zarnok NSW 2750" and doc.text_of(unit.street) == "Flurble Quux,"
        (m,) = paired.mentions
        assert m.entity_type is E.STREET_ADDRESS and m.value == "Flurble Quux, Zarnok NSW 2750"
        layout = paired.layouts[m.mention_id]
        assert layout.part is UnitPart.JOINED and layout.place_start == 2
        assert layout.roles == (None,) * 5  # not read by the grammar: identifying, so fail closed
        (targets,) = paired.targets.values()
        assert {"flurble", "quux", "zarnok"} <= set(targets.whole)

    def test_a_pair_reads_the_words_it_can(self) -> None:
        doc = Doc(column("Flurble Quux,", "Penrith NSW 2750", "", "Nothing follows."))
        street, place = doc.span(E.STREET_ADDRESS, 0, 0, 1), doc.span(E.LOCALITY, 1, 0, 2)
        pairs = {place.mention_id: street.mention_id}
        paired = join_addresses([street, place], doc.pages, _POLICY["client"], pairs, band_fraction=0.0)
        (unit,) = paired.units.values()
        assert unit.forced
        layout = paired.layouts[unit.mention_ids[0]]
        assert layout.roles == (None, None, Role.SUBURB, Role.STATE, Role.POSTCODE)

    def test_an_unpaired_place_is_a_place_only_unit(self) -> None:
        doc = Doc(column("Zarnok NSW 2750"))
        place = doc.span(E.LOCALITY, 0, 0, 2)
        result = join_addresses([place], doc.pages, _POLICY["client"], {place.mention_id: None}, band_fraction=0.0)
        (unit,) = result.units.values()
        assert unit.kind is UnitKind.PLACE_ONLY and unit.forced
        (m,) = result.mentions
        assert (m.entity_type, m.value) == (E.STREET_ADDRESS, "Zarnok NSW 2750")

    def test_a_pair_the_grammar_already_reads_changes_nothing(self) -> None:
        doc = Doc(column("12 Smith Street,", "Penrith NSW 2750"))
        street, place = doc.span(E.STREET_ADDRESS, 0, 0, 2), doc.span(E.LOCALITY, 1, 0, 0)
        mentions = [street, place, doc.span(E.POSTCODE, 1, 2, 2)]
        plain = join_addresses(mentions, doc.pages, _POLICY["client"], band_fraction=0.0)
        paired = join_addresses(
            mentions, doc.pages, _POLICY["client"], {place.mention_id: street.mention_id}, band_fraction=0.0
        )
        assert paired.mentions == plain.mentions and paired.units == plain.units
        assert not any(u.forced for u in plain.units.values())

    def test_a_pair_merges_the_units_it_touches(self) -> None:
        doc = Doc(column("12 Smith Street, Penrith NSW 2750", "", "Dubbo NSW 2830"))  # a paragraph gap
        street, place = doc.span(E.STREET_ADDRESS, 0, 0, 2), doc.span(E.LOCALITY, 1, 0, 0)
        assert len(join_addresses([street, place], doc.pages, _POLICY["client"], band_fraction=0.0).units) == 2
        merged = join_addresses(
            [street, place], doc.pages, _POLICY["client"], {place.mention_id: street.mention_id}, band_fraction=0.0
        )
        (unit,) = merged.units.values()
        assert doc.text_of(unit.token_ids) == "12 Smith Street, Penrith NSW 2750 Dubbo NSW 2830"
        assert len(unit.mention_ids) == 1

    def test_an_unknown_id_is_ignored(self) -> None:
        doc = Doc(column("Zarnok NSW 2750"))
        place = doc.span(E.LOCALITY, 0, 0, 2)
        result = join_addresses([place], doc.pages, _POLICY["client"], {"nope": None}, band_fraction=0.0)
        assert not result.units and result.mentions == (place,)
        result = join_addresses([place], doc.pages, _POLICY["client"], {place.mention_id: "nope"}, band_fraction=0.0)
        assert len(result.units) == 1  # an unknown street id is no street: the place stands alone

    def test_the_repeat_sweep_can_be_switched_off(self) -> None:
        """A suburb alone in a footer band repeats an address: the sweep makes it a unit, unless it is off."""
        grid = [GridLine("Address: 42 Wattle Grove Road, Penrith NSW 2750", 0, 0), GridLine("Penrith", 1, 52)]
        doc = Doc(grid)
        place = doc.span(E.LOCALITY, 1, 0, 0)
        on = join_addresses([place], doc.pages, _POLICY["client"])
        off = join_addresses([place], doc.pages, _POLICY["client"], repeat_sweep=False)
        assert [u.kind for u in on.units.values()] == [UnitKind.STREET, UnitKind.PLACE_ONLY]
        assert on.units[next(reversed(on.units))].repeat_of == next(iter(on.units))
        assert [m.entity_type for m in on.mentions] == [E.STREET_ADDRESS, E.STREET_ADDRESS]
        assert [u.kind for u in off.units.values()] == [UnitKind.STREET]
        assert [m.entity_type for m in off.mentions] == [E.STREET_ADDRESS, E.LOCALITY]


# ------------------------------------------------------------------------------------------------ targets and exempt
def targets_of(join: AddressJoin) -> list[tuple[set[tuple[str, ...]], set[str]]]:
    return [(set(t.compound), set(t.whole)) for t in join.targets.values()]


class TestTargets:
    def test_a_one_line_address(self) -> None:
        _doc, result = run_join(["12 Smith Street, Parramatta NSW 2150"])
        ((compound, whole),) = targets_of(result)
        assert whole == {"smith", "parramatta"}
        assert {
            ("12", "smith", "street", "parramatta", "nsw", "2150"),  # the unit
            ("12", "smith", "street"),  # the street line
            ("12", "smith"),  # number and street name
            ("parramatta", "nsw", "2150"),  # suburb, state, postcode
        } <= compound
        # never a bare number, a street type, a state or a lone postcode
        assert all(len(c) >= 2 for c in compound)
        assert not whole & {"12", "street", "nsw", "2150"}

    def test_a_block_has_one_target_per_line_that_holds_an_identifying_word(self) -> None:
        _doc, result = run_join(["Level 4", "Westpoint Tower", "45 Smith Street", "Penrith NSW 2750"])
        ((compound, whole),) = targets_of(result)
        assert whole == {"westpoint", "tower", "smith", "penrith"}
        assert ("westpoint", "tower") in compound
        assert ("45", "smith", "street") in compound
        assert ("penrith", "nsw", "2750") in compound
        assert ("level", "4") not in compound  # designators and numbers only
        assert ("level", "4", "westpoint", "tower", "45", "smith", "street", "penrith", "nsw", "2750") in compound

    def test_box_words_and_numbers(self) -> None:
        _doc, result = run_join(["PO Box 77 Wollongong NSW 2500"])
        ((compound, whole),) = targets_of(result)
        assert whole == {"wollongong"}
        assert ("po", "box", "77") in compound

    def test_connecting_words_are_not_whole_token_targets(self) -> None:
        _doc, result = run_join(["Lot 3 Old Bells Line of Road, Kurrajong NSW 2758"])
        ((compound, whole),) = targets_of(result)
        assert whole == {"old", "bells", "line", "kurrajong"}  # 'of' is a connecting word, 'Road' a street type
        assert ("lot", "3", "old", "bells", "line", "of", "road") in compound

    def test_a_form_unit_has_label_and_value_targets(self) -> None:
        _doc, result, _ = join_case(_BY_ID["form-boxes-on-own-lines"], "client", "bare")
        ((compound, whole),) = targets_of(result)
        assert ("street", "name", "smith", "street") in compound
        assert ("suburb/town", "penrith") in compound
        assert ("postcode", "2750") in compound
        assert not any(c[0] == "state" for c in compound)  # the State box holds a structure word only
        assert whole == {"smith", "penrith"}

    def test_a_form_row_without_box_tags_still_has_the_parts(self) -> None:
        _doc, result, _ = join_case(_BY_ID["form-one-row-colonless"], "client", "bare")
        ((compound, whole),) = targets_of(result)
        assert ("address", "12", "smith", "street") in compound
        assert ("suburb/town", "penrith") in compound and ("postcode", "2750") in compound
        assert whole == {"smith", "penrith"}

    def test_a_kept_place_next_to_an_address_in_the_same_suburb(self) -> None:
        """The address's suburb is a target and the same word kept in prose is exempt: it is no leak."""
        doc, result = run_join(["Address: 12 Smith Street, Penrith NSW 2750", "She grew up in Penrith before moving."])
        (targets,) = result.targets.values()
        assert "penrith" in targets.whole
        prose = doc.tid((1, 4))
        assert doc.texts[prose] == "Penrith" and prose in result.exempt.kept_places
        # a correct output: the address replaced, the prose place unchanged
        out = simulated_output(doc, result, "client")
        assert target_hits(doc, result, out, set(result.exempt.all)) == []
        # a left-over original place line is a hit
        out2 = dict(out)
        for t in doc.tids([(0, 4), (0, 5), (0, 6)]):
            out2[t] = doc.texts[t]
        assert target_hits(doc, result, out2, set(result.exempt.all))

    def test_a_foreign_owned_street_name_is_not_a_whole_token_target(self) -> None:
        _doc, result = run_join(["12 Smith Street, Penrith NSW 2750"], [(E.PERSON, 0, 1, 1)])
        ((compound, whole),) = targets_of(result)
        assert whole == {"penrith"}
        assert ("12", "smith", "street", "penrith", "nsw", "2750") in compound

    def test_fold_word(self) -> None:
        assert fold_word("Penrith,") == "penrith"
        assert fold_word("(O’Brien)") == "o'brien"
        assert fold_word("|") == ""
        assert fold_words(["12", "|", "Smith", "St."]) == ("12", "smith", "st")

    def test_targets_hold_no_text_in_their_repr(self) -> None:
        _doc, result = run_join(["12 Smith Street, Parramatta NSW 2150"])
        for obj in (result, *result.units.values(), *result.targets.values(), *result.layouts.values(), result.exempt):
            assert "Smith" not in repr(obj) and "smith" not in repr(obj)


class TestExemptPositions:
    def test_kept_places_and_kept_mentions(self) -> None:
        texts = ["She grew up in Penrith. Seen on 12 June 2024 at 12 Smith Street, Penrith NSW 2750."]
        doc, result = run_join(texts, [(E.LOCALITY, 0, 4, 4), (E.DATE, 0, 7, 9)])
        assert {doc.texts[t] for t in result.exempt.kept_places} == {"Penrith."}
        assert {doc.texts[t] for t in result.exempt.kept_mentions} == {"Penrith.", "12", "June", "2024"}
        assert result.exempt.all == result.exempt.kept_places | result.exempt.kept_mentions

    def test_what_a_profile_replaces_is_not_exempt(self) -> None:
        texts = ["She grew up in Penrith. Seen on 12 June 2024 at 12 Smith Street, Penrith NSW 2750."]
        specs = [(E.LOCALITY, 0, 4, 4), (E.DATE, 0, 7, 9)]
        _, broad = run_join(texts, specs, "broad")
        assert broad.exempt.kept_places == frozenset()  # broad replaces a locality on its own
        _, maximal = run_join(texts, specs, "maximal")
        assert maximal.exempt.kept_mentions == frozenset()  # maximal shifts dates and replaces localities

    def test_a_header_or_footer_place_is_not_a_kept_place(self) -> None:
        """A place in a header or footer band counts as an address position (a repeat), never as a place on its own."""
        text = "She works in Penrith and Katoomba every week."
        body, footer = Doc([GridLine(text, 0, 3)]), Doc([GridLine(text, 0, 52)])
        assert [
            doc.text_of(k.token_ids) for doc in (body,) for k in join_addresses([], doc.pages, _POLICY["client"]).kept
        ] == [
            "Penrith",
            "Katoomba",
        ]
        assert join_addresses([], footer.pages, _POLICY["client"]).kept == ()
        assert join_addresses([], footer.pages, _POLICY["client"], band_fraction=0.0).kept != ()


class TestPositionRules:
    def test_band_tags_override_the_position_rule(self) -> None:
        text = "She works in Penrith and Katoomba every week."
        footer, body = Doc([GridLine(text, 0, 52)]), Doc([GridLine(text, 0, 3)])
        line = footer.lines[0].line_id
        tagged_body = join_addresses([], footer.pages, _POLICY["client"], bands={line: Band.BODY})
        assert len(tagged_body.kept) == 2
        tagged_letterhead = join_addresses(
            [], body.pages, _POLICY["client"], bands={body.lines[0].line_id: Band.LETTERHEAD}
        )
        assert tagged_letterhead.kept == ()

    def test_sideways_text_does_not_join_upright_text(self) -> None:
        doc = Doc(column("45 Smith Street", "Penrith NSW 2750"))
        joined = join_addresses([], doc.pages, _POLICY["client"], band_fraction=0.0)
        assert len(joined.units) == 1
        page = doc.pages[0]
        a, b = page.lines
        tilted = Line(
            b.line_id,
            b.page,
            b.line_no,
            b.bbox,
            ((0.0, 0.0), (0.0, 100.0), (50.0, 100.0), (50.0, 0.0)),
            1.0,
            "t",
            b.tokens,
        )
        page2 = PageAnalysis(page.geometry, page.kind, (a, tilted))
        apart = join_addresses([], [page2], _POLICY["client"], band_fraction=0.0)
        assert not any(len(u.mention_ids) == 1 and len(u.roles) > 5 for u in apart.units.values())


# ------------------------------------------------------------------------------------------------ determinism
class TestDeterminism:
    def test_the_order_of_the_input_mentions_does_not_matter(self) -> None:
        case = _BY_ID["letterhead-and-footer-repeats"]
        doc, bands, boxes = case_doc(case)
        base = detected_parts(doc, case) + detected_places(doc, case) + neighbours(doc, case)
        first = join_addresses(base, doc.pages, _POLICY["client"], band_fraction=0.0, bands=bands, boxes=boxes)
        for seed in range(5):
            shuffled = list(base)
            random.Random(seed).shuffle(shuffled)
            again = join_addresses(shuffled, doc.pages, _POLICY["client"], band_fraction=0.0, bands=bands, boxes=boxes)
            assert again.mentions == first.mentions and again.units == first.units and again.layouts == first.layouts

    def test_joining_the_output_again_changes_nothing(self) -> None:
        for case in _CASES:
            doc, bands, boxes = case_doc(case)
            first = join_addresses(
                detected_parts(doc, case) + detected_places(doc, case) + neighbours(doc, case),
                doc.pages,
                _POLICY["client"],
                band_fraction=0.0,
                bands=bands,
                boxes=boxes,
            )
            again = join_addresses(
                first.mentions, doc.pages, _POLICY["client"], band_fraction=0.0, bands=bands, boxes=boxes
            )
            assert again.mentions == first.mentions, case["id"]
            assert again.units == first.units and again.targets == first.targets, case["id"]

    def test_output_is_sorted_by_first_token(self) -> None:
        for case in _CASES:
            doc, result, _ = join_case(case, "client", "neighbours")
            order = {t: i for i, t in enumerate(doc.order)}
            keys = [(order[m.token_ids[0]], order[m.token_ids[-1]]) for m in result.mentions]
            assert keys == sorted(keys), case["id"]
            assert list(result.units) == sorted(
                result.units, key=lambda u: order[min(result.units[u].roles, key=order.get)]
            )

    def test_ids_come_from_the_tokens_and_are_stable(self) -> None:
        case = _BY_ID["one-line-labelled"]
        a = join_case(case, "client", "parts")[1]
        b = join_case(case, "client", "bare")[1]
        assert set(a.units) == set(b.units) and a.layouts.keys() == b.layouts.keys()
        assert all(uid.startswith("AU-") for uid in a.units)
        assert all(mid.startswith("M-") for mid in a.layouts)
