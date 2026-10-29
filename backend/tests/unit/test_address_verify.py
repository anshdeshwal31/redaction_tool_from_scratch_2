"""Verification of whole addresses (FINAL_PLAN TOOL2 section 6, as amended by D and H): V2 (the output's text layer),
V4 (the re-read output) and V5 (the fakes), at the level of the verification functions, with the case table's README
item 4 run through them for every core and stretch case.  No models."""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_address_join import Doc, join_case
from test_surrogates import assign_fakes

from mlredact.config.loader import load_config
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, PageAnalysis, RenderOp, Token, token_id
from mlredact.core.types import ActionKind, EntityType
from mlredact.pdfout.builder import build_text_pdf
from mlredact.pdfout.textlayer import LayerMask, page_text_layer, text_layer_items
from mlredact.policy.addresses import AddressJoin, JoinedUnit, fold_word
from mlredact.text.address import Role
from mlredact.verify.addresses import (
    AddressFakes,
    AddressReread,
    carry_punctuation,
    check_address_fakes,
    reread_of,
    swapped,
    words_on_their_own,
)
from mlredact.verify.checks import check_text_leak
from mlredact.verify.needles import AddressNeedles, AddressText, count_sequences, excess, needles_of, tokens_of
from mlredact.verify.residual import ResidualHit, _explained, find_residuals, residual_ops
from mlredact.verify.surrogates import check_surrogates
from support import column

_TABLE = json.loads((Path(__file__).resolve().parents[1] / "data" / "address_cases.json").read_text("utf-8"))
_CASES: list[dict[str, Any]] = _TABLE["cases"]
_CLIENT = load_config("client")
LOCALITY_ROLES = frozenset({Role.SUBURB, Role.POSTCODE, Role.STATE_AND_POSTCODE})
STREET_NAME_ROLES = frozenset({Role.STREET_NAME})


# ------------------------------------------------------------------------------------------------ tokens and needles
class TestTokens:
    def test_digits_are_maximal_runs_and_never_joined(self) -> None:
        assert tokens_of("14/03/2021") == ["14", "03", "2021"]
        assert tokens_of("$12,750.00") == ["12", "750", "00"]
        assert tokens_of("Pain level 4 of 10") == ["pain", "level", "4", "of", "10"]
        assert tokens_of("NSW2750") == ["nsw", "2750"]
        assert tokens_of("O’Brien-Smith") == ["o", "brien", "smith"]

    def test_the_mask_is_its_own_token(self) -> None:
        assert tokens_of("Penrith ~ 2750") == ["penrith", "~", "2750"]

    def test_needles_are_whole_tokens_without_bare_numbers(self) -> None:
        doc = Doc(
            column(
                "Residential address: 40 Bridge Street",
                "Suburb/Town Paddington",
                "State NSW",
                "Postcode 2021",
            )
        )
        join = _join(doc)
        needles = needles_of(join.targets.values())
        assert ("paddington",) in needles.whole and ("bridge",) in needles.whole
        assert ("suburb", "town", "paddington") in needles.compound
        assert ("state", "nsw") not in needles.compound  # the State box's value stays: 'State NSW' is no target
        assert ("2021",) not in needles.sequences and ("40",) not in needles.sequences
        assert ("street",) not in needles.sequences

    def test_short_whole_words_are_not_searched_alone(self) -> None:
        class Targets:
            compound = (("mt", "druitt", "nsw", "2770"),)
            whole = ("mt", "druitt", "ab")

        needles = needles_of([Targets()])
        assert needles.whole == (("druitt",),)
        assert ("mt", "druitt", "nsw", "2770") in needles.compound

    def test_counts_never_cross_two_texts(self) -> None:
        needles = AddressNeedles(compound=(("penrith", "nsw", "2750"),), whole=(("penrith",),))
        counts = count_sequences(needles, ["Penrith NSW 2750", "Penrith", "NSW 2750", "She grew up in Penrith."])
        assert counts[("penrith",)] == 3 and counts[("penrith", "nsw", "2750")] == 1
        assert excess(needles, counts, {("penrith",): 3, ("penrith", "nsw", "2750"): 1}) == 0
        assert excess(needles, counts, {("penrith",): 2, ("penrith", "nsw", "2750"): 1}) == 1


class TestWordsOnTheirOwn:
    """Ordinary words equal to an address word are exempt positions (no V2 hold, no V4 re-plan); a word that stands
    where an address word stands is not."""

    DOC = column(
        "Address: 12 May Street, Rose Bay NSW 2029",
        "",
        "She may return to work in June.",
        "Bed 4 in Bay 2 of the ward.",
        "She was seen at Rose Bay's clinic.",
        "Then 14 May Avenue was closed.",
        "Rose and Bay 2029 are two words.",
        "The rose garden.",
    )

    def exempt(self, heights: Mapping[int, int]) -> list[str]:
        doc = Doc(self.DOC)
        join = _join(doc)
        lines = [ln for p in doc.pages for ln in p.lines]
        ids = words_on_their_own(join.units, join.targets, lines, heights, frozenset())
        return [tok.text for ln in lines for tok in ln.tokens if tok.token_id in ids]

    def test_ordinary_words_are_exempt_and_words_in_an_address_position_are_not(self) -> None:
        # Not exempt: 'May' after a number or before a street type, 'Bay' before a postcode, "Bay's" after 'Rose':
        # positions of an address left behind.  Exempt: the verb, a ward's bay, 'Rose' before 'and', the flower.
        assert self.exempt({}) == ["may", "Bay", "Rose", "rose"]

    def test_a_letterhead_or_footer_band_is_an_address_position(self) -> None:
        assert self.exempt({0: 12000}) == []  # every line lies in the top 6% of so tall a page


def _join(doc: Doc) -> AddressJoin:
    from mlredact.policy.addresses import join_addresses

    return join_addresses([], doc.pages, _CLIENT.policy)


# ------------------------------------------------------------------------------------------------ V2 on text layers
def _pdf(items: Sequence[tuple[BBox, str]]) -> bytes:
    from mlredact.core.model import PageGeometry

    geometry = PageGeometry(0, 595.276, 841.89, 300, 2481, 3508)
    return build_text_pdf([(595.276, 841.89, page_text_layer(geometry, items))])


def _item(row: int, x: int, text: str) -> tuple[BBox, str]:
    y = 200 + row * 70
    return BBox(x, y, x + 20 * len(text), y + 50), text


class TestTextLeak:
    """V2 for address units: every target occurrence must be accounted for by an exempt position."""

    NEEDLES = AddressNeedles(
        compound=(("paddington", "nsw", "2021"), ("suburb", "town", "paddington")), whole=(("paddington",),)
    )

    def check(self, shown: Sequence[tuple[BBox, str]], exempt: Sequence[tuple[BBox, str]]) -> bool:
        return check_text_leak(_pdf(shown), [], addresses=AddressText(self.NEEDLES, _pdf(exempt))).passed

    def test_kept_dates_amounts_and_a_kept_place_pass(self) -> None:
        shown = [
            _item(0, 100, "14/03/2021"),
            _item(1, 100, "$12,750.00"),
            _item(2, 100, "Pain level 4 of 10"),
            _item(3, 100, "She grew up in Paddington"),
            _item(4, 100, "Postcode 2021"),
        ]
        assert self.check(shown, shown)  # every word exempt
        masked = [shown[0], shown[1], shown[2], _item(3, 100, "She grew up in Paddington"), _item(4, 100, "~ ~")]
        assert self.check(shown, masked)

    def test_a_left_over_original_place_line_fails(self) -> None:
        shown = [_item(0, 100, "Paddington NSW 2021"), _item(1, 100, "She grew up in Paddington")]
        exempt = [_item(0, 100, "~ ~ ~"), _item(1, 100, "She grew up in Paddington")]
        assert not self.check(shown, exempt)

    def test_the_same_word_twice_needs_two_exempt_positions(self) -> None:
        shown = [_item(0, 100, "Paddington"), _item(1, 100, "in Paddington")]
        assert self.check(shown, shown)
        assert not self.check(shown, [_item(0, 100, "~"), _item(1, 100, "in Paddington")])

    def test_no_exemption_means_every_occurrence_is_a_hit(self) -> None:
        shown = [_item(0, 100, "Paddington")]
        assert not check_text_leak(_pdf(shown), [], addresses=AddressText(self.NEEDLES, None)).passed

    def test_a_number_never_joins_other_digits(self) -> None:
        needles = AddressNeedles(compound=(("flat", "2021"),), whole=())
        shown = [_item(0, 100, "14/03/2021"), _item(1, 100, "ref 20 21")]
        assert check_text_leak(_pdf(shown), [], addresses=AddressText(needles, _pdf([_item(0, 100, "~")]))).passed

    def test_metrics_report_the_address_needles(self) -> None:
        shown = [_item(0, 100, "Paddington")]
        result = check_text_leak(_pdf(shown), ["Zzzzzz"], addresses=AddressText(self.NEEDLES, _pdf([])))
        assert result.metrics["address_needles"] == 3 and result.metrics["address_hits"] == 1
        assert result.metrics["needles"] == 1 and result.metrics["hits"] == 0


class TestLayerMask:
    def tokens(self) -> list[Token]:
        return [
            Token(token_id(0, 0, i), 0, 0, i, BBox(100 + 120 * i, 100, 200 + 120 * i, 150), 0.9, 0.9, "t", w)
            for i, w in enumerate(["Dr", "Smith", "Penrith"])
        ]

    def test_without_a_mask_the_items_are_unchanged(self) -> None:
        toks = self.tokens()
        assert text_layer_items(toks, []) == text_layer_items(toks, [], None)
        assert [t for _, t in text_layer_items(toks, [])] == ["Dr", "Smith", "Penrith"]

    def test_a_word_no_exemption_covers_is_masked(self) -> None:
        toks = self.tokens()
        mask = LayerMask({toks[0].token_id}, {}, {})
        assert [t for _, t in text_layer_items(toks, [], mask)] == ["Dr", "~", "~"]

    def test_drawn_fakes_are_exempt_and_a_retypeset_line_is_masked_word_by_word(self) -> None:
        toks = self.tokens()
        ink = toks[0].bbox.union(toks[-1].bbox)
        op = RenderOp(
            0,
            ActionKind.RETYPESET_REGION,
            ink.pad(5, 5, BBox(0, 0, 3000, 3000)),
            "retypeset",
            "L-1",
            "Dr Jones Penrith",
            ink,
        )
        line = Line("p0.l0", 0, 0, ink, ((0.0, 0.0),) * 4, 0.9, "t", tuple(toks))  # type: ignore[arg-type]
        mask = LayerMask({toks[0].token_id}, {toks[1].token_id: "Jones"}, {(0, ink): line.tokens})
        assert [t for _, t in text_layer_items(toks, [op], mask)] == ["Dr Jones ~"]
        assert [t for _, t in text_layer_items(toks, [op])] == ["Dr Jones Penrith"]
        lonely = LayerMask(set(), {}, {})
        assert [t for _, t in text_layer_items(toks, [op], lonely)] == ["~"]


# ------------------------------------------------------------------------------------------------ the case table
def _simulate(
    case: Mapping[str, Any], keep_roles: frozenset[Role] = frozenset()
) -> tuple[Doc, AddressJoin, dict[str, str], dict[str, str]]:
    """A tool's output for a case: the real faker's fake for every unit word outside ``keep_roles`` (a leaking tool
    keeps those words), and a replacement for what the profile replaces among the neighbours.  Returns the doc, its
    join, the text written for every replaced token (the fakes, '' erasing a token) and the fakes alone."""
    doc, result, _ = join_case(case, "client", "neighbours")
    fakes = assign_fakes(doc, result)
    unit_roles = {t: role for u in result.units.values() for t, role in u.roles.items()}
    out = {t: text for t, text in fakes.items() if unit_roles.get(t) not in keep_roles}
    own = {t for u in result.units.values() for t in u.roles}
    n = 0
    for m in result.mentions:
        if (
            m.entity_type is EntityType.STREET_ADDRESS
            or _CLIENT.policy.entity_actions[m.entity_type] is ActionKind.KEEP
        ):
            continue
        for t in m.token_ids:
            if t not in own:
                n += 1
                original = doc.texts[t]
                out[t] = f"Other{n}x" + original[len(original.rstrip(",.;:")) :]  # a replacement keeps the punctuation
    return doc, result, out, fakes


def _layers(doc: Doc, out: Mapping[str, str], readable: frozenset[str]) -> tuple[bytes, bytes]:
    """The output's text layer as the builder writes it, and its exempt rendition (replaced tokens are drawn, so exempt;
    a kept token is exempt when it is in ``readable``)."""
    real: list[tuple[float, float, bytes | None]] = []
    masked: list[tuple[float, float, bytes | None]] = []
    mask = LayerMask(readable, {}, {})
    for page in doc.pages:
        g = page.geometry
        tokens = list(page.tokens())
        ops = [
            RenderOp(
                g.index, ActionKind.SURROGATE, t.bbox, "street_address", t.token_id, out[t.token_id] or None, t.bbox
            )
            for t in tokens
            if t.token_id in out
        ]
        real.append((g.width_pt, g.height_pt, page_text_layer(g, text_layer_items(tokens, ops))))
        masked.append((g.width_pt, g.height_pt, page_text_layer(g, text_layer_items(tokens, ops, mask))))
    return build_text_pdf(real), build_text_pdf(masked)


def _v2(case: Mapping[str, Any], keep_roles: frozenset[Role] = frozenset()) -> bool:
    doc, result, out, _ = _simulate(case, keep_roles)
    real, exempt = _layers(doc, out, frozenset(result.exempt.all))
    needles = needles_of(result.targets[u] for u in sorted(result.targets))
    return check_text_leak(real, [], addresses=AddressText(needles, exempt)).passed


def _affected(case: Mapping[str, Any], keep_roles: frozenset[Role]) -> bool:
    _, result, _ = join_case(case, "client", "neighbours")
    return any(role in keep_roles and t not in u.excluded for u in result.units.values() for t, role in u.roles.items())


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_v2_a_correct_output_passes_and_a_leaking_one_fails(case: dict[str, Any]) -> None:
    """README item 4 through the real V2: the simulated correct output (unit words replaced, kept places, labels,
    look-alikes and neighbours as their own rules leave them) has no hit; one that keeps the locality, and one that
    keeps the street name, are caught wherever they leak."""
    assert _v2(case), "a correct output was held"
    for keep in (LOCALITY_ROLES, STREET_NAME_ROLES):
        if _affected(case, keep):
            assert not _v2(case, keep), f"a leak of {sorted(r.value for r in keep)} was not caught"


# ------------------------------------------------------------------------------------------------ V4 on re-read output
def _reread_pages(doc: Doc, out: Mapping[str, str]) -> tuple[list[PageAnalysis], dict[int, list[RenderOp]]]:
    """The output as OCR would read it back: every line's words (a fake of several words becomes several tokens), laid
    over the line's box; and the planned operations over the replaced tokens' boxes."""
    pages: list[PageAnalysis] = []
    planned: dict[int, list[RenderOp]] = {}
    for page in doc.pages:
        lines: list[Line] = []
        for line in page.lines:
            words: list[tuple[str, bool, BBox]] = []
            for t in line.tokens:
                if t.token_id in out:
                    parts = out[t.token_id].split()
                    width = max(1, (t.bbox.x1 - t.bbox.x0) // max(1, len(parts)))
                    for i, w in enumerate(parts):
                        box = BBox(t.bbox.x0 + i * width, t.bbox.y0, t.bbox.x0 + (i + 1) * width, t.bbox.y1)
                        words.append((w, True, box))
                    planned.setdefault(page.geometry.index, []).append(
                        RenderOp(page.geometry.index, ActionKind.BLACKOUT, t.bbox, "x", f"R-{t.token_id}")
                    )
                else:
                    words.append((t.text, False, t.bbox))
            tokens = tuple(
                Token(token_id(line.page, line.line_no, i), line.page, line.line_no, i, box, 0.9, 0.9, "t", w)
                for i, (w, _replaced, box) in enumerate(words)
            )
            lines.append(dataclasses.replace(line, tokens=tokens))
        pages.append(dataclasses.replace(page, lines=tuple(lines)))
    return pages, planned


def _v4(case: Mapping[str, Any], keep_roles: frozenset[Role] = frozenset()) -> list[ResidualHit]:
    """The residuals the address checks add: V4's hits with the units' data, less the hits the rule detectors' own
    re-detection gives without it (the guards of step 5 end those: a look-alike read as an address)."""
    doc, result, out, fakes = _simulate(case, keep_roles)
    pages, planned = _reread_pages(doc, out)
    originals = {t: tok.text for line in doc.lines for tok in line.tokens for t in [tok.token_id]}
    unit_fakes = {uid: {t: fakes[t] for t in u.roles if t in fakes} for uid, u in result.units.items()}
    boxes: dict[int, list[BBox]] = {}
    for line in doc.lines:
        for tok in line.tokens:
            if tok.token_id in result.exempt.all:
                boxes.setdefault(tok.page, []).append(tok.bbox)
    words = frozenset(w for text in out.values() for w in map(fold_word, text.split()) if w)
    in_units = {t for u in result.units.values() for t in u.roles}
    outside = [text for t, text in originals.items() if t not in in_units]
    reread = reread_of(result.units, result.targets, originals, unit_fakes, frozenset(fakes), words, boxes, outside)
    replacements = [text for text in out.values() if text]
    base = set(find_residuals(pages, [], planned, _CLIENT, replacements, None))
    return [
        h for h in find_residuals(pages, [], planned, _CLIENT, replacements, None, addresses=reread) if h not in base
    ]


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_v4_a_correct_output_is_clean_and_a_leaking_one_is_residual(case: dict[str, Any]) -> None:
    """The same for V4: the probe and the join on the re-read output find nothing in a correct output, and find
    what a tool that keeps the locality, or the street name, leaves."""
    assert _v4(case) == [], "a correct output has a residual"
    for keep in (LOCALITY_ROLES, STREET_NAME_ROLES):
        if _affected(case, keep):
            assert _v4(case, keep), f"a leak of {sorted(r.value for r in keep)} is not a residual"


# ------------------------------------------------------------------------------------------------ V4 pieces
class TestExplained:
    def test_a_mention_within_one_fake_is_explained_but_one_that_contains_a_fake_is_not(self) -> None:
        fake = ["91oceanlanekempseynsw2440"]
        street = EntityType.STREET_ADDRESS
        assert _explained("91 Ocean Lane, Kempsey NSW 2440", fake, street)
        assert _explained("Ocean Lane", fake, street)
        assert not _explained("Address: 91 Ocean Lane, Penrith NSW 2750", fake, street)
        assert not _explained("Address: 91 Ocean Lane, Kempsey NSW 2440", fake, street)
        assert not _explained("Ms", fake, street)

    def test_a_mention_of_another_type_that_contains_a_fake_is_explained(self) -> None:
        """OCR glues a fake phone to the next label and re-reads the run as one phone (syn-mrc in oracle_regions)."""
        fake = ["0491570760"]
        assert _explained("0491 570 760Email:", fake, EntityType.PHONE)
        assert not _explained("0491 570 760Email:", fake, EntityType.STREET_ADDRESS)
        assert not _explained("0412 345 678", fake, EntityType.PHONE)


class TestSwaps:
    def reread(self) -> AddressReread:
        return AddressReread(
            structure=frozenset({"nsw", "po", "box"}),
            swaps={"penrith": "Kempsey", "2750": "2440", "smith": "Ocean"},
        )

    def test_a_run_of_words_takes_the_fakes_of_its_unit(self) -> None:
        r = self.reread()
        assert swapped(["Penrith", "NSW", "2750"], r) == "Kempsey NSW 2440"
        assert swapped(["PENRITH,"], r) == "KEMPSEY,"
        assert swapped(["penrith"], r) == "kempsey"

    def test_a_word_without_a_fake_blacks_the_run_out(self) -> None:
        r = self.reread()
        assert swapped(["Penrith", "Nowhere"], r) is None
        assert swapped(["NSW"], r) is None  # nothing to swap

    def test_punctuation_and_case_are_carried(self) -> None:
        assert carry_punctuation("(Penrith),", "Kempsey") == "(Kempsey),"
        assert carry_punctuation("PENRITH", "Kempsey") == "KEMPSEY"


class TestResidualOps:
    def test_a_hit_with_a_fake_is_drawn_and_any_other_is_blacked_out(self) -> None:
        box = BBox(10, 10, 200, 60)
        hits = [
            ResidualHit(0, box, "street_address", "removed_value", "Kempsey NSW 2440", "Penrith NSW 2750"),
            ResidualHit(0, BBox(10, 100, 50, 150), "street_address", "redetect"),
        ]
        ops = residual_ops(hits, "doc", 1)[0]
        assert [o.kind for o in ops] == [ActionKind.SURROGATE, ActionKind.BLACKOUT]
        assert ops[0].text == "Kempsey NSW 2440" and ops[0].ink_box == box and ops[0].source_text == "Penrith NSW 2750"
        assert ops[1].text is None and all(o.ref_id.startswith("R-") for o in ops)

    def test_the_repr_holds_no_text(self) -> None:
        hit = ResidualHit(0, BBox(0, 0, 1, 1), "street_address", "redetect", "Kempsey", "Penrith")
        assert "Kempsey" not in repr(hit) and "Penrith" not in repr(hit)


# ------------------------------------------------------------------------------------------------ V5
@functools.cache
def _fakes_of(case_id: str) -> tuple[Doc, AddressJoin, dict[str, str]]:
    case = next(c for c in _CASES if c["id"] == case_id)
    doc, result, _ = join_case(case, "client", "parts")
    return doc, result, assign_fakes(doc, result)


def _v5(case_id: str, edit: Mapping[str, str] | None = None) -> dict[str, int]:
    doc, result, fakes = _fakes_of(case_id)
    tokens = {t: doc.texts[t] for u in result.units.values() for t in u.roles}
    unit_fakes = {
        uid: {t: (edit or {}).get(t, fakes[t]) for t in u.roles if t in fakes} for uid, u in result.units.items()
    }
    return check_address_fakes(AddressFakes(result.units, tokens, unit_fakes, frozenset(fakes)))


def _word(case_id: str, role: Role, n: int = 0) -> str:
    _, result, _ = _fakes_of(case_id)
    return [t for u in result.units.values() for t, r in u.roles.items() if r is role][n]


class TestFakeChecks:
    def test_the_real_fakes_pass(self) -> None:
        for cid in ("one-line-basic", "block-person", "unit-slash", "matter-d8-two-addresses-one-suburb"):
            assert not any(_v5(cid).values()), cid

    def test_an_original_word_in_a_fake_is_caught_in_any_role(self) -> None:
        doc, _, _ = _fakes_of("one-line-basic")
        name, suburb = _word("one-line-basic", Role.STREET_NAME), _word("one-line-basic", Role.SUBURB)
        bad = _v5("one-line-basic", {name: doc.texts[suburb]})  # the original suburb as a street name
        assert bad["address_word_collision"] >= 1
        assert _v5("one-line-basic", {name: doc.texts[name] + "s"})["address_word_collision"] >= 1  # contains it

    def test_an_original_number_is_caught_as_a_whole_digit_run(self) -> None:
        doc, _, _ = _fakes_of("one-line-basic")
        number = _word("one-line-basic", Role.NUMBER)
        assert _v5("one-line-basic", {number: doc.texts[number]})["address_number_collision"] >= 1

    def test_the_postcode_must_differ_and_lie_in_the_state(self) -> None:
        doc, _, _ = _fakes_of("one-line-basic")
        pc = _word("one-line-basic", Role.POSTCODE)
        assert _v5("one-line-basic", {pc: doc.texts[pc]})["address_postcode"] == 1  # the original postcode
        assert _v5("one-line-basic", {pc: "3000"})["address_postcode"] == 1  # a Victorian postcode beside NSW
        assert (
            _v5("one-line-basic", {pc: "2" + doc.texts[pc][1:3] + str((int(doc.texts[pc][3]) + 1) % 10)})[
                "address_postcode"
            ]
            == 1
        )  # a digit after the state's lead digit is shared

    def test_two_different_originals_never_share_a_whole_fake(self) -> None:
        cid = "matter-d8-two-addresses-one-suburb"
        _, result, fakes = _fakes_of(cid)
        a, b = list(result.units.values())[:2]
        assert len(a.roles) == len(b.roles) and not a.key.same_address(b.key)
        assert not any(_v5(cid).values())
        edit = {tb: fakes[ta] for ta, tb in zip(a.roles, b.roles, strict=True)}
        assert _v5(cid, edit)["address_shared_fake"] >= 1

    def test_check_surrogates_takes_the_address_fakes(self) -> None:
        doc, result, fakes = _fakes_of("one-line-basic")
        tokens = {t: doc.texts[t] for u in result.units.values() for t in u.roles}
        unit_fakes = {uid: {t: fakes[t] for t in u.roles if t in fakes} for uid, u in result.units.items()}
        good = AddressFakes(result.units, tokens, unit_fakes, frozenset(fakes))
        assert check_surrogates([], [], addresses=good).passed
        pc = _word("one-line-basic", Role.POSTCODE)
        broken = {u: {**f, pc: doc.texts[pc]} for u, f in unit_fakes.items()}
        result_bad = check_surrogates([], [], addresses=AddressFakes(result.units, tokens, broken, frozenset(fakes)))
        assert not result_bad.passed and result_bad.metrics["address_postcode"] == 1


def test_units_are_a_frozen_dataclass_without_text_in_the_repr() -> None:
    doc = Doc(column("Address: 42 Wattle Grove Road, Penrith NSW 2750"))
    join = _join(doc)
    unit: JoinedUnit = next(iter(join.units.values()))
    assert "Wattle" not in repr(unit) and "Penrith" not in repr(unit)
    reread = reread_of(join.units, join.targets, doc.texts, {}, frozenset(), frozenset(), {})
    assert "wattle" not in repr(reread)
