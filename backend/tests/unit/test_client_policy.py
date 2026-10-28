"""The client profile (the client's policy decisions): actions, whole addresses and places kept on their own."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from mlredact.config.loader import builtin_profiles, config_hash, load_config
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.model import Evidence, Mention, PageAnalysis, token_id
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, RegionKind
from mlredact.detect.engine import detect_document
from mlredact.policy.addresses import AddressJoin, UnitPart, join_addresses
from mlredact.policy.engine import decide, decide_injected, split_kept_places
from mlredact.text.address import BoxRole, Role, UnitKind
from support import GridLine, column, pages_from_grid, pages_from_lines

CLIENT = load_config("client")
PARTS = load_config("client", overrides={"policy": {"addresses": "parts"}}).policy
A = ActionKind
E = EntityType


class TestProfile:
    def test_is_the_default(self) -> None:
        assert load_config().policy.profile_id == "client"

    def test_addresses_are_replaced_whole_and_places_on_their_own_are_kept(self) -> None:
        """The owner's decision of 2026-10-22: a departure from Q4.  ``locality: keep`` and ``postcode: keep`` now cover
        places mentioned on their own (see TestWholeAddresses)."""
        assert CLIENT.policy.addresses == "whole"
        assert CLIENT.policy.address_state == "keep"
        assert CLIENT.policy.entity_actions[E.STREET_ADDRESS] is A.SURROGATE
        assert CLIENT.policy.entity_actions[E.LOCALITY] is A.KEEP
        assert CLIENT.policy.entity_actions[E.POSTCODE] is A.KEEP

    @pytest.mark.parametrize(
        ("entity", "action"),
        [
            # Q1, Q2: every person and organisation is replaced, professionals included.
            (E.PERSON, A.SURROGATE),
            (E.ORGANISATION, A.SURROGATE),
            (E.PHONE, A.SURROGATE),
            (E.EMAIL, A.SURROGATE),
            (E.URL, A.SURROGATE),
            # Q3: dates, dates of birth and ages are kept.
            (E.DATE_OF_BIRTH, A.KEEP),
            (E.DATE_OF_DEATH, A.KEEP),
            (E.DATE, A.KEEP),
            (E.AGE, A.KEEP),
            # Q4: suburbs, towns and places of birth (and their postcodes) on their own are kept; addresses go whole.
            (E.LOCALITY, A.KEEP),
            (E.POSTCODE, A.KEEP),
            (E.STREET_ADDRESS, A.SURROGATE),
            # Q5: identifiers, claim and court numbers, provider numbers, ABNs, file paths.
            (E.MEDICARE, A.SURROGATE),
            (E.TFN, A.SURROGATE),
            (E.BANK_ACCOUNT, A.SURROGATE),
            (E.REFERENCE, A.SURROGATE),
            (E.COURT_FILE, A.SURROGATE),
            (E.PROVIDER_NUMBER, A.SURROGATE),
            (E.ABN, A.SURROGATE),
            (E.FILE_PATH, A.SURROGATE),
        ],
    )
    def test_entity_actions_follow_the_policy(self, entity: EntityType, action: ActionKind) -> None:
        assert CLIENT.policy.entity_actions[entity] is action
        assert not CLIENT.policy.keep_professionals

    def test_region_actions_follow_the_policy(self) -> None:
        regions = CLIENT.policy.region_actions
        # Q6/Q7: a fake rather than a box where one can be drawn without reading the original.
        for kind in (RegionKind.SIGNATURE, RegionKind.FACE, RegionKind.PHOTO):
            assert regions[kind] is A.SYNTHESISE
        # Q16: losing clinical content fails a document, so figures and body charts stay.
        assert regions[RegionKind.GRAPHIC] is A.KEEP
        assert regions[RegionKind.HANDWRITING] is A.REMOVE


def _mention(pages: list, line_no: int, etype: EntityType) -> tuple[Mention, dict]:
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    ids = tuple(t.token_id for t in pages[0].lines[line_no].tokens)
    ev = (Evidence("test", "1", "test", EvidenceStrength.STRONG, 1.0),)
    value = " ".join(tokens[i].text for i in ids)
    return Mention("M-1", etype, ids, ev, value), tokens


class TestKeptPlaces:
    """``addresses: parts``: the street line is replaced and, with ``locality: keep``, the place stays (the client's
    literal answer to Q4)."""

    def test_street_line_is_split_from_the_kept_place(self) -> None:
        pages = pages_from_lines(["12 Smith Street, Parramatta NSW 2150"])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        parts = split_kept_places([m], tokens, PARTS)
        assert [(p.entity_type, p.value) for p in parts] == [
            (E.STREET_ADDRESS, "12 Smith Street,"),
            (E.LOCALITY, "Parramatta NSW 2150"),
        ]

    def test_po_box_and_place_only_addresses(self) -> None:
        pages = pages_from_lines(["PO Box 77 Wollongong NSW 2500", "Parramatta NSW 2150"])
        box, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        place, _ = _mention(pages, 1, E.STREET_ADDRESS)
        parts = split_kept_places([box, place], tokens, PARTS)
        assert [(p.entity_type, p.value) for p in parts] == [
            (E.STREET_ADDRESS, "PO Box 77"),
            (E.LOCALITY, "Wollongong NSW 2500"),
            (E.LOCALITY, "Parramatta NSW 2150"),
        ]

    @pytest.mark.parametrize(
        ("line", "street"),
        [
            ("12 Victoria Road, Gladesville NSW 2111", "12 Victoria Road,"),  # street named after a state
            ("8 Broadway, Ultimo NSW 2007", "8 Broadway,"),  # no street type
            ("3 Smith St, Castle Hill NSW 2154", "3 Smith St,"),  # suburb with a street-type word
            ("8 Broadway Ultimo NSW", "8 Broadway"),  # no comma: the grammar knows Ultimo is a suburb
            ("12 Smith Street, Parramatta New South Wales 2150", "12 Smith Street,"),  # a spelled-out state
            ("12 Smith St, Parramatta N.S.W. 2150", "12 Smith St,"),  # a dotted state
            ("Unit 5, 12 Smith St Parramatta NSW", "Unit 5, 12 Smith St"),  # comma inside the street line
        ],
    )
    def test_the_street_line_ends_at_the_right_place(self, line: str, street: str) -> None:
        pages = pages_from_lines([line])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        parts = split_kept_places([m], tokens, PARTS)
        assert (parts[0].entity_type, parts[0].value) == (E.STREET_ADDRESS, street)

    @pytest.mark.parametrize("line", ["Unit 5, 8 Broadway Ultimo", "12 Victoria Road, Ryde"])
    def test_an_unclear_street_line_is_replaced_whole(self, line: str) -> None:
        pages = pages_from_lines([line])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        assert split_kept_places([m], tokens, PARTS) == [m]

    def test_a_replaced_postcode_is_split_off(self) -> None:
        overrides = {"policy": {"addresses": "parts", "entity_actions": {"postcode": "surrogate"}}}
        policy = load_config("client", overrides=overrides).policy
        pages = pages_from_lines(["12 Smith Street Parramatta NSW 2150"])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        parts = split_kept_places([m], tokens, policy)
        assert [p.entity_type for p in parts] == [E.STREET_ADDRESS, E.LOCALITY, E.POSTCODE]
        assert parts[-1].value == "2150"

    def test_addresses_without_a_state_and_other_profiles_are_left_alone(self) -> None:
        pages = pages_from_lines(["12 Smith Street Parramatta", "12 Smith Street, Parramatta NSW 2150"])
        no_state, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        assert split_kept_places([no_state], tokens, PARTS) == [no_state]
        full, _ = _mention(pages, 1, E.STREET_ADDRESS)
        assert split_kept_places([full], tokens, load_config("broad").policy) == [full]


class TestAddressSettings:
    """``addresses: whole | parts`` and ``address_state: keep | replace`` (owner ruling 5: every profile is whole)."""

    def test_every_profile_replaces_whole_addresses_and_keeps_the_state(self) -> None:
        profiles = builtin_profiles()
        assert profiles == ["broad", "claimant_focused", "client", "maximal"]
        for name in profiles:
            policy = load_config(name).policy
            assert (policy.addresses, policy.address_state) == ("whole", "keep"), name
            assert policy.entity_actions[E.STREET_ADDRESS] is not A.KEEP, name

    def test_each_profile_keeps_its_own_rule_for_places_on_their_own(self) -> None:
        got = {n: load_config(n).policy.entity_actions for n in builtin_profiles()}
        assert (got["client"][E.LOCALITY], got["client"][E.POSTCODE]) == (A.KEEP, A.KEEP)
        for name in ("broad", "claimant_focused", "maximal"):
            assert (got[name][E.LOCALITY], got[name][E.POSTCODE]) == (A.SURROGATE, A.SURROGATE), name

    def test_whole_with_a_kept_street_line_is_rejected(self) -> None:
        keep = {"entity_actions": {"street_address": "keep"}}
        with pytest.raises(EnvironmentErrorMl) as caught:
            load_config("client", overrides={"policy": keep})
        assert caught.value.code is ReasonCode.CONFIG_INVALID
        with pytest.raises(EnvironmentErrorMl):
            load_config("client", overrides={"policy": {**keep, "addresses": "whole"}})
        # the literal Q4 answer may keep it
        assert load_config("client", overrides={"policy": {**keep, "addresses": "parts"}}).policy.addresses == "parts"

    @pytest.mark.parametrize(
        "override", [{"addresses": "street_line"}, {"address_state": "move"}, {"address_state": ""}]
    )
    def test_unknown_values_are_rejected(self, override: dict[str, str]) -> None:
        with pytest.raises(EnvironmentErrorMl):
            load_config("client", overrides={"policy": override})

    def test_the_settings_are_part_of_the_config_hash(self) -> None:
        base = config_hash(load_config("client"))
        assert config_hash(load_config("client", overrides={"policy": {"addresses": "parts"}})) != base
        assert config_hash(load_config("client", overrides={"policy": {"address_state": "replace"}})) != base
        assert config_hash(load_config("client", overrides={"policy": {"address_state": "keep"}})) == base


def _detect_and_join(
    pages: list[PageAnalysis],
    profile: str = "client",
    extra: Sequence[Mention] = (),
    boxes: Mapping[str, BoxRole] | None = None,
) -> tuple[AddressJoin, dict[str, str]]:
    """The mentions the rule detectors give (no models), plus ``extra`` (what a model would add), joined."""
    config = load_config(profile)
    detection = detect_document(pages, config, "doc")
    join = join_addresses([*detection.mentions, *extra], pages, config.policy, band_fraction=0.05, boxes=boxes)
    return join, {t.token_id: t.text for p in pages for t in p.tokens()}


def _model_mention(pages: list[PageAnalysis], etype: EntityType, line: int, first: int, last: int) -> Mention:
    """A mention a model would give, over the words ``first``..``last`` of the line-th line."""
    tokens = [ln for p in pages for ln in p.lines][line].tokens[first : last + 1]
    ev = (Evidence("test.ner", "1", "test", EvidenceStrength.WEAK, 0.5),)
    return Mention(
        f"M-model-{tokens[0].token_id}", etype, tuple(t.token_id for t in tokens), ev, " ".join(t.text for t in tokens)
    )


def _values(join: AddressJoin, *types: EntityType) -> list[str]:
    return [m.value for m in join.mentions if m.entity_type in types]


def _acted(join: AddressJoin, profile: str = "client") -> list[str]:
    """The values the profile replaces."""
    return [d.mention.value for d in decide(join.mentions, load_config(profile).policy) if d.action is not A.KEEP]


class TestWholeAddresses:
    """``addresses: whole``: every address is one unit replaced as a whole and places on their own stay, through the
    mentions the rule detectors give (no models) and the weak spans a model would add."""

    @pytest.mark.parametrize(
        ("texts", "unit"),
        [
            (["12 Smith Street, Parramatta NSW 2150"], "12 Smith Street, Parramatta NSW 2150"),  # one line
            (["Address: 42 Wattle Grove Road, Penrith NSW 2750"], "42 Wattle Grove Road, Penrith NSW 2750"),  # label
            (["Mr J Citizen", "45 Smith Street", "Penrith NSW 2750"], "45 Smith Street Penrith NSW 2750"),  # block
            (["45 Smith Street", "", "Penrith NSW 2750"], "45 Smith Street Penrith NSW 2750"),  # a paragraph gap
            (["Smith Street, Penrith NSW 2750"], "Smith Street, Penrith NSW 2750"),  # a numberless street
            (
                ["Suite 3 Level 2 Riverside Medical Chambers 45 Smith Street, Penrith NSW 2750"],
                "Suite 3 Level 2 Riverside Medical Chambers 45 Smith Street, Penrith NSW 2750",
            ),  # words before the number
            (["1st Floor, 45 Smith Street, Penrith NSW 2750"], "1st Floor, 45 Smith Street, Penrith NSW 2750"),
            (
                ["Westpoint Tower, 45 Smith Street, Penrith NSW 2750"],
                "Westpoint Tower, 45 Smith Street, Penrith NSW 2750",
            ),
            (
                ["Westpoint Tower", "45 Smith Street", "Penrith NSW 2750"],
                "Westpoint Tower 45 Smith Street Penrith NSW 2750",
            ),  # a building line above
            (["12 Smith", "Street, Penrith NSW 2750"], "12 Smith Street, Penrith NSW 2750"),  # an OCR-split line
            (["Cnr Smith & Jones Streets, Penrith NSW 2750"], "Cnr Smith & Jones Streets, Penrith NSW 2750"),
            (["PO Box 77 Wollongong NSW 2500"], "PO Box 77 Wollongong NSW 2500"),
        ],
    )
    def test_joined_into_one_unit(self, texts: list[str], unit: str) -> None:
        pages = pages_from_grid(column(*texts))
        join, _ = _detect_and_join(pages)
        assert _values(join, E.STREET_ADDRESS, E.LOCALITY, E.POSTCODE) == [unit]
        assert unit in _acted(join) and len(join.units) == 1
        (m,) = [m for m in join.mentions if m.entity_type is E.STREET_ADDRESS]
        assert join.layouts[m.mention_id].part is UnitPart.JOINED

    @pytest.mark.parametrize("profile", ["client", "broad", "claimant_focused", "maximal"])
    def test_every_profile_replaces_the_address_as_one_unit(self, profile: str) -> None:
        pages = pages_from_grid(column("Address: 42 Wattle Grove Road, Penrith NSW 2750"))
        join, _ = _detect_and_join(pages, profile)
        assert _acted(join, profile) == ["42 Wattle Grove Road, Penrith NSW 2750"]
        assert _values(join, E.LOCALITY, E.POSTCODE) == []

    def test_the_table_cell_below(self) -> None:
        pages = pages_from_grid(
            [
                GridLine("Address", row=0),
                GridLine("45 Smith Street", row=0, x0=200),
                GridLine("Penrith NSW 2750", row=1, x0=200),
            ]
        )
        join, _ = _detect_and_join(pages)
        assert _acted(join) == ["45 Smith Street Penrith NSW 2750"]

    def test_the_place_on_the_next_page(self) -> None:
        grid = [
            GridLine("Please reply to", row=0),
            GridLine("45 Smith Street", row=1),
            GridLine("PENRITH NSW 2750", page=1),
        ]
        join, _ = _detect_and_join(pages_from_grid(grid))
        assert _acted(join) == ["45 Smith Street PENRITH NSW 2750"]

    def test_a_weak_model_span_over_the_whole_address_is_retyped(self) -> None:
        """'8 Broadway, Ultimo NSW 2007' under a weak LOCALITY span was kept whole by the client profile before."""
        pages = pages_from_grid(column("8 Broadway, Ultimo NSW 2007"))
        join, _ = _detect_and_join(pages, extra=[_model_mention(pages, E.LOCALITY, 0, 0, 4)])
        assert [(m.entity_type, m.value) for m in join.mentions] == [(E.STREET_ADDRESS, "8 Broadway, Ultimo NSW 2007")]
        assert _acted(join) == ["8 Broadway, Ultimo NSW 2007"]

    def test_a_phone_after_the_place_stays_out(self) -> None:
        pages = pages_from_grid(column("Address: 42 Wattle Grove Road, Penrith NSW 2750 T (02) 5550 6610"))
        address, phone = _model_mention(pages, E.STREET_ADDRESS, 0, 1, 7), _model_mention(pages, E.PHONE, 0, 8, 11)
        config = load_config("client")
        join = join_addresses([address, phone], pages, config.policy, band_fraction=0.05)
        assert [(m.entity_type, m.value) for m in join.mentions] == [
            (E.STREET_ADDRESS, "42 Wattle Grove Road, Penrith NSW 2750"),
            (E.PHONE, "T (02) 5550 6610"),
        ]

    def test_a_suburb_only_tail(self) -> None:
        pages = pages_from_grid(column("She lives at 4 Rose Lane, Katoomba with her husband."))
        join, _ = _detect_and_join(pages, extra=[_model_mention(pages, E.LOCALITY, 0, 5, 5)])
        assert _acted(join) == ["4 Rose Lane, Katoomba"]
        assert _values(join, E.LOCALITY) == []

    def test_a_footer_repeat_found_only_by_the_sweep(self) -> None:
        pages = pages_from_grid(
            [GridLine("Address: 42 Wattle Grove Road, Penrith NSW 2750", row=0), GridLine("Penrith", page=1, row=52)]
        )
        join, _ = _detect_and_join(pages, extra=[_model_mention(pages, E.LOCALITY, 1, 0, 0)])
        assert _acted(join) == ["42 Wattle Grove Road, Penrith NSW 2750", "Penrith"]
        first, second = join.units.values()
        assert second.repeat_of == first.unit_id and second.kind is UnitKind.PLACE_ONLY
        assert second.key.suburb == first.key.suburb

    def test_the_claim_form_with_one_box_per_line(self) -> None:
        cells = [
            ("Address:", BoxRole.LABEL),
            ("Unit", BoxRole.LABEL),
            ("5", BoxRole.VALUE),
            ("Street No.", BoxRole.LABEL),
            ("12", BoxRole.VALUE),
            ("Street name", BoxRole.LABEL),
            ("Smith Street", BoxRole.VALUE),
            ("Suburb/Town", BoxRole.LABEL),
            ("Penrith", BoxRole.VALUE),
            ("State", BoxRole.LABEL),
            ("NSW", BoxRole.VALUE),
            ("Postcode", BoxRole.LABEL),
            ("2750", BoxRole.VALUE),
        ]
        pages = pages_from_grid([GridLine(t, row=i) for i, (t, _b) in enumerate(cells)])
        boxes = {ln.line_id: box for ln, (_t, box) in zip(pages[0].lines, cells, strict=True)}
        # (the label rule would take the 'Unit' box as the value of 'Address:'; that is the detectors' to fix)
        join = join_addresses([], pages, CLIENT.policy, band_fraction=0.05, boxes=boxes)
        (unit,) = join.units.values()
        assert unit.form and unit.kind is UnitKind.STREET
        assert _acted(join) == ["5", "12", "Smith Street", "Penrith", "NSW", "2750"]
        parts = [join.layouts[i] for i in unit.mention_ids]
        assert [p.part for p in parts] == [UnitPart.STREET] * 3 + [UnitPart.PLACE] * 3
        assert [p.roles[0] for p in parts] == [
            Role.UNIT_NUMBER,
            Role.NUMBER,
            Role.STREET_NAME,
            Role.SUBURB,
            Role.STATE,
            Role.POSTCODE,
        ]

    def test_the_claim_form_with_a_row_of_colonless_boxes(self) -> None:
        cells = [
            ("Address", 40, 85, BoxRole.LABEL),
            ("12 Smith Street", 90, 210, BoxRole.VALUE),
            ("Suburb/Town", 215, 280, BoxRole.LABEL),
            ("Penrith", 285, 360, BoxRole.VALUE),
            ("State", 365, 395, BoxRole.LABEL),
            ("NSW", 400, 430, BoxRole.VALUE),
            ("Postcode", 435, 490, BoxRole.LABEL),
            ("2750", 495, 530, BoxRole.VALUE),
        ]
        pages = pages_from_grid([GridLine(t, row=0, x0=x0, x1=x1) for t, x0, x1, _b in cells])
        boxes = {ln.line_id: c[3] for ln, c in zip(pages[0].lines, cells, strict=True)}
        join = join_addresses([], pages, CLIENT.policy, band_fraction=0.05, boxes=boxes)
        (unit,) = join.units.values()
        assert _acted(join) == ["12 Smith Street", "Penrith", "NSW", "2750"]
        assert len(unit.mention_ids) == 4

    # ------------------------------------------------------------------------------------------ kept
    @pytest.mark.parametrize(
        ("text", "kept"),
        [
            ("She moved to Penrith in 2019 and was injured at Parramatta station.", ["Penrith", "Parramatta"]),
            ("Residents of postcode 2750 were surveyed.", ["2750"]),
            ("She flew to Sydney, then Victoria.", ["Sydney,", "Victoria."]),
            ("Date and place of birth: 12 June 1980, Penrith NSW 2750", ["Penrith NSW 2750"]),
            ("Date of accident: 12 June 2023, Penrith NSW 2750", ["Penrith NSW 2750"]),
            ("She was born in Parramatta NSW 2150 in 1980.", ["Parramatta NSW 2150"]),
            ("Declared at Penrith NSW 2750 on 3 May 2024 before me:", ["Penrith NSW 2750"]),
            ("She was injured on Victoria Road, Ryde.", ["Victoria Road,", "Ryde."]),
        ],
    )
    def test_places_on_their_own_are_kept(self, text: str, kept: list[str]) -> None:
        pages = pages_from_grid(column(text))
        join, words = _detect_and_join(pages)
        assert not join.units and _values(join, E.STREET_ADDRESS) == []
        assert _acted(join) == []
        recorded = {" ".join(words[t] for t in k.token_ids) for k in join.kept}
        assert set(kept) <= recorded
        exempt = {words[t] for t in join.exempt.kept_places}
        assert {w for phrase in kept for w in phrase.split()} <= exempt

    def test_a_model_locality_in_prose_is_kept(self) -> None:
        pages = pages_from_grid(column("She flew to Sydney last week."))
        join, _ = _detect_and_join(pages, extra=[_model_mention(pages, E.LOCALITY, 0, 3, 3)])
        assert [(m.entity_type, m.value) for m in join.mentions] == [(E.LOCALITY, "Sydney")]
        assert _acted(join) == []

    def test_a_standalone_suburb_beside_an_address_in_the_same_suburb(self) -> None:
        pages = pages_from_grid(
            column("Address: 12 Smith Street, Penrith NSW 2750", "She grew up in Penrith before moving.")
        )
        join, words = _detect_and_join(pages, extra=[_model_mention(pages, E.LOCALITY, 1, 4, 4)])
        assert _acted(join) == ["12 Smith Street, Penrith NSW 2750"]
        assert _values(join, E.LOCALITY) == ["Penrith"]
        assert [words[t] for t in join.exempt.kept_places] == ["Penrith"]
        (targets,) = join.targets.values()
        assert "penrith" in targets.whole  # a left-over address place would be a hit; the prose place is exempt

    def test_an_address_block_followed_by_a_place_of_birth_row(self) -> None:
        pages = pages_from_grid(column("45 Smith Street", "Penrith NSW 2750", "Place of birth - City: Dubbo"))
        join, words = _detect_and_join(pages)  # the 'City' label gives the LOCALITY
        assert _acted(join) == ["45 Smith Street Penrith NSW 2750"]
        assert _values(join, E.LOCALITY) == ["Dubbo"]
        assert "Dubbo" in {words[t] for t in join.exempt.kept_places}

    def test_the_kept_places_of_another_profile_are_replaced_by_its_own_rule(self) -> None:
        pages = pages_from_grid(column("She flew to Sydney last week."))
        join, _ = _detect_and_join(pages, "broad", extra=[_model_mention(pages, E.LOCALITY, 0, 3, 3)])
        assert _acted(join, "broad") == ["Sydney"]
        assert join.exempt.kept_places == frozenset()


def _injected(*items: tuple[EntityType, str]) -> list[Mention]:
    ev = (Evidence("bench.inject", "1", "inject.TEST", EvidenceStrength.STRONG, 1.0),)
    return [Mention(f"M-{i}", etype, (token_id(0, 0, i),), ev, value) for i, (etype, value) in enumerate(items)]


class TestInjectedMentions:
    """``decide_injected`` (benchmark oracle modes): the benchmark asks for every injected mention to be redacted."""

    def test_kept_types_are_replaced_and_nothing_is_kept(self) -> None:
        mentions = _injected(
            (E.LOCALITY, "Penrith"),
            (E.POSTCODE, "2750"),
            (E.DATE_OF_BIRTH, "14/03/1978"),
            (E.AGE, "42"),
            (E.PERSON, "Jane"),
            (E.PHONE, "0412 345 678"),
        )
        assert [d.action for d in decide(mentions, CLIENT.policy)] == [A.KEEP] * 4 + [A.SURROGATE] * 2
        decisions = decide_injected(mentions, CLIENT.policy)
        assert [d.mention for d in decisions] == mentions
        assert [d.action for d in decisions] == [A.SURROGATE] * 6
        assert decisions[4].action is CLIENT.policy.entity_actions[E.PERSON]
        assert decisions[5].action is CLIENT.policy.entity_actions[E.PHONE]

    def test_the_profile_s_own_transformations_stay_unless_they_would_change_nothing(self) -> None:
        maximal = load_config("maximal").policy
        mentions = _injected((E.DATE, "2 May 2024"), (E.AGE, "95"), (E.DATE, "1978"), (E.AGE, "42"))
        assert [d.action for d in decide(mentions, maximal)] == [A.DATE_SHIFT, A.GENERALISE, A.KEEP, A.KEEP]
        assert [d.action for d in decide_injected(mentions, maximal)] == [
            A.DATE_SHIFT,
            A.GENERALISE,
            A.SURROGATE,  # a bare year cannot be shifted
            A.SURROGATE,  # an age below the cap is not changed by top-coding
        ]

    def test_no_type_is_kept_under_any_profile(self) -> None:
        mentions = _injected(*((etype, "Peter Brown") for etype in EntityType))
        for profile in ("client", "broad", "claimant_focused", "maximal"):  # claimant_focused keeps organisations
            policy = load_config(profile).policy
            assert all(d.action is not A.KEEP for d in decide_injected(mentions, policy)), profile
