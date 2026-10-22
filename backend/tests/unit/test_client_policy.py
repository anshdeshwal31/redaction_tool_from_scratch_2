"""The client profile (the client's policy decisions): actions and kept places."""

from __future__ import annotations

import pytest

from mlredact.config.loader import load_config
from mlredact.core.model import Evidence, Mention, token_id
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength, RegionKind
from mlredact.policy.engine import decide, decide_injected, split_kept_places
from support import pages_from_lines

CLIENT = load_config("client")
A = ActionKind
E = EntityType


class TestProfile:
    def test_is_the_default(self) -> None:
        assert load_config().policy.profile_id == "client"

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
            # Q4: suburbs, towns and places of birth (and their postcodes) are kept; street lines go.
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
    def test_street_line_is_split_from_the_kept_place(self) -> None:
        pages = pages_from_lines(["12 Smith Street, Parramatta NSW 2150"])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        parts = split_kept_places([m], tokens, CLIENT.policy)
        assert [(p.entity_type, p.value) for p in parts] == [
            (E.STREET_ADDRESS, "12 Smith Street,"),
            (E.LOCALITY, "Parramatta NSW 2150"),
        ]

    def test_po_box_and_place_only_addresses(self) -> None:
        pages = pages_from_lines(["PO Box 77 Wollongong NSW 2500", "Parramatta NSW 2150"])
        box, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        place, _ = _mention(pages, 1, E.STREET_ADDRESS)
        parts = split_kept_places([box, place], tokens, CLIENT.policy)
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
            ("Unit 5, 12 Smith St Parramatta NSW", "Unit 5, 12 Smith St"),  # comma inside the street line
        ],
    )
    def test_the_street_line_ends_at_the_right_place(self, line: str, street: str) -> None:
        pages = pages_from_lines([line])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        parts = split_kept_places([m], tokens, CLIENT.policy)
        assert (parts[0].entity_type, parts[0].value) == (E.STREET_ADDRESS, street)

    @pytest.mark.parametrize("line", ["8 Broadway Ultimo NSW", "Unit 5, 8 Broadway Ultimo", "12 Victoria Road, Ryde"])
    def test_an_unclear_street_line_is_replaced_whole(self, line: str) -> None:
        pages = pages_from_lines([line])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        assert split_kept_places([m], tokens, CLIENT.policy) == [m]

    def test_a_replaced_postcode_is_split_off(self) -> None:
        policy = load_config("client", overrides={"policy": {"entity_actions": {"postcode": "surrogate"}}}).policy
        pages = pages_from_lines(["12 Smith Street Parramatta NSW 2150"])
        m, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        parts = split_kept_places([m], tokens, policy)
        assert [p.entity_type for p in parts] == [E.STREET_ADDRESS, E.LOCALITY, E.POSTCODE]
        assert parts[-1].value == "2150"

    def test_addresses_without_a_state_and_other_profiles_are_left_alone(self) -> None:
        pages = pages_from_lines(["12 Smith Street Parramatta", "12 Smith Street, Parramatta NSW 2150"])
        no_state, tokens = _mention(pages, 0, E.STREET_ADDRESS)
        assert split_kept_places([no_state], tokens, CLIENT.policy) == [no_state]
        full, _ = _mention(pages, 1, E.STREET_ADDRESS)
        assert split_kept_places([full], tokens, load_config("broad").policy) == [full]


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
