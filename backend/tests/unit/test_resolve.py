"""Person resolution, alias handling, surrogate assignment and labels on token streams (no OCR)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_address_join import Doc

from mlredact.config.loader import load_config
from mlredact.core.errors import EnvironmentErrorMl
from mlredact.core.model import Mention, Token
from mlredact.core.types import ActionKind, EntityType, ViewKind
from mlredact.detect.engine import detect_document
from mlredact.policy.engine import decide
from mlredact.resolve import persons as persons_module
from mlredact.resolve.persons import (
    Gender,
    NameToken,
    ParsedMention,
    PersonResolution,
    Role,
    name_gender,
    parse_name,
    resolve_persons,
    with_clusters,
)
from mlredact.surrogate.assign import MentionSurrogate, SurrogateAssigner, forbidden_keys
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory
from mlredact.surrogate.labels import assign_labels
from mlredact.surrogate.pools import load_pools
from mlredact.text.normalize import match_key
from mlredact.text.views import build_views
from support import column, pages_from_lines

CFG = load_config("broad")


def roles(tokens: list[str], title: str | None = None) -> list[tuple[str, Role]]:
    return [(t.core, t.role) for t in parse_name(tokens, title=title)]


class TestParseName:
    def test_inverted_allcaps_and_plain(self) -> None:
        assert roles(["SMITH,", "John"]) == [("SMITH", Role.SURNAME), ("John", Role.GIVEN)]
        assert roles(["John", "Andrew", "SMITH"])[-1] == ("SMITH", Role.SURNAME)
        assert roles(["Jane", "Citizen"]) == [("Jane", Role.GIVEN), ("Citizen", Role.SURNAME)]

    def test_initials_particles_and_possessives(self) -> None:
        assert roles(["J.", "Smith"]) == [("J.", Role.INITIAL), ("Smith", Role.SURNAME)]
        assert roles(["Ludwig", "van", "Beethoven"])[1:] == [("van", Role.SURNAME), ("Beethoven", Role.SURNAME)]
        assert roles(["Smith's"]) == [("Smith", Role.GIVEN)]

    def test_lone_word_depends_on_title(self) -> None:
        assert roles(["Smith"], title="Mr") == [("Smith", Role.SURNAME)]
        assert roles(["John"]) == [("John", Role.GIVEN)]


def _detect(lines: Sequence[str]):  # type: ignore[no-untyped-def]
    pages = pages_from_lines(lines)
    result = detect_document(pages, CFG, "doc")
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    return result, tokens


class TestResolution:
    LINES = (
        "Re: Mr John Andrew SMITH DOB: 14/03/1978",
        "Mr Smith attended with his wife Mrs Mary Smith.",
        "Smith reported that John was anxious; J. Smith signed the form.",
        "Dr Peter Brown examined him.",
    )

    def test_clusters_partial_mentions_and_gender(self) -> None:
        result, _tokens = _detect(self.LINES)
        res = result.persons
        by_cluster: dict[str | None, list[str]] = {}
        for pm in res.mentions.values():
            by_cluster.setdefault(pm.cluster, []).append(" ".join(t.core for t in pm.tokens))
        john = next(c for c, names in by_cluster.items() if any("John Andrew SMITH" in n for n in names))
        assert john is not None
        assert res.given_gender[match_key("John")] is Gender.MALE
        assert res.given_gender[match_key("Mary")] is Gender.FEMALE

    def test_standalone_surname_and_given_name_are_found(self) -> None:
        result, tokens = _detect(self.LINES)
        texts = {
            " ".join(tokens[t].text for t in m.token_ids) for m in result.mentions if m.entity_type is EntityType.PERSON
        }
        assert "Smith" in texts  # "Smith reported ..." (component propagation)
        assert "John" in texts  # "... that John was anxious"

    def test_eponym_context_is_not_a_name(self) -> None:
        result, tokens = _detect(["Re: Mr John SMITH", "Smith's fracture was noted and Smith is well."])
        line1 = [m for m in result.mentions if tokens[m.token_ids[0]].line_no == 1]
        texts = [" ".join(tokens[t].text for t in m.token_ids) for m in line1]
        assert "Smith's" not in texts and "Smith" in texts


class TestAssignment:
    def test_consistent_component_surrogates(self) -> None:
        result, tokens = _detect(TestResolution.LINES)
        decisions = decide(result.mentions, CFG.policy)
        removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
        factory = SurrogateFactory(Chooser(hashlib.sha256(b"s").digest()), forbidden_keys(removed, result.persons))
        sur = SurrogateAssigner(factory, result.persons, tokens).assign(decisions)
        by_value: dict[str, set[str]] = {}
        for d in decisions:
            if d.mention.entity_type is EntityType.PERSON and d.mention.mention_id in sur:
                by_value.setdefault(d.mention.value, set()).add(
                    " ".join(t for t in sur[d.mention.mention_id].tokens if t)
                )
        smith = by_value["Smith"]
        assert len(smith) == 1  # every standalone "Smith" maps to the same surrogate surname
        full = next(iter(by_value["John Andrew SMITH"])).split()
        assert full[-1].isupper() and full[-1].title() == next(iter(smith))  # family shares the surname
        initial = next(v for k, v in by_value.items() if k.startswith("J."))
        assert next(iter(initial)).startswith(full[0][0] + ".")  # "J. Smith" follows John's surrogate
        for originals, surrogates in by_value.items():
            assert all(match_key(s) != match_key(originals) for s in surrogates)

    def test_address_components(self) -> None:
        result, tokens = _detect(["Address: 42 Wattle Grove Road, Penrith NSW 2750"])
        decisions = decide(result.mentions, CFG.policy)
        factory = SurrogateFactory(Chooser(b"k" * 32), [match_key(d.mention.value) for d in decisions])
        sur = SurrogateAssigner(factory, result.persons, tokens).assign(decisions)
        (address,) = [s for d in decisions if (s := sur.get(d.mention.mention_id)) is not None]
        text = " ".join(t for t in address.tokens if t)
        assert "Road," in text and "NSW" in text and "Penrith" not in text and "Wattle" not in text

    def test_labels_are_consistent_per_person(self) -> None:
        result, _tokens = _detect(TestResolution.LINES)
        labels = assign_labels(decide(result.mentions, CFG.policy), result.persons)
        values = {s.tokens[0] for s in labels.values()}
        assert any(v.startswith("[PERSON ") for v in values) and "[DOB 1]" in values


class TestProfessionalRoles:
    LINES = (
        "Re: Mr John Andrew SMITH DOB: 14/03/1978",
        "Mr Smith attended with his wife Mrs Mary Smith.",
        "Dr Peter Brown FRACS, Orthopaedic Surgeon, examined him.",
        "Mr Smith was examined by his treating surgeon on Monday.",
        "Copy to: Ms Jane Citizen, Solicitor, for information.",
        "Dr Alan Smith reviewed the scans.",
    )

    def _kept_and_replaced(self, profile: str) -> tuple[set[str], set[str]]:
        from mlredact.resolve.roles import professional_mentions

        cfg = load_config(profile)
        result, tokens = _detect(self.LINES)
        pros = professional_mentions(result.persons, result.mentions, result.line_view)
        decisions = decide(result.mentions, cfg.policy, pros)
        kept, replaced = set(), set()
        for d in decisions:
            if d.mention.entity_type is EntityType.PERSON:
                text = " ".join(tokens[t].text for t in d.mention.token_ids).rstrip(",.")
                (kept if d.action is ActionKind.KEEP else replaced).add(text)
        return kept, replaced

    def test_claimant_focused_keeps_only_clear_professionals(self) -> None:
        kept, replaced = self._kept_and_replaced("claimant_focused")
        assert any("Peter Brown" in k for k in kept)
        assert any("Jane Citizen" in k for k in kept)
        assert not any("Smith" in k or "SMITH" in k for k in kept)  # claimant, wife, and Dr Alan Smith (family rule)
        assert any("John Andrew SMITH" in r for r in replaced) and any("Alan Smith" in r for r in replaced)

    def test_broad_never_keeps_people(self) -> None:
        kept, _replaced = self._kept_and_replaced("broad")
        assert kept == set()


# The lines of tests/determinism/test_hash_seed.py's script.
HASH_SEED_LINES = (
    "Re: Mr John Andrew SMITH DOB: 14/03/1978 Claim No: WC1234567",
    "Address: 42 Wattle Grove Road, Penrith NSW 2750 Phone 0412 345 678",
    "Mr Smith attended with his wife Mrs Mary Smith. J. Smith signed; John was anxious.",
    "Dr Peter Brown FRACS, Medicare 2953 70150 1, email john.smith78@bigpond.com",
    "Tinel's sign was positive. Smith's fracture was excluded.",
)


def _person(mid: str, given: str, surname: str, gender: Gender) -> ParsedMention:
    tokens = (NameToken("", given, "", Role.GIVEN), NameToken("", surname, "", Role.SURNAME))
    return ParsedMention(mid, tokens, None, gender, None)


class TestInjectedClusters:
    """``with_clusters``: the benchmark's injected clusters in place of the resolver's own."""

    @pytest.mark.parametrize("lines", [TestResolution.LINES, TestProfessionalRoles.LINES, HASH_SEED_LINES])
    def test_the_resolvers_own_clusters_change_nothing(self, lines: Sequence[str]) -> None:
        resolution = _detect(lines)[0].persons
        own = {m: p.cluster for m, p in resolution.mentions.items() if p.cluster}
        assert own
        assert with_clusters(resolution, own) == resolution

    def test_an_initial_follows_the_given_name_of_its_injected_cluster(self) -> None:
        result, tokens = _detect(["Re: Mr John Smith and his sister Ms Jane Smith", "J. Smith signed the form."])
        ids = {m.value: m.mention_id for m in result.mentions}
        john, jane, initial = ids["John Smith"], ids["Jane Smith"], ids["J. Smith"]
        own = {m: p.cluster for m, p in result.persons.mentions.items() if p.cluster}
        assert initial not in own  # two given names start with J: the resolver cannot tell

        def first_letters(clusters: dict[str, str]) -> tuple[str, str, str]:
            persons = with_clusters(result.persons, clusters)
            decisions = decide(result.mentions, CFG.policy)
            factory = SurrogateFactory(Chooser(hashlib.sha256(b"s").digest()), forbidden_keys([], persons))
            sur: dict[str, MentionSurrogate] = SurrogateAssigner(factory, persons, tokens).assign(decisions)
            return sur[initial].tokens[0][0], sur[jane].tokens[0][0], sur[john].tokens[0][0]

        mark, jane_fake, john_fake = first_letters(own)
        assert jane_fake != john_fake and mark not in (jane_fake, john_fake)
        assert first_letters({**own, initial: own[jane]})[0] == jane_fake
        assert first_letters({**own, initial: own[john]})[0] == john_fake

    def test_given_names_take_the_gender_of_their_injected_cluster(self) -> None:
        mentions = {
            "a": _person("a", "Sam", "Taylor", Gender.MALE),
            "b": _person("b", "Sam", "Jones", Gender.FEMALE),
            "c": _person("c", "Sam", "Jones", Gender.UNKNOWN),
        }
        resolution = PersonResolution(mentions, {}, {})
        with_male = with_clusters(resolution, {"a": "e1", "c": "e1", "b": "e2"})
        with_female = with_clusters(resolution, {"a": "e1", "b": "e2", "c": "e2"})
        assert with_male.clusters == {"e1": ("a", "c"), "e2": ("b",)}
        assert with_male.given_gender == {"sam": Gender.MALE}  # two votes for e1 (male), one for e2
        assert with_female.given_gender == {"sam": Gender.FEMALE}
        assert [p.gender for p in with_male.mentions.values()] == [Gender.MALE, Gender.FEMALE, Gender.UNKNOWN]

    def test_mentions_without_an_injected_cluster_belong_to_none(self) -> None:
        resolution = _detect(TestResolution.LINES)[0].persons
        some = sorted(resolution.mentions)[0]
        alone = with_clusters(resolution, {some: "e1", "not-a-mention": "e2"})
        assert alone.clusters == {"e1": (some,)}
        assert [m for m, p in alone.mentions.items() if p.cluster] == [some]
        none = with_clusters(resolution, {})
        assert none.clusters == {} and all(p.cluster is None for p in none.mentions.values())
        # Each mention then votes alone: Mr Smith's John and Mrs Mary Smith's Mary keep their own cue.
        assert none.given_gender[match_key("Mary")] is Gender.FEMALE


# ------------------------------------------------------------------------------------- gender of fake given names
Span = tuple[int, int, int]  # (line, first word, last word) of a person mention


def _resolve(lines: Sequence[str], spans: Sequence[Span]) -> tuple[list[Mention], PersonResolution, dict[str, Token]]:
    doc = Doc(column(*lines))
    mentions = [doc.span(EntityType.PERSON, *s) for s in spans]
    tokens = {t.token_id: t for p in doc.pages for t in p.tokens()}
    return mentions, resolve_persons(mentions, build_views(doc.pages)[ViewKind.LINE]), tokens


def _fakes(lines: Sequence[str], spans: Sequence[Span], key: bytes = b"m" * 32) -> dict[str, str]:
    """Original mention text -> its fake, under one matter's scope key."""
    mentions, persons, tokens = _resolve(lines, spans)
    decisions = decide(mentions, CFG.policy)
    factory = SurrogateFactory(Chooser(key), forbidden_keys(mentions, persons))
    sur = SurrogateAssigner(factory, persons, tokens).assign(decisions)
    return {m.value: " ".join(t for t in sur[m.mention_id].tokens if t) for m in mentions}


def _gender_of(fake: str) -> str:
    first = match_key(fake.split()[0])
    pools = load_pools()
    male, female = ({match_key(n) for n in p} for p in (pools.given_male, pools.given_female))
    return "m" if first in male else "f" if first in female else "?"


# The fictional benchmark claim (synthetic@0.1.0): the husband is named after a kinship word, and the pronouns
# around him are about his wife.
HUSBAND = ("Ms Jane Smithers takes Sertraline daily.", "Her husband Tom Smithers attended with her.", "She is well.")
HUSBAND_SPANS = ((0, 1, 2), (1, 2, 3))


class TestGenderOfGivenNames:
    @pytest.mark.parametrize(
        ("lines", "spans", "expected"),
        [
            (HUSBAND, HUSBAND_SPANS, ["f", "m"]),
            (["Tom Smithers, her husband, attended with her."], [(0, 0, 1)], ["m"]),
            (["Tom Smithers (husband) attended. She said so."], [(0, 0, 1)], ["m"]),
            (["Her husband Dr Tom Smithers attended with her."], [(0, 3, 4)], ["m"]),
            (["His wife Pat Smithers and his son Lou Smithers came."], [(0, 2, 3), (0, 7, 8)], ["f", "m"]),
            (["Mr Tom Smithers said she was well."], [(0, 1, 2)], ["m"]),
            (["Mrs Lou Smithers said he was well."], [(0, 1, 2)], ["f"]),
            # The kinship word must end the phrase: "her husband Tom" names Tom, not Jane.
            (["Jane Smithers, her husband Tom Bloggs and Pat Doe (wife) came."], [(0, 0, 1), (0, 4, 5), (0, 7, 8)],
             ["f", "m", "f"]),
            # A kinship word in a list ("son and daughter, ...") names no one in particular.
            (["His son and daughter, Zorbo Quist and Mia Quist, came."], [(0, 4, 5), (0, 7, 8)], ["u", "f"]),
            # Nor in a list without a comma, or a list of kinship words.
            (["Son and daughter Zorbo Quist and Xeno Quist."], [(0, 3, 4), (0, 6, 7)], ["u", "u"]),
            (["His son, daughter, Zorbo Quist and Xeno Quist came."], [(0, 3, 4), (0, 6, 7)], ["u", "u"]),
            # A kinship word after a name in a list of people is an item of the list, not the name's aside.
            (["Attendees: Zorbo Quist, her husband, and Dr Lou Brown."], [(0, 1, 2)], ["u"]),
            (["Present: Zorbo Quist, her son, her daughter."], [(0, 1, 2)], ["u"]),
            (["The claimant was accompanied by Zorbo Quist, his wife, and two children."], [(0, 5, 6)], ["u"]),
            (["Pat Doe, Zorbo Quist, her husband, came."], [(0, 2, 3)], ["u"]),
            # In a plain sentence the aside still counts.
            (["She lives with Zorbo Quist, her son, in Penrith."], [(0, 3, 4)], ["m"]),
            # A kinship word never gives a name the list genders the other sex.
            (["Attendees: Jane Smithers, her husband, and Dr Lou Brown."], [(0, 1, 2)], ["f"]),
            (["Present: Robyn Quist, her son, her daughter."], [(0, 1, 2)], ["f"]),
            (["Her daughter Peter Quist came."], [(0, 2, 3)], ["m"]),
            # A kinship word ending the sentence before names no one in the next one.
            (["Pat came with her son. Zorbo Quist drove."], [(0, 5, 6)], ["u"]),
            # Pronouns are not read: an unlisted name with no title or kinship word stays unknown.
            (["Zorbo Quist attended. He said he was well."], [(0, 0, 1)], ["u"]),
        ],
    )  # fmt: skip
    def test_the_document_states_the_gender(
        self, lines: Sequence[str], spans: Sequence[Span], expected: list[str]
    ) -> None:
        mentions, persons, _tokens = _resolve(lines, spans)
        assert [persons.mentions[m.mention_id].gender.value for m in mentions] == expected

    def test_the_name_list_gives_the_gender_of_common_names(self) -> None:
        assert name_gender("Peter") is Gender.MALE  # surrogate pool
        assert name_gender("Mary") is Gender.FEMALE  # NSW list only
        assert name_gender("Thomas") is Gender.MALE
        assert name_gender("Kellie-Ann") is Gender.FEMALE  # both parts female
        assert name_gender("KERRY") is Gender.FEMALE  # as before: in the female pool only
        assert name_gender("Zoë") is name_gender("Chloé") is name_gender("Renée") is Gender.FEMALE  # accents
        for name in ("Alex", "Sam", "Jordan", "Shannon", "Lee", "Zorbo", "Kelli-Ann", "Jean-Zorbo"):
            assert name_gender(name) is Gender.UNKNOWN, name

    def test_a_missing_name_list_is_the_tools_resource_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(persons_module, "resource_root", lambda: tmp_path)  # no names/ folder
        persons_module._listed_genders.cache_clear()
        try:
            with pytest.raises(EnvironmentErrorMl):
                name_gender("Zorbo")
        finally:
            monkeypatch.undo()
            persons_module._listed_genders.cache_clear()
        assert name_gender("Mary") is Gender.FEMALE

    def test_a_husband_gets_a_male_fake_and_his_wife_a_female_one(self) -> None:
        fakes = _fakes(HUSBAND, HUSBAND_SPANS)
        assert _gender_of(fakes["Jane Smithers"]) == "f"
        assert _gender_of(fakes["Tom Smithers"]) == "m"
        assert fakes["Jane Smithers"].split()[1] == fakes["Tom Smithers"].split()[1]  # the family keeps one surname

    @pytest.mark.parametrize(
        "line",
        [
            "Mr Tom Smithers attended with her.",
            "Her husband Tom Smithers attended.",
            "Tom Smithers, her husband, came.",
        ],
    )
    def test_a_man_by_title_or_kinship_word_gets_a_male_fake(self, line: str) -> None:
        first = line.split().index("Tom")
        for i in range(8):
            fake = _fakes([line], [(0, first, first + 1)], key=bytes([i]) * 32)[
                "Tom Smithers" + ("," if "," in line else "")
            ]
            assert _gender_of(fake) == "m", fake

    def test_a_woman_gets_a_female_fake(self) -> None:
        for i in range(8):
            fake = _fakes(["Her daughter Lou Quist came; he drove."], [(0, 2, 3)], key=bytes([i]) * 32)["Lou Quist"]
            assert _gender_of(fake) == "f", fake

    def test_an_unknown_name_is_faked_as_before(self) -> None:
        """No title, kinship word or listed name: the fake comes from both gendered pools, keyed as before."""
        key = b"u" * 32
        fake = _fakes(["Zorbo Quist attended. She was well."], [(0, 0, 1)], key=key)["Zorbo Quist"]
        mentions, persons, _tokens = _resolve(["Zorbo Quist attended. She was well."], [(0, 0, 1)])
        assert persons.given_gender == {"zorbo": Gender.UNKNOWN}
        expected = SurrogateFactory(Chooser(key), forbidden_keys(mentions, persons))._unique_pool(
            "given:u", "zorbo", load_pools().given_male + load_pools().given_female, names=True
        )
        assert fake.split()[0] == expected

    def test_unknown_people_keep_their_fake_beside_other_unknown_people(self) -> None:
        """A person of unknown gender keeps one fake whether alone or among five others of unknown gender in a
        document as often as before the gender change: unknown gender draws from both gendered pools, as before."""
        given = ("Tom", "Bob", "Jim", "Mike", "Lizzie", "Bill")
        people = [
            f"{g} {s}" for g, s in zip(given, ("Quist", "Varga", "Okafor", "Brandt", "Moreau", "Haddad"), strict=True)
        ]
        differ = 0
        for n in range(20):
            key = bytes([n, 7]) * 16
            together = _fakes([f"{p} attended." for p in people], [(i, 0, 1) for i in range(len(people))], key=key)
            for p in people:
                alone = _fakes([f"{p} attended."], [(0, 0, 1)], key=key)[p]
                differ += together[p].split()[0] != alone.split()[0]
        assert differ <= 4, differ  # of 120 person-documents: 4, exactly as on the base code (ac9dfc9)

    def test_fakes_are_deterministic_and_one_per_person_across_a_matter(self) -> None:
        """Two documents of one matter (one scope key) that name the husband differently give him one fake."""
        first = _fakes(HUSBAND, HUSBAND_SPANS)
        assert _fakes(HUSBAND, HUSBAND_SPANS) == first
        other = _fakes(["Re: Mr Tom Smithers and Ms Jane Smithers"], [(0, 2, 3), (0, 6, 7)])
        assert other["Tom Smithers"] == first["Tom Smithers"]
        assert other["Jane Smithers"] == first["Jane Smithers"]
        assert _fakes(HUSBAND, HUSBAND_SPANS, key=b"x" * 32) != first  # another matter, other fakes
