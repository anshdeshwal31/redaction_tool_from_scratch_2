"""Person resolution, alias handling, surrogate assignment and labels on token streams (no OCR)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from mlredact.config.loader import load_config
from mlredact.core.types import ActionKind, EntityType
from mlredact.detect.engine import detect_document
from mlredact.policy.engine import decide
from mlredact.resolve.persons import (
    Gender,
    NameToken,
    ParsedMention,
    PersonResolution,
    Role,
    parse_name,
    with_clusters,
)
from mlredact.surrogate.assign import MentionSurrogate, SurrogateAssigner, forbidden_keys
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory
from mlredact.surrogate.labels import assign_labels
from mlredact.text.normalize import match_key
from support import pages_from_lines

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
