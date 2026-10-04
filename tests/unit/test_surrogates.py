"""Surrogate generation: determinism, safety by construction, format preservation, consistency."""

from __future__ import annotations

import hashlib

import pytest

from mlredact.core.errors import EnvironmentErrorMl
from mlredact.core.types import EntityType
from mlredact.detect.rules.checksums import abn_valid, ihi_valid, medicare_valid, provider_number_valid, tfn_valid
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory, apply_case, canonical_date
from mlredact.surrogate.keys import load_master_key
from mlredact.surrogate.pools import load_pools
from mlredact.text.normalize import digits_only, match_key
from mlredact.verify.surrogates import acma_phone, check_surrogates

KEY = hashlib.sha256(b"test-scope").digest()


def factory(forbidden: tuple[str, ...] = (), key: bytes = KEY) -> SurrogateFactory:
    return SurrogateFactory(Chooser(key), [match_key(f) for f in forbidden])


class TestChooser:
    def test_deterministic_and_keyed(self) -> None:
        a, b = Chooser(KEY), Chooser(KEY)
        assert [a.index("d", "v", 97, i) for i in range(5)] == [b.index("d", "v", 97, i) for i in range(5)]
        other = Chooser(hashlib.sha256(b"other").digest())
        assert [a.index("d", "v", 10**6, i) for i in range(5)] != [other.index("d", "v", 10**6, i) for i in range(5)]

    def test_index_in_range(self) -> None:
        c = Chooser(KEY)
        assert all(0 <= c.index("d", str(i), 7) < 7 for i in range(200))

    def test_digit_string(self) -> None:
        s = Chooser(KEY).digit_string("d", "v", 12)
        assert len(s) == 12 and s.isdigit()


class TestKeys:
    def test_development_key_requires_opt_in_and_is_refused_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLREDACT_SURROGATE_KEY", raising=False)
        with pytest.raises(EnvironmentErrorMl):
            load_master_key(key_file=None, allow_development_key=False, production=False)
        with pytest.raises(EnvironmentErrorMl):
            load_master_key(key_file=None, allow_development_key=True, production=True)
        dev = load_master_key(key_file=None, allow_development_key=True, production=False)
        assert dev.key_id.startswith("k-") and len(dev.key_id) == 18

    def test_environment_key_wins_and_short_keys_are_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLREDACT_SURROGATE_KEY", "ab" * 32)
        k1 = load_master_key(key_file=None, allow_development_key=True, production=True)
        assert k1.scope_key("a") != k1.scope_key("b")
        monkeypatch.setenv("MLREDACT_SURROGATE_KEY", "ab" * 8)
        with pytest.raises(EnvironmentErrorMl):
            load_master_key(key_file=None, allow_development_key=False, production=False)

    def test_secret_never_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLREDACT_SURROGATE_KEY", "cd" * 32)
        assert "cdcd" not in repr(load_master_key(key_file=None, allow_development_key=False, production=False))


class TestIdentifiers:
    @pytest.mark.parametrize(
        ("etype", "value", "valid"),
        [
            (EntityType.MEDICARE, "2953 70150 1", lambda s: medicare_valid(digits_only(s)[:10])),
            (EntityType.TFN, "123 456 782", lambda s: tfn_valid(digits_only(s))),
            (EntityType.ABN, "51 824 753 556", lambda s: abn_valid(digits_only(s))),
            (EntityType.IHI, "8003 6081 6669 0503", lambda s: ihi_valid(digits_only(s))),
            (EntityType.PROVIDER_NUMBER, "212345AW", lambda s: provider_number_valid(s)),
        ],
    )
    def test_format_kept_and_checksum_deliberately_invalid(self, etype: EntityType, value: str, valid) -> None:  # type: ignore[no-untyped-def]
        sur = factory((value,)).identifier(etype, value)
        assert sur != value
        assert [c.isdigit() for c in sur] == [c.isdigit() for c in value]
        assert [c.isalpha() for c in sur] == [c.isalpha() for c in value]
        assert not valid(sur)

    def test_ihi_prefix_kept(self) -> None:
        assert digits_only(factory().identifier(EntityType.IHI, "8003 6081 6669 0503")).startswith("800360")

    def test_same_value_same_characters_in_each_layout(self) -> None:
        f = factory()
        a = f.identifier(EntityType.REFERENCE, "WC1234567")
        b = f.identifier(EntityType.REFERENCE, "WC 1234567,")
        assert match_key(a) == match_key(b) and b.endswith(",") and " " in b

    def test_numbers_keep_non_zero_lead(self) -> None:
        f = factory()
        assert all(f.number_like("n", str(v))[0] != "0" for v in range(100, 140))


class TestContact:
    @pytest.mark.parametrize(
        "value", ["0412 345 678", "(02) 9876 5432", "+61 412 555 666", "1800 123 456", "07 3333 2222"]
    )
    def test_phones_are_in_acma_fiction_ranges(self, value: str) -> None:
        assert acma_phone(factory().phone(value))

    def test_email_domains_reserved_and_role_mailboxes_kept(self) -> None:
        f = factory()
        assert f.email("admin@smithlawyers.com.au").startswith("admin@")
        assert f.email("john.smith78@bigpond.com", "peter.walsh").startswith("peter.walsh@")
        assert f.email("x.y@z.com").split("@")[1] in {"example.com", "example.org", "example.net"}


class TestDates:
    def test_same_year_and_consistent_across_layouts(self) -> None:
        f = factory(("14/03/1978", "14 March 1978"))
        numeric = f.date_of_birth("14/03/1978")
        textual = f.date_of_birth("14 March 1978")
        assert numeric.endswith("1978") and textual.endswith("1978")
        assert canonical_date(numeric) == canonical_date(textual) != "1978-03-14"

    def test_formats_and_ordinals(self) -> None:
        f = factory()
        assert len(f.date_of_birth("3/7/62")) <= len("31/12/62") and f.date_of_birth("3/7/62").endswith("62")
        assert f.date_of_birth("14th March 1978")[:4].rstrip("abcdefghijklmnopqrstuvwxyz ").isdigit()

    def test_unrecognised_layout_never_returns_original(self) -> None:
        f = factory(("Born 1978",))
        out = f.date_of_birth("Born 1978")
        assert out != "Born 1978" and out.endswith("1978")

    @pytest.mark.parametrize(
        ("text", "iso"),
        [
            ("14/03/1978", "1978-03-14"),
            ("3.7.62", "1962-07-03"),
            ("14 March 1978", "1978-03-14"),
            ("14th Mar, 1978", "1978-03-14"),
        ],
    )
    def test_canonical_date(self, text: str, iso: str) -> None:
        assert canonical_date(text) == iso


class TestNamesAndPlaces:
    def test_case_and_collision_avoidance(self) -> None:
        pools = load_pools()
        forbidden = tuple(pools.surnames[:-1])  # everything but the last pool entry is "original"
        f = factory(forbidden)
        out = f.surname("zzz", "SMITH")
        assert out.isupper() and out.title() == pools.surnames[-1].title()

    def test_distinct_originals_get_distinct_surrogates(self) -> None:
        f = factory()
        names = {f.surname(f"original{i}", "Smith") for i in range(60)}
        assert len(names) == 60

    def test_deterministic_across_factories(self) -> None:
        assert factory().given_name("john", "m", "John") == factory().given_name("john", "m", "John")

    def test_locality_same_state(self) -> None:
        loc = factory().locality("Penrith", "NSW")
        assert loc.state == "NSW" and loc.postcode.startswith("2")

    def test_organisation_keeps_generic_tail(self) -> None:
        assert factory().organisation("Smith & Partners Lawyers").endswith("& Partners Lawyers")

    def test_apply_case(self) -> None:
        assert apply_case("SMITH", "Walsh") == "WALSH"
        assert apply_case("smith", "Walsh") == "walsh"
        assert apply_case("Smith", "Walsh") == "Walsh"


class TestV5:
    def test_flags_collisions_valid_checksums_and_real_phones(self) -> None:
        originals = ["smith", "johnsmith"]
        assert check_surrogates([(EntityType.PERSON, "Smyth", ("Smyth",))], originals).passed is False
        assert check_surrogates([(EntityType.TFN, "123 456 782", ())], []).passed is False
        assert check_surrogates([(EntityType.PHONE, "0412 345 678", ())], []).passed is False
        assert check_surrogates([(EntityType.EMAIL, "a@gmail.com", ())], []).passed is False
        ok = check_surrogates(
            [(EntityType.PHONE, "0491 570 006", ()), (EntityType.PERSON, "Walsh", ("Walsh",))], originals
        )
        assert ok.passed

    def test_structure_words_are_not_collisions(self) -> None:
        # "Street" and "NSW" are kept from the original on purpose; only replaced tokens are compared.
        res = check_surrogates(
            [(EntityType.STREET_ADDRESS, "48 Acacia Street, Dubbo NSW 2830", ("48", "Acacia", "Dubbo", "2830"))],
            ["street", "nsw", "42", "wattle", "penrith", "2750"],
        )
        assert res.passed
