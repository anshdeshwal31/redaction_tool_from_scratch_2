"""Surrogate generation: determinism, safety by construction, format preservation, consistency."""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_address_join import Doc, join_case

from mlredact.config.loader import load_config
from mlredact.core.errors import EnvironmentErrorMl
from mlredact.core.model import Mention
from mlredact.core.types import ActionKind, EntityType, ViewKind
from mlredact.detect.rules.checksums import abn_valid, ihi_valid, medicare_valid, provider_number_valid, tfn_valid
from mlredact.policy.addresses import AddressJoin, JoinedUnit, join_addresses
from mlredact.policy.engine import Decision
from mlredact.resolve.persons import PersonResolution, resolve_persons
from mlredact.resources import loader
from mlredact.surrogate import pools as pools_module
from mlredact.surrogate.addresses import AddressFaker, UnitView, UnitWord, postcode_ok, view_of
from mlredact.surrogate.assign import SurrogateAssigner, forbidden_keys
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.exclusions import Exclusions
from mlredact.surrogate.generators import SurrogateFactory, apply_case, canonical_date
from mlredact.surrogate.keys import load_master_key
from mlredact.surrogate.pools import load_pools
from mlredact.text.address import TYPE_FORMS, AddressKey, canonical_street_type, is_abbreviated_type
from mlredact.text.gazetteer import RANGES, fold_name, load_gazetteer, name_variants
from mlredact.text.normalize import digits_only, match_key
from mlredact.text.views import build_views
from mlredact.verify.needles import tokens_of
from mlredact.verify.surrogates import acma_phone, check_surrogates
from support import column

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


# ====================================================================================================
# Whole-address fakes (FINAL_PLAN TOOL2 section 5, amendments D and E): the case table at the fake level, through the
# join and the assigner, and the rules of section 5 one by one.  Pure: no models.
# ====================================================================================================
_TABLE = json.loads((Path(__file__).resolve().parents[1] / "data" / "address_cases.json").read_text("utf-8"))
_CASES: list[dict[str, Any]] = _TABLE["cases"]
_BY_ID = {c["id"]: c for c in _CASES}
_KEYS = [hashlib.sha256(f"address-scope-{i}".encode()).digest() for i in range(3)]
_POLICY = load_config("client").policy
_GAZ = load_gazetteer()
_NO_PERSONS = PersonResolution({}, {}, {})
_STRUCTURE = frozenset(_TABLE["structure_roles"])
_RUN_ROLES = frozenset({"street_name", "building", "box_extra", "dx_exchange", "suburb", "state"})
_NUMBER_ROLES = frozenset({"number", "unit_number", "box_number", "dx_number", "unit_number+number"})
_STATE_OF = {
    "nsw": "NSW", "n.s.w": "NSW", "n5w": "NSW", "new south wales": "NSW", "vic": "VIC", "v.i.c": "VIC",
    "victoria": "VIC", "qld": "QLD", "q.l.d": "QLD", "queensland": "QLD", "sa": "SA", "s.a": "SA",
    "south australia": "SA", "wa": "WA", "w.a": "WA", "western australia": "WA", "tas": "TAS", "t.a.s": "TAS",
    "tasmania": "TAS", "nt": "NT", "n.t": "NT", "northern territory": "NT", "act": "ACT", "a.c.t": "ACT",
    "australian capital territory": "ACT",
}  # fmt: skip
_PREFIX = {"NSW": "2", "ACT": "2", "VIC": "3", "QLD": "4", "SA": "5", "WA": "6", "TAS": "7", "NT": "08"}
_EDGES = ",;:()[]|!?\"'"
_NO_ABBREVIATION = frozenset({"RIDGE", "QUAY", "CREST", "GREEN", "VISTA"})  # the faker writes these in full
_HANDLED = {
    "keep": ("keep", "replace"),
    "replace": ("replace",),
}


def expected_class(words: Sequence[str], state: str, written: str | None) -> str | None:
    """The class of an original suburb as the case table's resolver reads it: the row of the written postcode, else a
    primary row, else the most populous row of the name in the state (rows straight from the list)."""
    digits = (written or "").strip(_EDGES + ".").translate(str.maketrans("OoIlSB", "001158"))
    names = set(name_variants(fold_name(" ".join(words))))
    rows = [r for r in _GAZ.rows if r.state == state and r.folded in names]
    if not rows:
        return None
    pick = [r for r in rows if r.postcode == digits] or [r for r in rows if r.primary] or rows
    return max(pick, key=lambda r: (r.population, r.postcode)).klass


@functools.cache
def joined(case_id: str) -> tuple[Doc, AddressJoin]:
    doc, result, _ = join_case(_BY_ID[case_id], "client", "parts")
    return doc, result


def assign_fakes(
    doc: Doc,
    result: AddressJoin,
    scope: int = 0,
    state: str = "keep",
    *,
    persons: PersonResolution = _NO_PERSONS,
    extra: Sequence[Decision] = (),
) -> dict[str, str]:
    """The replacement text of every token of the unit mentions, as the assigner gives it."""
    decisions = [
        Decision(m, ActionKind.SURROGATE) for m in result.mentions if m.entity_type is EntityType.STREET_ADDRESS
    ]
    tokens = {t.token_id: t for line in doc.lines for t in line.tokens}
    factory = SurrogateFactory(Chooser(_KEYS[scope]), forbidden_keys([d.mention for d in decisions], persons))
    assigner = SurrogateAssigner(
        factory,
        persons,
        tokens,
        layouts=result.layouts,
        units=result.units,
        address_state=state,  # type: ignore[arg-type]
    )
    sur = assigner.assign([*decisions, *extra])
    fakes: dict[str, str] = {}
    for d in decisions:
        for tid, text in zip(d.mention.token_ids, sur[d.mention.mention_id].tokens, strict=True):
            fakes[tid] = text
    return fakes


@functools.cache
def case_fakes(case_id: str, scope: int = 0, state: str = "keep") -> Mapping[str, str]:
    doc, result = joined(case_id)
    return assign_fakes(doc, result, scope, state)


def table_units(case_id: str) -> list[tuple[dict[str, Any], JoinedUnit]]:
    doc, result = joined(case_id)
    by_set = {frozenset(u.roles): u for u in result.units.values()}
    out = []
    for e in _BY_ID[case_id]["expect"]["addresses"]:
        ids = doc.tids([tuple(p) for p in e["street_line"]] + [tuple(p) for p in e["locality"]])
        out.append((e, by_set[frozenset(ids)]))
    return out


def edges(text: str) -> tuple[str, str, str]:
    core = text.strip(_EDGES)
    if not core:
        return text, "", ""
    i = text.index(core)
    return text[:i], core, text[i + len(core) :]


def style(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return "upper"
    return "lower" if letters and all(c.islower() for c in letters) else "title"


def one_edit(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1 :] == a for i in range(len(b)))


def pc_ok(state: str | None, original: str, fake: str, fake_state: str | None = None) -> bool:
    """Amendment D3, written out again: the original's leading digits matching its state's prefix may stay, and, in
    another state (replace mode), so may the digits that state fixes."""
    if fake == original:
        return False
    prefix = _PREFIX.get(state or "", "")
    k = 0
    while k < len(prefix) and k < len(original) and original[k] == prefix[k]:
        k += 1
    if fake_state and fake_state != state and fake.startswith(_PREFIX[fake_state]):
        k = max(k, len(_PREFIX[fake_state]))
    return all(a != b for a, b in zip(original[k:], fake[k:], strict=False))


def kinds(text: str) -> list[str]:
    return [
        "d" if p.isdigit() else "l" if p.isalpha() else "s" for p in re.findall(r"\d+|[A-Za-z]+|[^0-9A-Za-z]+", text)
    ]


def number_errors(o: str, f: str) -> list[str]:
    """Amendment D4 and section 5: the shape kept, every digit different at its position, a non-zero lead, a letter
    another letter of the same case, ranges ascending."""
    po, pf = re.findall(r"\d+|[A-Za-z]+|[^0-9A-Za-z]+", o), re.findall(r"\d+|[A-Za-z]+|[^0-9A-Za-z]+", f)
    if kinds(o) != kinds(f):
        return [f"shape {kinds(o)} != {kinds(f)}"]
    errs = []
    for a, b in zip(po, pf, strict=True):
        if a.isdigit():
            if len(a) != len(b):
                errs.append("digit count")
            if any(x == y for x, y in zip(a, b, strict=False)):
                errs.append("a digit stays at its position")
            if b.startswith("0"):
                errs.append("zero lead")
        elif a.isalpha():
            if a == b or a.isupper() != b.isupper():
                errs.append("letter suffix")
        elif a != b:
            errs.append("separator")
    nums_o, nums_f = re.findall(r"\d+", o), re.findall(r"\d+", f)
    ascending = re.fullmatch(r"\d+[A-Za-z]?[-–]\d+[A-Za-z]?", o) and int(nums_o[0]) < int(nums_o[1])
    if ascending and int(nums_f[0]) >= int(nums_f[1]):
        errs.append("range not ascending")
    return errs


def level_errors(o: str, f: str) -> list[str]:
    mo, mf = re.fullmatch(r"(\d{1,2})(st|nd|rd|th)?", o, re.I), re.fullmatch(r"(\d{1,2})(st|nd|rd|th)?", f, re.I)
    if not mo or not mf:
        return ["level shape"]
    errs = []
    if not 1 <= int(mf.group(1)) <= 25:
        errs.append("level outside 1-25")
    if any(a == b for a, b in zip(mo.group(1), mf.group(1), strict=False)):
        errs.append("a level digit stays at its position")
    if bool(mo.group(2)) != bool(mf.group(2)):
        errs.append("ordinal")
    return errs


def identifying(case_id: str) -> tuple[set[str], set[str]]:
    """Fold-normalised identifying words and normalised identifying digit runs of every unit of the case."""
    words: set[str] = set()
    numbers: set[str] = set()
    for e in _BY_ID[case_id]["expect"]["addresses"]:
        structure = {tuple(p) for p in e["structure"]}
        partial = {tuple(x["at"]): x["change"] for x in e.get("partial_structure", [])}
        for li, wi, text, role in e["words"]:
            if (li, wi) in structure or role in _STRUCTURE:
                continue
            text = partial.get((li, wi), text)
            runs = re.findall(r"\d+", text)
            if runs:
                numbers.update(r.lstrip("0") or "0" for r in runs)
            elif edges(text)[1]:
                words.update(fold_name(edges(text)[1]).split())
    return words, numbers


def d1_errors(case_id: str, fakes: Mapping[str, str]) -> list[str]:
    """Amendment D1 / README 'fake_constraints': no fake word or number repeats an identifying word or number of any
    unit of the case; no fake word contains one of 3+ letters or is one edit from one of 5+ letters."""
    idw, idn = identifying(case_id)
    doc, _ = joined(case_id)
    errs = []
    for e, _unit in table_units(case_id):
        skip = {doc.tid(tuple(p)) for p in e["structure"]}
        for li, wi, _text, _role in e["words"]:
            tid = doc.tid((li, wi))
            if tid in skip:
                continue
            f = fakes[tid]
            runs = re.findall(r"\d+", f)
            if runs:
                errs += [f"number {r} repeats an identifying number" for r in runs if (r.lstrip("0") or "0") in idn]
                continue
            for w in fold_name(f).split():
                if any(w == o or (len(o) >= 3 and o in w) or (len(o) >= 5 and one_edit(w, o)) for o in idw):
                    errs.append(f"fake word {w} repeats an identifying word")
    return errs


def state_code(text: str) -> str | None:
    return _STATE_OF.get(edges(text)[1].rstrip(".").lower())


def state_of(words: Sequence[str]) -> str | None:
    """The state a run of state words writes: 'New South Wales', 'N.S.W.', or a doubled 'NSW NSW'."""
    whole = state_code(" ".join(words))
    if whole is not None:
        return whole
    each = {state_code(w) for w in words}
    return each.pop() if len(each) == 1 else None


def unit_errors(case_id: str, e: dict[str, Any], unit: JoinedUnit, fakes: Mapping[str, str], mode: str) -> list[str]:
    """README 'What to assert per case', item 2, for one unit: structure words stay, every other word changes under
    the fake's constraints (numbers, levels, names, types, suffixes, suburb and class, postcode, state)."""
    doc, _ = joined(case_id)
    rows = [(doc.tid((li, wi)), text, role) for li, wi, text, role in e["words"]]
    structure = {doc.tid(tuple(p)) for p in e["structure"]}
    partial = {doc.tid(tuple(x["at"])): x for x in e.get("partial_structure", [])}
    overseas = bool(e.get("overseas"))
    errs: list[str] = []
    # the items: a run of name-like words is one item
    items: list[tuple[str, list[tuple[str, str, str]]]] = []
    for tid, text, role in rows:
        if items and role in _RUN_ROLES and items[-1][0] == role:
            items[-1][1].append((tid, text, fakes[tid]))
        else:
            items.append((role, [(tid, text, fakes[tid])]))
    fake_suburb = " ".join(edges(f)[1] for role, its in items if role == "suburb" for _t, _o, f in its if f)
    suburb_rows = [r for r in _GAZ.fake_pool() if fold_name(r.name) == fold_name(fake_suburb)] if fake_suburb else []
    orig_state = unit.key.state
    pcs = [t for _tid, t, role in rows if role == "postcode" and not overseas]
    if orig_state is None and pcs:
        digits = pcs[0].strip(_EDGES + ".").translate(str.maketrans("OoIlSB", "001158"))
        if digits.isdigit():
            st = [s for s, rng in RANGES.items() if any(lo <= int(digits) <= hi for lo, hi in rng)]
            orig_state = st[0] if len(st) == 1 else None
    for role, its in items:
        o_words = [edges(o)[1] for _t, o, _f in its]
        f_words = [edges(f)[1] for _t, _o, f in its if f]
        o, f = " ".join(o_words), " ".join(f_words)
        first_t, first_o, first_f = its[0]
        last_o, last_f = its[-1][1], its[-1][2]
        tid = first_t
        if not overseas and role not in _STRUCTURE and tid in structure:
            continue
        if tid in structure or (role in _STRUCTURE and tid not in partial):
            if role == "state" and mode == "replace":
                if state_of(f_words) in (None, state_of(o_words)):
                    errs.append("replace mode left a state word")
                continue
            gpo = role == "box_kind" and first_o.lower().replace(".", "") == "gpo"
            if gpo and first_f.lower().replace(".", "") == "po":
                if fake_suburb and _GAZ.is_capital(fake_suburb.split()):
                    errs.append("GPO became PO beside a capital suburb")
            elif [x for _t, _o, x in its] != [x for _t, x, _f in its]:
                errs.append(f"structure word {role} changed")
            continue
        if not f:
            errs.append(f"{role} erased")
            continue
        # punctuation of the first and last word of the item is kept
        if edges(first_o)[0] != edges(first_f)[0] and first_f:
            errs.append(f"{role} leading punctuation")
        if edges(last_o)[2] != edges(last_f)[2] and last_f:
            errs.append(f"{role} trailing punctuation")
        if last_f and edges(last_o)[1].endswith(".") and not edges(last_f)[1].endswith("."):
            errs.append(f"{role} trailing full stop")  # a sentence's, or an abbreviation's ('St.' -> 'Cres.')
        if role in ("number", "unit_number", "box_number", "dx_number", "unit_number+number"):
            keep = partial[tid]["keep"] if tid in partial else ""
            if not f.startswith(keep):
                errs.append("kept level code changed")
            else:
                errs += [f"{role}: {x}" for x in number_errors(o[len(keep) :], f[len(keep) :])]
        elif role == "level_number":
            errs += [f"{role}: {x}" for x in level_errors(o, f)]
        elif role == "street_name":
            if fold_name(o) == fold_name(f):
                errs.append("street name stays")
            if style(o) != style(f.split()[0] if len(f.split()) > 1 and False else f) and len(f.split()) == 1:
                errs.append("street name case")
        elif role == "street_type":
            co, cf = canonical_street_type(first_o), canonical_street_type(first_f)
            if cf is None or co == cf:
                errs.append("street type not changed to another canonical type")
            elif style(first_o) != style(edges(first_f)[1]) and len(edges(first_o)[1]) > 1:
                errs.append("street type case")
            elif is_abbreviated_type(first_o) != is_abbreviated_type(first_f):
                has_abbr = any(is_abbreviated_type(x) for x in TYPE_FORMS[cf])
                if has_abbr and cf not in _NO_ABBREVIATION:
                    errs.append("street type style")
        elif role == "street_suffix":
            if fold_name(o) == fold_name(f) or (style(o) != style(f) and len(o) > 1):
                errs.append("street suffix")
        elif role == "building":
            if len(o_words) != len(f_words) or any(
                fold_name(a) == fold_name(b) for a, b in zip(o_words, f_words, strict=False)
            ):
                errs.append("building words")
        elif role in ("box_extra", "dx_exchange"):
            if fold_name(o) == fold_name(f):
                errs.append(f"{role} stays")
        elif role == "suburb" and not overseas:
            if style(o) != style(f) and len(o.replace(" ", "")) > 1:
                errs.append("suburb case")
            if not suburb_rows:
                errs.append("fake suburb not in the fake pool")
            else:
                states = {r.state for r in suburb_rows}
                if mode == "keep" and orig_state is not None and orig_state not in states:
                    errs.append("fake suburb of another state")
                if mode == "replace" and orig_state in states and len(states) == 1:
                    errs.append("replace mode kept the state")
                if orig_state is not None:
                    klass = expected_class(o_words, orig_state, pcs[0] if pcs else None)
                    if klass is not None and mode == "keep" and klass not in {r.klass for r in suburb_rows}:
                        errs.append("fake suburb of another class")
        elif role == "suburb":  # an overseas town: a pool name, no foreign gazetteer
            if fold_name(o) == fold_name(f) or set(fold_name(o).split()) & set(fold_name(f).split()):
                errs.append("town stays")
        elif role == "postcode" and overseas:
            for a, b in zip(first_o, first_f, strict=False):
                if a == b and a.isalnum():
                    errs.append("overseas postcode character stays")
            if len(first_o) != len(first_f) or kinds(first_o) != kinds(first_f):
                errs.append("overseas postcode shape")
        elif role in ("postcode", "state+postcode"):
            digits_o = (
                re.sub(r"\D", "", first_o.translate(str.maketrans("OoIlSB", "001158")))
                if role == "postcode"
                else partial[tid]["change"]
            )
            digits_f = re.sub(r"\D", "", first_f)
            if role == "state+postcode" and not (
                edges(first_f)[1].startswith(partial[tid]["keep"]) or mode == "replace"
            ):
                errs.append("glued state letters changed")
            if not re.search(r"\d{4}", digits_f) or len(digits_f) != 4:
                errs.append("postcode not four digits")
            elif (
                not any(pc_ok(orig_state, digits_o, digits_f, r.state) for r in suburb_rows if r.postcode == digits_f)
                and suburb_rows
            ):
                errs.append(f"postcode rule: {digits_o} -> {digits_f}")
            elif suburb_rows and digits_f not in {r.postcode for r in suburb_rows}:
                errs.append("postcode is not the fake suburb's own")
            if role == "postcode" and not re.fullmatch(r"[\W_]*\d{4}[\W_]*", first_f):
                errs.append("postcode letters")
        else:
            errs.append(f"unhandled role {role}")
    return errs


def whole_fake(case_id: str, unit_id: str, fakes: Mapping[str, str]) -> tuple[str, ...]:
    unit = next(u for e, u in table_units(case_id) if e["id"] == unit_id)
    return tuple(fold_name(" ".join(fakes[t] for t in unit.roles)).split())


def fake_parts(case_id: str, e: dict[str, Any], fakes: Mapping[str, str]) -> dict[str, str]:
    """The canonical parts of a unit's fake (what 'one fake per address' compares): street line, unit, level, place."""
    doc, _ = joined(case_id)
    parts: dict[str, list[str]] = {}
    for li, wi, _text, role in e["words"]:
        f = fakes[doc.tid((li, wi))]
        core = edges(f)[1]
        if role in ("street_name", "building", "suburb", "box_extra", "dx_exchange"):
            parts.setdefault(role, []).append(fold_name(core))
        elif role == "street_type":
            parts["type"] = [canonical_street_type(core) or ""]
        elif role == "street_suffix":
            parts["suffix"] = [fold_name(core)[:1]]
        elif role == "number":
            parts["number"] = [core.upper()]
        elif role == "unit_number+number":
            left, _, right = core.partition("/")
            parts["unit_number"], parts["number"] = [left.upper()], [right.upper()]
        elif role in ("unit_number", "level_number", "box_number", "dx_number"):
            parts[role] = [core.upper()]
        elif role == "postcode" or role == "state+postcode":
            parts["postcode"] = [re.sub(r"\D", "", core)]
    return {k: " ".join(v) for k, v in parts.items()}


_ALL_UNITS = [(c["id"], e["id"]) for c in _CASES for e in c["expect"]["addresses"]]


# ------------------------------------------------------------------------------------------ the case table
@pytest.mark.parametrize("case_id", [c["id"] for c in _CASES])
def test_case_fakes(case_id: str) -> None:
    """README items 2 and 3 for every core and stretch case, under three scope keys and both address_state modes:
    structure words stay (the state too under 'keep'), every other word changes under the fake's constraints, no fake
    word or number repeats an identifying word or number of the case (D1), the case's same_fake groups hold and units
    outside a group get different whole fakes."""
    case = _BY_ID[case_id]
    errs: list[str] = []
    for scope in range(3):
        for mode in ("keep", "replace"):
            fakes = case_fakes(case_id, scope, mode)
            for e, unit in table_units(case_id):
                errs += [f"{scope}/{mode}/{e['id']}: {x}" for x in unit_errors(case_id, e, unit, fakes, mode)]
            errs += [f"{scope}/{mode}: {x}" for x in d1_errors(case_id, fakes)]
            wholes = {e["id"]: whole_fake(case_id, e["id"], fakes) for e, _u in table_units(case_id)}
            groups = case["expect"].get("same_fake", [])
            group_of = {uid: i for i, g in enumerate(groups) for uid in g}
            ids = sorted(wholes)
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    if group_of.get(a, -1) == group_of.get(b, -2):
                        continue
                    if wholes[a] == wholes[b]:
                        errs.append(f"{scope}/{mode}: units {a} and {b} share a whole fake")
            for g in groups:
                parts = [
                    fake_parts(case_id, next(e for e, _u in table_units(case_id) if e["id"] == uid), fakes) for uid in g
                ]
                for p in parts[1:]:
                    errs += [
                        f"{scope}/{mode}: same_fake group differs in {k}"
                        for k in parts[0].keys() & p.keys()
                        if parts[0][k] != p[k]
                    ]
    assert errs == []


def test_case_fakes_replace_mode_moves_each_state_to_one_other_state() -> None:
    """address_state: replace.  All addresses of one original state move to one other state, in every case and under one
    scope key; the written state is the fake locality's, in the original's form."""
    moves: dict[str, set[str]] = {}
    for case in _CASES:
        fakes = case_fakes(case["id"], 0, "replace")
        doc, _ = joined(case["id"])
        for e, _unit in table_units(case["id"]):
            if e.get("overseas"):
                continue
            states = [
                (state_code(t), state_code(fakes[doc.tid((li, wi))])) for li, wi, t, r in e["words"] if r == "state"
            ]
            glued = [
                (
                    re.match(r"[A-Za-z]+", t).group(0).upper(),
                    re.match(r"[A-Za-z.]+", fakes[doc.tid((li, wi))]).group(0).upper(),
                )  # type: ignore[union-attr]
                for li, wi, t, r in e["words"]
                if r == "state+postcode"
            ]
            for o, f in [*states, *glued]:
                if o is not None:
                    moves.setdefault(o, set()).add(f or "")
    assert moves
    assert all(len(v) == 1 and k not in v for k, v in moves.items()), moves


_MATTER_SCOPE = {"MX": 0, "MY": 1}


def test_matters() -> None:
    """One fake per address across the matter, whatever its spelling; a different address in the same suburb gets a
    different whole fake; another matter key gives another fake."""
    scope_of = {cid: _MATTER_SCOPE[m["id"]] for m in _TABLE["matters"] for cid in m["cases"]}
    for m in _TABLE["matters"]:

        def get(ref: str) -> tuple[dict[str, str], tuple[str, ...]]:
            cid, uid = ref.rsplit("/", 1)
            fakes = case_fakes(cid, scope_of[cid])
            e = next(e for e, _u in table_units(cid) if e["id"] == uid)
            return fake_parts(cid, e, fakes), whole_fake(cid, uid, fakes)

        for group in m.get("same_fake", []):
            first, _ = get(group[0])
            for ref in group[1:]:
                parts, _ = get(ref)
                shared = first.keys() & parts.keys()
                assert shared, ref
                assert {k: first[k] for k in shared} == {k: parts[k] for k in shared}, ref
        for a, b in m.get("different_fake", []):
            assert get(a)[1] != get(b)[1], (a, b)


def test_another_scope_key_gives_another_fake_for_every_unit() -> None:
    for case in _CASES:
        for e, _unit in table_units(case["id"]):
            wholes = {whole_fake(case["id"], e["id"], case_fakes(case["id"], s)) for s in range(3)}
            assert len(wholes) == 3, (case["id"], e["id"])


def test_a_form_postcode_follows_its_suburb() -> None:
    """The boxes of a form are separate mentions; each gets the matching part of the unit's ONE fake."""
    seen = 0
    for case in _CASES:
        for e, unit in table_units(case["id"]):
            if not e.get("form") or not any(r == "suburb" for *_x, r in e["words"]):
                continue
            doc, _ = joined(case["id"])
            assert len(unit.mention_ids) > 1
            for scope in range(3):
                fakes = case_fakes(case["id"], scope)
                suburb = " ".join(fakes[doc.tid((li, wi))] for li, wi, _t, r in e["words"] if r == "suburb")
                postcode = next(fakes[doc.tid((li, wi))] for li, wi, _t, r in e["words"] if r == "postcode")
                assert postcode in {r.postcode for r in _GAZ.fake_pool() if fold_name(r.name) == fold_name(suburb)}
            seen += 1
    assert seen >= 3


# ------------------------------------------------------------------------------------------ the rules, one by one
def fake_lines(
    *lines: str,
    scope: int = 0,
    state: str = "keep",
    mentions: Callable[[Doc], list[Mention]] | None = None,
    persons: PersonResolution = _NO_PERSONS,
) -> list[tuple[str, str]]:
    """(original, fake) text of every address unit of a document of single-column lines."""
    doc = Doc(column(*lines))
    found = mentions(doc) if mentions is not None else []
    result = join_addresses(found, doc.pages, _POLICY, band_fraction=0.0)
    others = [Decision(m, ActionKind.SURROGATE) for m in found]
    fakes = assign_fakes(doc, result, scope, state, persons=persons, extra=others)
    return [
        (doc.text_of(list(u.roles)), " ".join(fakes[t] for t in u.roles if fakes.get(t))) for u in result.units.values()
    ]


def spelling_free(text: str) -> tuple[str, ...]:
    """A fake of the NSW address without its state words and postcode, for comparing spellings of one address."""
    text = re.sub(r"(?i)\b(nsw|new south wales)\s*\d{0,4}\b|\bnsw\d{4}\b|\b\d{4}\s*$", " ", text)
    return canonical_words(text)


def one_fake(*lines: str, **kwargs: Any) -> str:
    out = fake_lines(*lines, **kwargs)
    assert len(out) == 1, out
    return out[0][1]


def canonical_words(text: str) -> tuple[str, ...]:
    """A fake with its spelling taken out (case, punctuation, 'St' and 'Street'), for comparing fakes of one address."""
    return tuple(canonical_street_type(w) or fold_name(w) for w in (edges(x)[1] for x in text.split()) if w)


class TestFakePool:
    def test_localities_tsv_is_replaced_by_the_suburb_list(self) -> None:
        folder = loader.resource_root() / "surrogates"
        assert not (folder / "localities.tsv").exists()
        assert "localities" not in (folder / "SHA256SUMS").read_text("utf-8")
        assert loader.verify_resources() > 0

    def test_the_pool_is_the_lists_fake_pool_only(self) -> None:
        pools = load_pools()
        assert pools.localities_in() == _GAZ.fake_pool()
        assert pools.localities_in("NSW", "metro") == _GAZ.fake_pool("NSW", "metro")
        assert all(r.fake_pool and r.primary and r.source == "sal" for r in pools.localities_in())
        cross = {(s, pc) for s, _name, pc in _GAZ.cross_border}
        assert not any((r.state, r.postcode) in cross for r in pools.localities_in() if r.in_state_range is False)

    def test_a_fake_locality_is_always_a_pool_row_of_its_state(self) -> None:
        f = factory()
        pool = load_pools()
        for state in ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"):
            for i in range(25):
                loc = f.locality(f"original {i}", state)
                assert loc in pool.localities_in(state) and loc.fake_pool

    def test_a_class_is_kept_when_the_pool_has_it(self) -> None:
        f = factory()
        assert all(f.locality(f"o{i}", "NSW", "metro").metro for i in range(20))
        assert not any(f.locality(f"o{i}", "NSW", "regional").metro for i in range(20))


class TestExclusions:
    ex = Exclusions.of(["Wattle", "Grove", "Mt Druitt", "St", "Kelvin"], ["2", "02", "275"])

    @pytest.mark.parametrize(
        ("word", "ok"),
        [
            ("Wattle", False),  # equal
            ("WATTLE", False),  # case folded
            ("Wattlebird", False),  # begins with an identifying word of 3+ letters
            ("Bluegrove", False),  # contains one
            ("Kelvin", False),
            ("Kelvyn", False),  # one edit from a word of 5+ letters
            ("Mount Annan", False),  # 'Mt' reads as 'Mount'
            ("Druitt", False),
            ("Acacia", True),
            ("Street", True),  # 'St' (2 letters) only blocks the equal word
            ("St", False),
            ("Grov", False),  # one edit from "grove" (5 letters)
            ("Gro", True),  # two edits
        ],
    )
    def test_words(self, word: str, ok: bool) -> None:
        assert self.ex.word_ok(word) is ok

    def test_numbers_are_whole_digit_runs(self) -> None:
        assert not self.ex.number_ok("2")
        assert not self.ex.number_ok("02")
        assert not self.ex.number_ok("12/2")
        assert self.ex.number_ok("27")
        assert self.ex.number_ok("2750")  # 275 is not a run of 2750
        assert not self.ex.number_ok("275")


class TestPostcodeRule:
    @pytest.mark.parametrize(
        ("state", "original", "fake", "ok"),
        [
            ("NSW", "2750", "2196", True),  # the state's '2' may stay
            ("NSW", "2750", "2070", False),  # '0' ... position 3 is '0' in both
            ("NSW", "2750", "2750", False),
            ("NSW", "2750", "3196", True),
            ("NSW", "1300", "2751", True),  # a PO-box block shares no digit: '1' vs '2', '3' vs '7', ...
            ("NSW", "1300", "2341", False),  # position 1 is '3' in both
            ("NSW", "3644", "2341", False),  # a cross-border code: '3' is not NSW's prefix, so '4' at 2 clashes
            ("NSW", "3644", "2127", True),
            ("NT", "0870", "0845", True),  # '08' is fixed
            ("NT", "0870", "0875", False),
            ("VIC", "3000", "3552", True),
            ("ACT", "2601", "2915", True),
        ],
    )
    def test_original_prefix(self, state: str, original: str, fake: str, ok: bool) -> None:
        assert postcode_ok(state, original, fake) is ok

    def test_another_state_may_repeat_the_digits_it_fixes(self) -> None:
        assert not postcode_ok("NSW", "2830", "0872")  # the second digit is 8 in both
        assert postcode_ok("NSW", "2830", "0872", "NT")

    def test_an_original_outside_its_states_prefix_may_share_the_digits_the_state_fixes(self) -> None:
        """A cross-border NT postcode (4825): every NT postcode starts '08', and '8' is the original's second digit."""
        assert postcode_ok("NT", "4825", "0870") and not postcode_ok("NT", "4825", "0820")
        assert not postcode_ok("NSW", "3644", "2341")  # NSW fixes only the '2': the '4' at 2 still clashes

    @pytest.mark.parametrize("scope", range(len(_KEYS)))
    def test_a_cross_border_nt_address_stays_in_the_nt(self, scope: int) -> None:
        fake = one_fake("Address: 5 King Street, Alpurrurulam NT 4825", scope=scope)
        assert re.search(r"\bNT 08\d\d$", fake), fake


class TestNumbers:
    @pytest.mark.parametrize(
        "value",
        [
            "5",
            "9",
            "12",
            "123",
            "4600",
            "12345",
            "12A",
            "12a",
            "5B",
            "A",
            "c",
            "02",
            "1234567",
            "5/3",
            "02/53",
            "7-9",
            "1-3",
        ],
    )
    def test_every_digit_changes_at_its_position_and_the_shape_stays(self, value: str) -> None:
        f = factory()
        for i in range(40):
            out = f.address_number(value, f"key{i}")
            assert number_errors(value, out) == [], (value, out)

    def test_a_range_keeps_its_gap_when_it_can(self) -> None:
        f = factory()
        for i in range(60):
            a, b = f.address_number("12-14", f"key{i}").split("-")
            assert int(b) - int(a) == 2 and len(a) == len(b) == 2

    @pytest.mark.parametrize(("value", "digits"), [("8-10", (1, 2)), ("98-104", (2, 3)), ("9-12", (1, 2))])
    def test_a_range_across_digit_counts_keeps_each_digit_count(self, value: str, digits: tuple[int, int]) -> None:
        f = factory()
        for i in range(60):
            out = f.address_number(value, f"key{i}")
            a, b = out.split("-")
            assert (len(a), len(b)) == digits and int(a) < int(b) and number_errors(value, out) == []

    def test_a_letter_suffix_stays_a_letter_of_the_same_case(self) -> None:
        f = factory()
        assert all(re.fullmatch(r"\d\dA?", f.address_number("12A", f"k{i}")) is None for i in range(20))
        assert all(re.fullmatch(r"\d\d[A-Z]", f.address_number("12A", f"k{i}")) for i in range(20))
        assert all(re.fullmatch(r"\d\d[a-z]", f.address_number("12a", f"k{i}")) for i in range(20))

    def test_a_choice_is_keyed_and_the_exclusions_hold(self) -> None:
        f = factory()
        assert f.address_number("12", "k") == factory().address_number("12", "k")
        assert len({f.address_number("12345", f"k{i}") for i in range(30)}) > 25
        ex = Exclusions.of([], [str(n) for n in range(10, 100) if n % 7])
        assert all(int(f.address_number("12", f"k{i}", ex)) % 7 == 0 for i in range(30))

    def test_levels_are_another_level_without_a_digit_at_its_position(self) -> None:
        f = factory()
        for n in [*range(1, 26), 30, 99]:
            for i in range(10):
                out = f.level_number(str(n), f"k{i}")
                assert level_errors(str(n), out) == [], (n, out)
                assert 1 <= int(out) <= 25 and int(out) != n
                assert all(a != b for a, b in zip(out, str(n), strict=False))
        assert f.level_number("2", "k") != "2" and not f.level_number("2", "k").startswith("2")

    def test_an_ordinal_stays_an_ordinal(self) -> None:
        f = factory()
        for i in range(20):
            assert re.fullmatch(r"\d(st|nd|rd|th)", f.level_number("2nd", f"k{i}"))
            assert re.fullmatch(r"\d(ST|ND|RD|TH)", f.level_number("2ND", f"k{i}"))


class TestTypesNamesAndStates:
    def test_a_street_type_changes_in_the_originals_style(self) -> None:
        f = factory()
        for i in range(30):
            t = f.street_type("St", f"k{i}")
            assert canonical_street_type(t) not in (None, "STREET") and edges(t)[1][0].isupper()
            assert (
                f.street_type("ST", f"k{i}") == f.street_type("ST", f"k{i}") and f.street_type("ST", f"k{i}").isupper()
            )
            assert f.street_type("street", f"k{i}").islower()
            assert not is_abbreviated_type(f.street_type("Street", f"k{i}"))
            assert f.street_type("Streets", f"k{i}").lower().endswith(("s", "es"))

    def test_the_same_type_whatever_its_spelling(self) -> None:
        f = factory()
        for i in range(30):
            assert canonical_street_type(f.street_type("St", f"k{i}")) == canonical_street_type(
                f.street_type("Street", f"k{i}")
            )

    def test_an_untyped_street_gets_a_common_type_written_in_full(self) -> None:
        f = factory()
        types = {f.street_type(None, f"k{i}") for i in range(40)}
        assert types <= {"Street", "Road", "Avenue", "Drive", "Court", "Place", "Crescent", "Lane"} and len(types) > 3

    def test_a_type_is_never_one_a_unit_of_the_document_writes(self) -> None:
        f = factory()
        ex = Exclusions.of(["Grove", "Kelvin"], [])
        assert all(canonical_street_type(f.street_type("Rd", f"k{i}", ex)) != "GROVE" for i in range(60))
        only_lane = Exclusions.of([w for forms in TYPE_FORMS.values() for w in forms if forms[0] != "lane"], [])
        assert f.street_type("Rd", "k", only_lane) in ("Ln", "Lane")

    def test_an_untyped_street_takes_any_fake_type_when_the_document_writes_every_common_one(self) -> None:
        """'8 Broadway' in a document whose other streets are a Street, Road, ... and Lane: no common type is left, so
        the type comes from the wider pool rather than repeat one of them (which V5 would hold)."""
        f = factory()
        common = Exclusions.of(
            [
                w
                for c in ("STREET", "ROAD", "AVENUE", "DRIVE", "COURT", "PLACE", "CRESCENT", "LANE")
                for w in TYPE_FORMS[c]
            ],
            [],
        )
        for i in range(30):
            fake = f.street_type(None, f"k{i}", common)
            assert common.word_ok(fake) and canonical_street_type(fake) is not None, fake

    def test_a_suffix_is_another_directional_in_the_originals_style(self) -> None:
        f = factory()
        for i in range(30):
            assert f.street_suffix("North", f"k{i}") in ("South", "East", "West")
            assert f.street_suffix("NORTH", f"k{i}") in ("SOUTH", "EAST", "WEST")
            assert f.street_suffix("Nth", f"k{i}") in ("Sth", "E", "W")
            assert f.street_suffix("N", f"k{i}") in ("S", "E", "W")

    def test_a_building_name_changes_every_word_and_the_building_word(self) -> None:
        f = factory()
        for i in range(30):
            name, word = f.building_name(["Westpoint", "Tower"], f"k{i}")
            assert name.lower() != "westpoint" and word in (
                "House",
                "Chambers",
                "Centre",
                "Building",
                "Complex",
                "Arcade",
                "Plaza",
            )
            assert f.building_name(["Kilbride", "Station"], f"k{i}")[1] in (
                "Farm", "Homestead", "Estate", "Vineyard", "Orchard", "Downs", "Park"
            )  # fmt: skip
        assert f.building_name(["RIVERSIDE", "MEDICAL", "CHAMBERS"], "k")[0].isupper()

    @pytest.mark.parametrize(
        ("original", "code", "written"),
        [
            ("NSW", "VIC", "VIC"),
            ("NSW,", "QLD", "QLD"),
            ("N.S.W.", "VIC", "V.I.C."),
            ("N.S.W", "WA", "W.A"),
            ("New South Wales", "VIC", "Victoria"),
            ("NEW SOUTH WALES", "QLD", "QUEENSLAND"),
            ("Vic", "QLD", "Qld"),
            ("Qld.", "TAS", "Tas."),
            ("Vic", "NSW", "NSW"),  # a mixed-case original: only Vic, Qld and Tas are written so
            ("Tas.", "SA", "SA."),
            ("Victoria", "NT", "Northern Territory"),
            ("ACT", "SA", "SA"),
        ],
    )
    def test_state_form(self, original: str, code: str, written: str) -> None:
        assert SurrogateFactory.state_form(original, code) == written

    def test_overseas_postcodes_change_every_character_in_kind(self) -> None:
        f = factory()
        for value in ("1010", "OX1 4AA", "K1A 0B1", "SW1A 1AA"):
            out = f.overseas_postcode(value, "k")
            assert kinds(value) == kinds(out) and len(value) == len(out)
            assert all(a != b for a, b in zip(value, out, strict=True) if a.isalnum())


class TestNeverRaises:
    """An empty choice widens (the class, then the state) instead of raising 'surrogate pool exhausted'."""

    def test_locality_widens_from_class_to_state_to_any_state(self) -> None:
        act = [r.name for r in _GAZ.fake_pool("ACT")]
        f = factory(tuple(act))
        loc = f.locality("x", "ACT", "metro")  # every ACT name is an original: widens to another state
        assert loc.name not in act
        none = f.locality("x", "ACT", "metro", accept=lambda _loc: False)  # nothing accepted: the keyed last resort
        assert none in load_pools().localities_in()
        assert factory().locality("x", "ACT", "regional").state == "ACT"  # ACT has no regional rows: widens the class

    def test_street_names_never_run_out(self) -> None:
        pools = load_pools()
        f = factory(pools.street_names)
        assert f.street_name("k") in pools.street_names
        assert f.street_name("k", lambda _w: False) in pools.street_names

    def test_the_assigner_survives_a_pool_cut_to_two_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = _GAZ.fake_pool("ACT", "metro")[:2]
        real = pools_module.locality_pool

        def cut(state: str | None = None, klass: str | None = None) -> Any:
            return rows if state == "ACT" and klass == "metro" else real(state, klass)

        monkeypatch.setattr(pools_module, "locality_pool", cut)
        lines = [f"{n} Smith Street, Gungahlin ACT 2912" for n in (3, 5, 7, 9, 11, 13, 15, 17)]
        out = fake_lines(*lines)
        assert len(out) == 8 and all(f for _o, f in out)
        assert len({f.split(",")[0] for _o, f in out}) == 8  # different originals, different fakes


class TestNoPlaceFakeContainsAnOriginal:
    """V2 reads a removed value of 4+ letters inside any word as a leak, so a fake suburb or street name never contains
    one ('Hill' in 'Hillsdale', 'Orange' in 'Orangeville')."""

    ORIGINALS = ("Hill", "Orange", "Wood", "Park")

    def test_fake_localities_and_street_names_contain_no_original(self) -> None:
        keys = [match_key(o) for o in self.ORIGINALS]
        pools = load_pools()
        # The pools hold names that contain one, so the rule is what keeps them out.
        assert any(k in match_key(loc.name) for loc in pools.localities_in("NSW") for k in keys)
        assert any(k in match_key(w) for w in pools.street_names for k in keys)
        f = factory(self.ORIGINALS)
        for i in range(300):
            loc = f.locality("x", "NSW", key=f"k{i}")
            street = f.street_name(f"k{i}")
            assert loc.state == "NSW"
            assert not any(k in match_key(loc.name) for k in keys), loc.name
            assert not any(k in match_key(street) for k in keys), street

    def test_an_original_of_three_letters_does_not_narrow_the_choice(self) -> None:
        pools = load_pools()
        f, plain = factory(("Ash",)), factory()
        assert [f.street_name(f"k{i}") for i in range(50)] == [plain.street_name(f"k{i}") for i in range(50)]
        assert any("ash" in match_key(w) for w in pools.street_names)


class TestAddressFakes:
    def test_variants_of_one_address_give_one_fake(self) -> None:
        spellings = [
            "12 Smith Street, Penrith NSW 2750",
            "12 SMITH STREET PENRITH NSW 2750",
            "12 Smith St., Penrith, NSW 2750",
            "12 smith st penrith nsw 2750",
            "12 Smith Street, Penrith New South Wales 2750",
            "12 Smith Street Penrith NSW2750",
            "12 Smith Street, Penrith 2750",
            "12 Smith Street, Penrith",
        ]
        fakes = {spelling_free(one_fake(s)) for s in spellings}
        assert len(fakes) == 1, fakes

    def test_a_street_line_seen_without_its_locality_gives_the_same_fake_street_line(self) -> None:
        full = one_fake("Level 4, 12 Smith Street, Penrith NSW 2750")
        alone = one_fake("Level 4, 12 Smith Street.")
        assert canonical_words(full)[:5] == canonical_words(alone)[:5]  # 'level 4', number, name, type

    def test_another_scope_key_gives_another_fake(self) -> None:
        fakes = {one_fake("12 Smith Street, Penrith NSW 2750", scope=s) for s in range(3)}
        assert len(fakes) == 3

    def test_two_addresses_in_one_suburb_get_independent_fake_suburbs(self) -> None:
        lines = [f"{n} Smith Street, Penrith NSW 2750" for n in range(10, 22)]
        suburbs = [edges(f.split(",")[1].split(" NSW")[0])[1] for _o, f in fake_lines(*lines)]
        assert len(set(suburbs)) >= 7

    def test_the_same_address_in_a_document_that_names_its_fake_suburb_elsewhere_keeps_its_fake(self) -> None:
        first = one_fake("12 Smith Street, Penrith NSW 2750")
        suburb = edges(first.split(",")[1])[1].rsplit(" NSW", 1)[0].strip()
        with_prose = fake_lines(
            "12 Smith Street, Penrith NSW 2750", "", f"She grew up in {suburb}.", f"{suburb} was home."
        )
        assert [f for _o, f in with_prose] == [first]

    def test_a_fake_never_repeats_a_word_of_another_unit(self) -> None:
        """D1: a unit that writes the fake's suburb as its street name changes the fake."""
        first = one_fake("12 Smith Street, Penrith NSW 2750")
        suburb = edges(first.split(",")[1])[1].rsplit(" NSW", 1)[0].strip()
        two = fake_lines("12 Smith Street, Penrith NSW 2750", f"9 {suburb} Road, Ryde NSW 2112")
        assert suburb.lower() not in two[0][1].lower() and suburb.lower() not in two[1][1].lower()
        assert two[0][1] != first

    def test_two_different_originals_never_get_one_whole_fake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even when the draws of two addresses coincide, the second is redrawn."""
        lines = [f"{n} Smith Street, Penrith NSW 2750" for n in (12, 14)]
        real = AddressFaker._fake

        def same(self: AddressFaker, u: UnitView, redraw: int) -> dict[str, str]:
            out = real(self, u, redraw)
            if redraw == 0:
                out = {t: "X" if i else "" for i, t in enumerate(out)}
            return out

        monkeypatch.setattr(AddressFaker, "_fake", same)
        a, b = (f for _o, f in fake_lines(*lines))
        assert a != b

    def test_the_result_does_not_depend_on_the_order_of_the_units(self) -> None:
        doc = Doc(
            column(
                "12 Smith Street, Penrith NSW 2750", "", "9 Hill Road, Ryde NSW 2112", "", "PO Box 9, Katoomba NSW 2780"
            )
        )
        result = join_addresses([], doc.pages, _POLICY, band_fraction=0.0)
        texts = {t: tok for t, tok in ((t.token_id, t.text) for line in doc.lines for t in line.tokens)}
        views = [view_of(u, texts) for u in result.units.values()]
        factory_a = SurrogateFactory(Chooser(_KEYS[0]), [])
        a = AddressFaker(factory_a, views).fakes
        b = AddressFaker(SurrogateFactory(Chooser(_KEYS[0]), []), list(reversed(views))).fakes
        assert a == b

    def test_a_form_postcode_follows_its_suburb_across_boxes(self) -> None:
        case = _BY_ID["form-claim-grid"]
        doc, result = joined("form-claim-grid")
        assert len(result.units) == 1 and len(next(iter(result.units.values())).mention_ids) > 1
        fakes = case_fakes("form-claim-grid", 1)
        words = case["expect"]["addresses"][0]["words"]
        suburb = next(fakes[doc.tid((li, wi))] for li, wi, _t, r in words if r == "suburb")
        postcode = next(fakes[doc.tid((li, wi))] for li, wi, _t, r in words if r == "postcode")
        assert (suburb, postcode) in {(r.name, r.postcode) for r in _GAZ.fake_pool("NSW")}

    def test_a_gpo_box_becomes_a_po_box_unless_the_fake_suburb_is_a_capital(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert one_fake("GPO Box 9, Gladesville NSW 2111").startswith("PO Box ")
        assert one_fake("G.P.O. Box 9, Gladesville NSW 2111").startswith("P.O. Box ")
        assert one_fake("GPO BOX 9 GLADESVILLE NSW 2111").startswith("PO BOX ")
        assert one_fake("GPO Box 9.").startswith("GPO Box ")  # no fake suburb: stays
        sydney = next(r for r in _GAZ.fake_pool("NSW") if r.folded == "sydney")
        real = pools_module.locality_pool
        monkeypatch.setattr(
            pools_module, "locality_pool", lambda s=None, k=None: (sydney,) if s == "NSW" else real(s, k)
        )
        assert one_fake("GPO Box 9, Gladesville NSW 2111").startswith("GPO Box ")

    def test_replace_mode_moves_all_addresses_of_a_state_to_one_other_state(self) -> None:
        lines = [f"{n} Smith Street, Penrith NSW 2750" for n in (12, 14, 16, 18)] + [
            "5 Hill Road, Cairns QLD 4870",
            "7 Hill Road, Ipswich QLD 4305",
        ]
        out = fake_lines(*lines, state="replace")
        states = [f.split()[-2] for _o, f in out]
        assert len(set(states[:4])) == 1 and len(set(states[4:])) == 1
        assert states[0] != "NSW" and states[4] != "QLD"
        fake_postcodes = [f.split()[-1] for _o, f in out]
        for state, pc in zip(states, fake_postcodes, strict=True):
            assert any(r.postcode == pc and r.state == state for r in _GAZ.fake_pool())

    def test_replace_mode_writes_the_state_in_the_originals_form(self) -> None:
        assert one_fake("12 Smith Street, Penrith New South Wales 2750", state="replace").split()[-4:-1] != [
            "New", "South", "Wales"
        ]  # fmt: skip
        assert re.search(
            r"\b[A-Z]{2,3}2\d{3}\b|\b[A-Z]{2,3}\d{4}\b", one_fake("12 Smith Street, Penrith NSW2750", state="replace")
        )
        assert re.fullmatch(
            r"[A-Z](\.[A-Z])+\.", one_fake("12 Smith Street, Penrith N.S.W. 2750", state="replace").split()[-2]
        )

    def test_a_missing_state_is_inferred_to_choose_the_suburb_and_never_written(self) -> None:
        out = one_fake("12 Smith Street, Penrith")
        assert out.split(",")[1].strip() in {r.name for r in _GAZ.fake_pool("NSW")}
        assert not re.search(r"\b(NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\b|\d{4}", out)

    def test_a_place_only_unit_is_keyed_by_the_name_heading_its_block(self) -> None:
        def under(name: str, scope: int = 0) -> str:
            def orgs(doc: Doc) -> list[Mention]:
                return [doc.span(EntityType.ORGANISATION, 0, 0, len(name.split()) - 1)]

            return one_fake(name, "PENRITH NSW 2750", mentions=orgs, scope=scope)

        names = ["Smith Lawyers", "Jones Partners", "Taylor Legal", "Brown Medical", "Walker Clinic", "Hill Surgery"]
        suburbs = {under(n) for n in names}
        assert len(suburbs) >= 4  # independent per heading
        assert under("Smith Lawyers") == under("Smith Lawyers")
        assert under("Smith Lawyers Pty Ltd") == under("Smith Lawyers")  # generic words are not part of the name
        assert under("Smith Lawyers") != under("Smith Lawyers", scope=1)
        assert one_fake("PENRITH NSW 2750") == one_fake("PENRITH NSW 2750", scope=0)

    def test_a_person_heads_a_block(self) -> None:
        def under(name: str) -> str:
            def person(doc: Doc) -> list[Mention]:
                return [doc.span(EntityType.PERSON, 0, 0, len(name.split()) - 1)]

            return one_fake(name, "PENRITH NSW 2750", mentions=person)

        assert (
            len({under(n) for n in ("Jane Citizen", "Peter Walsh", "Anna Lee", "Tom Parker", "Sue Brown", "Ann Ng")})
            >= 4
        )

    @pytest.mark.parametrize("scope", range(len(_KEYS)))
    def test_a_person_heading_written_with_an_initial_keys_the_place_as_the_full_name(self, scope: int) -> None:
        def under(name: str) -> str:
            lines = (name, "PENRITH NSW 2750")
            n = len(name.split()) - 1

            def person(doc: Doc) -> list[Mention]:
                return [doc.span(EntityType.PERSON, 0, 0, n)]

            doc = Doc(column(*lines))
            persons = resolve_persons(person(doc), build_views(doc.pages)[ViewKind.LINE])
            return one_fake(*lines, mentions=person, persons=persons, scope=scope)

        assert under("John Smith") == under("J. Smith") == under("J Smith") == under("Mr John Smith")

    @pytest.mark.parametrize("scope", range(len(_KEYS)))
    @pytest.mark.parametrize("place", ["Richmond NSW 2753", "Kew NSW 2440", "Penrith NSW 2750"])
    def test_a_repeat_without_a_state_takes_the_state_of_the_address_it_repeats(self, place: str, scope: int) -> None:
        """'Richmond' alone would be read as VIC (the name's most populous row): the join links the repeat to the full
        address, so both get one fake suburb, in NSW."""
        suburb = place.split(" NSW")[0]
        out = fake_lines(f"12 Smith Street, {place}", "", f"Please reply to 12 Smith Street, {suburb}.", scope=scope)
        assert len(out) == 2, out
        full, repeat = (f for _o, f in out)
        assert repeat.endswith(".") and full.startswith(repeat[:-1] + " NSW"), (full, repeat)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("12 Smith Street, Balwyn North VIC 3104", "12 Smith Street, North Balwyn VIC 3104"),
            ("Locked Bag 7, Penrith NSW 2750", "Locked Mail Bag 7, Penrith NSW 2750"),
            ("Private Bag 7, Penrith NSW 2750", "PMB 7, Penrith NSW 2750"),
            ("12 Smith-Jones Road, Penrith NSW 2750", "12 Smith Jones Road, Penrith NSW 2750"),
            ("Level 4, 12 Smith Street, Penrith NSW 2750", "4th Floor, 12 Smith Street, Penrith NSW 2750"),
        ],
    )
    def test_spellings_of_one_address_get_one_fake_across_a_matter(self, a: str, b: str) -> None:
        """One address in two documents of a matter (one scope key), written two ways: one fake, in each spelling."""
        spelling = {"level", "floor", "locked", "private", "mail", "bag", "pmb"}

        def canon(fake: str) -> tuple[list[str], list[str]]:
            words = [w for w in tokens_of(fake) if not w.isdigit() and w not in spelling | {"st", "nd", "rd", "th"}]
            return [d.lstrip("0") for d in re.findall(r"\d+", fake)], words

        for scope in range(len(_KEYS)):
            assert canon(one_fake(a, scope=scope)) == canon(one_fake(b, scope=scope)), scope

    def test_unread_words_of_a_forced_unit_are_pool_words_and_numbers(self) -> None:
        """A unit built by the pairing can hold words the grammar did not read: they are identifying, so they change."""
        words = tuple(
            UnitWord(f"t{i}", text, None, place=i >= 4)
            for i, text in enumerate(["Lot", "12B", "Hidden", "Rise,", "Faketown", "NSW", "2750"])
        )
        faker = AddressFaker(factory(), [UnitView("U", AddressKey(), words)])
        out = faker.fakes["U"]
        assert all(out[w.token_id] != w.text for w in words if w.text != "NSW")
        assert out["t5"] == "NSW" and re.fullmatch(r"\d{2}[A-Z]", out["t1"]) and re.fullmatch(r"\d{4}", out["t6"])

    def test_an_overseas_unit_keeps_its_shape_and_its_country(self) -> None:
        out = one_fake("12 Queen Street, Auckland 1010, New Zealand.")
        assert out.endswith("New Zealand.") and re.fullmatch(
            r"\d\d? [A-Z][a-z]+ \w+, [A-Z][a-z]+ \d{4}, New Zealand\.", out
        ), out
        uk = one_fake("10 High Street, Oxford OX1 4AA, United Kingdom")
        assert re.fullmatch(r"\d\d? \w+ \w+, \w+ [A-Z]{2}\d [0-9][A-Z]{2}, United Kingdom", uk), uk


class TestDxAndExchanges:
    def test_a_dx_number_follows_the_number_rule_and_the_exchange_is_a_locality_of_the_same_state(self) -> None:
        out = fake_lines("PO Box 55, Penrith NSW 2751", "", "DX 8012 Penrith")
        assert len(out) == 2
        dx = out[1][1].split()
        assert dx[0] == "DX" and number_errors("8012", dx[1]) == []
        assert " ".join(dx[2:]) in {r.name for r in _GAZ.fake_pool("NSW")}
        assert "penrith" not in out[1][1].lower()


class TestPartsMode:
    """A STREET_ADDRESS mention the join did not lay out (``addresses: parts``, or a caller that passes no join) is
    replaced part by part, as before, from the new pool and without the uniqueness pool."""

    @staticmethod
    def assign_plain(text: str, kind: EntityType = EntityType.STREET_ADDRESS, scope: int = 0) -> tuple[str, ...]:
        doc = Doc(column(text))
        mention = doc.span(kind, 0, 0, len(text.split()) - 1)
        tokens = {t.token_id: t for line in doc.lines for t in line.tokens}
        fac = SurrogateFactory(Chooser(_KEYS[scope]), forbidden_keys([mention], _NO_PERSONS))
        out = SurrogateAssigner(fac, _NO_PERSONS, tokens).assign([Decision(mention, ActionKind.SURROGATE)])
        return out[mention.mention_id].tokens

    def test_a_street_address_without_a_layout_takes_its_suburb_and_postcode_from_the_suburb_list(self) -> None:
        out = self.assign_plain("42 Wattle Grove Road, PENRITH NSW 2750")
        assert out[0] != "42" and out[1:3] != ("Wattle", "Grove") and out[3] == "Road,"
        assert out[5] == "NSW"
        rows = [r for r in _GAZ.fake_pool("NSW") if fold_name(r.name) == fold_name(out[4])]
        assert rows and out[4].isupper() and out[6] in {r.postcode for r in rows}

    def test_the_locality_branch_keeps_the_originals_case(self) -> None:
        for text in ("PENRITH", "Penrith", "penrith"):
            (fake,) = self.assign_plain(text, EntityType.LOCALITY)
            assert style(fake) == style(text)
            assert fake.lower() != "penrith"

    def test_a_fill_keeps_the_case_of_the_words_it_replaces(self) -> None:
        out = self.assign_plain("12 SMITH STREET, Penrith NSW 2750")
        assert out[1].isupper() and out[1] != "SMITH"

    def test_many_street_names_do_not_exhaust_the_pool(self) -> None:
        lines = [f"{n} Street{n}Name Road, Penrith NSW 2750" for n in range(1, 120)]
        doc = Doc(column(*lines))
        mentions = [doc.span(EntityType.STREET_ADDRESS, i, 0, 5) for i in range(len(lines))]
        tokens = {t.token_id: t for line in doc.lines for t in line.tokens}
        fac = SurrogateFactory(Chooser(_KEYS[0]), forbidden_keys(mentions, _NO_PERSONS))
        out = SurrogateAssigner(fac, _NO_PERSONS, tokens).assign([Decision(m, ActionKind.SURROGATE) for m in mentions])
        assert len(out) == len(mentions)

    def test_the_assigner_reports_the_fakes_it_drew(self) -> None:
        doc, result = joined("one-line-basic")
        decisions = [
            Decision(m, ActionKind.SURROGATE) for m in result.mentions if m.entity_type is EntityType.STREET_ADDRESS
        ]
        tokens = {t.token_id: t for line in doc.lines for t in line.tokens}
        fac = SurrogateFactory(Chooser(_KEYS[0]), forbidden_keys([d.mention for d in decisions], _NO_PERSONS))
        assigner = SurrogateAssigner(fac, _NO_PERSONS, tokens, layouts=result.layouts, units=result.units)
        assert assigner.address_fakes == {}
        sur = assigner.assign(decisions)
        assert set(assigner.address_fakes) == set(result.units)
        for d in decisions:
            layout = result.layouts[d.mention.mention_id]
            assert sur[d.mention.mention_id].tokens == tuple(
                assigner.address_fakes[layout.unit_id][t] for t in d.mention.token_ids
            )
