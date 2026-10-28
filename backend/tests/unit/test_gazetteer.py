"""The packaged locality list (suburb list v2): its integrity check, name folding, aliases, cross-border pairs,
postcodes, classes and the fake pool.  The list is real public data (ABS, CC BY 4.0); no document text is involved."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.resources import loader
from mlredact.text import gazetteer as gaz_module
from mlredact.text.gazetteer import (
    RANGES,
    STATE_CODES,
    edit1,
    fold_name,
    in_range,
    load_gazetteer,
    name_variants,
    range_states,
)

GAZ = load_gazetteer()
FOLDER = loader.resource_root() / "gazetteer"


def test_the_list_is_cached_and_complete() -> None:
    assert load_gazetteer() is GAZ
    assert len(GAZ.rows) == 15_707
    assert sum(1 for r in GAZ.rows if r.source == "sal") == 15_389
    assert sum(1 for r in GAZ.rows if r.source == "official_locality") == 318
    assert len(GAZ.aliases) == 17
    assert len(GAZ.cross_border) == 55
    assert {r.state for r in GAZ.rows} == set(STATE_CODES)


def test_the_resource_folder_ships_with_its_attribution_and_checksums() -> None:
    names = {p.name for p in FOLDER.iterdir()}
    assert {"au_localities.csv", "aliases.csv", "cross_border.csv", "ATTRIBUTION.txt", "PROVENANCE.md"} <= names
    assert "SHA256SUMS" in names
    assert "CC BY 4.0" in (FOLDER / "ATTRIBUTION.txt").read_text("utf-8")
    assert loader.verify_resources() >= 5  # the generic check covers the folder too


def test_a_changed_file_fails_the_integrity_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "gazetteer"
    shutil.copytree(FOLDER, copy)
    (copy / "aliases.csv").write_text("alias,state,locality,postcode,reason\n", encoding="utf-8")
    monkeypatch.setattr(gaz_module, "resource_root", lambda: tmp_path)
    with pytest.raises(EnvironmentErrorMl) as err:
        load_gazetteer.__wrapped__()
    assert err.value.code is ReasonCode.RESOURCE_HASH_MISMATCH


def test_a_missing_checksum_entry_fails_the_integrity_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "gazetteer"
    shutil.copytree(FOLDER, copy)
    sums = [ln for ln in (copy / "SHA256SUMS").read_text("utf-8").splitlines() if "cross_border" not in ln]
    (copy / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    monkeypatch.setattr(gaz_module, "resource_root", lambda: tmp_path)
    with pytest.raises(EnvironmentErrorMl):
        load_gazetteer.__wrapped__()


@pytest.mark.parametrize(
    ("written", "folded"),
    [
        ("St. Mary’s-Bay", "st marys bay"),
        ("ST IVES", "st ives"),
        ("Saint Ives", "st ives"),
        ("Mt Druitt", "mount druitt"),
        ("MT. DRUITT", "mount druitt"),
        ("Pt Lonsdale", "point lonsdale"),
        ("Nth Ryde", "north ryde"),
        ("Sth Hurstville", "south hurstville"),
        ("O'Connor", "oconnor"),
        ("Port-Kembla", "port kembla"),
    ],
)
def test_name_folding(written: str, folded: str) -> None:
    assert fold_name(written) == folded


def test_reversed_directionals_fold() -> None:
    assert name_variants("north balwyn") == ["north balwyn", "balwyn north"]
    assert name_variants("balwyn north") == ["balwyn north", "north balwyn"]
    assert name_variants("penrith") == ["penrith"]
    assert GAZ.states_of(["North", "Balwyn"]) == {"VIC"}
    assert GAZ.states_of(["Balwyn", "North"]) == {"VIC"}


def test_lookups_are_per_state() -> None:
    assert GAZ.states_of(["Penrith"]) == {"NSW"}
    assert GAZ.states_of(["Penrith"], state="NSW") == {"NSW"}
    assert GAZ.states_of(["Penrith"], state="VIC") == frozenset()
    assert {"NSW", "VIC", "QLD"} <= GAZ.states_of(["Richmond"])
    assert GAZ.states_of(["Richmond"], state="TAS") == {"TAS"}
    assert GAZ.states_of([]) == frozenset()
    assert GAZ.states_of(["Smith", "Street"]) == frozenset()
    assert GAZ.is_locality(["St", "Ives"], "NSW")
    assert GAZ.is_locality(["Mount", "Victoria"], "NSW")
    assert not GAZ.is_locality(["Victoria"], "NSW")


def test_both_sources_are_names_of_their_state() -> None:
    official = next(r for r in GAZ.rows if r.source == "official_locality")
    assert GAZ.states_of([official.folded], state=official.state) == {official.state}
    assert not official.fake_pool


def test_aliases_name_the_postal_towns_the_abs_lacks() -> None:
    assert GAZ.states_of(["Brisbane"]) == {"QLD"}
    assert GAZ.states_of(["Canberra"]) == {"ACT"}
    assert GAZ.states_of(["Canberra", "City"]) == {"ACT"}
    assert GAZ.states_of(["Darwin"]) == {"NT"}
    assert GAZ.states_of(["Townsville"], state="QLD") == {"QLD"}
    assert GAZ.states_of(["Ballarat"], state="VIC") == {"VIC"}
    assert GAZ.postcodes_of(["Brisbane"], "QLD") == {4000}
    assert GAZ.klass_of(["Brisbane"], "QLD") == "metro"
    assert GAZ.klass_of(["Townsville"], "QLD") == "regional"


def test_cross_border_pairs_and_ranges() -> None:
    assert GAZ.ok_pair("NSW", 2619)  # Jerrabomberra: real, inside the ACT's range
    assert GAZ.ok_pair("ACT", 2620)  # Tharwa
    assert GAZ.ok_pair("VIC", 3644) and GAZ.ok_pair("NSW", 3644)  # Barooga, a Victorian code in NSW
    assert GAZ.ok_pair("NSW", 1300)  # a PO-box block
    assert GAZ.ok_pair("NSW", 2750)
    assert not GAZ.ok_pair("NSW", 3000)
    assert not GAZ.ok_pair("VIC", 2750)
    assert not GAZ.ok_pair("WA", 2750)
    assert ("ACT", "Tharwa", "2620") in GAZ.cross_border
    assert in_range("NSW", 2000) and in_range("ACT", 2600) and not in_range("ACT", 2620)
    assert range_states(2600) == {"ACT"} and range_states(2000) == {"NSW"} and range_states(100) == frozenset()
    assert all(in_range(code, lo) and in_range(code, hi) for code, spans in RANGES.items() for lo, hi in spans)


def test_postcodes_per_locality() -> None:
    assert GAZ.postcodes_of(["Penrith"], "NSW") == {2750}
    assert GAZ.postcodes_of(["Ryde"], "NSW") == {2112, 2114}
    assert GAZ.postcodes_of(["North", "Ryde"], "NSW") == {2113}
    assert GAZ.postcodes_of(["Ryde", "North"], "NSW") == {2113}  # the reversed form folds
    assert GAZ.postcodes_of(["Penrith"], "VIC") == frozenset()


def test_class_metro_or_regional() -> None:
    assert GAZ.klass_of(["Penrith"], "NSW") == "metro"
    assert GAZ.klass_of(["Dubbo"], "NSW") == "regional"
    assert GAZ.klass_of(["Penrith"], "VIC") is None
    assert all(r.klass in ("metro", "regional") for r in GAZ.rows)


def test_capitals() -> None:
    assert all(GAZ.is_capital([c]) for c in ("Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Hobart"))
    assert GAZ.is_capital(["canberra"]) and GAZ.is_capital(["DARWIN"])
    assert not GAZ.is_capital(["Penrith"]) and not GAZ.is_capital(["Sydney", "Olympic", "Park"])


def test_fuzzy_lookup_allows_one_edit_on_names_of_five_letters_or_more() -> None:
    assert GAZ.states_of(["Pennith"]) == frozenset()
    assert GAZ.states_of(["Pennith"], fuzzy=True) == {"NSW"}
    assert GAZ.states_of(["Pennith"], fuzzy=True, state="VIC") == frozenset()
    assert GAZ.states_of(["Bree"], fuzzy=True) == frozenset()  # too short to guess
    assert edit1("penrith", "pennith") and edit1("ryde", "rydes") and not edit1("penrith", "penrose")


def test_fake_pool_rows_only() -> None:
    pool = GAZ.fake_pool()
    assert len(pool) == 6_934
    assert all(r.fake_pool and r.primary and r.in_state_range and r.source == "sal" for r in pool)
    assert all(r.population >= 200 for r in pool)
    assert all(in_range(r.state, int(r.postcode)) for r in pool)
    # never a cross-border or detection-only row
    cross = {(s, int(pc)) for s, _, pc in GAZ.cross_border}
    assert not any((r.state, int(r.postcode)) in cross for r in pool)
    assert not any(r.source != "sal" for r in pool)
    # one row per locality and postcode (a state can hold two localities of one name), sorted, stable
    assert len({(r.state, r.folded, r.postcode) for r in pool}) == len(pool)
    assert [(r.state, r.folded, r.postcode) for r in pool] == sorted((r.state, r.folded, r.postcode) for r in pool)
    assert GAZ.fake_pool() == pool


def test_fake_pool_filters() -> None:
    nsw = GAZ.fake_pool("NSW")
    assert nsw and {r.state for r in nsw} == {"NSW"}
    regional = GAZ.fake_pool("NSW", "regional")
    metro = GAZ.fake_pool("NSW", "metro")
    assert len(regional) + len(metro) == len(nsw)
    assert all(not r.metro for r in regional) and all(r.metro for r in metro)
    assert all(GAZ.fake_pool(s) for s in STATE_CODES)
    assert all(
        len(GAZ.fake_pool(s, k)) >= 12 for s in STATE_CODES for k in ("metro", "regional") if s != "ACT" or k == "metro"
    )
    assert GAZ.fake_pool("ACT", "regional") == ()
