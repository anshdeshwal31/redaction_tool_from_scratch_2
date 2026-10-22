"""The published evaluator contract (``schemas/manifest.v1.json``) stays in step with the code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from mlredact.core.errors import ReasonCode
from mlredact.core.types import ActionKind, EntityType, JobStatus, PageKind, RegionKind, VerificationCheck
from mlredact.manifest.build import SCHEMA_VERSION

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "manifest.v1.json"


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text("utf-8"))  # type: ignore[no-any-return]


def test_enumerations_match_the_code(schema: dict[str, Any]) -> None:
    defs = schema["$defs"]
    props = schema["properties"]
    assert defs["entity_type"]["enum"] == [e.value for e in EntityType]
    assert defs["action"]["enum"] == [a.value for a in ActionKind]
    assert defs["reason"]["enum"] == [r.value for r in ReasonCode]
    assert defs["region_kind"]["enum"] == [k.value for k in RegionKind]
    assert props["status"]["enum"] == [s.value for s in JobStatus]
    assert props["pages"]["items"]["properties"]["kind"]["enum"] == [k.value for k in PageKind]
    check_enum = props["verification"]["properties"]["checks"]["items"]["properties"]["check"]["enum"]
    assert check_enum == [c.value for c in VerificationCheck]


def test_schema_version_is_compatible(schema: dict[str, Any]) -> None:
    import re

    assert re.fullmatch(schema["properties"]["schema_version"]["pattern"], SCHEMA_VERSION)


def test_an_oracle_regions_manifest_follows_the_schema(schema: dict[str, Any], tmp_path: Path) -> None:
    """``mlredact bench --mode oracle_regions`` on the synthetic letter (as tests/unit/test_bench_oracle.py runs it),
    with plan regions tool 2's own planning never makes: over a name and the word before it (the line is re-typeset),
    and over a word of no mention (it is removed as an illegible region, a residual box)."""
    from test_bench_oracle import WITH_NER, bench, letter

    from inject_support import words_poly

    docs = letter()
    regions = [r for r in docs["plan.json"]["regions"] if r["mentions"] != ["person-citizen"]]
    for region_id, words, named in (("x-0", ("Ms", "Citizen,"), ["person-citizen"]), ("x-1", ("Dear",), [])):
        poly = words_poly(docs, 11, *words)
        regions.append({"id": region_id, "page": 0, "poly": poly, "action": "replace", "mentions": named})
    docs["plan.json"]["regions"] = regions
    run = bench(tmp_path, "oracle_regions", docs, config=WITH_NER)
    assert run.result.exit_code == 0, run.result.output
    manifest = run.json("manifest.json")
    assert ActionKind.RETYPESET_REGION.value in {kind for m in manifest["mentions"] for kind in m["render"]}
    assert [box["type"] for box in manifest["residual_boxes"]] == [RegionKind.ILLEGIBLE.value]
    errors = [f"{list(e.absolute_path)}: {e.message}" for e in Draft202012Validator(schema).iter_errors(manifest)]
    assert errors == []
