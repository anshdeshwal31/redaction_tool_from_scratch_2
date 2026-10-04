"""The published evaluator contract (``schemas/manifest.v1.json``) stays in step with the code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

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
