"""Layered configuration loading: defaults < profile < deployment files < job overrides."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mlredact.config.schema import AppConfig
from mlredact.core.canonical import canonical_json, sha256_hex
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode

_PROFILE_PACKAGE = "mlredact.config.profiles"


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    return data


def builtin_profiles() -> list[str]:
    root = resources.files(_PROFILE_PACKAGE)
    return sorted(p.name.removesuffix(".yaml") for p in root.iterdir() if p.name.endswith(".yaml"))


def load_config(
    profile: str = "broad",
    files: Sequence[Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Build the effective configuration.

    ``profile`` names a built-in policy profile; ``files`` are deployment YAML files applied in
    order; ``overrides`` are (restricted) job-level overrides applied last.
    """
    if profile not in builtin_profiles():
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    data: dict[str, Any] = {}
    profile_text = resources.files(_PROFILE_PACKAGE).joinpath(f"{profile}.yaml").read_text("utf-8")
    data = _deep_merge(data, _read_yaml(profile_text))
    for path in files:
        data = _deep_merge(data, _read_yaml(path.read_text("utf-8")))
    if overrides:
        data = _deep_merge(data, overrides)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        # Validation messages may echo values; report the count only.
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, errors=exc.error_count()) from None


# Operational settings that provably cannot change the output bytes (they only change how fast or
# where work happens, or what is checked before work starts).  Thread counts are NOT listed: they
# can change floating-point reduction order and therefore belong to the output identity.
_OPERATIONAL_RUNTIME_FIELDS = ("workers", "page_timeout_s", "models_dir", "expected_profile")
# Where the surrogate key comes from is operational; *which* key was used is recorded separately as
# ``key_id`` in provenance.
_OPERATIONAL_SURROGATE_FIELDS = ("key_file", "allow_development_key")


def output_identity(config: AppConfig) -> dict[str, Any]:
    """The configuration that determines output bytes (hashed into provenance)."""
    data = config.model_dump(mode="json")
    for name in _OPERATIONAL_RUNTIME_FIELDS:
        data["runtime"].pop(name, None)
    for name in _OPERATIONAL_SURROGATE_FIELDS:
        data["surrogate"].pop(name, None)
    return data


def config_hash(config: AppConfig) -> str:
    return sha256_hex(canonical_json(output_identity(config)))
