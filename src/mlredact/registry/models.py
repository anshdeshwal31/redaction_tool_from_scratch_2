"""Pinned model registry (plan §15/§17).

Every model file is identified by name + SHA-256.  Processing code only ever *resolves* models
(verify hash, return path) — it never downloads.  :func:`fetch` exists for development and image
builds; production images ship models pre-fetched and run without network access.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.security.logging import get_logger

_log = get_logger("mlredact.registry")
_REGISTRY_RESOURCE = "models.yaml"
_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str
    sha256: str
    url: str
    size: int | None = None


@dataclass(frozen=True, slots=True)
class ModelEntry:
    model_id: str
    task: str
    version: str
    license: str
    format: str
    files: tuple[ModelFile, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "version": self.version,
            "license": self.license,
            "files": [{"name": f.name, "sha256": f.sha256} for f in self.files],
        }


def load_registry() -> dict[str, ModelEntry]:
    text = resources.files("mlredact.registry").joinpath(_REGISTRY_RESOURCE).read_text("utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    out: dict[str, ModelEntry] = {}
    for model_id, spec in sorted(raw["models"].items()):
        files = tuple(
            ModelFile(name=f["name"], sha256=f["sha256"].lower(), url=f["url"], size=f.get("size"))
            for f in spec["files"]
        )
        out[model_id] = ModelEntry(
            model_id=model_id,
            task=spec["task"],
            version=str(spec["version"]),
            license=spec["license"],
            format=spec["format"],
            files=files,
        )
    return out


def models_dir(configured: str) -> Path:
    """``MLREDACT_MODELS_DIR`` overrides the configured directory."""
    return Path(os.environ.get("MLREDACT_MODELS_DIR", configured)).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


_verified: dict[tuple[str, int, int], str] = {}


def _verified_hash(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    digest = _verified.get(key)
    if digest is None:
        digest = _sha256_file(path)
        _verified[key] = digest
    return digest


def resolve(model_id: str, directory: Path) -> dict[str, Path]:
    """Return verified local paths for every file of ``model_id``; raise if missing or tampered."""
    registry = load_registry()
    entry = registry.get(model_id)
    if entry is None:
        raise EnvironmentErrorMl(ReasonCode.MODEL_MISSING)
    paths: dict[str, Path] = {}
    for f in entry.files:
        path = directory / model_id / f.name
        if not path.is_file():
            _log.error("model.missing", model=model_id)
            raise EnvironmentErrorMl(ReasonCode.MODEL_MISSING)
        if _verified_hash(path) != f.sha256:
            _log.error("model.hash_mismatch", model=model_id)
            raise EnvironmentErrorMl(ReasonCode.MODEL_HASH_MISMATCH)
        paths[f.name] = path
    return paths


def fetch(model_id: str, directory: Path, *, timeout_s: int = 600) -> None:
    """Download (if needed) and verify every file of ``model_id``.  Development/build use only."""
    entry = load_registry()[model_id]
    target_dir = directory / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in entry.files:
        target = target_dir / f.name
        if target.is_file() and _sha256_file(target) == f.sha256:
            continue
        if not f.url.startswith("https://"):
            raise EnvironmentErrorMl(ReasonCode.MODEL_MISSING)
        fd, tmp_name = tempfile.mkstemp(dir=target_dir, prefix=".download-")
        tmp = Path(tmp_name)
        try:
            h = hashlib.sha256()
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(f.url, timeout=timeout_s) as resp:  # noqa: S310
                while chunk := resp.read(_CHUNK):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != f.sha256:
                raise EnvironmentErrorMl(ReasonCode.MODEL_HASH_MISMATCH)
            tmp.replace(target)
        finally:
            if tmp.exists():
                tmp.unlink()
        _log.info("model.fetched", model=model_id)
