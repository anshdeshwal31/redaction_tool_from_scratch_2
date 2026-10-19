"""Access to pinned runtime resources with SHA-256 verification."""

from __future__ import annotations

import hashlib
from functools import cache
from importlib import resources
from pathlib import Path

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode

_PACKAGE = "mlredact.resources"


def resource_root() -> Path:
    return Path(str(resources.files(_PACKAGE)))


def _parse_sums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        out[name.lstrip("*")] = digest.lower()
    return out


@cache
def verify_resources() -> int:
    """Verify every file listed in every ``SHA256SUMS`` under the resource tree.  Returns count."""
    count = 0
    for sums in sorted(resource_root().rglob("SHA256SUMS")):
        for name, digest in sorted(_parse_sums(sums.read_text("utf-8")).items()):
            path = sums.parent / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise EnvironmentErrorMl(ReasonCode.RESOURCE_HASH_MISMATCH)
            count += 1
    return count


def font_path(family: str = "Sans", style: str = "Regular") -> Path:
    """Liberation fonts (SIL OFL): family in {Sans, Serif, Mono}, style in {Regular, Bold, Italic}."""
    path = resource_root() / "fonts" / "liberation" / f"Liberation{family}-{style}.ttf"
    if not path.is_file():
        raise EnvironmentErrorMl(ReasonCode.RESOURCE_HASH_MISMATCH)
    return path
