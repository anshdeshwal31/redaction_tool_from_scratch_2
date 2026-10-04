"""Canonical serialisation, hashing and content-derived identifiers.

Everything that ends up in a deterministic artefact (manifest, config hash, IDs) goes through this
module so that byte-level output depends only on content, never on dict insertion order, platform
float formatting quirks, wall-clock time or random state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from mlredact.security.sensitive import Sensitive

_ID_DIGEST_BYTES = 12  # 96-bit IDs: collision-free in practice within a document, short in manifests


def _normalise(value: Any) -> Any:
    """Convert to plain JSON types, rejecting anything non-deterministic or sensitive."""
    if isinstance(value, Sensitive):
        raise TypeError("Sensitive values must never be serialised into canonical artefacts")
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not allowed in canonical JSON")
        # Integral floats are emitted as ints so 1.0 and 1 hash identically.
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("canonical JSON keys must be strings")
            out[k] = _normalise(v)
        return out
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_normalise(v) for v in value]
    if isinstance(value, set | frozenset):
        raise TypeError("sets have no canonical order; convert to a sorted list explicitly")
    raise TypeError(f"type {type(value).__name__} is not canonically serialisable")


def canonical_json(value: Any) -> bytes:
    """Serialise ``value`` to canonical UTF-8 JSON (sorted keys, no whitespace, shortest floats)."""
    normalised = _normalise(value)
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


def canonical_json_pretty(value: Any) -> bytes:
    """Human-readable canonical JSON (still deterministic) for artefacts people may open."""
    normalised = _normalise(value)
    text = json.dumps(normalised, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_id(prefix: str, *parts: str | int | bytes) -> str:
    """Deterministic identifier from *structural* parts (positions, hashes, indices).

    Callers must never pass PII values here: IDs appear in the non-sensitive manifest, and an
    unkeyed hash of a name could be reversed by dictionary attack.  Parts are length-prefixed so
    that ``("ab", "c")`` and ``("a", "bc")`` cannot collide.
    """
    h = hashlib.blake2b(digest_size=_ID_DIGEST_BYTES, person=b"mlredact-id-v1")
    h.update(prefix.encode("ascii"))
    for part in parts:
        if isinstance(part, bytes):
            raw, tag = part, b"b"
        elif isinstance(part, bool):  # bool is an int subclass; keep it distinct
            raw, tag = (b"1" if part else b"0"), b"o"
        elif isinstance(part, int):
            raw, tag = str(part).encode("ascii"), b"i"
        else:
            raw, tag = part.encode("utf-8"), b"s"
        h.update(tag + len(raw).to_bytes(8, "big") + raw)
    return f"{prefix}-{h.hexdigest()}"
