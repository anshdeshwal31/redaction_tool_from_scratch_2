"""Surrogate key management (plan §10).

Surrogates are derived *statelessly*: ``K_scope = HMAC(K_master, "scope:" + scope_id)`` and every
choice is an HMAC of ``(domain, canonical original value, counter)`` under ``K_scope``.  There is no
real->fake mapping table to protect, the same value always maps to the same surrogate within a
scope (e.g. all documents of one claim), and without ``K_master`` nobody can test candidate names
against the output (no dictionary attack).

``K_master`` comes from the secret store (env ``MLREDACT_SURROGATE_KEY`` as hex, or a key file).
Only its *id* (a hash prefix) is ever recorded.  A fixed development key exists for tests and must
be enabled explicitly; it is refused when ``runtime.expected_profile`` is set (production).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from pathlib import Path

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode

_DEV_KEY = hashlib.sha256(b"mlredact development surrogate key - NOT FOR PRODUCTION").digest()
_MIN_KEY_BYTES = 32


@dataclass(frozen=True, slots=True)
class MasterKey:
    key_id: str
    _secret: bytes = field(repr=False)

    def scope_key(self, scope_id: str) -> bytes:
        return hmac.new(self._secret, b"scope:" + scope_id.encode("utf-8"), hashlib.sha256).digest()


def _from_hex(text: str) -> bytes:
    try:
        raw = bytes.fromhex(text.strip())
    except ValueError:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID) from None
    if len(raw) < _MIN_KEY_BYTES:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    return raw


def load_master_key(*, key_file: str | None, allow_development_key: bool, production: bool) -> MasterKey:
    env = os.environ.get("MLREDACT_SURROGATE_KEY")
    if env:
        secret = _from_hex(env)
    elif key_file:
        path = Path(key_file)
        if not path.is_file():
            raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
        secret = _from_hex(path.read_text("ascii"))
    elif allow_development_key and not production:
        secret = _DEV_KEY
    else:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    key_id = "k-" + hashlib.sha256(b"key-id:" + secret).hexdigest()[:16]
    return MasterKey(key_id, secret)
