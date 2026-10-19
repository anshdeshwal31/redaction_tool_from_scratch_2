"""Signed, expiring links for the browser's direct uploads and downloads.

The frontend's server asks the API for a job (with the shared token); the API answers with paths
that carry a short-lived HMAC token.  The browser then uploads and downloads straight to and from
this API, so documents never pass through the frontend's host.  A token is bound to one purpose,
one job and one file name.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Literal

Purpose = Literal["upload", "file"]


def link_key(api_token: str) -> bytes:
    # A separate key from the bearer token itself: leaking a link never reveals the token.
    return hmac.new(api_token.encode(), b"mlredact-api-links-v1", hashlib.sha256).digest()


def _mac(key: bytes, purpose: Purpose, job: str, name: str, expires: int) -> str:
    return hmac.new(key, f"{purpose}\n{job}\n{name}\n{expires}".encode(), hashlib.sha256).hexdigest()


def sign(key: bytes, purpose: Purpose, job: str, name: str, ttl_s: int, now: float | None = None) -> tuple[str, int]:
    """Returns (token, expiry as a Unix time)."""
    expires = int(now if now is not None else time.time()) + ttl_s
    return f"{expires}.{_mac(key, purpose, job, name, expires)}", expires


def verify(key: bytes, token: str, purpose: Purpose, job: str, name: str, now: float | None = None) -> str | None:
    """None when the token is valid, otherwise a reason code."""
    expires_s, _, mac = token.partition(".")
    # isascii first: isdigit accepts digits such as '²' that int() rejects, and compare_digest raises
    # on a non-ASCII str.
    if not (expires_s.isascii() and expires_s.isdigit()) or len(mac) != 64 or not mac.isascii():
        return "E_API_LINK_INVALID"
    expected = _mac(key, purpose, job, name, int(expires_s))
    if not hmac.compare_digest(mac, expected):
        return "E_API_LINK_INVALID"
    if int(expires_s) < (now if now is not None else time.time()):
        return "E_API_LINK_EXPIRED"
    return None
