"""Deterministic keyed choices (HMAC-SHA256 counter mode, rejection sampling for uniformity)."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


class Chooser:
    def __init__(self, scope_key: bytes) -> None:
        self._key = scope_key

    def _block(self, domain: str, value_key: str, counter: int) -> bytes:
        msg = domain.encode() + b"\x00" + value_key.encode("utf-8") + b"\x00" + counter.to_bytes(8, "big")
        return hmac.new(self._key, msg, hashlib.sha256).digest()

    def integers(self, domain: str, value_key: str) -> Iterator[int]:
        """Endless stream of 64-bit integers for (domain, value)."""
        counter = 0
        while True:
            block = self._block(domain, value_key, counter)
            for i in range(0, 32, 8):
                yield int.from_bytes(block[i : i + 8], "big")
            counter += 1

    def index(self, domain: str, value_key: str, n: int, attempt: int = 0) -> int:
        """Uniform index in ``[0, n)``; ``attempt`` selects an independent draw (collision retry)."""
        if n <= 0:
            raise ValueError("cannot choose from an empty range")
        limit = (1 << 64) - ((1 << 64) % n)
        stream = self.integers(domain, f"{value_key}\x01{attempt}")
        for value in stream:
            if value < limit:
                return value % n
        raise AssertionError("unreachable")  # pragma: no cover

    def pick(self, domain: str, value_key: str, pool: Sequence[T], attempt: int = 0) -> T:
        return pool[self.index(domain, value_key, len(pool), attempt)]

    def digit_string(self, domain: str, value_key: str, length: int, attempt: int = 0) -> str:
        stream = self.integers(domain, f"{value_key}\x02{attempt}")
        out = []
        for value in stream:
            if value < (1 << 64) - ((1 << 64) % 10):
                out.append(str(value % 10))
                if len(out) == length:
                    return "".join(out)
        raise AssertionError("unreachable")  # pragma: no cover
