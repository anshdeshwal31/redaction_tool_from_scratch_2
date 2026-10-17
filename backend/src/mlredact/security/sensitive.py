"""A wrapper that keeps document-derived values out of reprs, f-strings, logs and serialisers.

Use :class:`Sensitive` for standalone values that travel between components (seed identifiers,
removal sets, surrogate source values).  Dataclasses that hold document text use
``field(repr=False)`` instead, so the hot path stays ergonomic while ``repr()`` remains safe.
"""

from __future__ import annotations

from typing import Any, NoReturn

_REDACTED = "Sensitive(<redacted>)"


class Sensitive[T]:
    """Opaque holder for a sensitive value.  Call :meth:`reveal` to use it deliberately."""

    __slots__ = ("_value",)
    _value: T

    def __init__(self, value: T) -> None:
        object.__setattr__(self, "_value", value)

    def reveal(self) -> T:
        return self._value

    # --- every implicit rendering path is redacted -------------------------------------------
    def __repr__(self) -> str:
        return _REDACTED

    def __str__(self) -> str:
        return _REDACTED

    def __format__(self, format_spec: str) -> str:
        return _REDACTED

    # --- value semantics so Sensitive values can live in sets / dict keys ----------------------
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sensitive):
            return bool(self._value == other._value)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Sensitive", self._value))

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("Sensitive is immutable")

    # Pickling is required for in-memory IPC with worker processes.  Pickled bytes never touch
    # disk unencrypted: the job workspace encrypts every artefact it persists.
    def __reduce__(self) -> tuple[type[Sensitive[T]], tuple[T]]:
        return (Sensitive, (self._value,))
