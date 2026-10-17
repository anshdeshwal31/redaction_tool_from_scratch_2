"""Error types and reason codes.

Errors raised by mlredact carry a :class:`ReasonCode` and *typed, non-textual* parameters only.
There is deliberately no free-text message field: document content must never be able to reach
an exception message, a log line or a quarantine record.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ReasonCode(StrEnum):
    """Stable, PII-free codes used in errors, logs, quarantine records and manifests."""

    # Intake
    INPUT_UNREADABLE = "E_INPUT_UNREADABLE"
    INPUT_ENCRYPTED = "E_INPUT_ENCRYPTED"
    INPUT_XFA_ONLY = "E_INPUT_XFA_ONLY"
    INPUT_TOO_LARGE = "E_INPUT_TOO_LARGE"
    INPUT_TOO_MANY_PAGES = "E_INPUT_TOO_MANY_PAGES"
    INPUT_UNSUPPORTED = "E_INPUT_UNSUPPORTED"
    INPUT_EMPTY = "E_INPUT_EMPTY"
    # Processing
    RENDER_FAILED = "E_RENDER_FAILED"
    PAGE_TOO_LARGE = "E_PAGE_TOO_LARGE"
    OCR_FAILED = "E_OCR_FAILED"
    PAGE_ILLEGIBLE = "E_PAGE_ILLEGIBLE"
    WORKER_TIMEOUT = "E_WORKER_TIMEOUT"
    WORKER_CRASHED = "E_WORKER_CRASHED"
    # Environment / configuration
    CONFIG_INVALID = "E_CONFIG_INVALID"
    MODEL_MISSING = "E_MODEL_MISSING"
    MODEL_HASH_MISMATCH = "E_MODEL_HASH_MISMATCH"
    RESOURCE_HASH_MISMATCH = "E_RESOURCE_HASH_MISMATCH"
    RUNTIME_PROFILE_MISMATCH = "E_RUNTIME_PROFILE_MISMATCH"
    # Verification (fail-closed release gate)
    VERIFY_STRUCTURE = "V1_STRUCTURE"
    VERIFY_TEXT_LEAK = "V2_TEXT_LEAK"
    VERIFY_PIXELS = "V3_PIXELS"
    VERIFY_RESIDUAL = "V4_RESIDUAL"
    VERIFY_SURROGATE = "V5_SURROGATE"
    VERIFY_INTEGRITY = "V6_INTEGRITY"
    # Catch-all
    INTERNAL = "E_INTERNAL"


SafeParam = int | float | bool | StrEnum | None

_MAX_PARAMS: Final = 16


class MlredactError(Exception):
    """Base class for all mlredact errors.

    Only a reason code and typed scalar parameters are accepted.  ``str(err)`` renders them, so it is
    always safe to log, but third-party exceptions must still go through
    :func:`mlredact.security.exceptions.sanitize`.
    """

    def __init__(self, code: ReasonCode, /, **params: SafeParam) -> None:
        if len(params) > _MAX_PARAMS:
            raise ValueError("too many error parameters")
        for key, value in params.items():
            if not isinstance(value, int | float | bool | StrEnum | type(None)):
                raise TypeError(f"error parameter {key!r} must be a scalar or enum, not {type(value).__name__}")
        self.code = code
        self.params: dict[str, SafeParam] = dict(sorted(params.items()))
        super().__init__(self._render())

    def _render(self) -> str:
        if not self.params:
            return self.code.value
        rendered = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.code.value} ({rendered})"

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        # Errors cross the worker-process boundary; the default ``cls(*args)`` would pass the
        # rendered string as the code (and an unpicklable result breaks the whole process pool).
        return _restore, (type(self), self.code, self.params)


def _restore(cls: type[MlredactError], code: ReasonCode, params: dict[str, SafeParam]) -> MlredactError:
    return cls(code, **params)


class InputRejected(MlredactError):
    """The input document cannot be processed (quarantine without output)."""


class ProcessingError(MlredactError):
    """A processing stage failed."""


class EnvironmentErrorMl(MlredactError):
    """The runtime environment (models, resources, profile, config) is not valid."""


class VerificationFailed(MlredactError):
    """The release gate failed; the output must not be released."""
