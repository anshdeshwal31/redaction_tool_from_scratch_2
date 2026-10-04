"""Exception sanitisation: turn any exception into a PII-free summary.

Third-party exception messages can embed document content (e.g. ``ValueError: invalid literal
'John Smith'``) and tracebacks can include source lines and reprs.  We therefore never log or
persist ``str(exc)`` for exceptions we did not construct, and never format tracebacks.
"""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from types import TracebackType

from mlredact.core.errors import MlredactError, ReasonCode

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.]")


@dataclass(frozen=True, slots=True)
class SanitizedError:
    code: ReasonCode
    error_type: str  # qualified class name only
    where: str  # "module:function:line" of the innermost mlredact frame, or "external:<unknown>:0"
    params: tuple[tuple[str, object], ...] = ()

    def as_fields(self) -> dict[str, object]:
        return {"reason": self.code, "error_type": self.error_type, "where": self.where}


def _innermost_own_frame(tb: TracebackType | None) -> str:
    where = "external:unknown:0"
    for frame, lineno in traceback.walk_tb(tb):
        module = frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("mlredact"):
            func = _SAFE_NAME.sub("_", frame.f_code.co_name)[:64] or "unknown"
            where = f"{_SAFE_NAME.sub('_', module)[:128]}:{func}:{lineno}"
    return where


def sanitize(exc: BaseException) -> SanitizedError:
    cls = type(exc)
    error_type = _SAFE_NAME.sub("_", f"{cls.__module__}.{cls.__qualname__}")[:96]
    where = _innermost_own_frame(exc.__traceback__)
    if isinstance(exc, MlredactError):
        return SanitizedError(exc.code, error_type, where, tuple(exc.params.items()))
    return SanitizedError(ReasonCode.INTERNAL, error_type, where)


def _excepthook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
    exc.__traceback__ = tb
    info = sanitize(exc)
    sys.stderr.write(f"mlredact: fatal {info.code.value} {info.error_type} at {info.where}\n")


def install_excepthook() -> None:
    """Replace the default excepthook so uncaught errors print a sanitised one-liner only."""
    sys.excepthook = _excepthook
