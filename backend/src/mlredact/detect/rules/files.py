"""File paths and file names (document footers such as "S:\\Clients\\Smith J\\IME report.docx")."""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate, trim_span
from mlredact.text.views import TextView

STRONG = EvidenceStrength.STRONG
EXTENSIONS = "docx?|dotx?|pdf|rtf|txt|odt|xlsx?|csv|pptx?|msg|eml|jpe?g|png|tiff?|heic|bmp|gif|zip|wpd|html?|xml"
_EXT_END = re.compile(rf"\.(?:{EXTENSIONS})(?![A-Za-z0-9])", re.IGNORECASE)
# Drive-letter or UNC path, up to the end of the line (cut below).
_WINDOWS = re.compile(r"(?<![\w\\])((?:[A-Za-z]:|\\\\[\w.$\-]+)\\[^\n<>|\"*?]*)")
# POSIX path: under a well-known root, or with a document extension.
_POSIX = re.compile(
    rf"(?<![\w/.:])(/(?:home|Users|mnt|media|srv|data|shares?|Volumes|var|tmp|opt)/[^\s\"'<>]+"
    rf"|/(?:[\w.\-]+/)+[\w.\-]+\.(?:{EXTENSIONS}))(?![\w])",
    re.IGNORECASE,
)
# Bare file name: a word with a document extension (not part of a path, URL or email address).
_NAME = re.compile(rf"(?<![\w.\\/@\-])(\w[\w\-.()]*\.(?:{EXTENSIONS}))(?![\w\\/]|\.\w)", re.IGNORECASE)


def _windows_end(text: str, start: int, end: int) -> int:
    """A path may contain spaces, so it ends after its last document extension or, without one,
    at the first space after its last backslash."""
    run = text[start:end]
    ext = list(_EXT_END.finditer(run))
    if ext:
        return start + ext[-1].end()
    last = run.rfind("\\")
    space = run.find(" ", last + 1)
    return start + (space if space >= 0 else len(run.rstrip()))


class FilePathDetector:
    detector_id = "rules.files"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        out: list[Candidate] = []
        spans: list[tuple[int, int, str]] = []
        for m in _WINDOWS.finditer(text):
            spans.append((m.start(1), _windows_end(text, m.start(1), m.end(1)), "file.windows"))
        for m in _POSIX.finditer(text):
            spans.append((m.start(1), m.end(1), "file.posix"))
        for m in _NAME.finditer(text):
            spans.append((m.start(1), m.end(1), "file.name"))
        for start, end, rule in spans:
            s, e = trim_span(text, start, end, " \t,;:()[]{}\"'")
            if e > s:
                out.append(candidate(self, EntityType.FILE_PATH, ViewKind.LINE, s, e, rule, STRONG))
        return out
