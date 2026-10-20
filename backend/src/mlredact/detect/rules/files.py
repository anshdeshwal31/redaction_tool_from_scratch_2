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
# The rest of a last folder name with spaces: words that start with a capital or a digit
# ("John Smith", "Smith J"), not the sentence that follows ("... and sent").
_TAIL = r"(?: +[A-Z0-9][^\s\"'<>/\\,;]*)*"
# POSIX path: under a well-known root, or with a document extension.  Folder names may contain
# spaces when another "/" follows ("/Users/jsmith/Smith Family/report.pdf").
_SEG = r"[^\s\"'<>/]+"
_POSIX = re.compile(
    rf"(?<![\w/.:])(/(?i:home|Users|mnt|media|srv|data|shares?|Volumes|var|tmp|opt)"
    rf"(?:/{_SEG}(?: +{_SEG})*(?=/))*(?:/{_SEG}(?: +{_SEG})*\.(?i:{EXTENSIONS})(?![A-Za-z0-9])|/{_SEG}{_TAIL})"
    rf"|/(?:[\w.\-]+(?: +[\w.\-]+)*/)+[\w.\-]+\.(?i:{EXTENSIONS}))(?![\w])"
)
_LAST_NAME = re.compile(rf"[^\s\"'<>/\\,;]*{_TAIL}")
# Bare file name: a word with a document extension (not part of a path, URL or email address).
_NAME = re.compile(rf"(?<![\w.\\/@\-])(\w[\w\-.()]*\.(?:{EXTENSIONS}))(?![\w\\/]|\.\w)", re.IGNORECASE)


def _windows_end(text: str, start: int, end: int) -> int:
    """A path may contain spaces, so it ends after its last document extension or, without one,
    after its last folder name ("C:\\Users\\John Smith")."""
    run = text[start:end]
    ext = list(_EXT_END.finditer(run))
    if ext:
        return start + ext[-1].end()
    last = run.rfind("\\") + 1
    tail = _LAST_NAME.match(run, last)
    return start + (tail.end() if tail else last)


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
