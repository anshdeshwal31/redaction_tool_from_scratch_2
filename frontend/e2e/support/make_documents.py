"""Writes the documents a real-engine end-to-end run submits, into the directory given:

* ``letter.pdf``     a synthetic IME letter with fictitious PII (the engine releases it), and
* ``encrypted.pdf``  the same letter, password-protected (rejected at intake: quarantined).

Run with the backend's Python: it uses backend/tools/synthdocs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend" / "tools"))

from synthdocs.generator import encrypted, make_ime_letter  # noqa: E402


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    letter = make_ime_letter(scanned=False).pdf
    (out / "letter.pdf").write_bytes(letter)
    (out / "encrypted.pdf").write_bytes(encrypted(letter))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
