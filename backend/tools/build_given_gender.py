"""Build ``src/mlredact/resources/names/given_gender.csv`` from the NSW popular baby names list (standard library only).

    python -I -B tools/build_given_gender.py <popular_baby_names_1952_to_2025.csv> <out folder>

The input is the NSW Registry of Births, Deaths and Marriages "Popular Baby Names 1952 to 2025" (CC BY 4.0, see
``resources/names/ATTRIBUTION.txt``): the 100 most registered names per year for boys and for girls.  Each name's
registrations are summed over all years for each gender.  A name is ``m`` or ``f`` when at least 90% of its
registrations are of that gender, otherwise ``u`` (used for both: the resolver then treats it as giving no gender).
Keys are tool 2's comparison keys (lower case, letters only: ``JO-ANNE`` is ``joanne``).  The output is sorted by key,
UTF-8, LF line endings, so two builds of one input are byte-identical.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

SHARE = 0.9
_KEY = re.compile(r"[^a-z]+")


def build(rows: list[dict[str, str]]) -> str:
    male: Counter[str] = Counter()
    female: Counter[str] = Counter()
    for r in rows:
        key = _KEY.sub("", r["Name"].strip().lower())
        count = int(r["Number"])
        if not key or count <= 0:
            raise SystemExit(f"bad row: {r}")
        if r["Gender"] == "Male":
            male[key] += count
        elif r["Gender"] == "Female":
            female[key] += count
        else:
            raise SystemExit(f"bad gender: {r}")
    out = ["name,male,female,gender"]
    for key in sorted(set(male) | set(female)):
        m, f = male[key], female[key]
        gender = "m" if m >= SHARE * (m + f) else "f" if f >= SHARE * (m + f) else "u"
        out.append(f"{key},{m},{f},{gender}")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(__doc__ or "")
        return 2
    src, folder = Path(argv[0]), Path(argv[1])
    raw = src.read_bytes()
    sys.stdout.write(f"input sha256 {hashlib.sha256(raw).hexdigest()} ({len(raw)} bytes)\n")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    text = build(rows)
    data = text.encode("utf-8")
    (folder / "given_gender.csv").write_bytes(data)
    genders = Counter(line.rsplit(",", 1)[1] for line in text.splitlines()[1:])
    sys.stdout.write(f"{len(rows)} rows -> {sum(genders.values())} names {dict(sorted(genders.items()))}\n")
    sys.stdout.write(f"given_gender.csv sha256 {hashlib.sha256(data).hexdigest()} ({len(data)} bytes)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
