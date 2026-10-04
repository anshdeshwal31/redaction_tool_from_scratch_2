"""Determinism (plan §13): document-level stages must not depend on Python's hash randomisation.

Detection, fusion, propagation, person resolution, policy and surrogate assignment run in fresh
interpreters with different ``PYTHONHASHSEED`` values; their canonical output must be identical.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import hashlib
from mlredact.config.loader import load_config
from mlredact.core.canonical import canonical_json
from mlredact.core.types import ActionKind
from mlredact.detect.engine import detect_document
from mlredact.policy.engine import decide
from mlredact.surrogate.assign import SurrogateAssigner, forbidden_keys
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import SurrogateFactory
from support import pages_from_lines

LINES = [
    "Re: Mr John Andrew SMITH DOB: 14/03/1978 Claim No: WC1234567",
    "Address: 42 Wattle Grove Road, Penrith NSW 2750 Phone 0412 345 678",
    "Mr Smith attended with his wife Mrs Mary Smith. J. Smith signed; John was anxious.",
    "Dr Peter Brown FRACS, Medicare 2953 70150 1, email john.smith78@bigpond.com",
    "Tinel's sign was positive. Smith's fracture was excluded.",
]
cfg = load_config("broad")
pages = pages_from_lines(LINES)
result = detect_document(pages, cfg, "doc")
decisions = decide(result.mentions, cfg.policy)
removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
tokens = {t.token_id: t for p in pages for t in p.tokens()}
factory = SurrogateFactory(Chooser(hashlib.sha256(b"k").digest()), forbidden_keys(removed, result.persons))
surrogates = SurrogateAssigner(factory, result.persons, tokens).assign(decisions)
doc = {
    "mentions": [[m.mention_id, m.entity_type.value, list(m.token_ids), m.value] for m in result.mentions],
    "low": [[l.entity_type.value, list(l.token_ids)] for l in result.low_evidence],
    "decisions": [[d.mention.mention_id, d.action.value] for d in decisions],
    "surrogates": {k: list(v.tokens) for k, v in sorted(surrogates.items())},
    "clusters": sorted([k, v.cluster or ""] for k, v in result.persons.mentions.items()),
}
print(hashlib.sha256(canonical_json(doc)).hexdigest())
"""


def _run(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tests"), str(ROOT / "tools"), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True, cwd=ROOT, timeout=300
    )
    return out.stdout.strip()


@pytest.mark.parametrize("seeds", [("0", "1", "2", "12345", "random")])
def test_document_stages_are_independent_of_hash_seed(seeds: tuple[str, ...]) -> None:
    digests = {_run(s) for s in seeds}
    assert len(digests) == 1, digests
