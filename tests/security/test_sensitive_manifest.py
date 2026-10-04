"""The sealed sensitive manifest: only the evaluator's key opens it, and tampering is detected."""

from __future__ import annotations

import json

import pytest
from cryptography.exceptions import InvalidTag

from mlredact.config.loader import load_config
from mlredact.detect.engine import detect_document
from mlredact.manifest.sensitive import build_sensitive, keypair, open_sealed, seal
from mlredact.policy.engine import decide
from support import pages_from_lines

CFG = load_config("broad")


def _document() -> dict[str, object]:
    pages = pages_from_lines(["Re: Mr John Andrew SMITH DOB: 14/03/1978", "Claim No: WC1234567"])
    result = detect_document(pages, CFG, "doc")
    return build_sensitive(
        input_sha256="0" * 64,
        pages=pages,
        decisions=decide(result.mentions, CFG.policy),
        surrogates={},
        persons=result.persons,
        low_evidence=result.low_evidence,
    )


def test_round_trip_with_the_evaluator_key() -> None:
    private, public = keypair()
    doc = _document()
    sealed = seal(doc, public)
    assert b"SMITH" not in sealed and b"WC1234567" not in sealed
    opened = open_sealed(sealed, private)
    assert opened == json.loads(json.dumps(doc))
    texts = {m["text"] for m in opened["mentions"]}
    assert "John Andrew SMITH" in texts and "WC1234567" in texts


def test_wrong_key_and_tampering_are_rejected() -> None:
    _private, public = keypair()
    other_private, _other_public = keypair()
    sealed = seal(_document(), public)
    with pytest.raises(InvalidTag):
        open_sealed(sealed, other_private)
    env = json.loads(sealed)
    env["recipient"] = "0" * 16  # header is authenticated
    with pytest.raises(InvalidTag):
        open_sealed(json.dumps(env).encode(), _private)


def test_each_sealing_uses_a_fresh_ephemeral_key() -> None:
    _private, public = keypair()
    doc = _document()
    assert seal(doc, public) != seal(doc, public)
