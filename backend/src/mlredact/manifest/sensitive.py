"""The sensitive manifest (Appendix A): original text for the external evaluator, sealed to its key.

It contains exactly what the safe manifest leaves out - each mention's original text, the OCR tokens
with their alternates, person clusters and low-evidence candidate text - and is only ever produced
in sealed form: plaintext exists in memory only, is never logged, and is encrypted to an evaluator
public key supplied with the job.

Envelope (``mlredact-sealed-v1``): an ephemeral X25519 key agreement with the recipient key, HKDF-SHA256
(salt = ephemeral public || recipient public), ChaCha20-Poly1305 over the canonical JSON, with the
canonical header as associated data.  A fresh ephemeral key per job means sealed output is not
deterministic, by design; the deterministic outputs are the redacted PDF and the safe manifest.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from mlredact.core.canonical import canonical_json
from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.model import PageAnalysis
from mlredact.detect.fusion import LowEvidence
from mlredact.policy.engine import Decision
from mlredact.resolve.persons import PersonResolution
from mlredact.surrogate.assign import MentionSurrogate

FORMAT = "mlredact-sealed-v1"
ALG = "X25519-HKDF-SHA256-ChaCha20Poly1305"
SCHEMA_VERSION = "1.0.0"
_INFO = b"mlredact sensitive manifest v1"


def build_sensitive(
    *,
    input_sha256: str,
    pages: Sequence[PageAnalysis],
    decisions: Sequence[Decision],
    surrogates: Mapping[str, MentionSurrogate],
    persons: PersonResolution | None,
    low_evidence: Sequence[LowEvidence],
) -> dict[str, Any]:
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    mentions = []
    for d in decisions:
        m = d.mention
        entry: dict[str, Any] = {
            "mention_id": m.mention_id,
            "type": m.entity_type.value,
            "action": d.action.value,
            "text": m.value,
            "tokens": [
                {"token_id": t, "text": tokens[t].text, "alternates": list(tokens[t].alternates)} for t in m.token_ids
            ],
        }
        sur = surrogates.get(m.mention_id)
        if sur is not None:
            entry["replacement"] = " ".join(s for s in sur.tokens if s)
        mentions.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "job": {"input_sha256": input_sha256},
        "mentions": mentions,
        "persons": [
            {"cluster": cluster, "mentions": list(mids)}
            for cluster, mids in sorted((persons.clusters if persons else {}).items())
        ],
        "low_evidence": [
            {
                "type": le.entity_type.value,
                "token_ids": list(le.token_ids),
                "text": " ".join(tokens[t].text for t in le.token_ids),
            }
            for le in low_evidence
        ],
        "pages": [
            {
                "index": p.geometry.index,
                "tokens": [
                    {
                        "token_id": t.token_id,
                        "text": t.text,
                        "alternates": list(t.alternates),
                        "px": list(t.bbox.as_tuple()),
                        "confidence": t.confidence,
                        "engine": t.engine,
                    }
                    for t in p.tokens()
                ],
            }
            for p in pages
        ],
    }


# ----------------------------------------------------------------------------------- sealing
def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def recipient_id(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def load_public_key(text: str) -> bytes:
    """A recipient key file: 64 hex characters (raw X25519 public key)."""
    raw = bytes.fromhex(text.strip())
    if len(raw) != 32:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    return raw


def keypair() -> tuple[bytes, bytes]:
    """(private, public) raw X25519 keys for an evaluator."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

    key = X25519PrivateKey.generate()
    private = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private, public


def _derive(shared: bytes, epk: bytes, recipient: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=epk + recipient, info=_INFO).derive(shared)


def seal(document: Mapping[str, Any], recipient_public: bytes) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    ephemeral = X25519PrivateKey.generate()
    epk = ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient_public))
    header = {"format": FORMAT, "alg": ALG, "recipient": recipient_id(recipient_public), "epk": _b64(epk)}
    nonce = os.urandom(12)
    ciphertext = ChaCha20Poly1305(_derive(shared, epk, recipient_public)).encrypt(
        nonce, canonical_json(dict(document)), canonical_json(header)
    )
    return canonical_json({**header, "nonce": _b64(nonce), "ciphertext": _b64(ciphertext)})


def open_sealed(envelope: bytes, private_raw: bytes) -> dict[str, Any]:
    """Evaluator side: decrypt and parse a sealed sensitive manifest."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    env = json.loads(envelope)
    if env.get("format") != FORMAT or env.get("alg") != ALG:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
    key = X25519PrivateKey.from_private_bytes(private_raw)
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    epk = base64.b64decode(env["epk"])
    shared = key.exchange(X25519PublicKey.from_public_bytes(epk))
    header = {k: env[k] for k in ("format", "alg", "recipient", "epk")}
    plaintext = ChaCha20Poly1305(_derive(shared, epk, public)).decrypt(
        base64.b64decode(env["nonce"]), base64.b64decode(env["ciphertext"]), canonical_json(header)
    )
    return json.loads(plaintext)  # type: ignore[no-any-return]
