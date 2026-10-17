"""Signed links and the stderr reason-code filter."""

from __future__ import annotations

import io

import pytest

from mlredact.api.engine import watch_error_code
from mlredact.api.links import link_key, sign, verify
from mlredact.api.settings import ApiSettings
from mlredact.core.errors import EnvironmentErrorMl

KEY = link_key("k" * 40)


def test_links_verify_only_for_their_purpose_job_and_name() -> None:
    token, _ = sign(KEY, "file", "a" * 32, "manifest.json", ttl_s=60)
    assert verify(KEY, token, "file", "a" * 32, "manifest.json") is None
    assert verify(KEY, token, "upload", "a" * 32, "manifest.json") == "E_API_LINK_INVALID"
    assert verify(KEY, token, "file", "b" * 32, "manifest.json") == "E_API_LINK_INVALID"
    assert verify(KEY, token, "file", "a" * 32, "redacted.pdf") == "E_API_LINK_INVALID"
    assert verify(link_key("x" * 40), token, "file", "a" * 32, "manifest.json") == "E_API_LINK_INVALID"


@pytest.mark.parametrize("token", ["", "abc", "123.", ".deadbeef", "9" * 12 + "." + "z" * 64])
def test_malformed_links_are_invalid(token: str) -> None:
    assert verify(KEY, token, "file", "a" * 32, "run.json") == "E_API_LINK_INVALID"


def test_links_expire() -> None:
    token, expires = sign(KEY, "upload", "a" * 32, "input", ttl_s=10, now=1000)
    assert expires == 1010
    assert verify(KEY, token, "upload", "a" * 32, "input", now=1005) is None
    assert verify(KEY, token, "upload", "a" * 32, "input", now=1011) == "E_API_LINK_EXPIRED"


def test_link_key_is_not_the_token() -> None:
    assert ("k" * 40).encode() != KEY


def code_from(data: bytes) -> str | None:
    return watch_error_code(io.BytesIO(data))()


@pytest.mark.parametrize(
    ("stderr", "code"),
    [
        (b"[INFO] RapidOCR: engine onnxruntime\nerror: E_MODEL_MISSING (x at y:z:1)\n", "E_MODEL_MISSING"),
        (b"error: V2_TEXT_LEAK (x at y)\r\n", "V2_TEXT_LEAK"),
        (b"error: E_INTERNAL (no newline at the end)", "E_INTERNAL"),
        (b"error: E_OCR_FAILED (a)\nerror: E_INTERNAL (b)\n", "E_OCR_FAILED"),
        (b"error: Mr John Citizen was not found\n", None),
        (b"    error: E_INTERNAL (indented lines are not the CLI's diagnostic)\n", None),
        (b"warning: E_INTERNAL\n", None),
        (b"x" * 20_000 + b"error: E_INTERNAL (tail of an overlong line)\n", None),
        (b"x" * 20_000 + b"tail\nerror: E_INTERNAL (next line)\n", "E_INTERNAL"),
    ],
)
def test_only_the_cli_reason_code_survives(stderr: bytes, code: str | None) -> None:
    assert code_from(stderr) == code


@pytest.mark.parametrize("token", ["", "short", "x" * 31])
def test_settings_refuse_a_weak_token(token: str) -> None:
    with pytest.raises(EnvironmentErrorMl):
        ApiSettings.from_env({"MLREDACT_API_TOKEN": token})


def test_settings_parse_the_environment(tmp_path: object) -> None:
    s = ApiSettings.from_env(
        {
            "MLREDACT_API_TOKEN": "t" * 32,
            "MLREDACT_API_ENGINE": '["python", "stub.py"]',
            "MLREDACT_API_CORS_ORIGINS": "https://app.example/, http://localhost:3100",
            "MLREDACT_API_MAX_UPLOAD_MB": "5",
        }
    )
    assert s.engine == ("python", "stub.py")
    assert s.cors_origins == ("https://app.example", "http://localhost:3100")
    assert s.max_upload_bytes == 5 * 1024 * 1024
    with pytest.raises(EnvironmentErrorMl):
        ApiSettings.from_env({"MLREDACT_API_TOKEN": "t" * 32, "MLREDACT_API_ENGINE": "not json"})
