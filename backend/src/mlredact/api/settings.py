"""API settings, read from ``MLREDACT_API_*`` environment variables."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode

MIN_TOKEN_CHARS = 32


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return min(hi, max(lo, int(raw)))
    except ValueError:
        raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID) from None


@dataclass(frozen=True)
class ApiSettings:
    # Shared secret for server-to-server calls (the frontend's server holds it) and the key that
    # signs browser upload/download links.
    token: str
    jobs_dir: Path
    # Command that runs the CLI; tests point it at a stub engine.
    engine: tuple[str, ...]
    engine_cwd: Path
    config_files: tuple[Path, ...]
    default_workers: int
    allow_dev_key: bool
    max_upload_bytes: int
    job_timeout_s: float
    upload_ttl_s: int
    link_ttl_s: int
    # Browser origins allowed to use signed links (the frontend's address).
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ApiSettings:
        env = os.environ if env is None else env
        token = env.get("MLREDACT_API_TOKEN", "").strip()
        if len(token) < MIN_TOKEN_CHARS:
            raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID, api_token_chars=len(token))
        cwd = Path.cwd()
        engine: tuple[str, ...] = (sys.executable, "-m", "mlredact.cli.main")
        if raw := env.get("MLREDACT_API_ENGINE", "").strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID) from None
            if not isinstance(parsed, list) or not parsed or not all(isinstance(p, str) for p in parsed):
                raise EnvironmentErrorMl(ReasonCode.CONFIG_INVALID)
            engine = tuple(parsed)
        configs = tuple(
            (cwd / p.strip()).resolve() for p in env.get("MLREDACT_API_CONFIG", "").split(os.pathsep) if p.strip()
        )
        origins = tuple(o.strip().rstrip("/") for o in env.get("MLREDACT_API_CORS_ORIGINS", "").split(",") if o.strip())
        return cls(
            token=token,
            jobs_dir=(cwd / env.get("MLREDACT_API_JOBS_DIR", ".jobs")).resolve(),
            engine=engine,
            engine_cwd=cwd,
            config_files=configs,
            default_workers=_int(env, "MLREDACT_API_WORKERS", 2, 1, 64),
            allow_dev_key=env.get("MLREDACT_API_ALLOW_DEV_KEY", "") == "1",
            max_upload_bytes=_int(env, "MLREDACT_API_MAX_UPLOAD_MB", 200, 1, 4096) * 1024 * 1024,
            job_timeout_s=float(_int(env, "MLREDACT_API_JOB_TIMEOUT_S", 3 * 3600, 1, 24 * 3600)),
            upload_ttl_s=_int(env, "MLREDACT_API_UPLOAD_TTL_S", 15 * 60, 30, 24 * 3600),
            link_ttl_s=_int(env, "MLREDACT_API_LINK_TTL_S", 15 * 60, 30, 24 * 3600),
            cors_origins=origins,
        )
