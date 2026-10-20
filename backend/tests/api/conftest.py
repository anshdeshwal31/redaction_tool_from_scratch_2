"""Fixtures for API tests: the app on a temporary jobs directory, with the stub engine behind it."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlredact.api.app import create_app
from mlredact.api.settings import ApiSettings

STUB = Path(__file__).with_name("stub_engine.py")
TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ORIGIN = "http://localhost:3100"


def make_settings(jobs_dir: Path, **changes: Any) -> ApiSettings:
    base = ApiSettings(
        token=TOKEN,
        jobs_dir=jobs_dir,
        engine=(sys.executable, str(STUB)),
        engine_cwd=Path.cwd(),
        config_files=(),
        default_workers=2,
        allow_dev_key=True,
        max_upload_bytes=1024 * 1024,
        job_timeout_s=60.0,
        upload_ttl_s=600,
        link_ttl_s=600,
        cors_origins=(ORIGIN,),
        original_retention_days=7,
    )
    return replace(base, **changes)


def stub_input(scenario: str, delay_ms: int = 0) -> bytes:
    return f"%STUB {scenario}{f' delay={delay_ms}' if delay_ms else ''}\n".encode()


def stub_pdf(scenario: str) -> bytes:
    """An input that also reads as a PDF (the API keeps only PDFs and scans as originals)."""
    return b"%PDF-1.7\n" + stub_input(scenario)


class Api:
    """A test client plus the helpers every test needs."""

    def __init__(self, client: TestClient, jobs_dir: Path) -> None:
        self.client = client
        self.jobs_dir = jobs_dir

    def create(self, body: bytes, **options: Any) -> dict[str, Any]:
        res = self.client.post("/v1/jobs", json={"inputBytes": len(body), **options}, headers=AUTH)
        assert res.status_code == 201, res.text
        data: dict[str, Any] = res.json()
        return data

    def submit(
        self, scenario: str = "released", *, delay_ms: int = 0, body: bytes | None = None, **options: Any
    ) -> str:
        body = stub_input(scenario, delay_ms) if body is None else body
        created = self.create(body, **options)
        res = self.client.put(created["uploadPath"], content=body, headers={"content-type": "application/octet-stream"})
        assert res.status_code == 202, res.text
        job_id: str = created["id"]
        return job_id

    def view(self, job_id: str) -> dict[str, Any]:
        res = self.client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert res.status_code == 200, res.text
        data: dict[str, Any] = res.json()
        return data

    def wait(
        self, job_id: str, until: Callable[[dict[str, Any]], bool] | None = None, timeout: float = 30
    ) -> dict[str, Any]:
        """Poll until the job leaves queued/running (or ``until`` holds); returns the job view."""
        done = until or (lambda v: v["job"]["state"] not in ("queued", "running", "awaiting_upload"))
        deadline = time.monotonic() + timeout
        while True:
            view = self.view(job_id)
            if done(view):
                return view
            if time.monotonic() > deadline:
                raise AssertionError(f"job {job_id} stuck in {view['job']['state']}")
            time.sleep(0.05)

    def job_file(self, job_id: str) -> str:
        return (self.jobs_dir / job_id / "job.json").read_text("utf-8")


@pytest.fixture
def api_factory(tmp_path: Path) -> Iterator[Callable[..., Api]]:
    clients: list[TestClient] = []

    def factory(**changes: Any) -> Api:
        jobs_dir = changes.pop("jobs_dir", tmp_path / "jobs")
        client = TestClient(create_app(make_settings(jobs_dir, **changes)))
        client.__enter__()  # runs the lifespan: starts the runner thread
        clients.append(client)
        return Api(client, jobs_dir)

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def api(api_factory: Callable[..., Api]) -> Api:
    return api_factory()
