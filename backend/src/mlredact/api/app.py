"""FastAPI application.

Two kinds of caller:

* **The frontend's server**, with ``Authorization: Bearer <MLREDACT_API_TOKEN>``: creates jobs,
  reads status, cancels, deletes, runs the self-check.  Never a browser: the token stays server-side.
* **A signed-in user's browser**, with a signed, expiring link from a job response: uploads the
  document and downloads artefacts directly, so files never pass through the frontend's host (Vercel
  caps a function's request and response at 4.5 MB).  Only these two routes allow CORS.

Every error body is ``{"error": "<CODE>"}``: no messages, and never an echo of the request (FastAPI's
default validation error would repeat seed values back).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import Future
from contextlib import asynccontextmanager
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from mlredact import __version__
from mlredact.api.engine import EngineResult, run_short
from mlredact.api.links import link_key, sign, verify
from mlredact.api.models import (
    ACTIVE_STATES,
    OUTPUT_FILES,
    CreatedJob,
    CreateJob,
    Defaults,
    EngineInfo,
    InputInfo,
    JobList,
    JobListItem,
    JobOptions,
    JobRecord,
    JobView,
    KeyInfo,
    QueueInfo,
    SelfcheckReport,
    SelfcheckRequest,
    SystemInfo,
)
from mlredact.api.runner import Runner
from mlredact.api.settings import ApiSettings
from mlredact.api.store import JobStore, is_job_id, iso_from_unix, new_job_id, now_iso
from mlredact.config.loader import builtin_profiles

_CHUNK_WRITE = 1 << 20


class ApiError(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


def require_token(request: Request) -> None:
    """Server-to-server routes: ``Authorization: Bearer <MLREDACT_API_TOKEN>``."""
    expected: str = request.app.state.settings.token
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(value.strip().encode(), expected.encode()):
        raise ApiError(401, "E_API_UNAUTHORIZED")


# Module level, not inside create_app: FastAPI resolves string annotations against module globals.
Authorised = Annotated[None, Depends(require_token)]


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else ApiSettings.from_env()
    store = JobStore(settings.jobs_dir)
    runner = Runner(settings, store)
    key = link_key(settings.token)
    profiles = builtin_profiles()
    # One self-check at a time: it reads every model, and the frontend gives up on a slow one long
    # before it ends, so a retry joins the run in progress instead of starting another.
    checks_lock = threading.Lock()
    checks: dict[str, Future[tuple[EngineResult, int]]] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        runner.start()
        try:
            yield
        finally:
            await anyio.to_thread.run_sync(runner.stop)

    app = FastAPI(
        title="mlredact API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,  # nothing to browse on a live server
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.runner = runner
    app.state.store = store

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "PUT"],
            allow_headers=["content-type"],
            expose_headers=["content-disposition", "content-length"],
            max_age=600,
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return _error(400, "E_API_BAD_REQUEST")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "E_API_NOT_FOUND", 405: "E_API_METHOD_NOT_ALLOWED"}
        return _error(exc.status_code, codes.get(exc.status_code, f"E_API_HTTP_{exc.status_code}"))

    @app.exception_handler(Exception)
    async def _internal(_request: Request, _exc: Exception) -> JSONResponse:
        return _error(500, "E_API_INTERNAL")

    def job_or_404(job_id: str) -> JobRecord:
        job = store.read(job_id) if is_job_id(job_id) else None
        if job is None:
            raise ApiError(404, "E_API_NOT_FOUND")
        return job

    # ------------------------------------------------------------------------------- routes
    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/system")
    def system(_: Authorised) -> dict[str, Any]:
        running, queued = runner.status()
        return SystemInfo(
            engine=EngineInfo(
                command=" ".join(settings.engine),
                cwd=str(settings.engine_cwd),
                config_files=[str(p) for p in settings.config_files],
            ),
            jobs_dir=str(settings.jobs_dir),
            profiles=profiles,
            defaults=Defaults(
                workers=settings.default_workers,
                max_upload_mb=settings.max_upload_bytes // (1024 * 1024),
                job_timeout_min=round(settings.job_timeout_s / 60),
            ),
            # Presence only: the key itself never leaves the engine's environment.
            surrogate_key=KeyInfo(
                from_env=bool(os.environ.get("MLREDACT_SURROGATE_KEY", "").strip()),
                dev_key_allowed=settings.allow_dev_key,
            ),
            queue=QueueInfo(running=running, queued=queued),
        ).dump()

    @app.post("/v1/system/selfcheck")
    def selfcheck(body: SelfcheckRequest, _: Authorised) -> dict[str, Any]:
        if body.profile not in profiles:
            raise ApiError(400, "E_API_BAD_PROFILE")
        with checks_lock:
            running = checks.get(body.profile)
            if running is None and checks:
                raise ApiError(409, "E_API_SELFCHECK_BUSY")  # another profile's check is running
            if running is None:
                running = checks[body.profile] = Future()
                owner = True
            else:
                owner = False
        if owner:
            try:
                args = ["selfcheck", "--profile", body.profile]
                for path in settings.config_files:
                    args += ["--config", str(path)]
                t0 = time.monotonic()
                outcome = run_short(settings, args, timeout_s=600)
                running.set_result((outcome, int((time.monotonic() - t0) * 1000)))
            except BaseException as exc:
                running.set_exception(exc)
                raise
            finally:
                with checks_lock:
                    del checks[body.profile]
        result, duration = running.result()
        if result.code == 0:
            try:
                return SelfcheckReport(ok=True, report=json.loads(result.stdout), duration_ms=duration).dump()
            except ValueError:
                pass
        code = (
            "E_API_ENGINE_NOT_FOUND"
            if result.spawn_failed
            else "E_API_TIMEOUT"
            if result.timed_out
            else result.error_code or "E_API_ENGINE_FAILED"
        )
        return SelfcheckReport(ok=False, error_code=code, duration_ms=duration).dump()

    @app.get("/v1/jobs")
    def list_jobs(_: Authorised) -> dict[str, Any]:
        positions = runner.queue_positions()
        items = [JobListItem(**job.model_dump(), queue_position=positions.get(job.id)) for job in store.list_jobs()]
        return JobList(jobs=items).dump()

    @app.post("/v1/jobs", status_code=201)
    def create_job(body: CreateJob, _: Authorised) -> dict[str, Any]:
        if body.profile not in profiles:
            raise ApiError(400, "E_API_BAD_PROFILE")
        if body.dev_key and not settings.allow_dev_key:
            raise ApiError(403, "E_API_DEV_KEY_DISABLED")
        if body.input_bytes > settings.max_upload_bytes:
            raise ApiError(413, "E_API_UPLOAD_TOO_LARGE")
        job_id = new_job_id()
        in_dir = store.input_dir(job_id)
        in_dir.mkdir(parents=True, mode=0o700)
        if body.seeds:
            seeds = [{"type": s.type, "value": s.value} for s in body.seeds]
            (in_dir / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")
        if body.evaluator_key:
            (in_dir / "evaluator.pub").write_text(body.evaluator_key + "\n", encoding="ascii")
        token, expires = sign(key, "upload", job_id, "input", settings.upload_ttl_s)
        job = JobRecord(
            id=job_id,
            created_at=now_iso(),
            state="awaiting_upload",
            expected_bytes=body.input_bytes,
            upload_expires_at=iso_from_unix(expires),
            options=JobOptions(
                profile=body.profile,
                style=body.style,
                workers=body.workers or settings.default_workers,
                scope="caller" if body.scope_id else "document",
                seed_count=len(body.seeds),
                sealed=body.evaluator_key is not None,
                dev_key=body.dev_key,
            ),
        )
        store.write(job)
        if body.scope_id:
            runner.remember_scope(job_id, body.scope_id)
        return CreatedJob(
            id=job_id,
            upload_path=f"/v1/jobs/{job_id}/input?token={token}",
            upload_expires_at=iso_from_unix(expires),
            max_bytes=settings.max_upload_bytes,
        ).dump()

    @app.put("/v1/jobs/{job_id}/input", status_code=202)
    async def upload(job_id: str, token: Annotated[str, Query()], request: Request) -> dict[str, str]:
        """Browser upload of the document, with the signed link from ``POST /v1/jobs``."""
        if not is_job_id(job_id):
            raise ApiError(404, "E_API_NOT_FOUND")
        if problem := verify(key, token, "upload", job_id, "input"):
            raise ApiError(403, problem)
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > settings.max_upload_bytes:
            raise ApiError(413, "E_API_UPLOAD_TOO_LARGE")
        claim = runner.begin_upload(job_id)
        if claim == "gone":
            raise ApiError(404, "E_API_NOT_FOUND")
        if claim == "busy":
            raise ApiError(409, "E_API_UPLOAD_IN_PROGRESS")
        part = store.input_dir(job_id) / "input.part"
        try:
            # Read under the claim, so a delete or the sweep cannot remove the job while it uploads.
            # Store calls run on a thread: they retry with sleeps, which must not hold up the loop.
            job = await anyio.to_thread.run_sync(job_or_404, job_id)
            if job.state != "awaiting_upload":
                raise ApiError(409, "E_API_UPLOAD_STATE")
            expected = job.expected_bytes  # None for a job created before sizes were recorded
            limit = expected or settings.max_upload_bytes
            digest = hashlib.sha256()
            size = 0
            # Straight to the job's input directory: no other copy of the document is made.
            with part.open("wb") as f:
                buffer = bytearray()
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise ApiError(413, "E_API_UPLOAD_TOO_LARGE")
                    if size > limit:  # more than was declared
                        raise ApiError(400, "E_API_UPLOAD_SIZE_MISMATCH")
                    digest.update(chunk)
                    buffer += chunk
                    if len(buffer) >= _CHUNK_WRITE:
                        await anyio.to_thread.run_sync(f.write, bytes(buffer))
                        buffer.clear()
                if buffer:
                    await anyio.to_thread.run_sync(f.write, bytes(buffer))
            if size == 0:
                raise ApiError(400, "E_API_EMPTY_UPLOAD")
            if expected and size != expected:  # a truncated upload
                raise ApiError(400, "E_API_UPLOAD_SIZE_MISMATCH")
            queued = job.model_copy(
                update={
                    "state": "queued",
                    "expected_bytes": None,
                    "upload_expires_at": None,
                    "input": InputInfo(bytes=size, sha256=digest.hexdigest()),
                }
            )

            def commit() -> None:
                part.replace(store.input_dir(job_id) / "input")
                store.write(queued)
                runner.enqueue(job_id)

            await anyio.to_thread.run_sync(commit)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        finally:
            runner.end_upload(job_id)
        return {"state": "queued"}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, _: Authorised) -> dict[str, Any]:
        job = job_or_404(job_id)
        done = job.state not in ACTIVE_STATES and job.state != "awaiting_upload"
        files = store.outputs(job_id) if done else []
        links: dict[str, str] = {}
        expires: int | None = None
        for name in files:
            token, expires = sign(key, "file", job_id, name, settings.link_ttl_s)
            links[name] = f"/v1/jobs/{job_id}/files/{name}?token={token}"
        return JobView(
            job=job,
            queue_position=runner.queue_position(job_id),
            files=files,
            links=links,
            links_expire_at=iso_from_unix(expires) if expires else None,
            run=store.read_output_json(job_id, "run.json") if done else None,
        ).dump()

    @app.get("/v1/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str, token: Annotated[str, Query()], download: bool = False) -> FileResponse:
        """Browser download of an artefact, with a signed link from ``GET /v1/jobs/{id}``."""
        if not is_job_id(job_id) or name not in OUTPUT_FILES:
            raise ApiError(404, "E_API_NOT_FOUND")
        if problem := verify(key, token, "file", job_id, name):
            raise ApiError(403, problem)
        path = store.output_path(job_id, name)
        if not path.is_file():
            raise ApiError(404, "E_API_NOT_FOUND")
        return FileResponse(
            path,
            media_type="application/pdf" if name.endswith(".pdf") else "application/json",
            # Named after the job, never after the uploaded file (filenames often contain names).
            filename=f"mlredact-{job_id[:8]}-{name}",
            content_disposition_type="attachment" if download else "inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, _: Authorised) -> dict[str, bool]:
        job_or_404(job_id)
        if not runner.cancel(job_id):
            raise ApiError(409, "E_API_NOT_ACTIVE")
        return {"ok": True}

    @app.delete("/v1/jobs/{job_id}")
    def delete(job_id: str, _: Authorised) -> JSONResponse:
        job_or_404(job_id)
        result = runner.delete(job_id)
        if result == "busy":
            raise ApiError(409, "E_API_UPLOAD_IN_PROGRESS")
        return JSONResponse({"result": result}, status_code=200 if result == "deleted" else 202)

    return app
