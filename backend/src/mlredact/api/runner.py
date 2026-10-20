"""The job queue: one job at a time, each a CLI run in its own process tree.

One job loads the NER models (~2.5 GB) and starts its own page workers, so jobs never overlap.  The
queue lives in this process; a restart marks unfinished jobs as interrupted and deletes their
inputs.  Uploads that never arrive expire and are swept away with their seeds.  A job that asked to
keep its original keeps it, once the job is released or quarantined, until it expires.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from mlredact.api import originals
from mlredact.api.engine import kill_tree, spawn, watch_error_code
from mlredact.api.models import ORIGINAL_FILE, JobRecord
from mlredact.api.settings import ApiSettings
from mlredact.api.store import JobStore, iso_from_unix, now_iso, summarise
from mlredact.config.loader import load_config
from mlredact.config.schema import LimitsConfig
from mlredact.security.logging import get_logger

_log = get_logger("mlredact.api")
SWEEP_EVERY_S = 15.0
# A reviewer compares these with their original; other outcomes have nothing to compare.
_KEEPS_ORIGINAL = frozenset({"released", "quarantined"})


@dataclass
class _Active:
    id: str
    proc: subprocess.Popen[bytes] | None = None
    cancelled: bool = False
    delete_on_exit: bool = False
    timed_out: bool = False
    # Set once the outcome is being recorded: a cancel can no longer change it, and a delete that
    # arrives from then on is carried out after the record is written.
    settling: bool = False
    removed: bool = False
    started: float = field(default_factory=time.monotonic)


def _log_id(job_id: str) -> str:
    return f"api-{job_id}"


class Runner:
    def __init__(self, settings: ApiSettings, store: JobStore) -> None:
        self._settings = settings
        self._store = store
        self._cv = threading.Condition()
        self._queue: deque[str] = deque()
        self._active: _Active | None = None
        # Values that must never be written to disk: the caller's scope id, until the job runs.
        self._scope_ids: dict[str, str] = {}
        self._uploading: set[str] = set()
        # Jobs a delete or the sweep is removing right now: an upload may not start into them.
        self._removing: set[str] = set()
        # Deleted jobs whose directory could not be removed yet (Windows handles); the sweep retries.
        self._pending_removal: set[str] = set()
        # Jobs whose outcome could not be recorded at all; the sweep retries marking them failed.
        self._unsettled: set[str] = set()
        # Serialises a queued job's cancel with a delete of the same job, so the cancel's record
        # write can never recreate a directory the delete has just removed.
        self._records = threading.Lock()
        self._stopping = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="mlredact-api-runner", daemon=True)
        # Its own thread: a long job must not hold back expiring uploads and their seeds.
        self._sweeper = threading.Thread(target=self._sweep_loop, name="mlredact-api-sweeper", daemon=True)

    # ------------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._recover()
        self._thread.start()
        self._sweeper.start()

    def stop(self) -> None:
        """Stop the worker and the sweep.  A job cut off here is recorded as interrupted."""
        with self._cv:
            self._stopping = True
            active = self._active
            self._cv.notify_all()
        self._stopped.set()
        if active is not None and active.proc is not None:
            kill_tree(active.proc)
        self._thread.join(timeout=30)
        self._sweeper.join(timeout=30)

    def _recover(self) -> None:
        for job in self._store.list_jobs():
            if job.state == "awaiting_upload":
                self._store.remove(job.id)  # never accepted a document; drop it with its seeds
            elif job.state in ("queued", "running"):
                self._store.remove_inputs(job.id)
                update = {"state": "interrupted", "error_code": "E_API_INTERRUPTED", "finished_at": now_iso()}
                self._store.write(job.model_copy(update=update))

    # ------------------------------------------------------------------------------- intake
    def remember_scope(self, job_id: str, scope_id: str) -> None:
        with self._cv:
            self._scope_ids[job_id] = scope_id

    def begin_upload(self, job_id: str) -> Literal["ok", "busy", "gone"]:
        """Claim a job for its upload.  The caller re-reads the job afterwards: a delete that finished
        just before the claim leaves nothing to upload into."""
        with self._cv:
            if job_id in self._removing:
                return "gone"
            if job_id in self._uploading:
                return "busy"
            self._uploading.add(job_id)
            return "ok"

    def end_upload(self, job_id: str) -> None:
        with self._cv:
            self._uploading.discard(job_id)

    def enqueue(self, job_id: str) -> None:
        with self._cv:
            self._queue.append(job_id)
            self._cv.notify_all()
        _log.info("api.job.queued", job=_log_id(job_id))

    def queue_position(self, job_id: str) -> int | None:
        with self._cv:
            return self._queue.index(job_id) if job_id in self._queue else None

    def queue_positions(self) -> dict[str, int]:
        """Every queued job's position, in one pass (a job list asks for all of them)."""
        with self._cv:
            return {job_id: i for i, job_id in enumerate(self._queue)}

    def status(self) -> tuple[str | None, int]:
        with self._cv:
            return (self._active.id if self._active else None), len(self._queue)

    # ------------------------------------------------------------------------------- control
    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job.  False when the job is not active."""
        proc = None
        with self._cv:
            if job_id in self._queue:
                self._queue.remove(job_id)
                self._scope_ids.pop(job_id, None)
                queued = True
            elif self._active is not None and self._active.id == job_id and not self._active.settling:
                self._active.cancelled = True
                proc = self._active.proc
                queued = False
            else:  # not queued or running, or already finishing: too late to cancel
                return False
        if queued:
            with self._records:
                self._store.remove_inputs(job_id)
                job = self._store.read(job_id)
                if job is not None:
                    self._store.write(job.model_copy(update={"state": "cancelled", "finished_at": now_iso()}))
        elif proc is not None:
            kill_tree(proc)
        return True

    def delete(self, job_id: str) -> Literal["deleted", "pending", "busy"]:
        """Delete a job and its artefacts.  A running job is stopped first and removed once it exits."""
        proc = None
        with self._cv:
            if job_id in self._uploading:
                return "busy"
            if self._active is not None and self._active.id == job_id:
                # Removed by the worker: by _settle, or after it when the job was already settling.
                self._active.delete_on_exit = True
                if not self._active.settling:
                    self._active.cancelled = True
                    proc = self._active.proc
                pending = True
            else:
                pending = False
                if job_id in self._queue:
                    self._queue.remove(job_id)
                self._scope_ids.pop(job_id, None)
                self._removing.add(job_id)
        if pending:
            if proc is not None:
                kill_tree(proc)
            return "pending"
        try:
            with self._records:
                self._store.remove(job_id)
        finally:
            with self._cv:
                self._removing.discard(job_id)
        return "deleted"

    # ------------------------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            with self._cv:
                if self._stopping:
                    return
                if not self._queue:
                    self._cv.wait()
                    continue
                job_id = self._queue.popleft()
                active = _Active(job_id)
                self._active = active
                scope = self._scope_ids.pop(job_id, None)
            try:
                self._run(active, scope)
            except Exception as exc:  # never let one job stop the queue
                _log.error("api.job.internal_error", job=_log_id(active.id), error_type=type(exc).__name__)
                try:
                    self._settle(active, code=None, spawn_failed=False, error_code="E_API_INTERNAL")
                except Exception as again:  # the sweep retries, so the job never stays "running"
                    _log.error("api.job.settle_failed", job=_log_id(active.id), error_type=type(again).__name__)
                    with self._cv:
                        self._unsettled.add(active.id)
            finally:
                with self._cv:
                    self._active = None
                    late_delete = active.delete_on_exit and not active.removed
                if late_delete:  # a delete that arrived while the outcome was being recorded
                    self._remove(active.id)

    def _args(self, job: JobRecord, scope_id: str | None) -> list[str]:
        o = job.options
        in_dir = self._store.input_dir(job.id)
        args = ["run", str(in_dir / "input"), "--out", str(self._store.output_dir(job.id))]
        args += ["--profile", o.profile, "--style", o.style, "--workers", str(o.workers)]
        for path in self._settings.config_files:
            args += ["--config", str(path)]
        if scope_id:
            args += ["--scope-id", scope_id]
        if o.seed_count:
            args += ["--seeds", str(in_dir / "seeds.json")]
        if o.dev_key:
            args.append("--dev-key")
        if o.sealed:
            args += ["--sensitive-to", str(in_dir / "evaluator.pub")]
        return args

    def _run(self, active: _Active, scope_id: str | None) -> None:
        job = self._store.read(active.id)
        if job is None or job.state != "queued":
            return
        self._store.write(job.model_copy(update={"state": "running", "started_at": now_iso()}))
        with self._cv:
            stop_now = active.cancelled or self._stopping
        if stop_now:  # cancelled, or the server is shutting down: never start the engine
            self._settle(active, code=None, spawn_failed=False, error_code=None)
            return
        self._store.output_dir(active.id).mkdir(parents=True, exist_ok=True, mode=0o700)
        _log.info("api.job.start", job=_log_id(active.id), workers=job.options.workers)
        try:
            proc = spawn(self._settings, self._args(job, scope_id))
        except OSError:
            self._settle(active, code=None, spawn_failed=True, error_code=None)
            return
        try:
            if self._attach(active, proc):  # a cancel or a shutdown arrived before the engine was attached
                kill_tree(proc)
            assert proc.stderr is not None
            error_code = watch_error_code(proc.stderr)
            try:
                proc.wait(timeout=self._settings.job_timeout_s)
            except subprocess.TimeoutExpired:
                active.timed_out = True
                kill_tree(proc)
                proc.wait()
        except BaseException:
            # Never leave an engine behind a job about to be marked failed: it would overlap the next
            # job, and its outputs could overwrite the record.
            kill_tree(proc)
            with contextlib.suppress(Exception):
                proc.wait(timeout=30)
            raise
        self._settle(active, code=proc.returncode, spawn_failed=False, error_code=error_code())

    def _attach(self, active: _Active, proc: subprocess.Popen[bytes]) -> bool:
        """Record the running process; True when the job was cancelled or the server began stopping
        meanwhile (other threads set those flags, so they are re-read under the lock).  stop() kills
        only an engine it can see, so one attached after it looked must be killed here."""
        with self._cv:
            active.proc = proc
            return active.cancelled or self._stopping

    def _settle(self, active: _Active, *, code: int | None, spawn_failed: bool, error_code: str | None) -> None:
        job_id = active.id
        with self._cv:
            active.settling = True
            delete = active.delete_on_exit
            stopping = self._stopping
        if delete:
            active.removed = True
            self._remove(job_id)
            return
        job = self._store.read(job_id)
        if job is None:
            # A failed delete is retried by the sweep.
            with contextlib.suppress(OSError):
                self._store.remove_inputs(job_id)
            return
        manifest = self._store.read_output_json(job_id, "manifest.json")
        update: dict[str, object] = {"finished_at": now_iso(), "exit_code": code}
        # The engine's own verdict wins over a cancel or timeout that raced with a finished run.
        if code == 0 and self._store.output_path(job_id, "redacted.pdf").is_file():
            update["state"] = "released"
        elif code == 2 and manifest is not None:
            update["state"] = "quarantined"
        elif active.cancelled:
            update["state"] = "cancelled"
        elif stopping:  # killed by a server shutdown, not by anything wrong with the document
            update["state"] = "interrupted"
            update["error_code"] = "E_API_INTERRUPTED"
        else:
            update["state"] = "failed"
            update["error_code"] = (
                "E_API_TIMEOUT"
                if active.timed_out
                else "E_API_ENGINE_NOT_FOUND"
                if spawn_failed
                else error_code or "E_API_ENGINE_FAILED"
            )
        if manifest is not None:
            update["summary"] = summarise(manifest, self._store.read_output_json(job_id, "run.json"))
        # Only once pages were analysed: an input rejected at intake has nothing to compare.
        wants_original = job.options.keep_original and update["state"] in _KEEPS_ORIGINAL
        if wants_original and manifest and manifest.get("pages") and (expires := self._keep_original(job)):
            update["original_expires_at"] = expires
        # A failed delete is retried by the sweep; the state update below must not be lost.
        with contextlib.suppress(OSError):
            self._store.remove_inputs(job_id)
        self._store.write(job.model_copy(update=update))
        _log.info(
            "api.job.done",
            job=_log_id(job_id),
            exit_code=code,
            duration_ms=round((time.monotonic() - active.started) * 1000.0, 1),
        )

    def _keep_original(self, job: JobRecord) -> str | None:
        """Keep the job's document for side-by-side review.  Returns when it expires, or None when it
        could not be kept (the job is recorded as usual, without it)."""
        days = self._settings.original_retention_days
        if days <= 0:
            return None
        try:
            try:
                limits = load_config(job.options.profile, files=self._settings.config_files).limits
            except Exception:  # the engine has already read this config; fall back to the defaults
                limits = LimitsConfig()
            directory = self._store.original_dir(job.id)
            directory.mkdir(mode=0o700, exist_ok=True)
            if originals.keep(self._store.input_dir(job.id) / "input", directory / ORIGINAL_FILE, limits):
                return iso_from_unix(time.time() + days * 86_400)
        except Exception as exc:
            _log.warning("api.job.original_not_kept", job=_log_id(job.id), error_type=type(exc).__name__)
        with contextlib.suppress(OSError):
            self._store.remove_original(job.id)
        return None

    def _remove(self, job_id: str) -> None:
        """Delete a job's directory; when that fails the sweep retries it, so the delete is not lost."""
        try:
            self._store.remove(job_id)
        except Exception as exc:  # called from the worker's finally: must not stop the queue
            with self._cv:
                self._pending_removal.add(job_id)
            _log.warning("api.job.remove_deferred", job=_log_id(job_id), error_type=type(exc).__name__)

    def _fail_unsettled(self, job_id: str) -> None:
        """Record a job whose outcome could not be written as failed, and delete its inputs."""
        self._store.remove_inputs(job_id)
        job = self._store.read(job_id)
        if job is not None and job.state in ("queued", "running"):
            update = {"state": "failed", "error_code": "E_API_INTERNAL", "finished_at": now_iso()}
            self._store.write(job.model_copy(update=update))

    def _sweep_loop(self) -> None:
        while not self._stopped.wait(SWEEP_EVERY_S):
            try:
                self._sweep()
            except Exception as exc:  # the next sweep tries again
                _log.error("api.sweep_failed", error_type=type(exc).__name__)

    def _sweep(self) -> None:
        """Drop uploads that never arrived, and inputs a finished job could not delete at the time."""
        now = time.time()
        with self._cv:
            busy = set(self._uploading) | set(self._queue) | ({self._active.id} if self._active else set())
            pending = set(self._pending_removal)
            unsettled = set(self._unsettled)
        for job_id in pending:
            try:
                self._store.remove(job_id)
            except OSError:
                continue
            with self._cv:
                self._pending_removal.discard(job_id)
        for job_id in unsettled - pending - busy:
            try:
                self._fail_unsettled(job_id)
            except OSError:
                continue
            with self._cv:
                self._unsettled.discard(job_id)
        for job in self._store.list_jobs():
            if job.id in busy or job.id in pending or job.id in unsettled:
                continue
            try:
                if job.state == "awaiting_upload":
                    expires = job.upload_expires_at
                    if expires is None or datetime.fromisoformat(expires).timestamp() < now:
                        with self._cv:
                            if job.id in self._uploading:  # began after the snapshot above
                                continue
                            # An upload that committed after list_jobs has already left this state
                            # (its record is written before the claim is released).
                            fresh = self._store.read(job.id)
                            if fresh is None or fresh.state != "awaiting_upload":
                                continue
                            self._removing.add(job.id)
                            # The caller's scope id goes with the job, never kept in memory.
                            self._scope_ids.pop(job.id, None)
                        try:
                            self._store.remove(job.id)
                        finally:
                            with self._cv:
                                self._removing.discard(job.id)
                elif job.state not in ("queued", "running") and self._store.input_dir(job.id).exists():
                    self._store.remove_inputs(job.id)
            except OSError:
                continue
