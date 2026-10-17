"""On-disk job store.

One directory per job::

    <jobs>/<id>/job.json   the record (no document text, no filename, no scope id, no seed values)
    <jobs>/<id>/in/        input, seeds, evaluator key: deleted as soon as the job ends
    <jobs>/<id>/out/       the CLI's fixed-name artefacts
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mlredact.api.models import OUTPUT_FILES, JobRecord, JobSummary

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_RETRYABLE = (PermissionError,)  # Windows refuses to replace a file another handle has open


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso_from_unix(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_job_id() -> str:
    return uuid.uuid4().hex


def is_job_id(job_id: str) -> bool:
    return bool(_JOB_ID.fullmatch(job_id))


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def job_dir(self, job_id: str) -> Path:
        if not is_job_id(job_id):
            raise ValueError("invalid job id")
        return self.root / job_id

    def input_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "in"

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "out"

    def output_path(self, job_id: str, name: str) -> Path:
        if name not in OUTPUT_FILES:
            raise ValueError("unknown artefact")
        return self.output_dir(job_id) / name

    # ---------------------------------------------------------------------------------- records
    def write(self, job: JobRecord) -> None:
        directory = self.job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = directory / "job.json"
        # A unique temp name per write: a cancel racing the final update must not share one.
        tmp = directory / f"job.json.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(job.dump(), indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):
            tmp.chmod(0o600)
        for attempt in range(41):
            try:
                tmp.replace(target)
                return
            except _RETRYABLE:
                if attempt == 40:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(0.01 + attempt * 0.005)

    def read(self, job_id: str) -> JobRecord | None:
        if not is_job_id(job_id):
            return None
        path = self.root / job_id / "job.json"
        for attempt in range(41):
            try:
                return JobRecord.model_validate_json(path.read_text("utf-8"))
            except _RETRYABLE:
                # Windows: the file is being replaced by a concurrent write this very moment.
                if attempt == 40:
                    return None
                time.sleep(0.005 + attempt * 0.005)
            except (OSError, ValidationError):
                return None
        return None

    def list_jobs(self) -> list[JobRecord]:
        jobs = [j for p in self.root.iterdir() if is_job_id(p.name) and (j := self.read(p.name)) is not None]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    # ---------------------------------------------------------------------------------- files
    def remove_inputs(self, job_id: str) -> None:
        _rmtree(self.input_dir(job_id))

    def remove(self, job_id: str) -> None:
        _rmtree(self.job_dir(job_id))

    def outputs(self, job_id: str) -> list[str]:
        return [n for n in OUTPUT_FILES if (self.output_dir(job_id) / n).is_file()]

    def read_output_json(self, job_id: str, name: str) -> dict[str, Any] | None:
        try:
            data = json.loads(self.output_path(job_id, name).read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


def _rmtree(path: Path) -> None:
    # Retries cover Windows, where a just-killed child may still hold a handle for a moment.
    for attempt in range(25):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 24:
                raise
            time.sleep(0.2)


def summarise(manifest: dict[str, Any], run: dict[str, Any] | None) -> JobSummary:
    mentions = manifest.get("mentions", [])
    checks = manifest.get("verification", {}).get("checks", [])
    timings = (run or {}).get("timings_ms", {})
    return JobSummary(
        pages=len(manifest.get("pages", [])) or int(manifest.get("job", {}).get("page_count", 0)),
        replaced=sum(1 for m in mentions if m.get("action") != "keep"),
        kept=sum(1 for m in mentions if m.get("action") == "keep"),
        regions=len(manifest.get("regions", [])),
        low_evidence=len(manifest.get("low_evidence", [])),
        residual=len(manifest.get("residual_boxes", [])),
        checks_passed=sum(1 for c in checks if c.get("passed")),
        checks_total=len(checks),
        reasons=list(manifest.get("reasons", [])),
        duration_ms=timings.get("total_ms"),
    )
