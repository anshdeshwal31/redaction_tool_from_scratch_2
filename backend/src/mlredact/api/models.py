"""Request and response bodies.  JSON uses camelCase (the frontend's convention); job records are
stored in the same shape.  The manifest and run record keep the engine's own snake_case."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from mlredact.core.types import EntityType

JobState = Literal[
    "awaiting_upload",
    "queued",
    "running",
    "released",
    "quarantined",
    "failed",
    "cancelled",
    "interrupted",
]
ACTIVE_STATES: frozenset[str] = frozenset({"queued", "running"})
RenderStyle = Literal["surrogate", "label", "blackout"]
OUTPUT_FILES: tuple[str, ...] = (
    "redacted.pdf",
    "manifest.json",
    "run.json",
    "quarantine.json",
    "sensitive-manifest.sealed.json",
)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


# ------------------------------------------------------------------------------------- records
class JobOptions(ApiModel):
    profile: str
    style: RenderStyle
    workers: int
    # Whether a caller scope id was given.  The id itself is never stored (like the manifest).
    scope: Literal["caller", "document"]
    seed_count: int
    sealed: bool
    dev_key: bool
    keep_original: bool = False


class InputInfo(ApiModel):
    bytes: int
    sha256: str


class JobSummary(ApiModel):
    pages: int
    replaced: int
    kept: int
    regions: int
    low_evidence: int
    residual: int
    checks_passed: int
    checks_total: int
    reasons: list[str]
    duration_ms: float | None = None


class JobRecord(ApiModel):
    id: str
    created_at: str
    state: JobState
    options: JobOptions
    input: InputInfo | None = None
    # The size declared when the job was created, while it waits for its upload: the upload must
    # match it exactly.
    expected_bytes: int | None = None
    upload_expires_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    # A content-free reason code when the job failed outside the engine's quarantine path.
    error_code: str | None = None
    summary: JobSummary | None = None


# ------------------------------------------------------------------------------------- requests
class Seed(ApiModel):
    type: str
    value: str = Field(min_length=1, max_length=500)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        EntityType(v)  # ValueError for unknown types
        return v

    @field_validator("value")
    @classmethod
    def _clean_value(cls, v: str) -> str:
        v = v.strip()
        if not v or _CONTROL.search(v):
            raise ValueError("invalid seed value")
        return v


class CreateJob(ApiModel):
    profile: str = "client"
    style: RenderStyle = "surrogate"
    workers: int | None = Field(default=None, ge=1, le=64)
    scope_id: str | None = Field(default=None, max_length=200)
    seeds: list[Seed] = Field(default_factory=list, max_length=500)
    evaluator_key: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    dev_key: bool = False
    # Keep the document after the job, for side-by-side review (released and quarantined jobs).
    keep_original: bool = False
    # Declared document size: the upload must match it exactly.
    input_bytes: int = Field(gt=0)

    @field_validator("scope_id")
    @classmethod
    def _clean_scope(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if _CONTROL.search(v):
            raise ValueError("invalid scope id")
        return v or None


class SelfcheckRequest(ApiModel):
    profile: str = "client"


# ------------------------------------------------------------------------------------- responses
class CreatedJob(ApiModel):
    id: str
    # Path (relative to the API's base URL) the browser PUTs the document to, with its signed token.
    upload_path: str
    upload_expires_at: str
    max_bytes: int


class JobListItem(JobRecord):
    queue_position: int | None = None


class JobList(ApiModel):
    jobs: list[JobListItem]


class JobView(ApiModel):
    job: JobRecord
    queue_position: int | None = None
    files: list[str]
    # Signed, expiring paths for each artefact, for the browser to fetch directly.
    links: dict[str, str]
    links_expire_at: str | None = None
    run: dict[str, Any] | None = None


class EngineInfo(ApiModel):
    command: str
    cwd: str
    config_files: list[str]


class Defaults(ApiModel):
    workers: int
    max_upload_mb: int
    job_timeout_min: int
    # 0: originals are never kept.
    original_retention_days: int


class KeyInfo(ApiModel):
    from_env: bool
    # A deployment config gives the engine a key: surrogate.key_file, or the development key.
    from_config: bool
    dev_key_allowed: bool


class QueueInfo(ApiModel):
    running: str | None = None
    queued: int


class SystemInfo(ApiModel):
    engine: EngineInfo
    jobs_dir: str
    profiles: list[str]
    defaults: Defaults
    surrogate_key: KeyInfo
    queue: QueueInfo


class SelfcheckReport(ApiModel):
    ok: bool
    error_code: str | None = None
    report: dict[str, Any] | None = None
    duration_ms: int
