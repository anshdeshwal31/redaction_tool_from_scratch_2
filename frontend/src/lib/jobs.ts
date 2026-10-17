/**
 * Job records as the backend API serves them (backend/src/mlredact/api/models.py, camelCase JSON).
 * Fields that are null on the backend are omitted from its JSON, hence the optional properties.
 */
import type { RenderStyle, RunRecord } from "./manifest";

export type JobState =
  | "awaiting_upload"
  | "queued"
  | "running"
  | "released"
  | "quarantined"
  | "failed"
  | "cancelled"
  | "interrupted";

/** States in which a job may still change: the UI keeps polling them. */
export const ACTIVE_STATES: ReadonlySet<JobState> = new Set(["awaiting_upload", "queued", "running"]);

export interface JobOptions {
  profile: string;
  style: RenderStyle;
  workers: number;
  /** Whether a caller scope id was supplied.  The id itself is never stored (like the manifest). */
  scope: "caller" | "document";
  seedCount: number;
  sealed: boolean;
  devKey: boolean;
}

/** Counts derived from the manifest when a job ends, so the job list never reads manifests. */
export interface JobSummary {
  pages: number;
  /** Mentions with an action other than keep. */
  replaced: number;
  kept: number;
  regions: number;
  lowEvidence: number;
  residual: number;
  checksPassed: number;
  checksTotal: number;
  reasons: string[];
  durationMs?: number;
}

export interface JobRecord {
  id: string;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  state: JobState;
  options: JobOptions;
  /** Set once the document has been uploaded. */
  input?: { bytes: number; sha256: string };
  uploadExpiresAt?: string;
  exitCode?: number;
  /** A content-free reason code when the job failed outside the engine's own quarantine path. */
  errorCode?: string;
  summary?: JobSummary;
}

export const OUTPUT_FILES = [
  "redacted.pdf",
  "manifest.json",
  "run.json",
  "quarantine.json",
  "sensitive-manifest.sealed.json",
] as const;
export type OutputFile = (typeof OUTPUT_FILES)[number];

export interface JobListItem extends JobRecord {
  queuePosition?: number;
}

export interface JobView {
  job: JobRecord;
  queuePosition?: number;
  files: OutputFile[];
  /** Absolute, signed, expiring URLs on the backend, one per artefact. */
  links: Partial<Record<OutputFile, string>>;
  linksExpireAt?: string;
  run?: RunRecord;
}

/** POST /api/jobs: the job, and where the browser uploads the document. */
export interface CreatedJob {
  id: string;
  uploadUrl: string;
  uploadExpiresAt: string;
  maxBytes: number;
}

export interface SystemInfo {
  engine: { command: string; cwd: string; configFiles: string[] };
  jobsDir: string;
  profiles: string[];
  defaults: { workers: number; maxUploadMb: number; jobTimeoutMin: number };
  surrogateKey: { fromEnv: boolean; devKeyAllowed: boolean };
  queue: { running?: string; queued: number };
}

export interface SelfcheckReport {
  ok: boolean;
  errorCode?: string;
  report?: {
    mlredact_version: string;
    profile: string;
    config_hash: string;
    runtime_profile: string;
    libraries: Record<string, string>;
    resources_verified: number;
    models_verified: string[];
  };
  durationMs: number;
}
