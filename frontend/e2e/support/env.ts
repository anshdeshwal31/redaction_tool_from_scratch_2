import { randomBytes, scryptSync } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";

/*
 * Settings for the end-to-end run, from E2E_* environment variables (see e2e/README.md).
 *
 *   E2E_ENGINE=stub   (default) the local backend runs backend/tests/api/stub_engine.py: no models,
 *                     each job's outcome is picked by the document's first line (%STUB released, ...)
 *   E2E_ENGINE=real   the real engine with its models.  The local backend is started unless
 *                     E2E_BACKEND_URL (+ E2E_BACKEND_TOKEN) points at one already running, e.g. EC2
 *                     through an SSH tunnel.
 */
export type Engine = "stub" | "real";

const engine = (process.env.E2E_ENGINE ?? "stub").trim().toLowerCase();
if (engine !== "stub" && engine !== "real") {
  throw new Error(`E2E_ENGINE must be "stub" or "real", not "${engine}"`);
}
export const ENGINE: Engine = engine;
export const REAL = ENGINE === "real";

const trimSlash = (u: string) => u.trim().replace(/\/+$/, "");

/** A backend that is already running; Playwright then starts only the frontend. */
export const EXTERNAL_BACKEND = process.env.E2E_BACKEND_URL ? trimSlash(process.env.E2E_BACKEND_URL) : null;
if (EXTERNAL_BACKEND && !REAL) {
  throw new Error("E2E_BACKEND_URL needs E2E_ENGINE=real: the stub engine runs only on the local backend");
}
if (EXTERNAL_BACKEND && (process.env.E2E_BACKEND_TOKEN ?? "").trim().length < 32) {
  throw new Error("E2E_BACKEND_URL needs E2E_BACKEND_TOKEN: that backend's MLREDACT_API_TOKEN");
}

// Ports of their own, so a dev server already running on 3100/8100 is left alone.
export const FRONTEND_PORT = Number(process.env.E2E_FRONTEND_PORT ?? 3199);
export const BACKEND_PORT = Number(process.env.E2E_BACKEND_PORT ?? 8199);
export const FRONTEND_URL = `http://127.0.0.1:${FRONTEND_PORT}`;
export const BACKEND_URL = EXTERNAL_BACKEND ?? `http://127.0.0.1:${BACKEND_PORT}`;

/** Shared secret between the two servers.  Test-only for the local backend. */
export const API_TOKEN = EXTERNAL_BACKEND ? process.env.E2E_BACKEND_TOKEN!.trim() : "e2e-token-".padEnd(48, "x");
export const NEXTAUTH_SECRET = "e2e-nextauth-secret-".padEnd(64, "x");

/** The one email-and-password user the frontend is started with. */
export const USER = { email: "e2e@mlredact.test", password: "e2e-password-not-a-secret" };

/** MLREDACT_UI_USERS entry, in the format of scripts/hash-password.mjs. */
export function usersEnv(): string {
  const salt = randomBytes(16);
  const key = scryptSync(USER.password, salt, 32, { N: 16384, r: 8, p: 1, maxmem: 64 * 1024 * 1024 });
  return `${USER.email}=scrypt.${salt.toString("hex")}.${key.toString("hex")}`;
}

// A real job loads its models first (about 35 s with NER), then takes roughly 20–40 s a page on CPU.
export const TIMEOUTS = REAL
  ? { test: 20 * 60_000, job: 15 * 60_000, selfcheck: 5 * 60_000 }
  : { test: 90_000, job: 30_000, selfcheck: 30_000 };

export const FRONTEND_DIR = path.resolve(__dirname, "../..");
export const BACKEND_DIR = path.resolve(FRONTEND_DIR, "../backend");
/** Everything the run writes: signed-in state, generated documents, the local backend's jobs. */
export const GENERATED_DIR = path.resolve(FRONTEND_DIR, "e2e/.generated");
export const STORAGE_STATE = path.join(GENERATED_DIR, "user.json");
export const REAL_DOCS_DIR = path.join(GENERATED_DIR, "documents");
export const JOBS_DIR = path.join(GENERATED_DIR, `jobs-${ENGINE}`);

/** The backend's Python: E2E_PYTHON, else backend/.venv (made by `uv sync`). */
export function backendPython(): string {
  if (process.env.E2E_PYTHON) return process.env.E2E_PYTHON;
  const candidates = [path.join(BACKEND_DIR, ".venv/Scripts/python.exe"), path.join(BACKEND_DIR, ".venv/bin/python")];
  const found = candidates.find((p) => existsSync(p));
  if (!found) throw new Error("No backend/.venv found: run `uv sync` in backend/, or set E2E_PYTHON");
  return found;
}
