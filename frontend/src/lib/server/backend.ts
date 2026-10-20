import "server-only";

import { NextResponse } from "next/server";

import { JOB_FILES, type JobFile } from "@/lib/jobs";

import { serverConfig } from "./config";

/*
 * Calls from this server to the backend API.  Only small JSON travels this way (Vercel caps a
 * function's request and response at 4.5 MB); documents and artefacts go straight between the
 * browser and the backend through signed links.
 */

const JOB_ID = /^[0-9a-f]{32}$/;

/** Path segments are validated before they are put in a backend URL. */
export const isJobId = (id: string) => JOB_ID.test(id);
export const isJobFile = (name: string): name is JobFile => (JOB_FILES as readonly string[]).includes(name);

export class BackendUnavailable extends Error {}
/** The backend did not answer within the call's time limit; it may still be working on the request. */
export class BackendTimeout extends BackendUnavailable {}

export async function backend(
  path: string,
  { method = "GET", body, timeoutMs = 30_000 }: { method?: string; body?: unknown; timeoutMs?: number } = {},
): Promise<Response> {
  const cfg = serverConfig();
  if (!cfg.apiToken) throw new BackendUnavailable("MLREDACT_API_TOKEN is not set");
  try {
    return await fetch(`${cfg.apiUrl}${path}`, {
      method,
      cache: "no-store",
      headers: {
        authorization: `Bearer ${cfg.apiToken}`,
        ...(body === undefined ? {} : { "content-type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    // AbortSignal.timeout rejects with a TimeoutError: the backend was reached but is still busy.
    if (err instanceof Error && err.name === "TimeoutError") throw new BackendTimeout("backend did not answer in time");
    throw new BackendUnavailable("backend unreachable");
  }
}

/** A backend-relative path (from a job response) as an absolute URL a browser can use. */
export function publicUrl(path: string): string {
  return `${serverConfig().publicApiUrl}${path}`;
}

/** Pass the backend's JSON (or its `{error}` code) through with the same status. */
export async function relay(res: Response, transform?: (data: unknown) => unknown): Promise<NextResponse> {
  const data: unknown = await res.json().catch(() => ({ error: "E_UI_BAD_BACKEND_RESPONSE" }));
  return NextResponse.json(res.ok && transform ? transform(data) : data, { status: res.status });
}

export function unavailable(): NextResponse {
  return NextResponse.json({ error: "E_UI_BACKEND_UNREACHABLE" }, { status: 502 });
}

/** Run a backend call, mapping a timeout to a 504 and an unreachable backend to a 502, each with a code. */
export async function withBackend(fn: () => Promise<NextResponse | Response>): Promise<NextResponse | Response> {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof BackendTimeout) return NextResponse.json({ error: "E_UI_BACKEND_TIMEOUT" }, { status: 504 });
    if (err instanceof BackendUnavailable) return unavailable();
    throw err;
  }
}
