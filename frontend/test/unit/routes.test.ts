/**
 * The API routes as thin, session-checked forwards to the backend.  next-auth's session lookup and
 * the network are mocked: the backend is a function recording what it was asked.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next-auth", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next-auth")>()),
  getServerSession: vi.fn(),
}));

import { getServerSession } from "next-auth";

import * as files from "@/app/api/jobs/[id]/files/[name]/route";
import * as job from "@/app/api/jobs/[id]/route";
import * as jobs from "@/app/api/jobs/route";

const TOKEN = "s".repeat(40);
const ID = "a".repeat(32);
const SITE = "https://redact.example.com";

type Call = { url: string; method: string; auth: string | null; body: unknown };
let calls: Call[] = [];
let reply: (call: Call) => Response;

beforeEach(() => {
  process.env.MLREDACT_API_URL = "http://backend.internal:8100";
  process.env.MLREDACT_API_PUBLIC_URL = "https://api.example.com";
  process.env.MLREDACT_API_TOKEN = TOKEN;
  vi.mocked(getServerSession).mockResolvedValue({ user: { email: "demo@example.com" }, expires: "" });
  calls = [];
  reply = () => Response.json({});
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const headers = new Headers(init.headers);
      const call = {
        url,
        method: init.method ?? "GET",
        auth: headers.get("authorization"),
        body: init.body ? JSON.parse(String(init.body)) : undefined,
      };
      calls.push(call);
      return reply(call);
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const get = (path: string) => new Request(`${SITE}${path}`, { headers: { host: "redact.example.com" } });
const post = (path: string, body: unknown, header = true) =>
  new Request(`${SITE}${path}`, {
    method: "POST",
    body: JSON.stringify(body),
    headers: { host: "redact.example.com", "content-type": "application/json", ...(header ? { "x-mlredact-ui": "1" } : {}) },
  });
const params = <T extends object>(p: T) => ({ params: Promise.resolve(p) });

describe("every route", () => {
  it("needs a signed-in user", async () => {
    vi.mocked(getServerSession).mockResolvedValue(null);
    const res = await jobs.GET(get("/api/jobs"));
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual({ error: "E_UI_UNAUTHENTICATED" });
    expect(calls).toEqual([]);
  });

  it("reports an unreachable backend with a code", async () => {
    reply = () => {
      throw new TypeError("fetch failed");
    };
    const res = await jobs.GET(get("/api/jobs"));
    expect([res.status, await res.json()]).toEqual([502, { error: "E_UI_BACKEND_UNREACHABLE" }]);
  });

  it("reports a backend that is still busy as a timeout, not as unreachable", async () => {
    reply = () => {
      throw new DOMException("The operation was aborted due to timeout", "TimeoutError");
    };
    const res = await jobs.GET(get("/api/jobs"));
    expect([res.status, await res.json()]).toEqual([504, { error: "E_UI_BACKEND_TIMEOUT" }]);
  });
});

describe("GET /api/jobs", () => {
  it("forwards with the server-held token, which never reaches the browser", async () => {
    reply = () => Response.json({ jobs: [] });
    const res = await jobs.GET(get("/api/jobs"));
    expect(await res.json()).toEqual({ jobs: [] });
    expect(calls).toEqual([{ url: "http://backend.internal:8100/v1/jobs", method: "GET", auth: `Bearer ${TOKEN}`, body: undefined }]);
  });
});

describe("POST /api/jobs", () => {
  it("needs the UI header", async () => {
    const res = await jobs.POST(post("/api/jobs", { inputBytes: 5 }, false));
    expect(res.status).toBe(403);
    expect(calls).toEqual([]);
  });

  it("creates the job and returns an absolute upload URL on the backend", async () => {
    reply = () =>
      Response.json({ id: ID, uploadPath: `/v1/jobs/${ID}/input?token=t`, uploadExpiresAt: "x", maxBytes: 9 }, { status: 201 });
    const options = { profile: "broad", inputBytes: 5, seeds: [{ type: "person", value: "Jane" }] };
    const res = await jobs.POST(post("/api/jobs", options));
    expect(res.status).toBe(201);
    expect((await res.json()).uploadUrl).toBe(`https://api.example.com/v1/jobs/${ID}/input?token=t`);
    expect(calls[0]).toMatchObject({ method: "POST", body: options });
  });

  it("passes the backend's error code through", async () => {
    reply = () => Response.json({ error: "E_API_BAD_PROFILE" }, { status: 400 });
    const res = await jobs.POST(post("/api/jobs", { inputBytes: 5, profile: "x" }));
    expect([res.status, await res.json()]).toEqual([400, { error: "E_API_BAD_PROFILE" }]);
  });
});

describe("GET /api/jobs/[id]", () => {
  it("refuses ids that could reach other backend routes", async () => {
    for (const id of ["../system", "A".repeat(32), "a".repeat(33), `${"a".repeat(31)}/`]) {
      expect((await job.GET(get("/api/jobs/x"), params({ id }))).status).toBe(404);
    }
    expect(calls).toEqual([]);
  });

  it("turns the backend's signed paths into absolute URLs", async () => {
    reply = () => Response.json({ job: { id: ID }, files: ["run.json"], links: { "run.json": `/v1/jobs/${ID}/files/run.json?token=t` } });
    const view = await (await job.GET(get(`/api/jobs/${ID}`), params({ id: ID }))).json();
    expect(view.links).toEqual({ "run.json": `https://api.example.com/v1/jobs/${ID}/files/run.json?token=t` });
  });
});

describe("GET /api/jobs/[id]/files/[name]", () => {
  beforeEach(() => {
    reply = () =>
      Response.json({ job: { id: ID }, files: ["redacted.pdf"], links: { "redacted.pdf": `/v1/jobs/${ID}/files/redacted.pdf?token=t` } });
  });

  it("redirects a download to a fresh signed link on the backend", async () => {
    const res = await files.GET(get(`/api/jobs/${ID}/files/redacted.pdf?download=1`), params({ id: ID, name: "redacted.pdf" }));
    expect(res.status).toBe(302);
    expect(res.headers.get("location")).toBe(`https://api.example.com/v1/jobs/${ID}/files/redacted.pdf?token=t&download=true`);
  });

  it("returns the link as JSON for scripts", async () => {
    const res = await files.GET(get(`/api/jobs/${ID}/files/redacted.pdf?as=json`), params({ id: ID, name: "redacted.pdf" }));
    expect(await res.json()).toEqual({ url: `https://api.example.com/v1/jobs/${ID}/files/redacted.pdf?token=t` });
  });

  it("serves only known artefacts that exist", async () => {
    expect((await files.GET(get("/x"), params({ id: ID, name: "job.json" }))).status).toBe(404);
    expect((await files.GET(get("/x"), params({ id: ID, name: "run.json" }))).status).toBe(404); // not in this job
  });
});
