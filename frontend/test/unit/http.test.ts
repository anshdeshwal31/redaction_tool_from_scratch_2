import { describe, expect, it } from "vitest";

import { checkMutation } from "@/lib/server/http";

const HOST = "redact.example.com";

function request(headers: Record<string, string>): Request {
  return new Request(`https://${HOST}/api/jobs`, { method: "POST", headers: { host: HOST, ...headers } });
}

async function refusal(req: Request): Promise<string | null> {
  const res = checkMutation(req);
  if (!res) return null;
  expect(res.status).toBe(403);
  return ((await res.json()) as { error: string }).error;
}

describe("mutations (cross-site requests)", () => {
  it("need the UI header, which another site cannot send without a CORS preflight", async () => {
    expect(await refusal(request({}))).toBe("E_UI_MISSING_HEADER");
    expect(await refusal(request({ "x-mlredact-ui": "yes" }))).toBe("E_UI_MISSING_HEADER");
    expect(await refusal(request({ "x-mlredact-ui": "1" }))).toBeNull();
  });

  it("refuse a foreign or malformed Origin", async () => {
    const base = { "x-mlredact-ui": "1" };
    expect(await refusal(request({ ...base, origin: "https://evil.example" }))).toBe("E_UI_BAD_ORIGIN");
    expect(await refusal(request({ ...base, origin: `https://${HOST}.evil.example` }))).toBe("E_UI_BAD_ORIGIN");
    expect(await refusal(request({ ...base, origin: "not a url" }))).toBe("E_UI_BAD_ORIGIN");
    expect(await refusal(request({ ...base, origin: `https://${HOST}` }))).toBeNull();
  });
});
