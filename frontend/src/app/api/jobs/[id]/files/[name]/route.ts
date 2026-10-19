import { NextResponse } from "next/server";

import { backend, isJobId, isOutputFile, publicUrl, relay, withBackend } from "@/lib/server/backend";
import { fail, guard } from "@/lib/server/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Ctx = { params: Promise<{ id: string; name: string }> };

/**
 * Hands out a fresh signed link to one artefact on the backend; the file itself never passes
 * through this server.
 *   ?download=1  redirects the browser there (a plain link or button works)
 *   ?as=json     returns {url}, for scripts that fetch the file (the PDF viewer, the manifest)
 */
export async function GET(req: Request, { params }: Ctx) {
  const refused = await guard(req, false);
  if (refused) return refused;
  const { id, name } = await params;
  if (!isJobId(id) || !isOutputFile(name)) return fail(404, "E_UI_NOT_FOUND");
  return withBackend(async () => {
    const res = await backend(`/v1/jobs/${id}`);
    if (!res.ok) return relay(res);
    const view = (await res.json()) as { links?: Record<string, string> };
    const path = view.links?.[name];
    if (!path) return fail(404, "E_UI_NOT_FOUND");
    const search = new URL(req.url).searchParams;
    const url = publicUrl(path) + (search.get("download") === "1" ? "&download=true" : "");
    if (search.get("as") === "json") return NextResponse.json({ url });
    return NextResponse.redirect(url, 302);
  });
}
