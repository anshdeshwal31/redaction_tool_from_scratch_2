import { backend, publicUrl, relay, withBackend } from "@/lib/server/backend";
import { fail, guard } from "@/lib/server/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const refused = await guard(req, false);
  if (refused) return refused;
  return withBackend(async () => relay(await backend("/v1/jobs")));
}

/**
 * Creates the job on the backend and returns a signed upload URL.  The browser then sends the
 * document straight to the backend; it never passes through this server.
 */
export async function POST(req: Request) {
  const refused = await guard(req, true);
  if (refused) return refused;
  const body: unknown = await req.json().catch(() => null);
  if (typeof body !== "object" || body === null || Array.isArray(body)) return fail(400, "E_UI_BAD_REQUEST");
  return withBackend(async () =>
    relay(await backend("/v1/jobs", { method: "POST", body }), (data) => {
      const created = data as { uploadPath: string };
      return { ...created, uploadUrl: publicUrl(created.uploadPath) };
    }),
  );
}
