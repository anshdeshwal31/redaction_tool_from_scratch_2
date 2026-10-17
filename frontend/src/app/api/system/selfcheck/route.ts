import { backend, relay, withBackend } from "@/lib/server/backend";
import { fail, guard } from "@/lib/server/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Hashing every pinned model takes a while; give the function room where the host allows it.
export const maxDuration = 300;

export async function POST(req: Request) {
  const refused = await guard(req, true);
  if (refused) return refused;
  const body = (await req.json().catch(() => ({}))) as { profile?: unknown };
  if (typeof body.profile !== "string") return fail(400, "E_UI_BAD_REQUEST");
  return withBackend(async () =>
    relay(
      await backend("/v1/system/selfcheck", { method: "POST", body: { profile: body.profile }, timeoutMs: 290_000 }),
    ),
  );
}
