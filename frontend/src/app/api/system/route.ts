import { backend, relay, withBackend } from "@/lib/server/backend";
import { guard } from "@/lib/server/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const refused = await guard(req, false);
  if (refused) return refused;
  return withBackend(async () => relay(await backend("/v1/system")));
}
