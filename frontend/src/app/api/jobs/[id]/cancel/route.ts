import { backend, isJobId, relay, withBackend } from "@/lib/server/backend";
import { fail, guard } from "@/lib/server/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const refused = await guard(req, true);
  if (refused) return refused;
  const { id } = await params;
  if (!isJobId(id)) return fail(404, "E_UI_NOT_FOUND");
  return withBackend(async () => relay(await backend(`/v1/jobs/${id}/cancel`, { method: "POST" })));
}
