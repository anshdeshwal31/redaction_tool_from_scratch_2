import { backend, isJobId, publicUrl, relay, withBackend } from "@/lib/server/backend";
import { fail, guard } from "@/lib/server/http";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Ctx = { params: Promise<{ id: string }> };

export async function GET(req: Request, { params }: Ctx) {
  const refused = await guard(req, false);
  if (refused) return refused;
  const { id } = await params;
  if (!isJobId(id)) return fail(404, "E_UI_NOT_FOUND");
  return withBackend(async () =>
    relay(await backend(`/v1/jobs/${id}`), (data) => {
      const view = data as { links?: Record<string, string> };
      const links = Object.fromEntries(Object.entries(view.links ?? {}).map(([name, path]) => [name, publicUrl(path)]));
      return { ...view, links };
    }),
  );
}

export async function DELETE(req: Request, { params }: Ctx) {
  const refused = await guard(req, true);
  if (refused) return refused;
  const { id } = await params;
  if (!isJobId(id)) return fail(404, "E_UI_NOT_FOUND");
  return withBackend(async () => relay(await backend(`/v1/jobs/${id}`, { method: "DELETE" })));
}
