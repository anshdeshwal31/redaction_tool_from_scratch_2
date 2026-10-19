import "server-only";

import { getServerSession } from "next-auth";
import { NextResponse } from "next/server";

import { authOptions } from "@/lib/auth";

/*
 * Every API route needs a signed-in user (next-auth session).  Mutations also need a custom
 * header: browsers cannot send it cross-site without a CORS preflight this server never grants, so
 * another site cannot post to these routes with the user's cookie.
 */

export const UI_HEADER = "x-mlredact-ui";

export function fail(status: number, code: string): NextResponse {
  return NextResponse.json({ error: code }, { status });
}

/** Header and Origin rules for mutations; null when the request may proceed. */
export function checkMutation(req: Request): NextResponse | null {
  if (req.headers.get(UI_HEADER) !== "1") return fail(403, "E_UI_MISSING_HEADER");
  const origin = req.headers.get("origin");
  if (origin) {
    let originHost = "";
    try {
      originHost = new URL(origin).host;
    } catch {
      return fail(403, "E_UI_BAD_ORIGIN");
    }
    if (originHost.toLowerCase() !== (req.headers.get("host") ?? "").toLowerCase()) return fail(403, "E_UI_BAD_ORIGIN");
  }
  return null;
}

/** Returns an error response when the request must be refused, otherwise null. */
export async function guard(req: Request, mutating: boolean): Promise<NextResponse | null> {
  const session = await getServerSession(authOptions);
  if (!session?.user) return fail(401, "E_UI_UNAUTHENTICATED");
  return mutating ? checkMutation(req) : null;
}
