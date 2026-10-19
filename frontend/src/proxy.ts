import { getToken } from "next-auth/jwt";
import { type NextRequest, NextResponse } from "next/server";

/*
 * Sends signed-out visitors to /login.  An optimistic check only: every API route checks the
 * session again on the server (lib/server/http.ts), so nothing relies on this alone.
 */
export async function proxy(req: NextRequest) {
  const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });
  if (token) return NextResponse.next();
  if (req.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "E_UI_UNAUTHENTICATED" }, { status: 401 });
  }
  const login = new URL("/login", req.url);
  login.searchParams.set("callbackUrl", req.nextUrl.pathname + req.nextUrl.search);
  return NextResponse.redirect(login);
}

export const config = {
  // Everything except sign-in itself, next-auth's own routes and static assets.
  matcher: ["/((?!login|api/auth|_next/static|_next/image|icon.svg|favicon.ico|pdf.worker.min.mjs).*)"],
};
