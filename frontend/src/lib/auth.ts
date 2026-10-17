import "server-only";

import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";

import { parseUsers, verifyPassword } from "@/lib/server/passwords";

/*
 * Sign-in for the frontend (next-auth v4, JWT sessions, no database):
 *  - email and password, for the users listed in MLREDACT_UI_USERS;
 *  - Google, when GOOGLE_CLIENT_ID/SECRET are set, for verified addresses on the allowlist
 *    (MLREDACT_UI_GOOGLE_ALLOWED_EMAILS / _DOMAINS).  With no allowlist, no Google account gets in.
 */

const list = (raw: string | undefined) =>
  new Set(
    (raw ?? "")
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  );

export function googleConfigured(): boolean {
  return Boolean(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET);
}

export function googleAllowed(email: string | null | undefined, verified: boolean | undefined): boolean {
  if (!email || verified !== true) return false;
  const address = email.trim().toLowerCase();
  const domain = address.split("@")[1] ?? "";
  return (
    list(process.env.MLREDACT_UI_GOOGLE_ALLOWED_EMAILS).has(address) ||
    (domain !== "" && list(process.env.MLREDACT_UI_GOOGLE_ALLOWED_DOMAINS).has(domain))
  );
}

export const authOptions: NextAuthOptions = {
  session: { strategy: "jwt", maxAge: 8 * 60 * 60 },
  pages: { signIn: "/login", error: "/login" },
  providers: [
    CredentialsProvider({
      name: "Email and password",
      credentials: { email: { type: "email" }, password: { type: "password" } },
      async authorize(credentials) {
        const email = credentials?.email?.trim().toLowerCase() ?? "";
        const password = credentials?.password ?? "";
        const hash = parseUsers(process.env.MLREDACT_UI_USERS).get(email);
        // Always run the hash check, even for unknown emails, so timing reveals nothing.
        const ok = await verifyPassword(password, hash);
        return ok && email ? { id: email, email, name: email } : null;
      },
    }),
    ...(googleConfigured()
      ? [
          GoogleProvider({
            clientId: process.env.GOOGLE_CLIENT_ID as string,
            clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
          }),
        ]
      : []),
  ],
  callbacks: {
    async signIn({ account, profile }) {
      if (account?.provider !== "google") return true;
      const p = profile as { email?: string; email_verified?: boolean } | undefined;
      return googleAllowed(p?.email, p?.email_verified);
    },
  },
};
