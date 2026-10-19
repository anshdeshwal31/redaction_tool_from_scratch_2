import type { Metadata } from "next";
import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";

import { LoginForm } from "@/components/login-form";
import { authOptions, googleConfigured } from "@/lib/auth";

export const metadata: Metadata = { title: "Sign in" };

type Props = { searchParams: Promise<{ callbackUrl?: string; error?: string }> };

/** Only same-site paths: a callback URL must never send a signed-in user to another site. */
function safeCallback(raw: string | undefined): string {
  if (!raw?.startsWith("/")) return "/";
  // Parse it the way the browser will: URL parsing strips tabs and newlines and reads '\' as '/', so
  // prefix checks alone let '/\t/evil.com' through as '//evil.com'.
  const base = "http://same.invalid";
  try {
    const url = new URL(raw, base);
    return url.origin === base ? url.pathname + url.search + url.hash : "/";
  } catch {
    return "/";
  }
}

export default async function LoginPage({ searchParams }: Props) {
  const { callbackUrl, error } = await searchParams;
  const target = safeCallback(callbackUrl);
  if (await getServerSession(authOptions)) redirect(target);
  return <LoginForm callbackUrl={target} google={googleConfigured()} initialError={error ?? null} />;
}
