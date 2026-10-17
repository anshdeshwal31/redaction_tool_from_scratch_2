import type { Metadata } from "next";
import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";

import { LoginForm } from "@/components/login-form";
import { authOptions, googleConfigured } from "@/lib/auth";

export const metadata: Metadata = { title: "Sign in" };

type Props = { searchParams: Promise<{ callbackUrl?: string; error?: string }> };

/** Only same-site paths: a callback URL must never send a signed-in user to another site. */
function safeCallback(raw: string | undefined): string {
  return raw && raw.startsWith("/") && !raw.startsWith("//") && !raw.startsWith("/\\") ? raw : "/";
}

export default async function LoginPage({ searchParams }: Props) {
  const { callbackUrl, error } = await searchParams;
  const target = safeCallback(callbackUrl);
  if (await getServerSession(authOptions)) redirect(target);
  return <LoginForm callbackUrl={target} google={googleConfigured()} initialError={error ?? null} />;
}
