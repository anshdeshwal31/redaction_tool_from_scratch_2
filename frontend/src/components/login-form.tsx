"use client";

import { AlertTriangle } from "lucide-react";
import { signIn } from "next-auth/react";
import { useState } from "react";

import { Button, Spinner } from "./ui";
import { Aurora, GlassChip } from "./visuals";

// next-auth error codes, in words.  Never say which part was wrong: that would confirm an email.
const ERRORS: Record<string, string> = {
  CredentialsSignin: "That email and password don't match an account.",
  AccessDenied: "This Google account isn't allowed to sign in. Ask your administrator to add it.",
  OAuthSignin: "Google sign-in could not start. Try again.",
  OAuthCallback: "Google sign-in did not complete. Try again.",
  SessionRequired: "Sign in to continue.",
};

const inputClass =
  "h-12 w-full border border-line-strong bg-surface-2 px-3.5 text-[0.9375rem] text-ink placeholder:text-muted focus:border-ink focus:outline-none";

function GoogleMark() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden="true">
      <path fill="#4285F4" d="M22.6 12.2c0-.8-.1-1.5-.2-2.2H12v4.2h5.9a5 5 0 0 1-2.2 3.3v2.7h3.6c2.1-1.9 3.3-4.8 3.3-8z" />
      <path fill="#34A853" d="M12 23c3 0 5.5-1 7.3-2.7l-3.6-2.8c-1 .7-2.2 1.1-3.7 1.1-2.9 0-5.3-1.9-6.2-4.5H2.1v2.8A11 11 0 0 0 12 23z" />
      <path fill="#FBBC05" d="M5.8 14.1a6.6 6.6 0 0 1 0-4.2V7.1H2.1a11 11 0 0 0 0 9.8l3.7-2.8z" />
      <path fill="#EA4335" d="M12 5.4c1.6 0 3.1.6 4.2 1.7l3.2-3.2A11 11 0 0 0 2.1 7.1l3.7 2.8C6.7 7.3 9.1 5.4 12 5.4z" />
    </svg>
  );
}

export function LoginForm({
  callbackUrl,
  google,
  initialError,
}: {
  callbackUrl: string;
  google: boolean;
  initialError: string | null;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState<"password" | "google" | null>(null);
  const [error, setError] = useState<string | null>(
    initialError ? (ERRORS[initialError] ?? "Sign-in failed. Try again.") : null,
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy("password");
    setError(null);
    const res = await signIn("credentials", { email, password, redirect: false, callbackUrl });
    if (res?.ok && !res.error) {
      // A full navigation, not router.replace: pages prefetched while signed out are cached as
      // redirects to /login, and reusing them would loop.
      window.location.assign(callbackUrl);
      return;
    }
    setError(ERRORS[res?.error ?? ""] ?? ERRORS.CredentialsSignin ?? null);
    setBusy(null);
  }

  return (
    <div className="grid min-h-[calc(100dvh-10rem)] items-stretch gap-2 py-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:py-16">
      <section className="flex flex-col justify-center bg-surface px-6 py-12 sm:px-12">
        <span className="text-xs text-muted">Sign in</span>
        <h1 className="display mt-4 text-[clamp(2.75rem,5vw,4.25rem)] leading-[0.98] text-ink">
          Welcome back.
        </h1>
        <p className="mt-4 max-w-sm text-[0.9375rem] leading-relaxed text-ink-2">
          Sign in to redact documents and review their results.
        </p>

        <form onSubmit={submit} className="mt-10 max-w-sm space-y-3" noValidate>
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium text-ink">Email</span>
            <input
              type="email"
              className={inputClass}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              required
              autoFocus
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium text-ink">Password</span>
            <input
              type="password"
              className={inputClass}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          <div className="min-h-6 pt-1 text-sm" aria-live="polite">
            {error ? (
              <span className="inline-flex items-start gap-1.5 text-crit-text" role="alert">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
                {error}
              </span>
            ) : null}
          </div>
          <Button
            type="submit"
            variant="primary"
            size="lg"
            className="w-full"
            disabled={busy !== null || !email || !password}
          >
            {busy === "password" ? <Spinner /> : null}
            Sign in
          </Button>
        </form>

        {google ? (
          <div className="max-w-sm">
            <div className="my-6 flex items-center gap-3 text-xs text-muted" aria-hidden>
              <span className="h-px flex-1 bg-line" />
              or
              <span className="h-px flex-1 bg-line" />
            </div>
            <Button
              size="lg"
              className="w-full"
              disabled={busy !== null}
              onClick={() => {
                setBusy("google");
                void signIn("google", { callbackUrl });
              }}
            >
              {busy === "google" ? <Spinner /> : <GoogleMark />}
              Continue with Google
            </Button>
          </div>
        ) : null}
      </section>

      <section className="relative hidden min-h-96 place-items-center overflow-hidden lg:grid" aria-hidden>
        <Aurora tone="dusk" />
        <div className="relative px-10 text-center text-white">
          <p className="display text-[clamp(2.5rem,4vw,3.75rem)] leading-[1.02]">
            Redact the record.
            <br />
            Verify every page.
          </p>
        </div>
        <GlassChip className="absolute bottom-5 left-5">Released only if every check passes</GlassChip>
      </section>
    </div>
  );
}
