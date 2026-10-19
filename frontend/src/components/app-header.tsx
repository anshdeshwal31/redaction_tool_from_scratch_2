"use client";

import { LogOut, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut } from "next-auth/react";
import { useEffect, useState } from "react";

import { Wordmark } from "./brand";
import { ButtonLink, cx } from "./ui";

const NAV = [
  { href: "/", label: "Overview", match: (p: string) => p === "/" },
  { href: "/redact", label: "New redaction", match: (p: string) => p.startsWith("/redact") },
  { href: "/jobs", label: "Jobs", match: (p: string) => p.startsWith("/jobs") },
  { href: "/system", label: "System", match: (p: string) => p.startsWith("/system") },
];

/** `user`: the signed-in email, or null on the sign-in page (no navigation then). */
export function AppHeader({ user }: { user: string | null }) {
  const pathname = usePathname();
  // A hairline appears under the header once the page scrolls beneath it.
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cx(
        "sticky top-0 z-30 border-b bg-page/80 backdrop-blur-xl transition-colors duration-300 supports-backdrop-filter:bg-page/65",
        scrolled ? "border-line" : "border-transparent",
      )}
    >
      <div className="mx-auto grid h-18 max-w-352 grid-cols-[auto_1fr_auto] items-center gap-4 px-4 sm:px-6 lg:px-8 md:grid-cols-[1fr_auto_1fr]">
        <Link href="/" className="group justify-self-start" aria-label="mlredact home">
          <Wordmark className="[&>svg]:transition-transform [&>svg]:duration-500 group-hover:[&>svg]:rotate-90" />
        </Link>
        <nav
          aria-label="Main"
          className={cx("flex min-w-0 items-center justify-center gap-1 overflow-x-auto sm:gap-3", !user && "invisible")}
        >
          {NAV.map(({ href, label, match }) => {
            const active = match(pathname);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "draw-line px-2 py-1.5 text-[0.9375rem] whitespace-nowrap transition-colors",
                  active ? "text-ink" : "text-ink-2 hover:text-ink",
                )}
              >
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="flex items-center justify-self-end gap-5">
          <span
            className="hidden items-center gap-1.5 text-xs text-muted lg:inline-flex"
            title="Documents go straight from your browser to your redaction server, never through this site's host."
          >
            <ShieldCheck className="size-3.5" aria-hidden />
            Private server
          </span>
          {!user || pathname.startsWith("/redact") ? null : (
            <ButtonLink href="/redact" variant="primary" className="hidden sm:inline-flex">
              Start a redaction
            </ButtonLink>
          )}
          {user ? (
            <button
              type="button"
              onClick={() => void signOut({ callbackUrl: "/login" })}
              className="inline-flex items-center gap-1.5 text-sm text-ink-2 transition-colors hover:text-ink"
              title={`Signed in as ${user}`}
            >
              <LogOut className="size-4" aria-hidden />
              <span className="hidden xl:inline">Sign out</span>
              <span className="sr-only xl:hidden">Sign out {user}</span>
            </button>
          ) : null}
        </div>
      </div>
    </header>
  );
}
