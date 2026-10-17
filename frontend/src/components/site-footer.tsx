import { WifiOff } from "lucide-react";
import Link from "next/link";

import { Wordmark } from "./brand";
import { ThemeToggle } from "./theme-toggle";

const COLUMNS: { title: string; links: { href: string; label: string }[] }[] = [
  {
    title: "Workspace",
    links: [
      { href: "/redact", label: "New redaction" },
      { href: "/jobs", label: "Jobs" },
      { href: "/system", label: "System" },
    ],
  },
  {
    title: "The pipeline",
    links: [
      { href: "/#how-it-works", label: "How it works" },
      { href: "/#guarantees", label: "Guarantees" },
      { href: "/#recent", label: "Recent jobs" },
    ],
  },
];

const PRINCIPLES = ["Rebuild, never edit", "Fail closed", "Content-free manifest", "Deterministic output"];

export function SiteFooter() {
  return (
    <footer className="mt-28">
      <div className="mx-auto max-w-352 px-4 sm:px-6 lg:px-8">
        <div data-stagger className="grid gap-12 px-0 py-14 sm:px-6 md:grid-cols-[1.6fr_1fr_1fr_1fr]">
          <div>
            <Wordmark />
            <p className="mt-6 max-w-[16rem] text-[0.9375rem] leading-relaxed text-ink-2">
              Local, deterministic PII redaction. Verified before release.
            </p>
          </div>
          {COLUMNS.map((col) => (
            <nav key={col.title} aria-label={col.title}>
              <h2 className="font-serif text-xl text-muted">{col.title}</h2>
              <ul className="mt-5 space-y-3.5">
                {col.links.map((l) => (
                  <li key={l.href}>
                    <Link
                      href={l.href}
                      className="inline-block text-[0.9375rem] text-ink-2 transition-[color,translate] duration-300 hover:translate-x-1 hover:text-ink"
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          ))}
          <div>
            <h2 className="font-serif text-xl text-muted">Principles</h2>
            <ul className="mt-5 space-y-3.5">
              {PRINCIPLES.map((p) => (
                <li key={p} className="text-[0.9375rem] text-ink-2">
                  {p}
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="flex flex-col items-center justify-between gap-4 border-t border-line py-7 text-sm text-muted sm:flex-row sm:px-6">
          <span>mlredact · operator interface</span>
          <ThemeToggle />
          <span className="inline-flex items-center gap-1.5">
            <WifiOff className="size-3.5" aria-hidden />
            No external requests
          </span>
        </div>
      </div>
    </footer>
  );
}
