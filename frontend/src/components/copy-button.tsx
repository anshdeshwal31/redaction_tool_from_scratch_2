"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        } catch {
          /* clipboard unavailable */
        }
      }}
      className="inline-flex size-6 shrink-0 items-center justify-center text-muted hover:bg-surface-2 hover:text-ink"
      aria-label={done ? "Copied" : label}
      title={done ? "Copied" : label}
    >
      {done ? <Check className="pop size-3.5 text-good" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
    </button>
  );
}

export function Hash({ value, n = 16 }: { value: string; n?: number }) {
  return (
    <span className="inline-flex max-w-full items-center gap-1">
      <span className="truncate font-mono text-[0.8125rem]" title={value}>
        {value.length > n ? `${value.slice(0, n)}…` : value}
      </span>
      <CopyButton value={value} />
    </span>
  );
}
