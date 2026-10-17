import { Ban, CircleSlash, Clock, ShieldAlert, ShieldCheck, XCircle } from "lucide-react";

import type { JobState } from "@/lib/jobs";

import { cx, Spinner } from "./ui";

/** Status always pairs its colour with an icon and a word, never colour alone. */
const STATES: Record<JobState, { label: string; className: string; icon: React.ReactNode }> = {
  awaiting_upload: {
    label: "Uploading",
    className: "bg-surface-2 text-ink-2 border-line",
    icon: <Spinner className="size-3.5" />,
  },
  released: {
    label: "Released",
    className: "bg-good-soft text-good-text border-good/30",
    icon: <ShieldCheck className="size-3.5" aria-hidden />,
  },
  quarantined: {
    label: "Quarantined",
    className: "bg-warn-soft text-warn-text border-warn/40",
    icon: <ShieldAlert className="size-3.5" aria-hidden />,
  },
  failed: {
    label: "Failed",
    className: "bg-crit-soft text-crit-text border-crit/30",
    icon: <XCircle className="size-3.5" aria-hidden />,
  },
  running: {
    label: "Running",
    className: "bg-surface-3 text-ink border-line-strong",
    icon: <Spinner className="size-3.5" />,
  },
  queued: {
    label: "Queued",
    className: "bg-surface-2 text-ink-2 border-line",
    icon: <Clock className="size-3.5 motion-safe:animate-pulse" aria-hidden />,
  },
  cancelled: {
    label: "Cancelled",
    className: "bg-surface-2 text-ink-2 border-line",
    icon: <Ban className="size-3.5" aria-hidden />,
  },
  interrupted: {
    label: "Interrupted",
    className: "bg-surface-2 text-ink-2 border-line",
    icon: <CircleSlash className="size-3.5" aria-hidden />,
  },
};

export function StatusBadge({ state, size = "sm" }: { state: JobState; size?: "sm" | "lg" }) {
  const s = STATES[state];
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 border font-medium whitespace-nowrap",
        size === "lg" ? "h-8 px-3 text-sm" : "h-6 px-2 text-xs",
        s.className,
      )}
    >
      {s.icon}
      {s.label}
    </span>
  );
}
