import { cx } from "./ui";

/** The mark: a struck-out line between two points, in the product's one gesture. */
export function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={cx("size-5 shrink-0", className)} fill="currentColor" aria-hidden="true">
      <rect x="10.75" y="3.5" width="2.5" height="2.5" />
      <rect x="3.5" y="10.75" width="2.5" height="2.5" />
      <rect x="8.25" y="10.75" width="7.5" height="2.5" />
      <rect x="18" y="10.75" width="2.5" height="2.5" />
      <rect x="10.75" y="18" width="2.5" height="2.5" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-2 text-ink", className)}>
      <Mark className="size-4.5" />
      <span className="font-serif text-[1.75rem] leading-none">mlredact</span>
    </span>
  );
}
