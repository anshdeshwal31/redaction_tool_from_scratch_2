import { ArrowRight } from "lucide-react";
import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  // A disabled primary steps back to a surface rather than fading to grey.
  primary: "bg-accent text-accent-ink hover:bg-accent-hover disabled:bg-surface-3 disabled:text-muted",
  secondary: "bg-surface-2 text-ink hover:bg-surface-3 disabled:opacity-40",
  ghost: "bg-transparent text-ink-2 hover:bg-surface-2 hover:text-ink disabled:opacity-40",
  danger: "bg-surface-2 text-crit-text hover:bg-crit-soft disabled:opacity-40",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 gap-1.5 px-3 text-xs",
  md: "h-10 gap-2 px-4 text-sm",
  lg: "h-13 gap-2 px-7 text-[0.9375rem]",
};

const BUTTON_BASE =
  "inline-flex items-center justify-center font-medium whitespace-nowrap transition-[color,background-color,border-color,scale] duration-150 motion-safe:active:scale-[0.97] disabled:pointer-events-none";

type ButtonStyle = { variant?: Variant; size?: Size };

export function Button({ variant = "secondary", size = "md", className, ...props }: ComponentProps<"button"> & ButtonStyle) {
  return <button type="button" className={cx(BUTTON_BASE, SIZES[size], VARIANTS[variant], className)} {...props} />;
}

export function ButtonLink({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ComponentProps<typeof Link> & ButtonStyle) {
  return <Link className={cx(BUTTON_BASE, SIZES[size], VARIANTS[variant], className)} {...props} />;
}

export function ButtonAnchor({ variant = "secondary", size = "md", className, ...props }: ComponentProps<"a"> & ButtonStyle) {
  return <a className={cx(BUTTON_BASE, SIZES[size], VARIANTS[variant], className)} {...props} />;
}

/** Text link with a trailing arrow that nudges forward on hover. */
export function ArrowLink({ className, children, ...props }: ComponentProps<typeof Link>) {
  return (
    <Link
      className={cx(
        "group inline-flex items-center gap-1.5 text-sm text-ink underline decoration-transparent underline-offset-4 transition-colors hover:decoration-current",
        className,
      )}
      {...props}
    >
      {children}
      <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden />
    </Link>
  );
}

export function Card({ className, ...props }: ComponentProps<"section">) {
  return <section className={cx("border border-line bg-surface", className)} {...props} />;
}

export function CardHeader({
  title,
  description,
  actions,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <header className={cx("flex flex-wrap items-start justify-between gap-3 px-6 pt-5 pb-4", className)}>
      <div className="min-w-0">
        <h2 className="font-serif text-[1.375rem] leading-tight text-ink">{title}</h2>
        {description ? <p className="mt-1 max-w-2xl text-sm text-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

/** The small label that sits above a section heading. */
export function Eyebrow({ className, ...props }: ComponentProps<"span">) {
  return (
    <span
      className={cx("inline-flex h-7 items-center bg-surface-2 px-3 text-xs font-medium text-ink-2", className)}
      {...props}
    />
  );
}

/** Centred section heading: eyebrow, serif display title, optional lead paragraph. */
export function SectionHeading({
  eyebrow,
  title,
  description,
  className,
  as: Tag = "h2",
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  className?: string;
  as?: "h1" | "h2";
}) {
  return (
    <div data-stagger className={cx("flex flex-col items-center text-center", className)}>
      {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
      <Tag className={cx("display text-[clamp(2.5rem,5.2vw,4.5rem)] text-balance text-ink", eyebrow ? "mt-6" : "")}>
        {title}
      </Tag>
      {description ? (
        <p className="mt-6 max-w-xl text-base leading-relaxed text-pretty text-ink-2 sm:text-[1.0625rem]">{description}</p>
      ) : null}
    </div>
  );
}

export function PageTitle({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="pt-10 pb-12 sm:pt-16 sm:pb-16">
      <SectionHeading as="h1" eyebrow={eyebrow} title={title} description={description} />
      {actions ? (
        <div data-reveal className="mt-8 flex flex-wrap items-center justify-center gap-2">
          {actions}
        </div>
      ) : null}
    </div>
  );
}

export function Mono({ className, ...props }: ComponentProps<"span">) {
  return <span className={cx("font-mono text-[0.8125rem]", className)} {...props} />;
}

export function Pill({ className, ...props }: ComponentProps<"span">) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 border border-line bg-surface-2 px-2 py-0.5 text-xs font-medium text-ink-2",
        className,
      )}
      {...props}
    />
  );
}

/** Stat tile: a serif figure over its label, centred. */
export function Stat({
  label,
  value,
  hint,
  size = "md",
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  size?: "md" | "lg";
  className?: string;
}) {
  return (
    <div
      className={cx(
        "flex min-w-0 flex-col items-center justify-center border border-line bg-surface px-4 text-center transition-[background-color,border-color,translate] duration-300 hover:-translate-y-1 hover:border-line-strong hover:bg-surface-2",
        size === "lg" ? "py-14 sm:py-16" : "py-7",
        className,
      )}
    >
      <div
        className={cx(
          "display tabular text-ink",
          size === "lg" ? "text-[clamp(3.5rem,6.5vw,5rem)] leading-none" : "text-[2.75rem] leading-none",
        )}
      >
        {value}
      </div>
      <div className={cx("text-ink-2", size === "lg" ? "mt-5 max-w-xs text-[0.9375rem]" : "mt-3 truncate text-sm")}>{label}</div>
      {hint ? <div className="mt-1 max-w-full truncate text-xs text-muted">{hint}</div> : null}
    </div>
  );
}

export function KeyValue({ items, className }: { items: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={cx("grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-6 gap-y-2.5 text-sm", className)}>
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 break-words text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      {icon ? <div className="text-muted">{icon}</div> : null}
      <div className="font-serif text-2xl text-ink">{title}</div>
      {children ? <div className="max-w-md text-sm text-muted">{children}</div> : null}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cx("size-4 animate-spin", className)} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" />
    </svg>
  );
}

export function Notice({
  tone,
  icon,
  title,
  children,
  className,
}: {
  tone: "info" | "good" | "warn" | "crit";
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const tones = {
    info: "bg-surface-2 border-l-ink-2",
    good: "bg-good-soft border-l-good",
    warn: "bg-warn-soft border-l-warn",
    crit: "bg-crit-soft border-l-crit",
  } as const;
  const iconTone = { info: "text-ink-2", good: "text-good-text", warn: "text-warn-text", crit: "text-crit-text" };
  return (
    <div
      role={tone === "crit" ? "alert" : "status"}
      className={cx("flex gap-3 border-l-2 px-5 py-4", tones[tone], className)}
    >
      {icon ? <div className={cx("mt-0.5 shrink-0", iconTone[tone])}>{icon}</div> : null}
      <div className="min-w-0 text-sm">
        <div className="font-medium text-ink">{title}</div>
        {children ? <div className="mt-1 text-ink-2">{children}</div> : null}
      </div>
    </div>
  );
}

/** A coloured swatch that carries identity beside a text label (text itself stays in ink). */
export function Swatch({ color, dashed = false, className }: { color: string; dashed?: boolean; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cx("inline-block size-2.5 shrink-0", className)}
      style={dashed ? { border: `1.5px dashed ${color}` } : { background: color }}
    />
  );
}
