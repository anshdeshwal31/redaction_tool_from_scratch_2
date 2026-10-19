export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "–";
  if (ms < 999.5) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 9.95) return `${s.toFixed(1)} s`;
  // Round once to the smallest unit shown, then split, so no part ever reads 60 (or 1000 ms).
  const total = Math.round(s);
  if (total < 60) return `${total} s`;
  if (total < 3600) return `${Math.floor(total / 60)} min ${(total % 60).toString().padStart(2, "0")} s`;
  const minutes = Math.round(total / 60);
  return `${Math.floor(minutes / 60)} h ${(minutes % 60).toString().padStart(2, "0")} min`;
}

/** m:ss / h:mm:ss for a live elapsed counter. */
export function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? m.toString().padStart(2, "0") : String(m);
  return `${h ? `${h}:` : ""}${mm}:${s.toString().padStart(2, "0")}`;
}

const dateFmt = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

export function formatDate(iso: string | undefined): string {
  return iso ? dateFmt.format(new Date(iso)) : "–";
}

export function formatRelative(iso: string, now = Date.now()): string {
  const diff = Math.round((now - new Date(iso).getTime()) / 1000);
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86_400) return `${Math.round(diff / 3600)} h ago`;
  if (diff < 7 * 86_400) return `${Math.round(diff / 86_400)} d ago`;
  return formatDate(iso);
}

export function shortHash(hash: string, n = 12): string {
  return hash.length > n ? hash.slice(0, n) : hash;
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}

export function percent(x: number | undefined, digits = 1): string {
  return x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`;
}

export function compact(n: number): string {
  return new Intl.NumberFormat(undefined, { notation: n >= 10_000 ? "compact" : "standard" }).format(n);
}
