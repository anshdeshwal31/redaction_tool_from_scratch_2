"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { MUTATION_HEADERS } from "./ui-header";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
  ) {
    super(code);
  }
}

async function parse<T>(res: Response): Promise<T> {
  const body = (await res.json().catch(() => ({}))) as T & { error?: string };
  if (res.status === 401 && typeof window !== "undefined") {
    // Session expired: sign in again and come back here.
    const back = window.location.pathname + window.location.search + window.location.hash;
    window.location.assign(`/login?callbackUrl=${encodeURIComponent(back)}`);
  }
  if (!res.ok) throw new ApiError(res.status, body.error ?? `HTTP_${res.status}`);
  return body;
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  return parse<T>(await fetch(url, { cache: "no-store", signal }));
}

export async function send<T>(url: string, method: "POST" | "DELETE", body?: object): Promise<T> {
  const init: RequestInit = { method, headers: { ...MUTATION_HEADERS } };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { ...MUTATION_HEADERS, "content-type": "application/json" };
  }
  return parse<T>(await fetch(url, init));
}

/** A link that downloads an artefact: this app checks the session, then redirects to the backend. */
export const downloadUrl = (id: string, name: string) => `/api/jobs/${id}/files/${name}?download=1`;

/** A fresh signed URL to fetch an artefact straight from the backend (links expire after minutes). */
export async function freshFileUrl(id: string, name: string): Promise<string> {
  return (await getJson<{ url: string }>(`/api/jobs/${id}/files/${name}?as=json`)).url;
}

/**
 * PUT the document straight to the backend's signed upload URL, reporting progress (0..1).  Only
 * the bytes are sent: the filename never leaves the browser.
 */
export function uploadFile(url: string, file: Blob, onProgress: (fraction: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("content-type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve();
      let code = `HTTP_${xhr.status}`;
      try {
        code = (JSON.parse(xhr.responseText) as { error?: string }).error ?? code;
      } catch {
        /* not JSON */
      }
      reject(new ApiError(xhr.status, code));
    };
    xhr.onerror = () => reject(new ApiError(0, "E_UI_UPLOAD_NETWORK"));
    xhr.send(file);
  });
}

/**
 * Fetch `url`, then refetch every `intervalMs` while `shouldPoll(data)` holds.  The previous data
 * stays on screen during refetches (no flash), and polling pauses while the tab is hidden.
 */
export function usePolling<T>(url: string, intervalMs: number, shouldPoll: (data: T) => boolean) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [tick, setTick] = useState(0);
  const shouldPollRef = useRef(shouldPoll);
  useEffect(() => {
    shouldPollRef.current = shouldPoll;
  });

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const next = await getJson<T>(url, ctrl.signal);
        setData(next);
        setError(null);
        if (shouldPollRef.current(next)) timer = setTimeout(load, document.hidden ? intervalMs * 5 : intervalMs);
      } catch (err) {
        if (ctrl.signal.aborted) return;
        setError(err instanceof ApiError ? err : new ApiError(0, "E_UI_NETWORK"));
        timer = setTimeout(load, intervalMs * 3);
      }
    };
    void load();
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [url, intervalMs, tick]);

  return { data, error, refresh };
}

/** A clock that re-renders every second while `active`. */
export function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}
