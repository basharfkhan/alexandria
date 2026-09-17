"use client";

import { useEffect, useState } from "react";

import { API_URL } from "@/lib/api";

const SLOW_AFTER_MS = 2500;
const RETRY_EVERY_MS = 5000;
const MAX_ATTEMPTS = 14; // ~70 s, comfortably past a free-tier cold start

type Status = "ok" | "waking" | "down";

/**
 * The API runs on a free instance that sleeps when idle and takes ~30-50 s to wake.
 * Ping it on load and, if it's slow, explain the wait instead of looking broken.
 */
export function ServerWakeBanner() {
  const [status, setStatus] = useState<Status>("ok");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let settled = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const slow = setTimeout(() => {
      if (!settled) setStatus("waking");
    }, SLOW_AFTER_MS);

    const ping = async (n: number) => {
      try {
        const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
        if (res.ok) {
          settled = true;
          setStatus("ok");
          return;
        }
      } catch {
        /* still waking, or offline */
      }
      if (settled) return;
      if (n + 1 >= MAX_ATTEMPTS) {
        settled = true;
        setStatus("down");
        return;
      }
      retry = setTimeout(() => ping(n + 1), RETRY_EVERY_MS);
    };
    ping(0);

    return () => {
      settled = true;
      clearTimeout(slow);
      clearTimeout(retry);
    };
  }, [attempt]);

  if (status === "ok") return null;

  return (
    <div role="status" aria-live="polite" className="border-b border-line bg-accent-soft text-accent">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-2.5 text-sm sm:px-6">
        {status === "waking" ? (
          <>
            <span className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            <span>
              <strong>Waking up the library…</strong> The server sleeps when nobody&apos;s reading (free hosting), so the
              first visit can take ~30-50 seconds.
            </span>
          </>
        ) : (
          <>
            <span>The library server isn&apos;t responding right now.</span>
            <button
              className="font-medium underline"
              onClick={() => {
                setStatus("waking");
                setAttempt((a) => a + 1);
              }}
            >
              Try again
            </button>
          </>
        )}
      </div>
    </div>
  );
}
