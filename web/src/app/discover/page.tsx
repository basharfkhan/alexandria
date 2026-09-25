"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { BookCover } from "@/components/BookCover";
import { FeedbackButtons, SIGNAL_LABEL } from "@/components/FeedbackButtons";
import { api, type Recommendation, type Recommendations, type Signal } from "@/lib/api";
import { useRequireUser } from "@/lib/auth";

const REASON_STYLE: Record<Recommendation["reason"], string> = {
  similar: "bg-moss-soft text-moss",
  collaborative: "bg-moss-soft text-moss",
  genre: "bg-accent-soft text-accent",
  popular: "bg-accent-soft text-accent",
  explore: "bg-ink text-paper",
};

export default function DiscoverPage() {
  const { ready } = useRequireUser({ requireOnboarded: true });
  const [data, setData] = useState<Recommendations | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [rated, setRated] = useState(0);
  const refetchTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const load = useCallback(async () => {
    try {
      setData(await api.recommendations(24));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load recommendations");
    }
  }, []);

  useEffect(() => {
    if (!ready) return;
    api
      .recommendations(24)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load recommendations"));
  }, [ready]);

  async function rate(rec: Recommendation, signal: Signal, position: number) {
    // Optimistically drop the card, then refresh the list once the user pauses rating.
    setData((d) => d && { ...d, items: d.items.filter((r) => r.book.id !== rec.book.id) });
    setToast(`${SIGNAL_LABEL[signal]}: ${rec.book.title} - updating your picks…`);
    setRated((n) => n + 1);
    try {
      await api.feedback(rec.book.id, signal, "recommendation", position);
    } catch {
      setToast("Could not save that rating");
    }
    clearTimeout(refetchTimer.current);
    refetchTimer.current = setTimeout(async () => {
      await load();
      setToast(null);
    }, 1200);
  }

  if (!ready) return null;

  const personalization = Math.round((data?.personalization ?? 0) * 100);

  return (
    <div className="animate-rise">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-serif text-3xl font-semibold sm:text-4xl">Picked for you</h1>
          <p className="mt-1 text-muted">Rate anything you&apos;ve read - your shelf re-tunes itself as you go.</p>
        </div>
        <div className="w-full max-w-xs">
          <div className="flex justify-between text-xs text-muted">
            <span>Personalization</span>
            <span>{personalization}%</span>
          </div>
          <div className="mt-1 h-2 overflow-hidden rounded-full bg-line">
            <div className="h-full rounded-full bg-moss transition-all duration-700" style={{ width: `${Math.max(personalization, 4)}%` }} />
          </div>
          <p className="mt-1 text-xs text-muted">
            {personalization < 100 ? "Rate a few more books to sharpen your picks." : "Fully tuned to your history."}
          </p>
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-6 left-1/2 z-30 -translate-x-1/2 rounded-full bg-ink px-5 py-2.5 text-sm text-paper shadow-lg">
          {toast}
        </div>
      )}

      {error && (
        <p className="mt-8 rounded-lg bg-accent-soft px-4 py-3 text-accent">
          {error}{" "}
          <button className="underline" onClick={load}>
            Retry
          </button>
        </p>
      )}

      {!data && !error && (
        <div className="mt-8 grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="aspect-[2/3] animate-pulse rounded-md bg-line" />
          ))}
        </div>
      )}

      {data && (
        <ul className="mt-8 grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 lg:grid-cols-4">
          {data.items.map((rec, i) => (
            <li key={rec.book.id} className="animate-rise flex flex-col" style={{ animationDelay: `${Math.min(i, 12) * 30}ms` }}>
              <Link href={`/books/${rec.book.id}`} className="group">
                <BookCover book={rec.book} className="transition group-hover:-translate-y-1 group-hover:shadow-md" />
                <h2 className="mt-3 line-clamp-2 font-serif leading-snug font-semibold group-hover:text-accent">
                  {rec.book.title}
                  {rec.book.is_new && (
                    <span className="ml-2 align-middle rounded-full bg-moss-soft px-2 py-0.5 text-[10px] font-medium tracking-wide text-moss uppercase">
                      New
                    </span>
                  )}
                </h2>
              </Link>
              <p className="line-clamp-1 text-sm text-muted">{rec.book.authors}</p>
              <p className={`mt-2 self-start rounded-md px-2 py-1 text-xs ${REASON_STYLE[rec.reason]}`}>{rec.explanation}</p>
              <div className="mt-3">
                <FeedbackButtons compact onSelect={(s) => rate(rec, s, i)} />
              </div>
            </li>
          ))}
        </ul>
      )}

      {data && rated === 0 && (
        <p className="mt-10 text-center text-sm text-muted">
          Tip: ♥ loved · 👍 liked · 👎 didn&apos;t like · 🔖 want to read · ✕ not for me
        </p>
      )}
    </div>
  );
}
