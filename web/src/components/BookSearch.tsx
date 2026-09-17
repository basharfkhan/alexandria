"use client";

import { useEffect, useRef, useState } from "react";

import { api, type Book } from "@/lib/api";

/** Debounced catalog search that reports the picked book. */
export function BookSearch({
  onPick,
  placeholder = "Search by title or author…",
  excludeIds = [],
}: {
  onPick: (book: Book) => void;
  placeholder?: string;
  excludeIds?: number[];
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Book[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) return;
    const handle = setTimeout(() => {
      api
        .search(q)
        .then((r) => {
          setResults(r);
          setOpen(true);
        })
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const visible = query.trim().length >= 2 ? results.filter((b) => !excludeIds.includes(b.id)) : [];

  return (
    <div ref={boxRef} className="relative">
      <input
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setLoading(e.target.value.trim().length >= 2);
        }}
        onFocus={() => results.length && setOpen(true)}
        placeholder={placeholder}
        className="w-full rounded-lg border border-line bg-card px-4 py-2.5 outline-none focus:border-accent"
      />
      {loading && <span className="absolute top-3 right-3 text-xs text-muted">searching…</span>}
      {open && query.trim().length >= 2 && (
        <ul className="absolute z-30 mt-1 max-h-80 w-full overflow-auto rounded-lg border border-line bg-card shadow-lg">
          {visible.length === 0 && !loading && <li className="px-4 py-3 text-sm text-muted">No matches</li>}
          {visible.map((b) => (
            <li key={b.id}>
              <button
                type="button"
                onClick={() => {
                  onPick(b);
                  setQuery("");
                  setOpen(false);
                }}
                className="flex w-full items-baseline gap-2 px-4 py-2 text-left hover:bg-accent-soft"
              >
                <span className="font-medium">{b.title}</span>
                <span className="truncate text-xs text-muted">{b.authors}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
