"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { BookCover } from "@/components/BookCover";
import { BookSearch } from "@/components/BookSearch";
import { FeedbackButtons, SIGNALS } from "@/components/FeedbackButtons";
import { api, type Book, type LibraryItem, type Signal } from "@/lib/api";
import { useRequireUser } from "@/lib/auth";

export default function LibraryPage() {
  const { ready } = useRequireUser();
  const [items, setItems] = useState<LibraryItem[] | null>(null);
  const [adding, setAdding] = useState<Book | null>(null);

  const load = useCallback(() => api.library().then(setItems), []);

  useEffect(() => {
    if (ready) load();
  }, [ready, load]);

  async function update(book: Book, signal: Signal) {
    await api.feedback(book.id, signal, adding ? "search" : "app");
    setAdding(null);
    load();
  }

  async function remove(book: Book) {
    await api.removeFeedback(book.id);
    load();
  }

  if (!ready) return null;

  return (
    <div className="animate-rise space-y-10">
      <div>
        <h1 className="font-serif text-3xl font-semibold sm:text-4xl">My shelf</h1>
        <p className="mt-1 text-muted">Finished something? Add it here - every rating teaches your recommendations.</p>
      </div>

      <section className="rounded-xl border border-line bg-card p-5">
        <h2 className="mb-3 font-serif text-lg font-semibold">Log a book</h2>
        <BookSearch onPick={setAdding} placeholder="What did you read?" />
        {adding && (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="font-medium">{adding.title}</span>
            <FeedbackButtons onSelect={(s) => update(adding, s)} />
            <button onClick={() => setAdding(null)} className="text-sm text-muted hover:text-ink">
              Cancel
            </button>
          </div>
        )}
      </section>

      {items?.length === 0 && (
        <p className="text-muted">
          Your shelf is empty. <Link href="/discover" className="text-accent underline">Rate some recommendations</Link> to fill it.
        </p>
      )}

      {items &&
        SIGNALS.map(({ signal, label, icon }) => {
          const shelf = items.filter((it) => it.signal === signal);
          if (!shelf.length) return null;
          return (
            <section key={signal}>
              <h2 className="mb-4 font-serif text-xl font-semibold">
                {icon} {label} <span className="text-sm font-normal text-muted">({shelf.length})</span>
              </h2>
              <ul className="grid grid-cols-2 gap-5 sm:grid-cols-4 lg:grid-cols-6">
                {shelf.map(({ book }) => (
                  <li key={book.id} className="flex flex-col">
                    <Link href={`/books/${book.id}`}>
                      <BookCover book={book} />
                      <p className="mt-2 line-clamp-2 text-sm font-medium">{book.title}</p>
                    </Link>
                    <div className="mt-2">
                      <FeedbackButtons compact current={signal} onSelect={(s) => update(book, s)} />
                    </div>
                    <button onClick={() => remove(book)} className="mt-1 self-start text-xs text-muted hover:text-accent">
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
    </div>
  );
}
