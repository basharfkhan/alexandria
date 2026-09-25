"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { BookCover } from "@/components/BookCover";
import { FeedbackButtons } from "@/components/FeedbackButtons";
import { api, genreLabel, type Book, type BookDetail, type Signal } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function BookPage() {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const [book, setBook] = useState<BookDetail | null>(null);
  const [similar, setSimilar] = useState<Book[]>([]);
  const [signal, setSignal] = useState<Signal | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    const bookId = Number(id);
    api.book(bookId).then(setBook).catch(() => setNotFound(true));
    api.similar(bookId, 12).then(setSimilar).catch(() => setSimilar([]));
  }, [id]);

  useEffect(() => {
    if (!user || !book) return;
    api
      .library()
      .then((items) => setSignal(items.find((it) => it.book.id === book.id)?.signal ?? null))
      .catch(() => undefined);
  }, [user, book]);

  if (notFound) return <p className="text-muted">That book isn&apos;t in our catalog.</p>;
  if (!book || book.id !== Number(id)) return <div className="h-80 animate-pulse rounded-xl bg-line" />;

  async function rate(s: Signal) {
    setSignal(s);
    await api.feedback(book!.id, s, "app");
  }

  return (
    <div className="animate-rise space-y-12">
      <div className="grid gap-8 sm:grid-cols-[220px_1fr]">
        <div className="max-w-[220px]">
          <BookCover book={book} />
        </div>
        <div>
          <h1 className="font-serif text-3xl leading-tight font-semibold sm:text-4xl">{book.title}</h1>
          <p className="mt-2 text-lg text-muted">{book.authors}</p>
          <p className="mt-4 text-sm text-muted">
            ★ {book.avg_rating.toFixed(2)} · {book.ratings_count.toLocaleString()} ratings
            {book.year ? ` · ${book.year}` : ""}
          </p>
          {book.is_new && (
            <p className="mt-2 text-sm text-moss">
              Recently published - recommended from its description and readers of similar books.
            </p>
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            {book.genres.map((g) => (
              <span key={g} className="rounded-full bg-moss-soft px-3 py-1 text-sm text-moss">
                {genreLabel(g)}
              </span>
            ))}
          </div>
          {book.description && <Description text={book.description} key={book.id} />}
          {user ? (
            <div className="mt-8">
              <p className="mb-2 text-sm font-medium">Read it? Tell Alexandria:</p>
              <FeedbackButtons current={signal} onSelect={rate} />
            </div>
          ) : (
            <Link href="/login" className="mt-8 inline-block text-accent underline">
              Sign in to rate this book
            </Link>
          )}
        </div>
      </div>

      {similar.length > 0 && (
        <section>
          <h2 className="mb-5 font-serif text-2xl font-semibold">More like this</h2>
          <ul className="grid grid-cols-3 gap-4 sm:grid-cols-4 lg:grid-cols-6">
            {similar.map((b) => (
              <li key={b.id}>
                <Link href={`/books/${b.id}`} className="group">
                  <BookCover book={b} className="transition group-hover:-translate-y-1" />
                  <p className="mt-2 line-clamp-2 text-sm font-medium group-hover:text-accent">{b.title}</p>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function Description({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const long = text.length > 420;
  return (
    <div className="mt-6 max-w-2xl">
      <p className={`leading-relaxed whitespace-pre-line ${long && !expanded ? "line-clamp-5" : ""}`}>{text}</p>
      <div className="mt-2 flex items-center gap-3 text-xs text-muted">
        {long && (
          <button onClick={() => setExpanded((e) => !e)} className="font-medium text-accent hover:underline">
            {expanded ? "Show less" : "Read more"}
          </button>
        )}
        <span>
          Description from{" "}
          <a href="https://openlibrary.org" target="_blank" rel="noreferrer" className="underline">
            Open Library
          </a>
        </span>
      </div>
    </div>
  );
}
