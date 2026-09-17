"use client";

import { useEffect, useState } from "react";

import { useFinishOnboarding } from "@/app/onboarding/useFinishOnboarding";
import { BookChip } from "@/components/BookChip";
import { BookSearch } from "@/components/BookSearch";
import { GenrePicker } from "@/components/GenrePicker";
import { api, type Book, type Genre } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function QuizOnboarding() {
  const { user } = useAuth();
  const [genres, setGenres] = useState<Genre[]>([]);
  const [selected, setSelected] = useState<string[]>(user?.favorite_genres ?? []);
  const [loved, setLoved] = useState<Book[]>([]);
  const [disliked, setDisliked] = useState<Book[]>([]);
  const [suggestions, setSuggestions] = useState<Book[]>([]);
  const { finish, saving, error } = useFinishOnboarding();

  useEffect(() => {
    api.genres().then(setGenres).catch(() => setGenres([]));
  }, []);

  // Offer popular books from the chosen genres as one-click "loved it" picks.
  useEffect(() => {
    const genre = selected.at(-1);
    api.popular(genre, 12).then(setSuggestions).catch(() => setSuggestions([]));
  }, [selected]);

  const toggle = (slug: string) =>
    setSelected((s) => (s.includes(slug) ? s.filter((g) => g !== slug) : [...s, slug]));
  const pickedIds = [...loved, ...disliked].map((b) => b.id);
  const addUnique = (list: Book[], b: Book) => (list.some((x) => x.id === b.id) ? list : [...list, b]);

  return (
    <div className="space-y-10">
      <section>
        <h2 className="font-serif text-xl font-semibold">1. Genres you enjoy</h2>
        <p className="mb-4 text-sm text-muted">Pick as many as you like.</p>
        <GenrePicker genres={genres} selected={selected} onToggle={toggle} />
      </section>

      <section>
        <h2 className="font-serif text-xl font-semibold">2. Books you loved</h2>
        <p className="mb-4 text-sm text-muted">Three or more gives the model a strong start.</p>
        <BookSearch onPick={(b) => setLoved((l) => addUnique(l, b))} excludeIds={pickedIds} />
        {loved.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {loved.map((b) => (
              <BookChip key={b.id} book={b} onRemove={() => setLoved((l) => l.filter((x) => x.id !== b.id))} />
            ))}
          </div>
        )}
        {suggestions.filter((b) => !pickedIds.includes(b.id)).length > 0 && (
          <div className="mt-5">
            <p className="mb-2 text-xs tracking-wide text-muted uppercase">Popular picks - tap any you&apos;ve loved</p>
            <div className="flex flex-wrap gap-2">
              {suggestions
                .filter((b) => !pickedIds.includes(b.id))
                .map((b) => (
                  <button
                    key={b.id}
                    onClick={() => setLoved((l) => addUnique(l, b))}
                    className="rounded-full border border-dashed border-line px-3 py-1 text-sm hover:border-moss hover:text-moss"
                  >
                    + {b.title}
                  </button>
                ))}
            </div>
          </div>
        )}
      </section>

      <section>
        <h2 className="font-serif text-xl font-semibold">3. Books that weren&apos;t for you <span className="text-sm font-normal text-muted">(optional)</span></h2>
        <p className="mb-4 text-sm text-muted">Helps us steer away from what you don&apos;t enjoy.</p>
        <BookSearch onPick={(b) => setDisliked((l) => addUnique(l, b))} excludeIds={pickedIds} />
        {disliked.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {disliked.map((b) => (
              <BookChip key={b.id} tone="accent" book={b} onRemove={() => setDisliked((l) => l.filter((x) => x.id !== b.id))} />
            ))}
          </div>
        )}
      </section>

      {error && <p className="rounded-lg bg-accent-soft px-3 py-2 text-sm text-accent">{error}</p>}
      <div className="flex items-center gap-4 border-t border-line pt-6">
        <button
          disabled={saving || (selected.length === 0 && loved.length === 0)}
          onClick={() =>
            finish({
              genres: selected,
              loved_book_ids: loved.map((b) => b.id),
              disliked_book_ids: disliked.map((b) => b.id),
              source: "onboarding",
            })
          }
          className="rounded-full bg-accent px-6 py-3 font-medium text-paper transition disabled:opacity-40"
        >
          {saving ? "Building your shelf…" : "Show my recommendations →"}
        </button>
        <span className="text-sm text-muted">
          {selected.length} genres · {loved.length} loved · {disliked.length} disliked
        </span>
      </div>
    </div>
  );
}
