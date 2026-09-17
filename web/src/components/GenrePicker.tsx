"use client";

import { genreLabel, type Genre } from "@/lib/api";

export function GenrePicker({
  genres,
  selected,
  onToggle,
}: {
  genres: Genre[];
  selected: string[];
  onToggle: (slug: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {genres.map((g) => {
        const on = selected.includes(g.slug);
        return (
          <button
            key={g.slug}
            type="button"
            aria-pressed={on}
            onClick={() => onToggle(g.slug)}
            className={`rounded-full border px-3.5 py-1.5 text-sm transition ${
              on ? "border-moss bg-moss text-paper" : "border-line bg-card hover:border-moss"
            }`}
          >
            {genreLabel(g.slug)}
            <span className={`ml-1.5 text-xs ${on ? "opacity-80" : "text-muted"}`}>{g.count}</span>
          </button>
        );
      })}
    </div>
  );
}
