import type { Book } from "@/lib/api";

const PALETTE = ["#7a2e2e", "#3f6b4f", "#2f4a6b", "#8a5a1f", "#5b3f6b", "#2e6b68"];

/** Goodreads cover when available, otherwise a generated typographic cover. */
export function BookCover({ book, className = "" }: { book: Book; className?: string }) {
  if (book.image_url) {
    return (
      // eslint-disable-next-line @next/next/no-img-element -- remote covers from many hosts; no optimisation needed
      <img
        src={book.image_url}
        alt={`Cover of ${book.title}`}
        loading="lazy"
        className={`aspect-[2/3] w-full rounded-md object-cover shadow-sm ${className}`}
      />
    );
  }
  const color = PALETTE[book.id % PALETTE.length];
  return (
    <div
      className={`flex aspect-[2/3] w-full flex-col justify-between rounded-md p-3 text-white shadow-sm ${className}`}
      style={{ background: `linear-gradient(160deg, ${color}, #1f1b16)` }}
      aria-label={`Cover of ${book.title}`}
    >
      <span className="line-clamp-4 font-serif text-sm leading-snug font-semibold">{book.title}</span>
      <span className="line-clamp-2 text-[10px] opacity-80">{book.authors}</span>
    </div>
  );
}
