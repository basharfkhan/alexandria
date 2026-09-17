import type { Book } from "@/lib/api";

export function BookChip({ book, onRemove, tone = "moss" }: { book: Book; onRemove?: () => void; tone?: "moss" | "accent" }) {
  const colors = tone === "moss" ? "bg-moss-soft text-moss" : "bg-accent-soft text-accent";
  return (
    <span className={`inline-flex max-w-full items-center gap-1.5 rounded-full px-3 py-1 text-sm ${colors}`}>
      <span className="truncate">{book.title}</span>
      {onRemove && (
        <button type="button" onClick={onRemove} aria-label={`Remove ${book.title}`} className="opacity-70 hover:opacity-100">
          ✕
        </button>
      )}
    </span>
  );
}
