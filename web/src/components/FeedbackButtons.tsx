"use client";

import type { Signal } from "@/lib/api";

export const SIGNALS: { signal: Signal; label: string; icon: string }[] = [
  { signal: "loved", label: "Loved it", icon: "♥" },
  { signal: "liked", label: "Liked it", icon: "👍" },
  { signal: "disliked", label: "Didn't like", icon: "👎" },
  { signal: "want_to_read", label: "Want to read", icon: "🔖" },
  { signal: "not_interested", label: "Not for me", icon: "✕" },
];

export const SIGNAL_LABEL = Object.fromEntries(SIGNALS.map((s) => [s.signal, s.label])) as Record<Signal, string>;

export function FeedbackButtons({
  current,
  onSelect,
  compact = false,
  disabled = false,
}: {
  current?: Signal | null;
  onSelect: (signal: Signal) => void;
  compact?: boolean;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {SIGNALS.map(({ signal, label, icon }) => {
        const active = current === signal;
        return (
          <button
            key={signal}
            type="button"
            title={label}
            aria-label={label}
            aria-pressed={active}
            disabled={disabled}
            onClick={() => onSelect(signal)}
            className={`rounded-full border text-xs transition disabled:opacity-50 ${
              compact ? "h-7 w-7" : "px-2.5 py-1"
            } ${
              active
                ? "border-accent bg-accent text-paper"
                : "border-line bg-card text-muted hover:border-accent hover:text-accent"
            }`}
          >
            {compact ? icon : `${icon} ${label}`}
          </button>
        );
      })}
    </div>
  );
}
