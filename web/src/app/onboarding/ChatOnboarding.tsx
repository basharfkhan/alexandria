"use client";

import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import { useFinishOnboarding } from "@/app/onboarding/useFinishOnboarding";
import { BookChip } from "@/components/BookChip";
import { api, genreLabel, type Book, type ChatMessage } from "@/lib/api";

const OPENING: ChatMessage = {
  role: "assistant",
  content:
    "Welcome to Alexandria! I'm your librarian. 📚 Tell me a bit about what you like to read - a few genres you gravitate toward, or a book you couldn't put down?",
};

export function ChatOnboarding() {
  const [messages, setMessages] = useState<ChatMessage[]>([OPENING]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [genres, setGenres] = useState<string[]>([]);
  const [loved, setLoved] = useState<Book[]>([]);
  const [disliked, setDisliked] = useState<Book[]>([]);
  const [removed, setRemoved] = useState<Set<number>>(new Set());
  const { finish, saving, error } = useFinishOnboarding();
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, thinking]);

  async function send(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || thinking) return;

    const next = [...messages, { role: "user" as const, content: text }];
    setMessages(next);
    setInput("");
    setThinking(true);
    setChatError(null);
    try {
      const res = await api.chat(next);
      setMessages([...next, { role: "assistant", content: res.reply }]);
      setReady(res.ready);
      setGenres(res.preferences.genres);
      setLoved(res.matched_loved);
      setDisliked(res.matched_disliked);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "The librarian is unavailable");
      setMessages(messages);
      setInput(text);
    } finally {
      setThinking(false);
    }
  }

  const keptLoved = loved.filter((b) => !removed.has(b.id));
  const keptDisliked = disliked.filter((b) => !removed.has(b.id));
  const hasSomething = genres.length + keptLoved.length > 0;
  const remove = (id: number) => setRemoved((s) => new Set(s).add(id));

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
      <div className="flex h-[520px] flex-col rounded-xl border border-line bg-card">
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <p
                className={`animate-rise max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                  m.role === "user" ? "rounded-br-sm bg-ink text-paper" : "rounded-bl-sm bg-paper"
                }`}
              >
                {m.content}
              </p>
            </div>
          ))}
          {thinking && <p className="text-sm text-muted italic">The librarian is thinking…</p>}
          <div ref={endRef} />
        </div>
        {chatError && <p className="mx-4 mb-2 rounded-lg bg-accent-soft px-3 py-2 text-sm text-accent">{chatError}</p>}
        <form onSubmit={send} className="flex gap-2 border-t border-line p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. I loved The Name of the Wind and anything by Brandon Sanderson"
            className="flex-1 rounded-full border border-line bg-paper px-4 py-2 text-sm outline-none focus:border-accent"
          />
          <button disabled={thinking || !input.trim()} className="rounded-full bg-ink px-4 text-sm text-paper disabled:opacity-40">
            Send
          </button>
        </form>
      </div>

      <aside className="space-y-5 rounded-xl border border-line bg-card p-4 text-sm">
        <h3 className="font-serif text-lg font-semibold">What I&apos;ve learned</h3>
        <Section title="Genres">
          {genres.map((g) => (
            <span key={g} className="rounded-full bg-moss-soft px-3 py-1 text-moss">
              {genreLabel(g)}
            </span>
          ))}
        </Section>
        <Section title="Loved">
          {keptLoved.map((b) => (
            <BookChip key={b.id} book={b} onRemove={() => remove(b.id)} />
          ))}
        </Section>
        <Section title="Not for you">
          {keptDisliked.map((b) => (
            <BookChip key={b.id} tone="accent" book={b} onRemove={() => remove(b.id)} />
          ))}
        </Section>
        {error && <p className="text-accent">{error}</p>}
        <button
          disabled={!hasSomething || saving}
          onClick={() =>
            finish({
              genres,
              loved_book_ids: keptLoved.map((b) => b.id),
              disliked_book_ids: keptDisliked.map((b) => b.id),
              source: "chat",
            })
          }
          className={`w-full rounded-full py-2.5 font-medium text-paper transition disabled:opacity-40 ${
            ready ? "animate-pulse bg-accent" : "bg-ink"
          }`}
        >
          {saving ? "Building your shelf…" : "Show my recommendations →"}
        </button>
      </aside>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode[] }) {
  return (
    <div>
      <p className="mb-2 text-xs tracking-wide text-muted uppercase">{title}</p>
      <div className="flex flex-wrap gap-1.5">
        {children.length ? children : <span className="text-muted italic">Nothing yet</span>}
      </div>
    </div>
  );
}
