"use client";

import Link from "next/link";

import { useAuth } from "@/lib/auth";

const STEPS = [
  {
    title: "Tell us your taste",
    body: "Pick genres and a few books you loved - or just chat with our AI librarian.",
  },
  {
    title: "Get a tailored shelf",
    body: "A hybrid model blends book similarity with what readers like you enjoyed.",
  },
  {
    title: "It learns as you read",
    body: "Every ♥, 👍 or 👎 re-tunes your recommendations instantly - no waiting for retraining.",
  },
];

export default function Home() {
  const { user } = useAuth();

  return (
    <div className="animate-rise space-y-16 py-8">
      <section className="max-w-3xl">
        <p className="mb-3 text-sm font-medium tracking-widest text-accent uppercase">Book recommendations</p>
        <h1 className="font-serif text-4xl leading-tight font-semibold sm:text-6xl">
          Find the book you&apos;ll <em className="text-accent">not</em> put down.
        </h1>
        <p className="mt-5 max-w-xl text-lg text-muted">
          Alexandria learns what you love to read and gets sharper with every book you rate.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href={user ? (user.onboarded ? "/discover" : "/onboarding") : "/login?mode=register"}
            className="rounded-full bg-accent px-6 py-3 font-medium text-paper transition hover:opacity-90"
          >
            {user ? "Go to my recommendations" : "Build my shelf - it's free"}
          </Link>
          {!user && (
            <Link href="/login" className="rounded-full border border-line px-6 py-3 font-medium hover:border-ink">
              I have an account
            </Link>
          )}
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {STEPS.map((s, i) => (
          <div key={s.title} className="rounded-xl border border-line bg-card p-6">
            <span className="font-serif text-3xl text-accent">{i + 1}</span>
            <h2 className="mt-2 font-serif text-xl font-semibold">{s.title}</h2>
            <p className="mt-2 text-sm text-muted">{s.body}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
