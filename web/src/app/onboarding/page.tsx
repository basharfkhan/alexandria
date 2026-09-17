"use client";

import { useState } from "react";

import { ChatOnboarding } from "@/app/onboarding/ChatOnboarding";
import { QuizOnboarding } from "@/app/onboarding/QuizOnboarding";
import { useRequireUser } from "@/lib/auth";

export default function OnboardingPage() {
  const { user, ready } = useRequireUser();
  const [mode, setMode] = useState<"quiz" | "chat">("quiz");

  if (!ready || !user) return null;

  return (
    <div className="animate-rise mx-auto max-w-3xl">
      <h1 className="font-serif text-3xl font-semibold sm:text-4xl">
        {user.onboarded ? "Tune your taste" : `Let's build your shelf, ${user.username}`}
      </h1>
      <p className="mt-2 text-muted">
        The more you share, the better your first recommendations. You can always refine later by rating books.
      </p>

      <div className="mt-6 inline-flex rounded-full border border-line bg-card p-1 text-sm">
        {(
          [
            ["quiz", "Quick picks"],
            ["chat", "Chat with the librarian ✨"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setMode(key)}
            className={`rounded-full px-4 py-1.5 transition ${mode === key ? "bg-ink text-paper" : "text-muted"}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-8">{mode === "quiz" ? <QuizOnboarding /> : <ChatOnboarding />}</div>
    </div>
  );
}
