"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState, type FormEvent } from "react";

import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const { login, register, user } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [mode, setMode] = useState<"login" | "register">(
    searchParams.get("mode") === "register" ? "register" : "login",
  );
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (user) router.replace(user.onboarded ? "/discover" : "/onboarding");
  }, [user, router]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (mode === "login" ? login : register)(username, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  const isRegister = mode === "register";

  return (
    <div className="animate-rise mx-auto max-w-sm py-10">
      <h1 className="font-serif text-3xl font-semibold">{isRegister ? "Create your shelf" : "Welcome back"}</h1>
      <p className="mt-2 text-sm text-muted">
        {isRegister ? "Just a username and password - no email needed." : "Sign in to see your recommendations."}
      </p>

      <form onSubmit={submit} className="mt-8 space-y-4">
        <label className="block text-sm">
          Username
          <input
            required
            minLength={3}
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="mt-1 w-full rounded-lg border border-line bg-card px-3 py-2.5 outline-none focus:border-accent"
          />
        </label>
        <label className="block text-sm">
          Password
          <input
            required
            type="password"
            minLength={8}
            autoComplete={isRegister ? "new-password" : "current-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1 w-full rounded-lg border border-line bg-card px-3 py-2.5 outline-none focus:border-accent"
          />
          {isRegister && <span className="mt-1 block text-xs text-muted">At least 8 characters.</span>}
        </label>
        {error && <p className="rounded-lg bg-accent-soft px-3 py-2 text-sm text-accent">{error}</p>}
        <button
          disabled={busy}
          className="w-full rounded-full bg-ink py-2.5 font-medium text-paper transition disabled:opacity-60"
        >
          {busy ? "One moment…" : isRegister ? "Create account" : "Sign in"}
        </button>
      </form>

      <button
        onClick={() => setMode(isRegister ? "login" : "register")}
        className="mt-6 w-full text-center text-sm text-muted hover:text-ink"
      >
        {isRegister ? "Already have an account? Sign in" : "New here? Create an account"}
      </button>
    </div>
  );
}
