"use client";

import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import { api, ApiError, tokenStore, type User } from "@/lib/api";

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<User>;
  register: (username: string, password: string) => Promise<User>;
  logout: () => void;
  refresh: () => Promise<User | null>;
}

const AuthContext = createContext<AuthState | null>(null);

async function loadMe(): Promise<User | null> {
  if (!tokenStore.get()) return null;
  try {
    return await api.me();
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) tokenStore.clear();
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const me = await loadMe();
    setUser(me);
    return me;
  }, []);

  useEffect(() => {
    loadMe().then((me) => {
      setUser(me);
      setLoading(false);
    });
  }, []);

  const withToken = useCallback(
    async (getToken: Promise<{ access_token: string }>) => {
      tokenStore.set((await getToken).access_token);
      const me = await refresh();
      if (!me) throw new Error("Could not load your profile");
      return me;
    },
    [refresh],
  );

  const value: AuthState = {
    user,
    loading,
    refresh,
    login: (u, p) => withToken(api.login(u, p)),
    register: (u, p) => withToken(api.register(u, p)),
    logout: () => {
      tokenStore.clear();
      setUser(null);
    },
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

/** Redirects to /login when signed out (and to /onboarding when `requireOnboarded`). */
export function useRequireUser({ requireOnboarded = false } = {}) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    if (!user) router.replace("/login");
    else if (requireOnboarded && !user.onboarded) router.replace("/onboarding");
  }, [user, loading, requireOnboarded, router]);

  return { user, ready: !loading && !!user && (!requireOnboarded || user.onboarded) };
}
