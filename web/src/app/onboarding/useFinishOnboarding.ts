"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { api, type OnboardingPayload } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function useFinishOnboarding() {
  const { refresh } = useAuth();
  const router = useRouter();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function finish(payload: OnboardingPayload) {
    setSaving(true);
    setError(null);
    try {
      await api.onboard(payload);
      await refresh();
      router.push("/discover");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your preferences");
      setSaving(false);
    }
  }

  return { finish, saving, error };
}
