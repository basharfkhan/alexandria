"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { Pharos } from "@/components/Pharos";
import { useAuth } from "@/lib/auth";

const LINKS = [
  { href: "/discover", label: "Discover" },
  { href: "/library", label: "My shelf" },
  { href: "/onboarding", label: "Tune taste" },
];

export function Nav() {
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  return (
    <header className="sticky top-0 z-20 bg-paper/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3 sm:px-6">
        <Link
          href={user ? "/discover" : "/"}
          className="flex shrink-0 items-center gap-2 font-display text-lg font-semibold tracking-[0.1em] sm:text-xl"
        >
          <Pharos className="h-6 w-6 shrink-0 text-accent" />
          <span className="max-[420px]:sr-only">
            <span className="text-accent">Alex</span>andria
          </span>
        </Link>
        {user && (
          <nav className="flex min-w-0 gap-1 overflow-x-auto text-sm">
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-full px-3 py-1.5 whitespace-nowrap transition ${
                  pathname.startsWith(l.href) ? "bg-ink text-paper" : "text-muted hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            ))}
          </nav>
        )}
        <div className="ml-auto shrink-0 text-sm whitespace-nowrap">
          {user ? (
            <button
              onClick={() => {
                logout();
                router.push("/");
              }}
              className="text-muted hover:text-ink"
            >
              Sign out
            </button>
          ) : (
            <Link href="/login" className="rounded-full bg-ink px-4 py-1.5 text-paper">
              Sign in
            </Link>
          )}
        </div>
      </div>
      <div className="cornice" />
    </header>
  );
}
