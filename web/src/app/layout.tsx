import type { Metadata } from "next";
import { Cinzel, Fraunces, Inter } from "next/font/google";

import { Nav } from "@/components/Nav";
import { ServerWakeBanner } from "@/components/ServerWakeBanner";
import { AuthProvider } from "@/lib/auth";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const fraunces = Fraunces({ variable: "--font-fraunces", subsets: ["latin"] });
// Roman inscriptional capitals, for the wordmark and the small carved labels.
const cinzel = Cinzel({ variable: "--font-cinzel", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Alexandria - find your next favorite book",
  description: "Personalized book recommendations that learn from every book you rate.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${fraunces.variable} ${cinzel.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col font-sans">
        <AuthProvider>
          <Nav />
          <ServerWakeBanner />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">{children}</main>
          <footer className="mt-8">
            <div className="cornice" />
            <div className="py-6 text-center text-xs text-muted">
              <span className="inscribed text-[0.65rem] text-gold">Bibliotheca Alexandrina</span>
              <p className="mt-2">Book data from Goodbooks-10k (CC BY-SA 4.0) and Open Library (CC0)</p>
            </div>
          </footer>
        </AuthProvider>
      </body>
    </html>
  );
}
