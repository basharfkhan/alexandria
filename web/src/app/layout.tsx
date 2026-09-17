import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";

import { Nav } from "@/components/Nav";
import { ServerWakeBanner } from "@/components/ServerWakeBanner";
import { AuthProvider } from "@/lib/auth";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const fraunces = Fraunces({ variable: "--font-fraunces", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Alexandria - find your next favorite book",
  description: "Personalized book recommendations that learn from every book you rate.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <AuthProvider>
          <Nav />
          <ServerWakeBanner />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6">{children}</main>
          <footer className="border-t border-line py-6 text-center text-xs text-muted">
            Alexandria · book data from Goodbooks-10k (CC BY-SA 4.0)
          </footer>
        </AuthProvider>
      </body>
    </html>
  );
}
