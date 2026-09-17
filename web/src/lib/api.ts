export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Signal = "loved" | "liked" | "want_to_read" | "disliked" | "not_interested";

export interface Book {
  id: number;
  title: string;
  authors: string;
  year: number | null;
  avg_rating: number;
  ratings_count: number;
  image_url: string | null;
  genres: string[];
}

export interface User {
  id: number;
  username: string;
  favorite_genres: string[];
  onboarded: boolean;
}

export interface Genre {
  slug: string;
  count: number;
}

export type Reason = "similar" | "collaborative" | "genre" | "popular" | "explore";

export interface Recommendation {
  book: Book;
  score: number;
  reason: Reason;
  explanation: string;
  because_of: Book | null;
}

export interface Recommendations {
  model_version: string | null;
  personalization: number;
  items: Recommendation[];
}

export interface LibraryItem {
  book: Book;
  signal: Signal;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface BookMention {
  title: string;
  author: string;
}

export interface ChatResponse {
  reply: string;
  ready: boolean;
  preferences: { genres: string[]; loved_books: BookMention[]; disliked_books: BookMention[] };
  matched_loved: Book[];
  matched_disliked: Book[];
}

export interface OnboardingPayload {
  genres: string[];
  loved_book_ids: number[];
  liked_book_ids?: number[];
  disliked_book_ids: number[];
  source: "onboarding" | "chat";
}

const TOKEN_KEY = "alexandria.token";

export const tokenStore = {
  get: (): string | null => {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  clear: () => localStorage.removeItem(TOKEN_KEY),
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = tokenStore.get();
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  register: (username: string, password: string) =>
    request<{ access_token: string }>("/auth/register", { method: "POST", body: json({ username, password }) }),
  login: (username: string, password: string) =>
    request<{ access_token: string }>("/auth/login", { method: "POST", body: json({ username, password }) }),
  me: () => request<User>("/auth/me"),

  genres: () => request<Genre[]>("/genres"),
  search: (q: string, limit = 8) => request<Book[]>(`/books/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  popular: (genre?: string, limit = 12) =>
    request<Book[]>(`/books/popular?limit=${limit}${genre ? `&genre=${encodeURIComponent(genre)}` : ""}`),
  book: (id: number) => request<Book>(`/books/${id}`),
  similar: (id: number, limit = 12) => request<Book[]>(`/books/${id}/similar?limit=${limit}`),

  onboard: (payload: OnboardingPayload) => request<User>("/me/onboarding", { method: "POST", body: json(payload) }),
  setGenres: (genres: string[]) => request<User>("/me/genres", { method: "PUT", body: json(genres) }),
  recommendations: (limit = 24) => request<Recommendations>(`/me/recommendations?limit=${limit}`),
  library: () => request<LibraryItem[]>("/me/library"),
  feedback: (bookId: number, signal: Signal, source: "recommendation" | "app" | "search" = "app", position?: number) =>
    request<LibraryItem>(`/me/books/${bookId}`, { method: "PUT", body: json({ signal, source, position }) }),
  removeFeedback: (bookId: number) => request<void>(`/me/books/${bookId}`, { method: "DELETE" }),

  chat: (messages: ChatMessage[]) =>
    request<ChatResponse>("/chat/onboarding", { method: "POST", body: json({ messages }) }),
};

export const genreLabel = (slug: string) =>
  slug
    .split("-")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
