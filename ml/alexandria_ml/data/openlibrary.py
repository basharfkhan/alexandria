"""Enrich Goodbooks-10k with book descriptions, subjects and covers from Open Library (CC0 data).

    python -m alexandria_ml.data.openlibrary            # ~1 hour for 10k books, resumable

Goodbooks has no descriptions, so content embeddings only see titles and shelf tags - the
weakest part of the recommender. For each book we:

1. Resolve its Open Library *work*: batched ISBN search (Goodbooks ISBN-10s lost their leading
   zeros when saved as numbers; ISBN-13s were saved as floats and are unusable), falling back
   to a title + author search that only accepts results whose author surname matches.
2. Fetch the work record: description, subjects and a cover id.

Results are appended to a JSONL cache, one line per book, so an interrupted run resumes where
it stopped. Requests are rate limited and identify the project in the User-Agent, per Open
Library's API guidelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from alexandria_ml.config import DATA_DIR, RAW_DIR

log = logging.getLogger(__name__)

API = "https://openlibrary.org"
USER_AGENT = "Alexandria-portfolio-recommender/0.1 (+https://github.com/basharfkhan/alexandria)"
CACHE_PATH = DATA_DIR / "openlibrary" / "works.jsonl"
ISBN_BATCH = 40


class RateLimitedClient:
    def __init__(self, max_per_second: float = 2.5, retries: int = 5):
        self.min_interval = 1.0 / max_per_second
        self.retries = retries
        self._last = 0.0

    def get_json(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{API}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                backoff = 2 ** attempt * (10 if exc.code == 429 else 2)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                backoff = 2 ** attempt * 2
            log.warning("request failed (attempt %d), retrying in %ds: %s", attempt + 1, backoff, url[:120])
            time.sleep(backoff)
        log.error("giving up on %s", url[:120])
        return None


def normalize_isbn10(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    if re.fullmatch(r"\d{1,10}", value) and len(value) >= 8:
        return value.zfill(10)
    return value if re.fullmatch(r"\d{9}[\dX]", value) else None


def _surname(authors: str) -> str:
    return authors.split(",")[0].strip().split()[-1].lower() if authors.strip() else ""


def clean_description(raw) -> str | None:
    """Open Library descriptions are free text, sometimes with markdown links and source footers."""
    text = raw.get("value") if isinstance(raw, dict) else raw
    if not isinstance(text, str):
        return None
    text = re.split(r"\n-{3,}|\n\*{3,}|\(\[source\]", text)[0]  # drop "----" sections and source footers
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"\[[^\]]*\]\[\d+\]|\[\d+\]:\s*\S+", "", text)  # reference-style links
    text = re.sub(r"(\*{1,3}|_{2,3})(\S.*?\S|\S)\1", r"\2", text)  # **bold**, *italic*, __bold__ -> text
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= 40 else None


def resolve_by_isbn(client: RateLimitedClient, isbns: list[str]) -> dict[str, dict]:
    res = client.get_json(
        "/search.json",
        {"q": f"isbn:({' OR '.join(isbns)})", "fields": "key,title,author_name,isbn", "limit": 200},
    )
    wanted, found = set(isbns), {}
    for doc in (res or {}).get("docs", []):
        for isbn in set(doc.get("isbn", [])) & wanted:
            found.setdefault(isbn, doc)
    return found


def resolve_by_title(client: RateLimitedClient, title: str, authors: str) -> dict | None:
    res = client.get_json(
        "/search.json",
        {"title": title, "author": authors.split(",")[0].strip(), "fields": "key,title,author_name", "limit": 5},
    )
    surname = _surname(authors)
    for doc in (res or {}).get("docs", []):
        if any(surname and surname in name.lower() for name in doc.get("author_name", [])):
            return doc
    return None


def load_cache(path: Path = CACHE_PATH) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {rec["book_id"]: rec for rec in map(json.loads, fh)}


def fetch(raw_dir: Path = RAW_DIR, cache_path: Path = CACHE_PATH, limit: int | None = None) -> Path:
    books = pd.read_csv(raw_dir / "books.csv", dtype={"isbn": str})
    if limit:
        books = books.head(limit)
    done = load_cache(cache_path)
    todo = books[~books.book_id.isin(done)]
    log.info("%d books cached, %d to fetch", len(done), len(todo))
    if todo.empty:
        return cache_path

    client = RateLimitedClient()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0, stats = time.time(), {"isbn": 0, "title": 0, "miss": 0, "description": 0}

    with cache_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), ISBN_BATCH):
            chunk = todo.iloc[start : start + ISBN_BATCH]
            isbns = {bid: normalize_isbn10(i) for bid, i in zip(chunk.book_id, chunk.isbn, strict=True)}
            by_isbn = resolve_by_isbn(client, [i for i in isbns.values() if i])

            for row in chunk.itertuples(index=False):
                doc, match = by_isbn.get(isbns[row.book_id]), "isbn"
                if doc is None:
                    title = row.original_title if isinstance(row.original_title, str) and row.original_title else row.title
                    doc, match = resolve_by_title(client, re.sub(r"\s*\([^)]*#[^)]*\)\s*$", "", title), row.authors), "title"

                record = {"book_id": int(row.book_id), "match": None, "work": None,
                          "description": None, "subjects": [], "cover_id": None}
                if doc is None:
                    stats["miss"] += 1
                else:
                    stats[match] += 1
                    work = client.get_json(f"{doc['key']}.json") or {}
                    record.update(
                        match=match,
                        work=doc["key"],
                        description=clean_description(work.get("description")),
                        subjects=[s for s in work.get("subjects", []) if isinstance(s, str)][:25],
                        cover_id=next((c for c in work.get("covers", []) if isinstance(c, int) and c > 0), None),
                    )
                    stats["description"] += record["description"] is not None
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            n = min(start + ISBN_BATCH, len(todo))
            rate = n / (time.time() - t0)
            log.info("%d/%d books  %s  (%.1f books/s, ~%.0f min left)",
                     n, len(todo), stats, rate, (len(todo) - n) / rate / 60)
    return cache_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="only the first N books (for a quick test)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch(limit=args.limit)
    cache = load_cache()
    with_desc = sum(1 for r in cache.values() if r["description"])
    log.info("cache: %d books, %d matched, %d with descriptions", len(cache),
             sum(1 for r in cache.values() if r["work"]), with_desc)


if __name__ == "__main__":
    main()
