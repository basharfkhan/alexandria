"""Extend the catalog with popular books published after Goodbooks-10k (2018 onwards).

    python -m alexandria_ml.data.recent_books                 # ~3,000 books, resumable
    python -m alexandria_ml.data.recent_books --package       # compress the cache into the repo

Goodbooks-10k stops in 2017, so the app could never recommend *Project Hail Mary*. Open Library
(CC0) knows those books: this module takes the most-shelved English titles per year, keeps the
metadata the recommender needs (title, author, subjects, description, cover, popularity), and
caches them like the enrichment fetcher.

These books have no ratings, so they get no collaborative factors from training - the pipeline
projects one from their nearest rated neighbours instead (see `alexandria_ml.cold_start`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from pathlib import Path

from alexandria_ml.config import DATA_DIR, ML_ROOT
from alexandria_ml.data.openlibrary import RateLimitedClient, clean_description, open_cache

log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "recent" / "recent_books.jsonl"
PACKAGED_CACHE = ML_ROOT / "enrichment" / "recent_books.jsonl.gz"

YEARS = range(2018, 2027)
SEARCH_FIELDS = "key,title,author_name,first_publish_year,readinglog_count,ratings_count,ratings_average,cover_i,subject,language"
PAGE_SIZE = 100
MIN_READINGLOG = 30  # below this, Open Library's long tail is mostly noise


def _is_usable(doc: dict) -> bool:
    title = (doc.get("title") or "").strip()
    return bool(
        title
        and len(title) < 200
        and doc.get("author_name")
        and doc.get("cover_i")
        and (doc.get("readinglog_count") or 0) >= MIN_READINGLOG
        and "eng" in (doc.get("language") or ["eng"])
        # Skip study guides and summaries of other books - they pollute recommendations.
        and not re.search(r"\b(summary|study guide|workbook|analysis of|sparknotes)\b", title, re.I)
    )


def search_year(client: RateLimitedClient, year: int, wanted: int) -> list[dict]:
    """Most-shelved English books first published in `year`."""
    found: list[dict] = []
    for page in range(1, 11):
        res = client.get_json(
            "/search.json",
            {"q": f"first_publish_year:{year}", "sort": "readinglog", "language": "eng",
             "fields": SEARCH_FIELDS, "limit": PAGE_SIZE, "page": page},
        )
        docs = (res or {}).get("docs", [])
        if not docs:
            break
        found.extend(doc for doc in docs if _is_usable(doc))
        if len(found) >= wanted:
            break
    log.info("  %d: %d candidates", year, len(found))
    return found[:wanted]


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open_cache(path) as fh:
        return {rec["work"]: rec for rec in map(json.loads, fh)}


def package_cache(cache_path: Path = CACHE_PATH, out: Path = PACKAGED_CACHE) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    records = load_cache(cache_path)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as fh:
        for record in records.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("packaged %d recent books -> %s (%.1f MB)", len(records), out, out.stat().st_size / 1024 / 1024)
    return out


def fetch(target: int = 3000, cache_path: Path = CACHE_PATH) -> Path:
    client = RateLimitedClient()
    cached = load_cache(cache_path)
    log.info("%d recent books cached; searching Open Library for up to %d", len(cached), target)

    per_year = max(1, target // len(YEARS))
    candidates: dict[str, dict] = {}
    for year in YEARS:
        for doc in search_year(client, year, per_year):
            candidates.setdefault(doc["key"], doc)

    todo = [doc for key, doc in candidates.items() if key not in cached]
    log.info("%d unique candidates, %d new to fetch", len(candidates), len(todo))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("a", encoding="utf-8") as out:
        for n, doc in enumerate(todo, start=1):
            work = client.get_json(f"{doc['key']}.json") or {}
            subjects = [s for s in (work.get("subjects") or doc.get("subject") or []) if isinstance(s, str)]
            record = {
                "work": doc["key"],
                "title": doc["title"].strip(),
                "authors": ", ".join(doc.get("author_name", [])[:3]),
                "year": doc.get("first_publish_year"),
                "readinglog_count": doc.get("readinglog_count") or 0,
                "ratings_count": doc.get("ratings_count") or 0,
                "ratings_average": doc.get("ratings_average"),
                "cover_id": doc.get("cover_i"),
                "subjects": subjects[:25],
                "description": clean_description(work.get("description")),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                out.flush()
                log.info("  fetched %d/%d", n, len(todo))
    return cache_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=3000)
    p.add_argument("--package", action="store_true", help="only compress the existing cache into the repo")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.package:
        package_cache()
        return
    fetch(target=args.target)
    cache = load_cache()
    with_description = sum(1 for r in cache.values() if r["description"])
    log.info("cache: %d recent books, %d with descriptions", len(cache), with_description)
    package_cache()


if __name__ == "__main__":
    main()
