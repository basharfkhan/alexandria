"""Merge Open Library's recent books into the Goodbooks catalog.

Goodbooks books carry a Goodreads `ratings_count` in the tens of thousands; Open Library's
"shelved by N readers" counts are far smaller and measured on a different population, so the two
cannot be compared directly. The recommender's popularity prior therefore reads a separate
`popularity` column: Goodbooks books use their rating count, recent books use their shelf count
mapped onto the *same distribution* by percentile (a book shelved more often than 90% of recent
books gets the popularity of a Goodbooks book at the 90th percentile).

`ratings_count` keeps the real number for display, so the UI never invents ratings.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from alexandria_ml.data.genres import assign_genres
from alexandria_ml.data.preprocess import DESCRIPTION_MAX_CHARS, OL_COVER_URL, clean_subjects

log = logging.getLogger(__name__)

RECENT_BOOK_ID_OFFSET = 1_000_000  # keeps ids clear of Goodbooks' 1..10000
DEFAULT_AVG_RATING = 3.9
# Recent books are dampened: their popularity maps into the lower part of the Goodbooks range,
# because a 2024 title has had far less time to accumulate readers than a 2008 classic.
POPULARITY_PERCENTILE_CAP = 0.75


def subject_to_tag(subject: str) -> str:
    """"Psychological fiction" -> "psychological-fiction", so genre keywords match."""
    return re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-")


def map_popularity(shelf_counts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map shelf counts onto the reference popularity distribution by percentile."""
    if len(shelf_counts) == 0:
        return np.array([], dtype=np.int64)
    order = shelf_counts.argsort().argsort()  # 0 = least shelved
    percentiles = (order + 0.5) / len(shelf_counts) * POPULARITY_PERCENTILE_CAP
    return np.quantile(reference, percentiles).astype(np.int64)


def merge_recent_books(books: pd.DataFrame, recent: list[dict]) -> pd.DataFrame:
    """Append recent books to the catalog frame, continuing item_idx and book_id."""
    if not recent:
        books = books.copy()
        books["popularity"] = books.ratings_count
        books["has_ratings"] = True
        return books

    rows = []
    for offset, rec in enumerate(recent):
        subjects = clean_subjects(rec.get("subjects") or [])
        genres = assign_genres({subject_to_tag(s): 1 for s in (rec.get("subjects") or [])})
        rows.append({
            "book_id": RECENT_BOOK_ID_OFFSET + offset,
            "title": rec["title"],
            "authors": rec["authors"] or "Unknown",
            "year": rec.get("year"),
            "avg_rating": rec.get("ratings_average") or DEFAULT_AVG_RATING,
            "ratings_count": rec.get("ratings_count") or 0,
            "image_url": OL_COVER_URL.format(rec["cover_id"]) if rec.get("cover_id") else "",
            "genres": genres,
            "tags": subjects,
            "description": (rec.get("description") or "")[:DESCRIPTION_MAX_CHARS] or None,
            "subjects": subjects,
            "shelf_count": rec.get("readinglog_count") or 0,
        })

    new = pd.DataFrame(rows)
    new["popularity"] = map_popularity(new.shelf_count.to_numpy(), books.ratings_count.to_numpy())
    new["has_ratings"] = False
    new["year"] = new.year.astype("Int64")
    new = new.drop(columns=["shelf_count"])

    existing = books.copy()
    existing["popularity"] = existing.ratings_count
    existing["has_ratings"] = True

    # Drop recent books that duplicate a Goodbooks title by the same author.
    key = lambda frame: (frame.title.str.lower().str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()  # noqa: E731
                         + "|" + frame.authors.str.lower().str.split(",").str[0].str.strip())
    new = new[~key(new).isin(set(key(existing)))]

    merged = pd.concat([existing, new], ignore_index=True)
    merged["item_idx"] = np.arange(len(merged))
    log.info(
        "catalog: %d Goodbooks books + %d recent books (%d with descriptions, popularity %d-%d)",
        len(existing), len(new), new.description.notna().sum(),
        new.popularity.min() if len(new) else 0, new.popularity.max() if len(new) else 0,
    )
    return merged
