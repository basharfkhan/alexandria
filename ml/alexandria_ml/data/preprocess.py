"""Clean raw Goodbooks CSVs into analysis-ready parquet tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alexandria_ml.config import POSITIVE_RATING
from alexandria_ml.data.genres import assign_genres, is_noise_tag

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    books: pd.DataFrame  # one row per book, `item_idx` = dense 0..n-1 index
    ratings: pd.DataFrame  # user_idx, item_idx, rating

    @property
    def n_items(self) -> int:
        return len(self.books)

    @property
    def n_users(self) -> int:
        return int(self.ratings.user_idx.max()) + 1


def _clean_title(title: str) -> str:
    return " ".join(str(title).split())


def build_dataset(raw_dir: Path, top_tags: int = 12) -> Dataset:
    books = pd.read_csv(raw_dir / "books.csv")
    ratings = pd.read_csv(raw_dir / "ratings.csv")
    tags = pd.read_csv(raw_dir / "tags.csv")
    book_tags = pd.read_csv(raw_dir / "book_tags.csv")

    # --- tags & genres --------------------------------------------------------
    book_tags = book_tags.merge(tags, on="tag_id")
    book_tags = book_tags[book_tags["count"] > 0]
    genre_map: dict[int, list[str]] = {}
    tag_map: dict[int, list[str]] = {}
    for gid, grp in book_tags.groupby("goodreads_book_id"):
        counts = dict(zip(grp.tag_name.astype(str), grp["count"], strict=True))
        genre_map[gid] = assign_genres(counts)
        clean = grp[~grp.tag_name.astype(str).map(is_noise_tag)].nlargest(top_tags, "count")
        tag_map[gid] = clean.tag_name.astype(str).tolist()

    # --- books ----------------------------------------------------------------
    books = books.drop_duplicates("book_id").sort_values("book_id").reset_index(drop=True)
    books["item_idx"] = np.arange(len(books))
    books["title"] = books["title"].map(_clean_title)
    books["authors"] = books["authors"].fillna("Unknown").astype(str)
    books["genres"] = books.goodreads_book_id.map(lambda g: genre_map.get(g, []))
    books["tags"] = books.goodreads_book_id.map(lambda g: tag_map.get(g, []))
    books["year"] = books.original_publication_year.astype("Int64")
    # Goodreads placeholder covers contain "nophoto"; the frontend renders its own.
    books["image_url"] = books.image_url.fillna("").astype(str).where(~books.image_url.astype(str).str.contains("nophoto"), "")
    books = books[
        ["item_idx", "book_id", "title", "authors", "year", "average_rating", "ratings_count", "image_url", "genres", "tags"]
    ].rename(columns={"average_rating": "avg_rating"})

    # --- ratings --------------------------------------------------------------
    ratings = ratings.drop_duplicates(["user_id", "book_id"], keep="last")
    ratings = ratings.merge(books[["book_id", "item_idx"]], on="book_id")
    user_codes, _ = pd.factorize(ratings.user_id, sort=True)
    ratings = pd.DataFrame(
        {"user_idx": user_codes.astype(np.int32), "item_idx": ratings.item_idx.astype(np.int32), "rating": ratings.rating.astype(np.int8)}
    )

    log.info(
        "dataset: %d books, %d users, %d ratings (%.0f%% positive), %d books without genres",
        len(books), ratings.user_idx.nunique(), len(ratings),
        100 * (ratings.rating >= POSITIVE_RATING).mean(), (books.genres.str.len() == 0).sum(),
    )
    return Dataset(books=books, ratings=ratings)


def save_dataset(ds: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.books.to_parquet(out_dir / "books.parquet", index=False)
    ds.ratings.to_parquet(out_dir / "ratings.parquet", index=False)


def load_dataset(processed_dir: Path) -> Dataset:
    books = pd.read_parquet(processed_dir / "books.parquet")
    books["genres"] = books.genres.map(list)
    books["tags"] = books.tags.map(list)
    return Dataset(books=books, ratings=pd.read_parquet(processed_dir / "ratings.parquet"))


def book_text(row: pd.Series) -> str:
    """Text used for content embeddings. Goodbooks has no blurbs, so tags stand in."""
    parts = [f"{row.title} by {row.authors}."]
    if len(row.genres):
        parts.append("Genres: " + ", ".join(row.genres) + ".")
    if len(row.tags):
        parts.append("Tags: " + ", ".join(t.replace("-", " ") for t in row.tags) + ".")
    return " ".join(parts)


def train_test_split_by_user(
    ratings: pd.DataFrame, test_frac: float = 0.2, min_positives: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a share of each eligible user's *positive* ratings as the test set.

    Goodbooks has no timestamps, so a random per-user holdout is the standard protocol.
    Users with fewer than ``min_positives`` positives stay entirely in train.
    """
    rng = np.random.default_rng(seed)
    pos = ratings[ratings.rating >= POSITIVE_RATING]
    counts = pos.groupby("user_idx").size()
    eligible = pos[pos.user_idx.isin(counts[counts >= min_positives].index)]

    rand = pd.Series(rng.random(len(eligible)), index=eligible.index)
    rank = rand.groupby(eligible.user_idx).rank(pct=True)
    test_index = eligible.index[rank <= test_frac]

    test = ratings.loc[test_index]
    train = ratings.drop(index=test_index)
    return train.reset_index(drop=True), test.reset_index(drop=True)
