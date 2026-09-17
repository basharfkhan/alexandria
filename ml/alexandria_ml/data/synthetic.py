"""Small synthetic dataset with the exact Goodbooks-10k schema.

Users have latent genre preferences, so a working model *should* beat popularity.
Used by tests and CI so the full pipeline runs in seconds without downloading 70MB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SYN_GENRES = ["fantasy", "science-fiction", "romance", "mystery", "history", "self-help"]
WORDS = ["shadow", "crown", "star", "heart", "secret", "empire", "river", "night", "garden", "code"]


def make_synthetic(out_dir: Path, n_books: int = 400, n_users: int = 600, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    book_genre = rng.integers(0, len(SYN_GENRES), size=n_books)
    popularity = rng.pareto(1.5, size=n_books) + 1

    books = pd.DataFrame(
        {
            "book_id": np.arange(1, n_books + 1),
            "goodreads_book_id": np.arange(1, n_books + 1) * 11,
            "authors": [f"Author {i % 90}" for i in range(n_books)],
            "original_publication_year": rng.integers(1900, 2024, size=n_books).astype(float),
            "title": [
                f"The {WORDS[i % len(WORDS)].title()} of {SYN_GENRES[g].title()} {i}"
                for i, g in enumerate(book_genre)
            ],
            "average_rating": rng.uniform(3.2, 4.6, size=n_books).round(2),
            "ratings_count": (popularity * 1000).astype(int),
            "image_url": "",
            "small_image_url": "",
        }
    )

    tags = pd.DataFrame({"tag_id": range(len(SYN_GENRES) + 2), "tag_name": SYN_GENRES + ["to-read", "owned"]})
    book_tags = pd.DataFrame(
        [
            {"goodreads_book_id": gid, "tag_id": tid, "count": int(rng.integers(50, 500))}
            for gid, g in zip(books.goodreads_book_id, book_genre, strict=True)
            for tid in (g, len(SYN_GENRES))
        ]
    )

    rows = []
    for user in range(1, n_users + 1):
        liked = rng.choice(len(SYN_GENRES), size=2, replace=False)
        affinity = np.where(np.isin(book_genre, liked), 1.0, 0.08) * popularity
        n_read = int(rng.integers(15, 45))
        read = rng.choice(n_books, size=n_read, replace=False, p=affinity / affinity.sum())
        for b in read:
            base = 4.3 if book_genre[b] in liked else 2.6
            rows.append((user, b + 1, int(np.clip(round(base + rng.normal(0, 0.8)), 1, 5))))
    ratings = pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])

    books.to_csv(out_dir / "books.csv", index=False)
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    tags.to_csv(out_dir / "tags.csv", index=False)
    book_tags.to_csv(out_dir / "book_tags.csv", index=False)
    return out_dir
