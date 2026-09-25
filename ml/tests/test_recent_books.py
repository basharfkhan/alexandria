import json

import numpy as np
import pandas as pd

from alexandria_ml.cold_start import project_cold_start
from alexandria_ml.data.preprocess import build_dataset
from alexandria_ml.data.recent_merge import RECENT_BOOK_ID_OFFSET, map_popularity, merge_recent_books
from alexandria_ml.data.synthetic import make_synthetic

GOODBOOKS = pd.DataFrame({
    "item_idx": [0, 1], "book_id": [1, 2], "title": ["The Martian", "Gone Girl"],
    "authors": ["Andy Weir", "Gillian Flynn"], "year": pd.array([2011, 2012], dtype="Int64"),
    "avg_rating": [4.4, 4.0], "ratings_count": [100_000, 50_000], "image_url": ["", ""],
    "genres": [["science-fiction"], ["thriller"]], "tags": [["space"], ["crime"]],
    "description": ["a", "b"], "subjects": [[], []],
})


def recent(title="Project Hail Mary", authors="Andy Weir", shelved=1367, **kw):
    return {"work": f"/works/{title[:4]}", "title": title, "authors": authors, "year": 2021,
            "readinglog_count": shelved, "ratings_count": 300, "ratings_average": 4.5,
            "cover_id": 123, "subjects": ["Science fiction", "Space flight"],
            "description": "A lone astronaut must save the earth.", **kw}


def test_recent_books_join_the_catalog():
    merged = merge_recent_books(GOODBOOKS, [recent()])
    added = merged[merged.book_id >= RECENT_BOOK_ID_OFFSET].iloc[0]

    assert added.title == "Project Hail Mary" and not added.has_ratings
    assert added.genres == ["science-fiction"], "subjects are mapped onto the genre vocabulary"
    assert added.image_url.endswith("123-M.jpg")
    assert merged.item_idx.tolist() == [0, 1, 2], "item_idx stays contiguous"
    assert merged[merged.book_id < RECENT_BOOK_ID_OFFSET].has_ratings.all()


def test_duplicate_of_an_existing_book_is_dropped():
    merged = merge_recent_books(GOODBOOKS, [recent(title="The Martian (Movie Tie-In)")])
    assert len(merged) == 2


def test_popularity_is_mapped_onto_the_rated_distribution():
    """Shelf counts and Goodreads rating counts are different scales; ranking needs one scale."""
    merged = merge_recent_books(GOODBOOKS, [recent(shelved=10), recent(title="Other", shelved=5000)])
    new = merged[~merged.has_ratings].sort_values("popularity")
    assert new.popularity.is_monotonic_increasing
    assert new.popularity.max() <= GOODBOOKS.ratings_count.max()
    assert (new.ratings_count == 300).all(), "the displayed rating count stays the real one"

    mapped = map_popularity(np.array([1, 2, 3]), np.array([10, 100, 1000, 10_000]))
    assert len(mapped) == 3 and (np.diff(mapped) > 0).all()


def test_no_recent_books_leaves_the_catalog_unchanged():
    merged = merge_recent_books(GOODBOOKS, [])
    assert len(merged) == 2 and merged.has_ratings.all()
    assert merged.popularity.tolist() == GOODBOOKS.ratings_count.tolist()


def test_build_dataset_merges_recent_books(tmp_path):
    raw = make_synthetic(tmp_path / "raw", n_books=60, n_users=40)
    cache = tmp_path / "recent.jsonl"
    cache.write_text(json.dumps(recent(title="A Brand New Book")) + "\n", encoding="utf-8")

    ds = build_dataset(raw, recent_path=cache)
    assert ds.n_items == 61
    assert (~ds.books.has_ratings).sum() == 1
    assert ds.ratings.item_idx.max() < 60, "ratings still point at the rated books only"


def test_cold_start_projection_borrows_from_similar_books():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 16))
    content = np.vstack([
        centers[0] + 0.05 * rng.normal(size=(5, 16)),  # rated, topic A
        centers[1] + 0.05 * rng.normal(size=(5, 16)),  # rated, topic B
        centers[0] + 0.05 * rng.normal(size=(1, 16)),  # new, topic A
    ])
    factors = np.vstack([np.tile([1.0, 0.0], (5, 1)), np.tile([0.0, 1.0], (5, 1)), np.zeros((1, 2))])
    bias = np.concatenate([np.full(10, 0.5), [0.0]])
    has_ratings = np.array([True] * 10 + [False])

    new_factors, new_bias = project_cold_start(content, factors, bias, has_ratings)
    assert new_factors[10][0] > new_factors[10][1], "the new book leans towards the topic it resembles"
    assert 0 < new_bias[10] < 0.5, "borrowed evidence is shrunk below the neighbours' own bias"
    assert np.allclose(new_factors[:10], factors[:10]), "rated books are untouched"
