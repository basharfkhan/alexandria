import numpy as np
import pandas as pd
import pytest

from alexandria_core.recommender import CatalogArrays, HybridRecommender
from alexandria_ml.drift import (
    PROBE_READERS,
    compare,
    compare_inputs,
    fingerprint,
    input_stats,
    jaccard,
    resolve_probe_books,
    summarise,
    total_variation,
)


def catalog(n=60):
    """A small catalog that contains every probe reader's first title, plus filler."""
    titles = [t[0] for t in PROBE_READERS.values()]
    titles += [f"Filler Book {i}" for i in range(n - len(titles))]
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "book_id": np.arange(1, len(titles) + 1),
        "title": titles,
        "authors": [f"Author {i % 7}" for i in range(len(titles))],
        "genres": [["science-fiction"] if i % 2 else ["fantasy"] for i in range(len(titles))],
        "popularity": rng.integers(100, 50_000, len(titles)),
        "avg_rating": rng.uniform(3.2, 4.8, len(titles)),
        "has_ratings": True,
        "description": "a book",
    })


def recommender(books, seed=0):
    rng = np.random.default_rng(seed)
    n = len(books)
    content = rng.normal(size=(n, 16))
    content /= np.linalg.norm(content, axis=1, keepdims=True)
    return HybridRecommender(CatalogArrays(
        content=content,
        popularity=books.popularity.to_numpy(),
        avg_rating=books.avg_rating.to_numpy(),
        genres=books.genres.tolist(),
        cf_factors=rng.normal(size=(n, 8)),
        cf_bias=np.zeros(n),
        authors=books.authors.tolist(),
        titles=books.title.tolist(),
        cold_start=[False] * n,
    ))


def test_probe_readers_resolve_by_title_not_by_id():
    """Ids shift when the catalog changes; a title-based cohort survives that."""
    books = catalog()
    resolved = resolve_probe_books(books)
    assert set(resolved) == set(PROBE_READERS)
    assert all(len(v) >= 1 for v in resolved.values())

    shifted = books.assign(book_id=books.book_id + 1_000)
    assert resolve_probe_books(shifted) == resolved, "resolution is by position, not by book id"


def test_probe_reader_with_no_matching_book_is_skipped():
    books = catalog()
    books = books[~books.title.str.startswith("The Martian")].reset_index(drop=True)
    assert "hard_sf" not in resolve_probe_books(books)


def test_fingerprint_is_deterministic_for_the_same_model():
    books = catalog()
    model = recommender(books)
    assert fingerprint(books, model) == fingerprint(books, model)


def test_fingerprint_reports_what_the_cohort_was_shown():
    books = catalog()
    print_ = fingerprint(books, recommender(books), k=10)
    assert print_["n_readers"] == len(PROBE_READERS)
    assert all(len(ids) <= 10 for ids in print_["readers"].values())
    assert 0.0 <= print_["popularity_pct"] <= 1.0
    assert pytest.approx(sum(print_["genre_mix"].values()), abs=1e-6) == 1.0


def test_identical_models_show_no_drift():
    books = catalog()
    print_ = fingerprint(books, recommender(books))
    drift = compare(print_, print_)
    assert drift["overlap_mean"] == 1.0 and not drift["warnings"]


def test_a_different_model_drifts_and_warns():
    books = catalog()
    before = fingerprint(books, recommender(books, seed=0))
    after = fingerprint(books, recommender(books, seed=99))  # unrelated latent space
    drift = compare(before, after)
    assert drift["overlap_mean"] < 1.0
    assert any("recommendations" in w for w in drift["warnings"])


def test_popularity_drift_is_called_out():
    """The failure that shipped once: accuracy barely moves, the lists go bestseller."""
    before = {"readers": {"a": [1, 2, 3]}, "popularity_pct": 0.40,
              "cold_start_share": 0.0, "genre_mix": {"fantasy": 1.0}}
    after = {**before, "popularity_pct": 0.72}
    drift = compare(after, before)
    assert drift["popularity_pct_shift"] == pytest.approx(0.32)
    assert any("more popular" in w for w in drift["warnings"])


def test_genre_mix_shift_is_called_out():
    before = {"readers": {"a": [1]}, "popularity_pct": 0.5, "cold_start_share": 0.0,
              "genre_mix": {"fantasy": 0.5, "thriller": 0.5}}
    after = {**before, "genre_mix": {"fantasy": 0.9, "thriller": 0.1}}
    assert any("genre mix" in w for w in compare(after, before)["warnings"])


def test_jaccard_and_total_variation():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert total_variation({"a": 1.0}, {"a": 1.0}) == 0.0
    assert total_variation({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_input_stats_track_the_training_data():
    books = catalog()
    ratings = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 2], "rating": [5, 4, 5]})
    stats = input_stats(books, ratings)
    assert stats["n_books"] == len(books) and stats["n_ratings"] == 3 and stats["n_users"] == 2
    assert stats["ratings_per_user"] == 1.5 and stats["description_share"] == 1.0


def test_input_drift_warns_on_a_jump_but_not_on_a_nudge():
    base = {"n_books": 10_000, "n_ratings": 6_000_000, "n_users": 50_000}
    assert not compare_inputs({**base, "n_books": 10_200}, base)["warnings"]
    jumped = compare_inputs({**base, "n_books": 20_000}, base)
    assert jumped["changes"]["n_books"] == pytest.approx(1.0)
    assert any("n_books" in w for w in jumped["warnings"])


def test_summarise_handles_a_first_run():
    assert "first run" in summarise(None)
    quiet = summarise({"predictions": {"overlap_mean": 0.9, "overlap_min": 0.8,
                                       "popularity_pct_shift": 0.01, "genre_shift": 0.02,
                                       "warnings": []}})
    assert "nothing unusual" in quiet
