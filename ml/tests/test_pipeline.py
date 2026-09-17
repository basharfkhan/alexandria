import json

import numpy as np
import pytest

from alexandria_ml.data.genres import assign_genres, genres_for_tag, is_noise_tag
from alexandria_ml.data.preprocess import book_text, build_dataset, train_test_split_by_user
from alexandria_ml.data.synthetic import make_synthetic


def test_genre_mapping():
    assert genres_for_tag("sci-fi") == ["science-fiction"]
    # Regression: "science" used to match inside "science-fiction", labelling 1984 as a science book.
    assert genres_for_tag("science-fiction") == ["science-fiction"]
    assert genres_for_tag("popular-science") == ["science"]
    assert genres_for_tag("science-fiction-fantasy") == ["fantasy", "science-fiction"]
    assert genres_for_tag("historical-fiction") == ["historical-fiction"]
    assert "young-adult" in genres_for_tag("ya-fantasy")
    assert genres_for_tag("to-read") == []
    assert is_noise_tag("books-i-own") and is_noise_tag("2015-reads") and not is_noise_tag("fantasy")
    assert assign_genres({"fantasy": 900, "romance": 20, "to-read": 5000}) == ["fantasy"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    raw = make_synthetic(tmp_path_factory.mktemp("raw"), n_books=120, n_users=150)
    return build_dataset(raw)


def test_build_dataset(dataset):
    assert dataset.n_items == 120
    assert dataset.books.item_idx.tolist() == list(range(120))
    assert dataset.books.genres.map(len).gt(0).all()
    assert "to-read" not in dataset.books.tags.explode().unique()
    assert "Genres:" in book_text(dataset.books.iloc[0])


def test_open_library_enrichment(tmp_path):
    import json as _json

    from alexandria_ml.data.openlibrary import clean_description

    raw = make_synthetic(tmp_path / "raw", n_books=60, n_users=40)
    cache = tmp_path / "works.jsonl"
    cache.write_text(
        _json.dumps({"book_id": 1, "work": "/works/OL1W", "match": "isbn", "cover_id": 42,
                     "description": "A heist crew of thieves takes on a city of alchemy and debt.",
                     "subjects": ["series:Crows", "Thieves", "Heists", "Accessible book"]}) + "\n",
        encoding="utf-8",
    )
    ds = build_dataset(raw, enrichment_path=cache)
    first, second = ds.books.iloc[0], ds.books.iloc[1]
    assert first.description.startswith("A heist crew") and first.subjects == ["thieves", "heists"]
    assert first.image_url == "https://covers.openlibrary.org/b/id/42-M.jpg"
    assert not isinstance(second.description, str) and second.subjects == []
    text = book_text(first)
    assert "A heist crew" in text and "Subjects: thieves, heists." in text

    raw_description = {"value": "A long saga about [dragons](https://x.y) and the riders who love them.\n----------\nContains: stuff"}
    assert clean_description(raw_description) == "A long saga about dragons and the riders who love them."
    assert clean_description("too short") is None


def test_split_holds_out_positives_only(dataset):
    train, test = train_test_split_by_user(dataset.ratings)
    assert len(train) + len(test) == len(dataset.ratings)
    assert (test.rating >= 4).all()
    merged = train.merge(test, on=["user_idx", "item_idx"])
    assert merged.empty


def test_export_handles_non_ascii_titles(tmp_path):
    """Regression: on Windows the default cp1252 codec crashed on titles like 'İstanbul'."""
    import pandas as pd

    from alexandria_ml.export import export_artifacts

    books = pd.DataFrame([{
        "book_id": 1, "title": "İstanbul: Memories and the City", "authors": "Orhan Pamuk", "year": 2003,
        "avg_rating": 3.9, "ratings_count": 100, "image_url": "", "genres": ["memoir"], "tags": ["turkey"],
    }])
    export_artifacts(tmp_path, books, np.zeros((1, 384)), np.zeros((1, 64)), np.zeros(1), manifest={})
    assert json.loads((tmp_path / "books.json").read_text(encoding="utf-8"))[0]["title"].startswith("İstanbul")


def test_full_pipeline_beats_popularity(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEXANDRIA_DISABLE_MLFLOW", "1")
    monkeypatch.setattr("alexandria_ml.pipeline.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("alexandria_ml.pipeline.PROCESSED_DIR", tmp_path / "processed")
    from alexandria_ml.pipeline import main

    art = tmp_path / "artifacts"
    results = main(["--synthetic", "--embedder", "tfidf", "--epochs", "15", "--batch-size", "1024",
                    "--artifact-dir", str(art), "--no-final-fit"])

    assert results["hybrid_foldin"]["ndcg@20"] > results["popularity"]["ndcg@20"]
    assert results["bpr"]["recall@20"] > results["popularity"]["recall@20"]

    books = json.loads((art / "books.json").read_text())
    assert len(books) == np.load(art / "content_embeddings.npy").shape[0]
    assert np.load(art / "cf_factors.npy").shape == (len(books), 64)
    assert json.loads((art / "manifest.json").read_text())["content_dim"] == 384
