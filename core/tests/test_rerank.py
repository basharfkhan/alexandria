import numpy as np
import pytest

from alexandria_core import FEATURE_NAMES, CatalogArrays, HybridRecommender, Reranker, rerank_features
from alexandria_core.recommender import parse_series


@pytest.fixture
def catalog() -> CatalogArrays:
    """20 books: 0-9 fantasy (0-2 are one series by one author), 10-19 romance."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 8))
    labels = np.repeat([0, 1], 10)
    titles = [f"Quest {i} (Dragon Saga, #{i + 1})" if i < 3 else f"Book {i}" for i in range(20)]
    authors = ["Ann Author" if i < 3 else f"Writer {i}" for i in range(20)]
    return CatalogArrays(
        content=centers[labels] + 0.05 * rng.normal(size=(20, 8)),
        popularity=np.arange(20) * 100 + 10,
        avg_rating=np.full(20, 4.0),
        genres=[["fantasy"] if lab == 0 else ["romance"] for lab in labels],
        cf_factors=centers[labels][:, :4] + 0.05 * rng.normal(size=(20, 4)),
        cf_bias=np.zeros(20),
        authors=authors,
        titles=titles,
    )


def test_parse_series():
    assert parse_series("The Well of Ascension (Mistborn, #2)") == ("mistborn", 2.0)
    assert parse_series("The Slow Regard of Silent Things (The Kingkiller Chronicle, #2.5)")[1] == 2.5
    assert parse_series("Gone Girl")[0] is None


def test_features_capture_author_and_series(catalog):
    rec = HybridRecommender(catalog)
    feedback = {0: 2.0, 15: -1.0}  # loved book 1 of the Dragon Saga, disliked a romance
    blend = rec.blend(feedback, [])
    pool = rec.candidates(blend, feedback, exclude=[], size=50)  # 20 books - 2 rated = 18 candidates
    x = rerank_features(rec, blend, feedback, [], pool)
    assert x.shape == (18, len(FEATURE_NAMES))
    assert np.isfinite(x[:, [FEATURE_NAMES.index(f) for f in FEATURE_NAMES if f != "series_number"]]).all()

    col = {name: x[:, i] for i, name in enumerate(FEATURE_NAMES)}
    row = {int(item): i for i, item in enumerate(pool)}
    # Book 1 is the next book in a series the reader loved, by the same author.
    assert col["next_in_series"][row[1]] == 1.0
    assert col["same_series_liked"][row[1]] == 1.0
    assert col["same_author_liked"][row[1]] == 1.0
    assert col["next_in_series"][row[2]] == 0.0  # #3 is not "next" after #1
    assert col["same_author_disliked"][row[10]] == 0.0
    assert col["max_content_sim_liked"][row[1]] > col["max_content_sim_liked"][row[19]]
    assert col["n_liked"][0] == 1 and col["n_disliked"][0] == 1
    assert col["stage1_rank_pct"][0] == 0.0 and col["stage1_rank_pct"][-1] == 1.0


def test_reranker_reorders_candidates(catalog):
    rec = HybridRecommender(catalog)

    class PreferHighIndex:
        """Stand-in model: score = the item's own popularity feature."""

        def predict(self, x):
            return x[:, FEATURE_NAMES.index("popularity_z")]

    feedback = {0: 2.0}
    baseline = [r.index for r in rec.recommend(feedback, k=5, diversity=0)]
    reranked = [r.index for r in rec.recommend(feedback, k=5, diversity=0, reranker=Reranker(PreferHighIndex()))]
    assert reranked != baseline
    assert reranked == sorted(reranked, key=lambda i: -catalog.popularity[i])


def test_reranker_rejects_mismatched_features():
    with pytest.raises(ValueError, match="different feature set"):
        Reranker(object(), feature_names=["only_one_feature"])


def test_genre_only_cold_start_skips_reranker(catalog):
    rec = HybridRecommender(catalog)

    class Exploding:
        def predict(self, x):  # pragma: no cover - must never be called
            raise AssertionError("reranker must not run without liked books")

    recs = rec.recommend({}, genres=["romance"], k=5, diversity=0, reranker=Reranker(Exploding()))
    assert all(r.index >= 10 for r in recs)
