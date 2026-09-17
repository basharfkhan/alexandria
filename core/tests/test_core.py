import numpy as np
import pytest

from alexandria_core import CatalogArrays, HybridRecommender, ImplicitFoldIn
from alexandria_core.metrics import evaluate_rankings, ndcg_at_k, recall_at_k
from alexandria_core.ranking import cf_weight, mmr_rerank, zscore


@pytest.fixture
def catalog() -> CatalogArrays:
    """Two clearly separated clusters: fantasy books 0-49, romance books 50-99."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2, 16))
    labels = np.repeat([0, 1], 50)
    content = centers[labels] + 0.1 * rng.normal(size=(100, 16))
    factors = centers[labels][:, :8] + 0.1 * rng.normal(size=(100, 8))
    return CatalogArrays(
        content=content,
        popularity=rng.integers(10, 1000, size=100),
        avg_rating=rng.uniform(3, 5, size=100),
        genres=[["fantasy"] if lab == 0 else ["romance"] for lab in labels],
        cf_factors=factors,
        cf_bias=np.zeros(100),
    )


def test_metrics_basic():
    assert recall_at_k([1, 2, 3], {2, 9}, k=3) == 0.5
    assert ndcg_at_k([2, 1], {2}, k=2) == pytest.approx(1.0)
    res = evaluate_rankings({0: [1, 2]}, {0: {2}}, n_items=10, ks=(2,))
    assert res["hit_rate@2"] == 1.0 and res["coverage@2"] == pytest.approx(0.2)


def test_zscore_and_cf_weight():
    assert np.allclose(zscore(np.ones(5)), 0)
    assert cf_weight(0) == 0
    assert cf_weight(1) < cf_weight(10) < 0.6


@pytest.mark.parametrize("seed", range(20))
def test_foldin_moves_toward_likes_and_away_from_dislikes(seed):
    Y = np.random.default_rng(seed).normal(size=(50, 8))
    fold = ImplicitFoldIn(Y)
    liked = fold.user_vector(np.array([3]), np.array([2.0]))
    cosine = (Y @ liked) / np.linalg.norm(Y, axis=1)
    assert 3 in np.argsort(-cosine)[:3]

    with_dislike = fold.user_vector(np.array([3, 7]), np.array([2.0, -1.0]))
    scores = Y @ with_dislike
    share_ranked_above = (scores > scores[7]).mean()
    # Either the dislike lowered the book's score, or it was already near the bottom of the ranking.
    assert scores[7] < Y[7] @ liked or share_ranked_above >= 0.9


def test_foldin_empty_feedback():
    assert np.allclose(ImplicitFoldIn(np.ones((5, 3))).user_vector(np.array([]), np.array([])), 0)


def test_mmr_diversifies():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    picked = mmr_rerank(np.array([0, 1, 2]), np.array([1.0, 0.99, 0.9]), emb, k=2, lambda_=0.5)
    assert list(picked) == [0, 2]


def test_genre_cold_start(catalog):
    rec = HybridRecommender(catalog)
    recs = rec.recommend({}, genres=["romance"], k=10, diversity=0)
    assert all(r.index >= 50 for r in recs)
    assert recs[0].reason == "genre"


def test_feedback_steers_and_excludes(catalog):
    rec = HybridRecommender(catalog)
    feedback = {0: 2.0, 1: 1.0, 60: -1.0}
    recs = rec.recommend(feedback, k=10, exclude=[2])
    ids = [r.index for r in recs]
    assert not set(ids) & {0, 1, 2, 60}
    assert sum(i < 50 for i in ids) >= 8
    assert recs[0].because_of in {0, 1}


def test_explore_slots_are_marked(catalog):
    rec = HybridRecommender(catalog)
    recs = rec.recommend({0: 2.0}, k=10, explore_slots=2, seed=42)
    assert len(recs) == 10
    assert sum(r.reason == "explore" for r in recs) == 2


def test_similar(catalog):
    rec = HybridRecommender(catalog)
    assert all(j < 50 for j, _ in rec.similar(5, k=5))
