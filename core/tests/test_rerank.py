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
    # stage1_weight=0: the model alone decides the order.
    pure = Reranker(PreferHighIndex(), stage1_weight=0.0)
    reranked = [r.index for r in rec.recommend(feedback, k=5, diversity=0, reranker=pure)]
    assert reranked != baseline
    assert reranked == sorted(reranked, key=lambda i: -catalog.popularity[i])


def test_stage1_weight_anchors_the_ranking(catalog):
    """A large stage-1 weight pulls the learned order back towards the first-stage ranking."""
    rec = HybridRecommender(catalog)

    class PreferHighIndex:
        def predict(self, x):
            return x[:, FEATURE_NAMES.index("popularity_z")]

    feedback = {0: 2.0}
    baseline = [r.index for r in rec.recommend(feedback, k=8, diversity=0)]
    pure = [r.index for r in rec.recommend(feedback, k=8, diversity=0,
                                           reranker=Reranker(PreferHighIndex(), stage1_weight=0.0))]
    anchored = [r.index for r in rec.recommend(feedback, k=8, diversity=0,
                                               reranker=Reranker(PreferHighIndex(), stage1_weight=20.0))]
    overlap = lambda a, b: len(set(a[:5]) & set(b[:5]))  # noqa: E731
    assert overlap(anchored, baseline) == 5, "a heavy anchor keeps stage 1's top picks"
    assert overlap(pure, baseline) < overlap(anchored, baseline)


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


def test_new_books_get_reserved_slots(catalog):
    """Never-rated books cannot displace rated ones: they fill a fixed number of slots."""
    catalog.cold_start = [i % 2 == 1 for i in range(20)]  # every other book has no ratings
    rec = HybridRecommender(catalog)

    without = [r.index for r in rec.recommend({0: 2.0}, k=8, diversity=0)]
    with_slots = rec.recommend({0: 2.0}, k=8, diversity=0, new_book_slots=2)

    assert all(i % 2 == 0 for i in without), "no new books unless slots are reserved"
    assert sum(r.reason == "new_release" for r in with_slots) == 2
    assert all(catalog.cold_start[r.index] for r in with_slots if r.reason == "new_release")
    assert len(with_slots) == 8, "reserved slots replace ranked picks, they don't extend the page"
    ranked = [r.index for r in with_slots if r.reason != "new_release"]
    assert ranked == without[: len(ranked)], "the ranked picks keep their order"


def test_new_book_slots_respect_the_author_cap(catalog):
    catalog.cold_start = [i >= 10 for i in range(20)]
    catalog.authors = ["Solo Author"] * 10 + ["New Writer"] * 10
    rec = HybridRecommender(catalog)
    recs = rec.recommend({0: 2.0}, k=8, diversity=0, max_per_author=3, new_book_slots=3)
    assert sum(r.reason == "new_release" for r in recs) == 3


def test_no_new_books_in_catalog_means_no_reserved_slots(catalog):
    """A catalog of only rated books fills every slot with ranked picks."""
    rec = HybridRecommender(catalog)
    recs = rec.recommend({0: 2.0}, k=6, diversity=0, new_book_slots=2)
    assert len(recs) == 6 and not any(r.reason == "new_release" for r in recs)
