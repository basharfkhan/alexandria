import json

import pytest

from alexandria_ml.registry import evaluate_candidate, live_model, load_registry, record


def manifest(version="v2", ndcg=0.31, recall=0.28, cold=0.158, coverage=0.46, ranker=True):
    rows = {
        "rerank_served": {"ndcg@20": ndcg, "recall@20": recall, "coverage@20": coverage},
        "rerank_cold5": {"ndcg@20": cold},
    }
    if not ranker:
        rows = {"hybrid_served": rows["rerank_served"], "hybrid_cold5": rows["rerank_cold5"]}
    return {"model_version": version, "created_at": "2026-09-25T00:00:00+00:00", "metrics": rows,
            "ranker": {"best_iteration": 186} if ranker else None}


@pytest.fixture
def live_registry(tmp_path):
    registry = {"production": None, "history": []}
    return record(registry, manifest("v1"), evaluate_candidate(registry, manifest("v1")), True, tmp_path / "r.json")


def test_first_model_is_promoted(tmp_path):
    decision = evaluate_candidate({"production": None, "history": []}, manifest())
    assert decision.promote and "no model in production" in decision.reasons[0]


def test_equal_or_better_model_is_promoted(live_registry):
    assert evaluate_candidate(live_registry, manifest("v2", ndcg=0.33)).promote
    assert evaluate_candidate(live_registry, manifest("v2")).promote  # identical metrics


def test_regression_is_rejected(live_registry):
    decision = evaluate_candidate(live_registry, manifest("v2", ndcg=0.25))
    assert not decision.promote
    assert "ndcg@20 fell" in decision.reasons[0]
    assert "REJECT" in decision.summary()


def test_small_wobble_is_tolerated(live_registry):
    """Negative sampling makes runs differ slightly; 1% down is not a regression."""
    assert evaluate_candidate(live_registry, manifest("v2", ndcg=0.31 * 0.99)).promote


def test_coverage_collapse_is_rejected(live_registry):
    """Guards the popularity-drift failure: accuracy up, catalog coverage down."""
    decision = evaluate_candidate(live_registry, manifest("v2", ndcg=0.35, coverage=0.30))
    assert not decision.promote and "coverage@20" in decision.reasons[0]


def test_cold_start_regression_is_rejected(live_registry):
    assert not evaluate_candidate(live_registry, manifest("v2", cold=0.10)).promote


def test_stage1_only_model_compares_against_stage1_rows(live_registry):
    """A run without a ranker is still comparable (hybrid_served stands in for rerank_served)."""
    assert evaluate_candidate(live_registry, manifest("v2", ranker=False)).promote


def test_rejected_model_does_not_become_production(tmp_path, live_registry):
    path = tmp_path / "registry.json"
    decision = evaluate_candidate(live_registry, manifest("v2", ndcg=0.10))
    registry = record(live_registry, manifest("v2", ndcg=0.10), decision, promoted=False, path=path)
    assert registry["production"] == "v1"
    assert live_model(registry)["model_version"] == "v1"
    assert [h["promoted"] for h in registry["history"]] == [True, False]
    assert json.loads(path.read_text(encoding="utf-8"))["production"] == "v1"
    assert load_registry(path)["production"] == "v1"


def test_gate_prefers_the_like_for_like_row(live_registry):
    """Adding unrated books costs measured accuracy, so the gate reads rerank_rated_only."""
    candidate = manifest("v2", ndcg=0.25)  # served row drops (new books took slots)
    candidate["metrics"]["rerank_rated_only"] = {"ndcg@20": 0.32, "recall@20": 0.29, "coverage@20": 0.47}
    assert evaluate_candidate(live_registry, candidate).promote

    candidate["metrics"]["rerank_rated_only"]["ndcg@20"] = 0.20  # the model itself got worse
    assert not evaluate_candidate(live_registry, candidate).promote
