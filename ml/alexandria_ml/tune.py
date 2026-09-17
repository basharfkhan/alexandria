"""Hyper-parameter search for the serving-time hybrid recommender.

    python -m alexandria_ml.tune                  # Goodbooks-10k (run the pipeline once first)

Protocol - the pipeline's test split is never touched:
    ratings --(seed 42)--> train | test
    train   --(seed 7)---> fit   | validation
BPR is trained on `fit`; every configuration is scored on `validation` users with the
exact serving code path. Objective = mean NDCG@20 of the full-history and 5-rating
(cold-start) scenarios, so a config can't win by only serving heavy users well.

Stage 1 tunes the fold-in (regularisation, confidence, global Gram term, item bias);
stage 2 tunes the blend weights on top of the best stage-1 setting.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import time

import pandas as pd
import torch

from alexandria_core import HybridRecommender
from alexandria_core.recommender import Weights
from alexandria_ml.config import DATA_DIR, ML_ROOT, POSITIVE_RATING, PROCESSED_DIR
from alexandria_ml.data.preprocess import book_text, load_dataset, train_test_split_by_user
from alexandria_ml.evaluate import build_context, catalog_arrays, hybrid_metrics
from alexandria_ml.features.embeddings import cached_embeddings
from alexandria_ml.models.bpr import BPRMF, BPRConfig, train_bpr
from alexandria_ml.tracking import Tracker

log = logging.getLogger("alexandria.tune")

STAGE1 = {
    "foldin_global_gram": [True, False],
    "foldin_reg": [0.1, 1.0, 10.0],
    "foldin_alpha": [1.0, 5.0, 20.0],
    "cf_bias": [0.0, 1.0],
}
STAGE2 = {
    "cf_max": [0.6, 0.8, 0.95],
    "cf_half_life": [2.0, 5.0],
    "popularity": [0.0, 0.15, 0.3, 0.5, 0.8],
    "quality": [0.0, 0.1],
}
WEIGHT_FIELDS = {f.name for f in dataclasses.fields(Weights)}


def load_or_train_bpr(
    fit: pd.DataFrame, n_users: int, n_items: int, cfg: BPRConfig, dataset: str
) -> BPRMF:
    path = DATA_DIR / "cache" / f"bpr_fit_{dataset}_d{cfg.dim}_e{cfg.epochs}_s{cfg.seed}.pt"
    model = BPRMF(n_users, n_items, cfg.dim)
    if path.exists():
        model.load_state_dict(torch.load(path))
        log.info("loaded cached BPR model from %s", path)
        return model.eval()
    pos = fit[fit.rating >= POSITIVE_RATING]
    model = train_bpr(pos.user_idx.to_numpy(), pos.item_idx.to_numpy(), n_users, n_items, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def build(catalog, params: dict) -> HybridRecommender:
    weights = Weights(**{k: v for k, v in params.items() if k in WEIGHT_FIELDS})
    fold = {k: v for k, v in params.items() if k.startswith("foldin_")}
    return HybridRecommender(catalog, weights=weights, **fold)


def grid(space: dict) -> list[dict]:
    return [dict(zip(space, values, strict=True)) for values in itertools.product(*space.values())]


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="goodbooks")
    p.add_argument("--max-users", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    processed = PROCESSED_DIR / args.dataset
    ds = load_dataset(processed)
    content, method = cached_embeddings(ds.books.apply(book_text, axis=1).tolist(), processed)

    train, _test = train_test_split_by_user(ds.ratings, seed=42)
    fit, val = train_test_split_by_user(train, seed=7)
    cfg = BPRConfig(epochs=args.epochs)
    model = load_or_train_bpr(fit, ds.n_users, ds.n_items, cfg, args.dataset)
    factors, bias = model.item_factors()
    catalog = catalog_arrays(ds.books, content, factors, bias)
    ctx = build_context(fit, val, ds.n_items, max_users=args.max_users, seed=7)

    tracker = Tracker(experiment="alexandria-tuning", run_name=f"tune-{args.dataset}")
    tracker.log_params({"dataset": args.dataset, "embedder": method, "max_users": args.max_users, **cfg.to_dict()})

    rows: list[dict] = []

    def evaluate(params: dict, stage: int) -> float:
        t0 = time.time()
        hybrid = build(catalog, params)
        full = hybrid_metrics(hybrid, ctx, "full")
        cold = hybrid_metrics(hybrid, ctx, "cold5", n_ratings=5)
        objective = (full["ndcg@20"] + cold["ndcg@20"]) / 2
        rows.append({
            "stage": stage, **params, "objective": objective,
            "full_ndcg@20": full["ndcg@20"], "full_recall@20": full["recall@20"], "full_coverage@20": full["coverage@20"],
            "cold5_ndcg@20": cold["ndcg@20"], "cold5_recall@20": cold["recall@20"], "cold5_coverage@20": cold["coverage@20"],
        })
        tracker.log_metrics({"objective": objective}, step=len(rows))
        log.info("[stage %d | %3d] objective=%.4f  %s  (%.0fs)", stage, len(rows), objective, params, time.time() - t0)
        return objective

    stage1 = grid(STAGE1)
    best1 = max(stage1, key=lambda prm: evaluate(prm, 1))
    stage2 = [{**best1, **prm} for prm in grid(STAGE2)]
    best = max(stage2, key=lambda prm: evaluate(prm, 2))

    results = pd.DataFrame(rows).sort_values("objective", ascending=False)
    out = ML_ROOT / "tuning"
    out.mkdir(exist_ok=True)
    results.to_csv(out / f"results_{args.dataset}.csv", index=False)
    (out / f"best_{args.dataset}.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    tracker.log_params({f"best.{k}": v for k, v in best.items()})
    tracker.log_metrics({k: float(v) for k, v in results.iloc[0].items() if k.startswith(("full_", "cold5_", "objective"))})
    tracker.finish(out)

    log.info("best config: %s", best)
    print(results.head(10).to_string(index=False))
    return best


if __name__ == "__main__":
    main()
