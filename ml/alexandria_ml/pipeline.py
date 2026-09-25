"""End-to-end training pipeline.

    python -m alexandria_ml.pipeline                 # real Goodbooks-10k data
    python -m alexandria_ml.pipeline --synthetic     # tiny fake dataset, runs in seconds

Steps: acquire data -> preprocess -> embed -> split -> train BPR -> evaluate
       -> refit on all data -> export artifacts (+ MLflow tracking).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd

from alexandria_core import HybridRecommender, Reranker
from alexandria_ml.cold_start import project_cold_start
from alexandria_ml.config import ARTIFACT_DIR, CF_DIM, DATA_DIR, POSITIVE_RATING, PROCESSED_DIR, RAW_DIR
from alexandria_ml.data.app_feedback import load_app_ratings
from alexandria_ml.data.download import download_goodbooks
from alexandria_ml.data.openlibrary import PACKAGED_CACHE
from alexandria_ml.data.preprocess import build_dataset, save_dataset, train_test_split_by_user
from alexandria_ml.data.recent_books import CACHE_PATH as RECENT_FETCHED
from alexandria_ml.data.recent_books import PACKAGED_CACHE as RECENT_CACHE
from alexandria_ml.data.synthetic import make_synthetic
from alexandria_ml.evaluate import catalog_arrays, evaluate_models
from alexandria_ml.export import export_artifacts
from alexandria_ml.features.embeddings import content_embeddings
from alexandria_ml.models.bpr import BPRConfig, cached_bpr
from alexandria_ml.ranker import RankerConfig, train_ranker
from alexandria_ml.tracking import Tracker

log = logging.getLogger("alexandria.pipeline")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="use a generated dataset instead of Goodbooks-10k")
    p.add_argument("--embedder", choices=["auto", "sbert", "tfidf"], default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--dim", type=int, default=CF_DIM, help="latent factors (must match the API's CF_DIM)")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--reg", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-eval-users", type=int, default=2000)
    p.add_argument("--no-final-fit", action="store_true", help="export the train-split model instead of refitting")
    p.add_argument("--no-ranker", action="store_true", help="skip the second-stage learning-to-rank model")
    p.add_argument("--no-recent-books", action="store_true",
                   help="catalog of rated books only (skip post-2017 titles from Open Library)")
    p.add_argument("--app-feedback", action="store_true",
                   help="also train on ratings collected by the live app (needs DATABASE_URL)")
    p.add_argument("--ranker-users", type=int, default=RankerConfig.max_users)
    p.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()

    tracker = Tracker(run_name="synthetic" if args.synthetic else "goodbooks-10k")
    cfg = BPRConfig(dim=args.dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg=args.reg)
    tracker.log_params({**cfg.to_dict(), "dataset": "synthetic" if args.synthetic else "goodbooks-10k"})

    # 1. Data
    dataset_name = "synthetic" if args.synthetic else "goodbooks"
    raw_dir = make_synthetic(DATA_DIR / "synthetic_raw") if args.synthetic else download_goodbooks(RAW_DIR)
    enrichment = None
    if not args.synthetic:
        fetched = DATA_DIR / "openlibrary" / "works.jsonl"
        enrichment = fetched if fetched.exists() else PACKAGED_CACHE
        if not enrichment.exists():
            log.warning("no Open Library enrichment - run `python -m alexandria_ml.data.openlibrary` for descriptions")
    recent = None
    if not args.synthetic and not args.no_recent_books:
        recent = RECENT_FETCHED if RECENT_FETCHED.exists() else RECENT_CACHE
    ds = build_dataset(raw_dir, enrichment_path=enrichment, recent_path=recent)
    processed = PROCESSED_DIR / dataset_name
    save_dataset(ds, processed)

    # 1b. Ratings collected by the live app become extra users (closing the feedback loop).
    if args.app_feedback:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            log.warning("--app-feedback given but DATABASE_URL is unset; training on Goodbooks only")
        else:
            app_ratings = load_app_ratings(database_url, ds.books, first_user_idx=ds.n_users)
            if len(app_ratings):
                ds.ratings = pd.concat([ds.ratings, app_ratings], ignore_index=True)
                tracker.log_params({"app_ratings": len(app_ratings), "app_users": app_ratings.user_idx.nunique()})

    # 2. Content embeddings
    content, method = content_embeddings(ds.books, processed, args.embedder)
    tracker.log_params({"embedder": method})
    log.info("embedded %d books with %s -> %s", len(ds.books), method, content.shape)

    # 3. Split + train
    train, test = train_test_split_by_user(ds.ratings)
    pos = train[train.rating >= POSITIVE_RATING]
    log.info("train ratings=%d (positives=%d)  test ratings=%d", len(train), len(pos), len(test))
    model = cached_bpr(
        train, ds.n_users, ds.n_items, cfg, f"{dataset_name}_train", DATA_DIR / "cache",
        on_epoch=lambda e, loss: tracker.log_metrics({"train_bpr_loss": loss}, step=e),
    )

    # Books with no ratings never appear in training, so their latent vectors stay at their random
    # initialisation: project one from their nearest rated neighbours before anything scores with them.
    has_ratings = ds.books.has_ratings.to_numpy()

    def item_factors(trained):
        factors, bias = trained.item_factors()
        return project_cold_start(content, factors, bias, has_ratings) if not has_ratings.all() else (factors, bias)

    # 4. Second-stage ranker: trained on a fit/validation split *inside* the training data, so the
    #    test split stays untouched and the ranker never sees the labels it is evaluated on.
    reranker, ranker_info = None, None
    if not args.no_ranker:
        fit, val = train_test_split_by_user(train, seed=7)
        fit_model = cached_bpr(fit, ds.n_users, ds.n_items, cfg, dataset_name, DATA_DIR / "cache")
        fit_factors, fit_bias = item_factors(fit_model)
        fit_hybrid = HybridRecommender(catalog_arrays(ds.books, content, fit_factors, fit_bias))
        booster, ranker_info = train_ranker(
            fit_hybrid, fit, val, RankerConfig(max_users=args.ranker_users, seed=cfg.seed)
        )
        reranker = Reranker(booster, booster.feature_name())
        tracker.log_params({f"ranker.{k}": v for k, v in ranker_info.items() if k != "feature_importance"})
        tracker.log_metrics({"ranker_holdout_ndcg@20": ranker_info["holdout_ndcg@20"]})

    # 5. Evaluate
    results = evaluate_models(
        ds.books, train, test, content, model, max_users=args.max_eval_users,
        reranker=reranker, item_factors=item_factors,
    )
    for name, metrics in results.items():
        tracker.log_metrics(metrics, prefix=f"{name}.")

    # 6. Refit on all positives for serving
    if not args.no_final_fit:
        log.info("refitting on all data for export")
        model = cached_bpr(ds.ratings, ds.n_users, ds.n_items, cfg, f"{dataset_name}_all", DATA_DIR / "cache")

    factors, bias = item_factors(model)
    out = export_artifacts(
        Path(args.artifact_dir),
        ds.books, content, factors, bias,
        manifest={
            "model_version": time.strftime("%Y%m%d-%H%M%S"),
            "dataset": "synthetic" if args.synthetic else "goodbooks-10k",
            "embedder": method,
            "bpr": cfg.to_dict(),
            "ranker": ranker_info,
            "metrics": results,
        },
        ranker=booster if reranker is not None else None,
    )
    tracker.log_artifacts(out)
    tracker.finish(out)
    log.info("done in %.1fs - artifacts in %s", time.time() - t0, out)
    return results


if __name__ == "__main__":
    main()
