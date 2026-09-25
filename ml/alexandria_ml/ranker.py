"""Second-stage learning-to-rank model (LightGBM LambdaMART).

Stage 1 (the hand-tuned hybrid blend) is good at *retrieval* but weak at ordering: on the
validation split 54% of a reader's held-out favourites land in its top 200, yet only 19% reach
the top 20. This module trains a ranker that reorders those 200 candidates using the features in
`alexandria_core.rerank` - signal agreement, similarity to individual liked books, author and
series continuity, and how much the reader has told us.

Data protocol (the pipeline's test split is never touched):

    ratings --(seed 42)--> train | test
    train   --(seed 7)---> fit   | validation

Histories come from `fit`, labels from `validation` (rating 5 -> 2, rating 4 -> 1). A share of
users is truncated to a handful of ratings so the model also learns the cold-start regime.
Users are split into train/early-stopping groups, so no user contributes to both.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from alexandria_core import FEATURE_NAMES, HybridRecommender, rerank_features
from alexandria_ml.config import POSITIVE_RATING
from alexandria_ml.evaluate import feedback_from_ratings

log = logging.getLogger(__name__)

LABEL_BY_RATING = {5: 2, 4: 1}


@dataclass
class RankerConfig:
    pool: int = 200
    max_users: int = 8000
    truncate_frac: float = 0.35  # share of users shown as newcomers (1-10 ratings)
    truncate_max: int = 10
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50
    early_stopping_rounds: int = 50
    seed: int = 42
    # Popularity-style features let the ranker rediscover "just recommend bestsellers"; see
    # docs/ARCHITECTURE.md. Empty tuple = use every feature.
    exclude_features: tuple[str, ...] = ()

    @property
    def feature_names(self) -> list[str]:
        return [f for f in FEATURE_NAMES if f not in self.exclude_features]

    def to_dict(self) -> dict:
        return {**asdict(self), "exclude_features": list(self.exclude_features)}


def _graded_labels(val: pd.DataFrame) -> dict[int, dict[int, int]]:
    positives = val[val.rating >= POSITIVE_RATING]
    labels: dict[int, dict[int, int]] = {}
    for user, item, rating in zip(positives.user_idx, positives.item_idx, positives.rating, strict=True):
        labels.setdefault(int(user), {})[int(item)] = LABEL_BY_RATING.get(int(rating), 1)
    return labels


def build_training_data(
    hybrid: HybridRecommender,
    fit: pd.DataFrame,
    val: pd.DataFrame,
    cfg: RankerConfig,
    users: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """Return (features, labels, group sizes, user ids) - one group per user."""
    rng = np.random.default_rng(cfg.seed)
    labels_by_user = _graded_labels(val)
    history = {int(u): g for u, g in fit[fit.user_idx.isin(labels_by_user)].groupby("user_idx")}

    if users is None:
        users = np.array(sorted(set(labels_by_user) & set(history)))
        if len(users) > cfg.max_users:
            users = rng.choice(users, size=cfg.max_users, replace=False)

    columns = [FEATURE_NAMES.index(f) for f in cfg.feature_names]
    xs, ys, groups, kept_users = [], [], [], []
    for user in users:
        user = int(user)
        ratings = history[user]
        if rng.random() < cfg.truncate_frac and len(ratings) > 1:
            n = int(rng.integers(1, min(cfg.truncate_max, len(ratings)) + 1))
            ratings = ratings.sample(n=n, random_state=user)
        feedback = feedback_from_ratings(ratings)
        if not any(w > 0 for w in feedback.values()):
            continue

        seen = set(history[user].item_idx.tolist())
        blend = hybrid.blend(feedback, [])
        pool = hybrid.candidates(blend, feedback, exclude=seen, size=cfg.pool)
        labels = np.array([labels_by_user[user].get(int(i), 0) for i in pool])
        if labels.sum() == 0:  # lambdarank needs at least one relevant item per group
            continue

        xs.append(rerank_features(hybrid, blend, feedback, [], pool)[:, columns])
        ys.append(labels)
        groups.append(len(pool))
        kept_users.append(user)

    log.info("ranker training data: %d users, %d rows, %.1f%% positive",
             len(groups), sum(groups), 100 * np.concatenate(ys).astype(bool).mean())
    return np.vstack(xs), np.concatenate(ys), groups, np.array(kept_users)


def train_ranker(
    hybrid: HybridRecommender,
    fit: pd.DataFrame,
    val: pd.DataFrame,
    cfg: RankerConfig | None = None,
    holdout_frac: float = 0.15,
    users: np.ndarray | None = None,
):
    """Train the LambdaMART ranker; returns (booster, info dict)."""
    import lightgbm as lgb

    cfg = cfg or RankerConfig()
    x, y, groups, users = build_training_data(hybrid, fit, val, cfg, users)

    rng = np.random.default_rng(cfg.seed)
    is_holdout = rng.random(len(groups)) < holdout_frac
    row_mask = np.concatenate([
        np.full(size, flag) for size, flag in zip(groups, is_holdout, strict=True)
    ])
    train_groups = [g for g, flag in zip(groups, is_holdout, strict=True) if not flag]
    eval_groups = [g for g, flag in zip(groups, is_holdout, strict=True) if flag]
    log.info("ranker: %d training groups, %d early-stopping groups", len(train_groups), len(eval_groups))

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=cfg.seed,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(
        x[~row_mask], y[~row_mask], group=train_groups, feature_name=cfg.feature_names,
        eval_X=x[row_mask], eval_y=y[row_mask], eval_group=[eval_groups], eval_at=[20],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(100)],
    )

    booster = ranker.booster_
    importance = sorted(
        zip(cfg.feature_names, ranker.feature_importances_.tolist(), strict=True), key=lambda kv: -kv[1]
    )
    info = {
        **cfg.to_dict(),
        "n_users": len(users),
        "best_iteration": int(ranker.best_iteration_ or cfg.n_estimators),
        "holdout_ndcg@20": float(ranker.best_score_["valid_0"]["ndcg@20"]),
        "feature_importance": dict(importance),
    }
    log.info("ranker trained: %d trees, holdout NDCG@20=%.4f", info["best_iteration"], info["holdout_ndcg@20"])
    log.info("top features: %s", ", ".join(f"{name}={score}" for name, score in importance[:8]))
    return booster, info
