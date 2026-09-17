"""Top-K ranking metrics for offline evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items retrieved, normalised by min(|relevant|, k)."""
    if not relevant:
        return float("nan")
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    return float(bool(set(ranked[:k]) & relevant))


def evaluate_rankings(
    rankings: Mapping[int, Sequence[int]],
    test_items: Mapping[int, set[int]],
    n_items: int,
    ks: Sequence[int] = (10, 20),
) -> dict[str, float]:
    """Average ranking metrics over users plus catalog coverage.

    Coverage (share of the catalog that appears in anyone's top-K) guards against a
    model that scores well by recommending the same bestsellers to everybody.
    """
    results: dict[str, float] = {}
    users = [u for u in rankings if test_items.get(u)]
    for k in ks:
        results[f"recall@{k}"] = float(np.mean([recall_at_k(rankings[u], test_items[u], k) for u in users]))
        results[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(rankings[u], test_items[u], k) for u in users]))
        results[f"hit_rate@{k}"] = float(np.mean([hit_rate_at_k(rankings[u], test_items[u], k) for u in users]))
        recommended = {item for u in users for item in rankings[u][:k]}
        results[f"coverage@{k}"] = len(recommended) / n_items
    results["n_users"] = float(len(users))
    return results
