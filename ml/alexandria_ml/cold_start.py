"""Give never-rated books a collaborative vector borrowed from their nearest rated neighbours.

Books published after Goodbooks-10k (2017) have no ratings, so matrix factorisation learns
nothing for them: their latent vector stays at its random initialisation and they can never be
recommended for taste reasons. This is the classic *item cold start*.

The fix is a content-to-collaborative projection: a new book inherits the weighted average of the
latent vectors of the rated books whose text it most resembles. *Project Hail Mary* ends up close
to *The Martian*, *Artemis* and *Ready Player One*, so readers who loved those can be shown it.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

NEIGHBOURS = 10
# Borrowed evidence is weaker than learned evidence, so shrink it slightly: without this, a new
# book can outrank the very books it borrowed from.
SHRINKAGE = 0.85


def project_cold_start(
    content: np.ndarray,
    cf_factors: np.ndarray,
    cf_bias: np.ndarray,
    has_ratings: np.ndarray,
    neighbours: int = NEIGHBOURS,
    shrinkage: float = SHRINKAGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (factors, bias) with rows for never-rated books filled in from similar rated books."""
    cf_factors, cf_bias = cf_factors.copy(), cf_bias.copy()
    cold = np.flatnonzero(~has_ratings)
    warm = np.flatnonzero(has_ratings)
    if len(cold) == 0 or len(warm) == 0:
        return cf_factors, cf_bias

    normalized = content / np.maximum(np.linalg.norm(content, axis=1, keepdims=True), 1e-9)
    k = min(neighbours, len(warm))
    for start in range(0, len(cold), 512):  # chunked: the full similarity matrix would be large
        batch = cold[start : start + 512]
        sims = normalized[batch] @ normalized[warm].T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        weights = np.clip(np.take_along_axis(sims, top, axis=1), 0, None)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        neighbour_items = warm[top]
        cf_factors[batch] = shrinkage * np.einsum("nk,nkf->nf", weights, cf_factors[neighbour_items])
        cf_bias[batch] = shrinkage * (weights * cf_bias[neighbour_items]).sum(axis=1)
        del sims

    log.info("projected collaborative vectors for %d never-rated books from %d neighbours each",
             len(cold), k)
    return cf_factors, cf_bias
