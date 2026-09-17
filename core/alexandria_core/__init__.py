"""Alexandria core: the recommendation logic shared by offline evaluation and online serving.

Keeping this package numpy-only means the exact code that is benchmarked offline
is the code that answers API requests - no train/serve skew.
"""

from alexandria_core.foldin import ImplicitFoldIn
from alexandria_core.recommender import (
    FEEDBACK_WEIGHTS,
    CatalogArrays,
    HybridRecommender,
    Recommendation,
)

__all__ = [
    "FEEDBACK_WEIGHTS",
    "CatalogArrays",
    "HybridRecommender",
    "ImplicitFoldIn",
    "Recommendation",
]
