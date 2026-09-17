"""Online user "fold-in" for latent-factor models.

Training a matrix-factorization model learns an embedding for every *training* user.
A brand-new app user has no embedding, and retraining the model every time someone
clicks "loved it" is not practical. Fold-in solves for that user's vector on the fly,
holding the item factors fixed, using the implicit-feedback ALS objective
(Hu, Koren & Volinsky, 2008):

    min_u  sum_i c_i (p_i - u . y_i)^2  +  reg * ||u||^2

where p_i = 1 for positive feedback and c_i is the confidence. Items the user never
touched get p=0 and c=1, which is folded into the precomputed Gram matrix
Y^T Y, so each solve costs O(n_feedback * f^2 + f^3) - sub-millisecond for f=64.

Explicit dislikes are an extension to the original (implicit-only) formulation: they get
a *negative* target (p = -negative_target) instead of 0. With p = 0 a disliked book that
already scored below zero would be pulled back *up* toward zero - the opposite of intent.
"""

from __future__ import annotations

import numpy as np


class ImplicitFoldIn:
    def __init__(
        self, item_factors: np.ndarray, reg: float = 0.1, alpha: float = 20.0, negative_target: float = 1.0
    ):
        self.Y = np.asarray(item_factors, dtype=np.float64)
        self.reg = reg
        self.alpha = alpha
        self.negative_target = negative_target
        self.YtY = self.Y.T @ self.Y
        self._eye = np.eye(self.Y.shape[1])

    @property
    def dim(self) -> int:
        return self.Y.shape[1]

    def user_vector(self, item_idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Solve for a user vector from (item index, signed feedback weight) pairs.

        Positive weights pull the user toward an item; negative weights push them away.
        """
        item_idx = np.asarray(item_idx, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        if item_idx.size == 0:
            return np.zeros(self.dim)

        Yi = self.Y[item_idx]
        conf = 1.0 + self.alpha * np.abs(weights)
        pref = np.where(weights > 0, 1.0, -self.negative_target)

        A = self.YtY + (Yi.T * (conf - 1.0)) @ Yi + self.reg * self._eye
        b = Yi.T @ (conf * pref)
        return np.linalg.solve(A, b)
