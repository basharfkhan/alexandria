"""Bayesian Personalized Ranking matrix factorisation in PyTorch.

BPR (Rendle et al., 2009) learns user/item embeddings from implicit feedback by
optimising pairwise ranking: an item the user liked should score higher than a
randomly sampled item they did not interact with.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class BPRConfig:
    dim: int = 64
    epochs: int = 20
    batch_size: int = 8192
    lr: float = 5e-3
    reg: float = 1e-5
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(-1) + self.item_bias(items).squeeze(-1)

    @torch.no_grad()
    def all_scores(self, users: torch.Tensor) -> torch.Tensor:
        return self.user(users) @ self.item.weight.T + self.item_bias.weight.T

    def item_factors(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.item.weight.detach().cpu().numpy().astype(np.float32),
            self.item_bias.weight.detach().cpu().numpy().ravel().astype(np.float32),
        )


def cached_bpr(
    fit, n_users: int, n_items: int, cfg: BPRConfig, dataset: str, cache_dir: Path, on_epoch=None
) -> BPRMF:
    """Train (or reuse) the BPR model for a fit split - shared by tuning and ranker training."""
    path = cache_dir / f"bpr_fit_{dataset}_d{cfg.dim}_e{cfg.epochs}_s{cfg.seed}.pt"
    if path.exists():
        model = BPRMF(n_users, n_items, cfg.dim)
        model.load_state_dict(torch.load(path))
        log.info("loaded cached BPR model from %s", path)
        return model.eval()

    from alexandria_ml.config import POSITIVE_RATING

    pos = fit[fit.rating >= POSITIVE_RATING]
    model = train_bpr(pos.user_idx.to_numpy(), pos.item_idx.to_numpy(), n_users, n_items, cfg, on_epoch=on_epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def train_bpr(
    users: np.ndarray, items: np.ndarray, n_users: int, n_items: int, cfg: BPRConfig, on_epoch=None
) -> BPRMF:
    """Train on positive (user, item) pairs with uniform negative sampling."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BPRMF(n_users, n_items, cfg.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    u_all = torch.tensor(users, dtype=torch.long)
    i_all = torch.tensor(items, dtype=torch.long)
    n = len(users)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.as_tensor(rng.permutation(n))
        neg_all = torch.as_tensor(rng.integers(0, n_items, size=n), dtype=torch.long)
        total = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            u, i, j = u_all[idx].to(device), i_all[idx].to(device), neg_all[idx].to(device)

            diff = model(u, i) - model(u, j)
            loss = -nn.functional.logsigmoid(diff).mean()
            reg = cfg.reg * (
                model.user(u).pow(2).sum() + model.item(i).pow(2).sum() + model.item(j).pow(2).sum()
            ) / len(idx)

            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
            total += loss.item() * len(idx)

        avg = total / n
        log.info("epoch %2d/%d  bpr_loss=%.4f", epoch, cfg.epochs, avg)
        if on_epoch:
            on_epoch(epoch, avg)
    return model.cpu().eval()
