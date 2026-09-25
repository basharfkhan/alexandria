"""Hybrid recommender: content similarity + collaborative filtering + priors.

Scoring for a user with feedback F and chosen genres G:

    score = w_content * z(content) + w_cf * z(cf) + w_genre * genre_match
            + w_pop * z(log popularity) + w_quality * z(avg rating)

* content   - cosine between the item's text embedding and the user's taste profile
              (weighted mean of liked items minus disliked items, plus genre anchors).
* cf        - dot product with a user vector folded in from feedback (see foldin.py).
* w_cf grows with the amount of explicit feedback, so cold-start users lean on content
  and genre signals, and the latent-factor model takes over as they rate more books.

The top candidates are then diversified with MMR, and a few "explore" slots from
further down the ranking are mixed in so the profile does not collapse into a bubble.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from alexandria_core.foldin import ImplicitFoldIn
from alexandria_core.ranking import cf_weight, mmr_rerank, zscore

if TYPE_CHECKING:
    from alexandria_core.rerank import Reranker

FEEDBACK_WEIGHTS: dict[str, float] = {
    "loved": 2.0,
    "liked": 1.0,
    "want_to_read": 0.5,
    "disliked": -1.0,
    "not_interested": -0.5,
}

GENRE_ANCHOR_SIZE = 50
RERANK_POOL = 200

_SERIES = re.compile(r"\(([^()#]*?),?\s*#\s*(\d+(?:\.\d+)?)[^()]*\)\s*$")


def parse_series(title: str) -> tuple[str | None, float]:
    """Goodreads titles encode series as "The Well of Ascension (Mistborn, #2)"."""
    m = _SERIES.search(title or "")
    if not m:
        return None, float("nan")
    name = re.sub(r"^the\s+", "", m.group(1).strip().lower())
    return (name or None), float(m.group(2))


@dataclass
class CatalogArrays:
    """Column-oriented view of the catalog; row i is the same book in every array."""

    content: np.ndarray  # (n, d) text embeddings
    popularity: np.ndarray  # (n,) number of ratings
    avg_rating: np.ndarray  # (n,)
    genres: Sequence[Sequence[str]]  # genre slugs per book
    cf_factors: np.ndarray | None = None  # (n, f)
    cf_bias: np.ndarray | None = None  # (n,)
    authors: Sequence[str] | None = None  # used to cap how many picks one author gets
    titles: Sequence[str] | None = None  # used to detect series ("Title (Series, #2)")
    cold_start: Sequence[bool] | None = None  # books with no ratings (post-2017 titles)


@dataclass
class Recommendation:
    index: int
    score: float
    reason: str  # "similar" | "collaborative" | "genre" | "popular" | "explore" | "new_release"
    because_of: int | None = None  # catalog index of the book that explains it
    genre: str | None = None
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class Blend:
    raw: dict  # per-signal raw scores from HybridRecommender.score()
    parts: dict[str, np.ndarray]  # weighted, standardised signal contributions
    total: np.ndarray  # first-stage score per catalog item


@dataclass
class Weights:
    """Blend weights. Defaults were tuned on a Goodbooks-10k validation split (see ml/alexandria_ml/tune.py).

    `popularity` trades accuracy against catalog coverage: 0.5 scored ~3% higher NDCG@20 than 0.3
    but recommended from 27% of the catalog instead of 39%, so 0.3 was chosen.
    """

    genre: float = 0.6
    popularity: float = 0.3
    quality: float = 0.1
    negative_content: float = 0.5
    genre_anchor: float = 0.5
    cf_max: float = 0.8
    cf_half_life: float = 2.0
    # Scale on BPR's learned item bias. It is essentially a popularity score; at 1.0 it swamped the
    # personal signal and collapsed coverage to ~2%, so popularity enters only via `popularity`.
    cf_bias: float = 0.0


class HybridRecommender:
    def __init__(
        self,
        catalog: CatalogArrays,
        weights: Weights | None = None,
        foldin_reg: float = 1.0,
        foldin_alpha: float = 5.0,
        foldin_global_gram: bool = True,
    ):
        self.w = weights or Weights()
        self.n_items = len(catalog.popularity)

        content = np.asarray(catalog.content, dtype=np.float32)
        norms = np.linalg.norm(content, axis=1, keepdims=True)
        self.content = content / np.maximum(norms, 1e-9)

        self.pop_z = zscore(np.log1p(np.asarray(catalog.popularity, dtype=np.float64)))
        self.quality_z = zscore(np.asarray(catalog.avg_rating, dtype=np.float64))
        self.item_genres = [list(g) for g in catalog.genres]
        # Primary author only: "Stephen King, Owen King" and "Stephen King" count as the same author.
        self.item_author = (
            [a.split(",")[0].strip().lower() for a in catalog.authors] if catalog.authors is not None else None
        )

        self.is_cold_start = (
            np.asarray(catalog.cold_start, dtype=bool) if catalog.cold_start is not None
            else np.zeros(self.n_items, dtype=bool)
        )

        series = [parse_series(t) for t in catalog.titles] if catalog.titles is not None else []
        self.item_series = [s for s, _ in series] if series else [None] * self.n_items
        self.item_series_no = np.array([n for _, n in series]) if series else np.full(self.n_items, np.nan)

        self.has_cf = catalog.cf_factors is not None
        self.cf_bias = np.zeros(self.n_items)
        if self.has_cf:
            self.cf_factors = np.asarray(catalog.cf_factors, dtype=np.float64)
            self.cf_bias = (
                np.asarray(catalog.cf_bias, dtype=np.float64)
                if catalog.cf_bias is not None
                else np.zeros(self.n_items)
            )
            self.foldin = ImplicitFoldIn(
                self.cf_factors, reg=foldin_reg, alpha=foldin_alpha, global_gram=foldin_global_gram
            )

        self._build_genre_index()

    # ------------------------------------------------------------------ genres
    def _build_genre_index(self) -> None:
        members: dict[str, list[int]] = {}
        for idx, genres in enumerate(self.item_genres):
            for g in genres:
                members.setdefault(g, []).append(idx)

        self.genre_members = {g: np.array(ix) for g, ix in members.items()}
        self.genre_anchors: dict[str, np.ndarray] = {}
        for g, ix in self.genre_members.items():
            top = ix[np.argsort(-self.pop_z[ix])[:GENRE_ANCHOR_SIZE]]
            anchor = self.content[top].mean(axis=0)
            self.genre_anchors[g] = anchor / max(np.linalg.norm(anchor), 1e-9)

    @property
    def genres(self) -> list[str]:
        return sorted(self.genre_members, key=lambda g: -len(self.genre_members[g]))

    # ----------------------------------------------------------------- scoring
    def score(
        self, feedback: Mapping[int, float], genres: Iterable[str] = ()
    ) -> dict[str, np.ndarray | float]:
        """Return per-signal score arrays over the whole catalog."""
        genres = [g for g in genres if g in self.genre_anchors]
        idx = np.fromiter(feedback.keys(), dtype=np.int64, count=len(feedback))
        wts = np.fromiter(feedback.values(), dtype=np.float64, count=len(feedback))

        # Content profile
        profile = np.zeros(self.content.shape[1], dtype=np.float64)
        pos, neg = wts > 0, wts < 0
        if pos.any():
            profile += (self.content[idx[pos]] * wts[pos, None]).sum(axis=0) / wts[pos].sum()
        if neg.any():
            profile -= self.w.negative_content * (
                (self.content[idx[neg]] * -wts[neg, None]).sum(axis=0) / -wts[neg].sum()
            )
        if genres:
            anchor = np.mean([self.genre_anchors[g] for g in genres], axis=0)
            scale = self.w.genre_anchor if pos.any() else 1.0
            profile += scale * anchor

        content = self.content @ profile.astype(np.float32) if profile.any() else np.zeros(self.n_items)

        # Collaborative signal
        n_explicit = int((np.abs(wts) >= 1.0).sum())
        w_cf = 0.0
        cf = np.zeros(self.n_items)
        if self.has_cf and pos.any():
            w_cf = cf_weight(n_explicit, self.w.cf_max, self.w.cf_half_life)
            user_vec = self.foldin.user_vector(idx, wts)
            cf = self.cf_factors @ user_vec + self.w.cf_bias * self.cf_bias

        genre_match = np.zeros(self.n_items)
        for g in genres:
            genre_match[self.genre_members[g]] = 1.0

        return {
            "content": content,
            "cf": cf,
            "genre": genre_match,
            "w_cf": w_cf,
            "n_explicit": float(n_explicit),
        }

    def blend(self, feedback: Mapping[int, float], genres: Sequence[str]) -> Blend:
        """First-stage scores: the weighted hybrid of every signal over the whole catalog."""
        s = self.score(feedback, genres)
        w_cf = float(s["w_cf"])
        has_taste = bool(feedback) or bool(genres)
        parts = {
            "content": (1.0 - w_cf) * zscore(s["content"]) if has_taste else np.zeros(self.n_items),
            "cf": w_cf * zscore(s["cf"]) if w_cf > 0 else np.zeros(self.n_items),
            "genre": self.w.genre * s["genre"],
            "popularity": self.w.popularity * self.pop_z * (1.0 if has_taste else 4.0),
            "quality": self.w.quality * self.quality_z,
        }
        return Blend(raw=s, parts=parts, total=sum(parts.values()))

    def candidates(
        self,
        blend: Blend,
        feedback: Mapping[int, float],
        exclude: Iterable[int],
        size: int,
        cold_start: bool | None = False,
    ) -> np.ndarray:
        """Top ``size`` unseen books by first-stage score, best first.

        ``cold_start`` selects the catalog half: False = books with ratings (the main pool),
        True = never-rated books only (the new-release slots), None = both.
        """
        blocked = np.zeros(self.n_items, dtype=bool)
        if cold_start is not None:
            blocked |= self.is_cold_start if not cold_start else ~self.is_cold_start
        for i in list(exclude) + list(feedback.keys()):
            blocked[i] = True
        total = np.where(blocked, -np.inf, blend.total)
        size = min(size, int((~blocked).sum()))
        if size <= 0:
            return np.array([], dtype=np.int64)
        pool = np.argpartition(-total, size - 1)[:size]
        return pool[np.argsort(-total[pool])]

    def recommend(
        self,
        feedback: Mapping[int, float],
        genres: Iterable[str] = (),
        k: int = 20,
        exclude: Iterable[int] = (),
        diversity: float = 0.25,
        explore_slots: int = 0,
        seed: int | None = None,
        max_per_author: int | None = None,
        new_book_slots: int = 0,
        reranker: Reranker | None = None,
    ) -> list[Recommendation]:
        genres = [g for g in genres if g in self.genre_anchors]
        blend = self.blend(feedback, genres)
        parts = blend.parts
        # Never-rated books (post-2017 titles) only reach the catalog on collaborative vectors
        # borrowed from similar rated books, so they get reserved slots instead of competing for
        # every one: ranking quality for the rest of the page is then unaffected by catalog size.
        pool = self.candidates(blend, feedback, exclude, max(RERANK_POOL, k * 10))
        k = min(k, len(pool) + new_book_slots)
        if k == 0:
            return []
        new_book_slots = min(new_book_slots, max(k - 1, 0)) if self.is_cold_start.any() else 0

        # Second stage: a learned ranker reorders the candidate pool. It is trained on readers with
        # at least one liked book, so genre-only cold starts keep the first-stage order.
        total = blend.total.copy()
        if reranker is not None and any(w > 0 for w in feedback.values()):
            rank_scores = reranker.score(self, blend, feedback, genres, pool)
            total[pool] = rank_scores
            pool = pool[np.argsort(-rank_scores, kind="stable")]

        n_explore = min(explore_slots, max(k - new_book_slots - 1, 0))
        n_main = k - n_explore - new_book_slots
        if diversity > 0:
            ranked = mmr_rerank(pool, total[pool], self.content, len(pool), lambda_=1.0 - diversity)
        else:
            ranked = pool
        main = self._select(ranked, n_main, max_per_author)

        explore: np.ndarray = np.array([], dtype=np.int64)
        if n_explore:
            rng = np.random.default_rng(seed)
            tail = np.setdiff1d(pool[n_main * 2 :], main)
            if len(tail):
                explore = rng.choice(tail, size=min(n_explore, len(tail)), replace=False)

        pos_items = np.array([i for i, w in feedback.items() if w > 0], dtype=np.int64)
        recs = [self._explain(int(i), float(total[i]), parts, pos_items, genres) for i in main]

        if new_book_slots:
            fresh_pool = self.candidates(blend, feedback, exclude, new_book_slots * 8, cold_start=True)
            if reranker is not None and len(fresh_pool) and any(w > 0 for w in feedback.values()):
                fresh_scores = reranker.score(self, blend, feedback, genres, fresh_pool)
                fresh_pool = fresh_pool[np.argsort(-fresh_scores, kind="stable")]
                total[fresh_pool] = np.sort(fresh_scores)[::-1]
            fresh = self._select(fresh_pool, new_book_slots, max_per_author)
            for j, i in enumerate(fresh):
                rec = self._explain(int(i), float(total[i]), parts, pos_items, genres)
                rec.reason = "new_release"
                recs.insert(min(len(recs), (j + 1) * 3), rec)

        # Interleave exploration picks evenly through the list.
        for j, i in enumerate(explore):
            rec = self._explain(int(i), float(total[i]), parts, pos_items, genres)
            rec.reason = "explore"
            position = min(len(recs), (j + 1) * max(1, len(recs) // (len(explore) + 1)))
            recs.insert(position, rec)
        return recs

    def _select(self, ranked: np.ndarray, n: int, max_per_author: int | None) -> np.ndarray:
        """Take the first ``n`` items, allowing at most ``max_per_author`` each (backfilling if short)."""
        if not max_per_author or self.item_author is None:
            return ranked[:n]
        counts: dict[str, int] = {}
        kept, overflow = [], []
        for i in ranked:
            author = self.item_author[i]
            if counts.get(author, 0) < max_per_author:
                counts[author] = counts.get(author, 0) + 1
                kept.append(i)
                if len(kept) == n:
                    break
            else:
                overflow.append(i)
        kept.extend(overflow[: n - len(kept)])
        return np.array(kept, dtype=np.int64)

    def _explain(
        self,
        i: int,
        score: float,
        parts: Mapping[str, np.ndarray],
        pos_items: np.ndarray,
        genres: Sequence[str],
    ) -> Recommendation:
        components = {name: float(arr[i]) for name, arr in parts.items()}
        rec = Recommendation(index=i, score=score, reason="popular", components=components)

        if len(pos_items):
            rec.reason = "collaborative" if components["cf"] > components["content"] else "similar"
            if rec.reason == "collaborative":
                # Explain with the liked book whose *latent factors* are closest (what CF actually used).
                vecs = self.cf_factors[pos_items]
                sims = (vecs @ self.cf_factors[i]) / np.maximum(np.linalg.norm(vecs, axis=1), 1e-9)
            else:
                sims = self.content[pos_items] @ self.content[i]
            rec.because_of = int(pos_items[int(np.argmax(sims))])
        elif genres:
            matched = [g for g in genres if g in self.item_genres[i]]
            if matched:
                rec.reason, rec.genre = "genre", matched[0]
            else:
                rec.reason = "similar"
        return rec

    def similar(self, i: int, k: int = 10) -> list[tuple[int, float]]:
        sims = self.content @ self.content[i]
        sims[i] = -np.inf
        k = min(k, self.n_items - 1)
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(j), float(sims[j])) for j in top]
