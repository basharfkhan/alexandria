"""Replay historical readers through a running Alexandria API.

    python -m alexandria_ml.simulate_traffic --api http://localhost:8000 --readers 20

The app has no production traffic yet, so the feedback loop (app ratings -> retraining) has
nothing to learn from. This script gives it something honest: each simulated reader is a real
Goodbooks user, onboarded with a few books they actually loved, who then answers recommendations
the way that user rated those books historically (5★ -> loved, 4★ -> liked, 1-2★ -> disliked,
unrated -> ignored).

It also reports an online-style metric offline evaluation cannot: what share of recommended books
the reader had an opinion on, and how many of those were positive.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import secrets
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from alexandria_ml.config import PROCESSED_DIR
from alexandria_ml.data.preprocess import load_dataset

log = logging.getLogger("alexandria.simulate")

SIGNAL_BY_RATING = {5: "loved", 4: "liked", 2: "disliked", 1: "disliked"}


def request(api: str, path: str, method: str = "GET", token: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{api}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
            return json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> {exc.code}: {exc.read()[:200]!r}") from exc


def simulate_reader(api: str, books, ratings_by_item: dict[int, int], genres: list[str], rounds: int, rng) -> dict:
    """Create one reader, onboard them, then answer `rounds` pages of recommendations."""
    username = f"sim_{secrets.token_hex(4)}"
    token = request(api, "/auth/register", "POST",
                    body={"username": username, "password": secrets.token_urlsafe(12)})["access_token"]

    loved = [item for item, rating in ratings_by_item.items() if rating == 5]
    rng.shuffle(loved)
    seed_items = loved[:3]
    request(api, "/me/onboarding", "POST", token=token, body={
        "genres": genres,
        "loved_book_ids": [int(books.book_id.iloc[i]) for i in seed_items],
        "source": "onboarding",
    })

    shown = rated = positive = 0
    for _ in range(rounds):
        recs = request(api, "/me/recommendations?limit=20", token=token)["items"]
        if not recs:
            break
        item_of = dict(zip(books.book_id, books.item_idx, strict=True))
        for position, rec in enumerate(recs):
            shown += 1
            item = item_of.get(rec["book"]["id"])
            rating = ratings_by_item.get(item)
            signal = SIGNAL_BY_RATING.get(rating) if rating is not None else None
            if signal is None:  # this reader never rated that book - no opinion to give
                continue
            request(api, f"/me/books/{rec['book']['id']}", "PUT", token=token,
                    body={"signal": signal, "source": "recommendation", "position": position})
            rated += 1
            positive += signal in ("loved", "liked")
    return {"username": username, "shown": shown, "rated": rated, "positive": positive}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--readers", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3, help="pages of recommendations answered per reader")
    p.add_argument("--dataset", default="goodbooks")
    p.add_argument("--min-ratings", type=int, default=40, help="only replay users with enough history")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    health = request(args.api, "/health")
    log.info("simulating %d readers against %s (%s books)", args.readers, args.api, health["books"])

    ds = load_dataset(PROCESSED_DIR / args.dataset)
    rng = random.Random(args.seed)
    counts = ds.ratings.groupby("user_idx").size()
    candidates = counts[counts >= args.min_ratings].index.to_numpy()
    chosen = np.random.default_rng(args.seed).choice(candidates, size=args.readers, replace=False)

    genre_options = {g for genres in ds.books.genres for g in genres}
    totals = {"shown": 0, "rated": 0, "positive": 0}
    for n, user in enumerate(chosen, start=1):
        history = ds.ratings[ds.ratings.user_idx == user]
        ratings_by_item = dict(zip(history.item_idx.tolist(), history.rating.tolist(), strict=True))
        liked_genres = [g for i, r in ratings_by_item.items() if r >= 4 for g in ds.books.genres.iloc[i]]
        top_genres = [g for g, _ in sorted({g: liked_genres.count(g) for g in set(liked_genres)}.items(),
                                           key=lambda kv: -kv[1])[:2] if g in genre_options]
        result = simulate_reader(args.api, ds.books, ratings_by_item, top_genres, args.rounds, rng)
        for key in totals:
            totals[key] += result[key]
        log.info("  %2d/%d %s: %d shown, %d rated (%d positive)",
                 n, args.readers, result["username"], result["shown"], result["rated"], result["positive"])

    opinion_rate = totals["rated"] / max(totals["shown"], 1)
    precision = totals["positive"] / max(totals["rated"], 1)
    log.info("\n%d recommendations shown | %.0f%% had an opinion | %.0f%% of those were positive",
             totals["shown"], 100 * opinion_rate, 100 * precision)
    return totals


if __name__ == "__main__":
    main()
