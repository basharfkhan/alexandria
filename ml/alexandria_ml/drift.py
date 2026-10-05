"""Drift monitoring for the retraining loop.

The promotion gate answers "is this model worse than the last one". It cannot answer "has this
model quietly become a different model", which is the failure a weekly retrain actually invites:

* **Slow regression.** Ten retrains that each lose 1.9% all pass a 2% gate and leave the model 17%
  worse than where it started. Comparing against a pinned *reference* model closes that (see
  ``registry.REFERENCE_GATE``).
* **Prediction drift.** Accuracy metrics are averages over thousands of users and move very little
  when the character of the recommendations changes. The popularity-drift failure that shipped once
  already looked fine on NDCG. A fixed probe cohort makes it visible: the same readers, the same
  ratings, every run, with their top-20 compared run to run.
* **Input drift.** The catalog grows, app ratings join the training data, and a surprise in the
  inputs is easier to read here than in the metrics downstream of it.

None of this needs production traffic, which is the point: the probe readers are synthetic and
fixed, so any change in what they are shown comes from the model, not from who happened to visit.

The fingerprint is small enough to live in the manifest and the registry, so a run compares itself
against stored output rather than needing the previous model's artifacts.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from alexandria_core.recommender import HybridRecommender
from alexandria_core.rerank import Reranker

log = logging.getLogger(__name__)

# Readers defined by books they loved, not by user index: ids and indices shift when the catalog or
# the ratings change, titles do not. Each is a recognisable taste so a drifting list is obvious to
# read, and together they span the genres the catalog is strongest in.
PROBE_READERS: dict[str, list[str]] = {
    "hard_sf": ["The Martian", "Ender's Game", "Ready Player One"],
    "epic_fantasy": ["The Way of Kings", "The Name of the Wind", "Mistborn"],
    "thriller": ["Gone Girl", "The Girl on the Train", "The Girl with the Dragon Tattoo"],
    "classic_lit": ["Pride and Prejudice", "Jane Eyre", "Wuthering Heights"],
    "ya_dystopia": ["The Hunger Games", "Divergent", "The Maze Runner"],
    "horror": ["The Shining", "It", "Pet Sematary"],
    "book_club": ["The Kite Runner", "Life of Pi", "The Book Thief"],
    "romance": ["Me Before You", "The Notebook", "The Fault in Our Stars"],
    "nonfiction": ["Sapiens", "Outliers", "Freakonomics"],
    "golden_age_mystery": ["And Then There Were None", "Murder on the Orient Express", "Rebecca"],
}

PROBE_K = 20
LIKE = 1.0

# A new model is *expected* to change its recommendations, so these raise warnings rather than
# blocking promotion. They are tuned to be quiet across ordinary retrains and loud when the
# character of the output changes.
OVERLAP_WARN = 0.5  # mean Jaccard overlap of top-20 against the previous model
POPULARITY_WARN = 0.10  # shift in the mean popularity percentile of recommended books
GENRE_WARN = 0.15  # total-variation distance between genre mixes
INPUT_WARN = 0.25  # relative change in a training-input statistic


def resolve_probe_books(books: pd.DataFrame) -> dict[str, list[int]]:
    """Map each probe reader's titles onto catalog positions (the most-rated match wins)."""
    titles = books.title.str.lower()
    resolved: dict[str, list[int]] = {}
    for reader, wanted in PROBE_READERS.items():
        found = []
        for title in wanted:
            # Goodbooks titles carry series suffixes ("Mistborn: The Final Empire (Mistborn, #1)"),
            # so match on prefix and break ties by popularity.
            matches = books.index[titles.str.startswith(title.lower())]
            if len(matches):
                found.append(int(books.popularity.loc[matches].idxmax()))
        if found:
            resolved[reader] = found
    missing = set(PROBE_READERS) - set(resolved)
    if missing:
        log.warning("probe readers with no matching books in this catalog: %s", ", ".join(sorted(missing)))
    return resolved


def fingerprint(
    books: pd.DataFrame,
    recommender: HybridRecommender,
    reranker: Reranker | None = None,
    k: int = PROBE_K,
    **serving,
) -> dict:
    """What a fixed cohort of readers is shown by this model, summarised for later comparison."""
    probes = resolve_probe_books(books)
    book_ids = books.book_id.to_numpy()
    popularity = books.popularity.to_numpy()
    order = np.sort(popularity)
    cold_start = (~books.has_ratings).to_numpy() if "has_ratings" in books else np.zeros(len(books), bool)

    readers, percentiles, genres, cold = {}, [], Counter(), 0
    for reader, liked in probes.items():
        recs = recommender.recommend(
            dict.fromkeys(liked, LIKE), k=k, exclude=liked, reranker=reranker, **serving
        )
        picked = [r.index for r in recs]
        readers[reader] = [int(book_ids[i]) for i in picked]
        percentiles.extend(np.searchsorted(order, popularity[picked]) / max(len(order), 1))
        cold += int(cold_start[picked].sum())
        for i in picked:
            genres.update(books.genres.iloc[i][:1])  # primary genre only

    n_recs = sum(len(v) for v in readers.values()) or 1
    return {
        "readers": readers,
        "n_readers": len(readers),
        "popularity_pct": float(np.mean(percentiles)) if percentiles else 0.0,
        "cold_start_share": cold / n_recs,
        "genre_mix": {g: round(c / n_recs, 4) for g, c in sorted(genres.items())},
    }


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 1.0


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Half the L1 distance between two genre mixes: 0 = identical, 1 = no overlap at all."""
    return 0.5 * sum(abs(a.get(g, 0.0) - b.get(g, 0.0)) for g in set(a) | set(b))


def compare(current: dict, previous: dict) -> dict:
    """Prediction drift between two fingerprints. Warnings, never a promotion blocker."""
    shared = sorted(set(current["readers"]) & set(previous["readers"]))
    overlaps = {r: jaccard(current["readers"][r], previous["readers"][r]) for r in shared}
    popularity_shift = current["popularity_pct"] - previous["popularity_pct"]
    genre_shift = total_variation(current["genre_mix"], previous["genre_mix"])

    warnings = []
    mean_overlap = float(np.mean(list(overlaps.values()))) if overlaps else 1.0
    if mean_overlap < OVERLAP_WARN:
        warnings.append(f"probe readers keep only {mean_overlap:.0%} of their recommendations")
    if abs(popularity_shift) > POPULARITY_WARN:
        direction = "more" if popularity_shift > 0 else "less"
        warnings.append(f"recommendations are {abs(popularity_shift):.0%} {direction} popular")
    if genre_shift > GENRE_WARN:
        warnings.append(f"genre mix moved {genre_shift:.0%}")

    return {
        "overlap_mean": round(mean_overlap, 4),
        "overlap_min": round(min(overlaps.values()), 4) if overlaps else 1.0,
        "overlap_by_reader": {r: round(v, 4) for r, v in sorted(overlaps.items())},
        "popularity_pct_shift": round(popularity_shift, 4),
        "cold_start_share_shift": round(current["cold_start_share"] - previous["cold_start_share"], 4),
        "genre_shift": round(genre_shift, 4),
        "warnings": warnings,
    }


def input_stats(books: pd.DataFrame, ratings: pd.DataFrame) -> dict:
    """Shape of what the model was trained on, so a surprise in the inputs is visible here."""
    popularity = books.popularity.to_numpy() if "popularity" in books else books.ratings_count.to_numpy()
    rated = books.has_ratings.sum() if "has_ratings" in books else len(books)
    described = books.description.notna().sum() if "description" in books else 0
    return {
        "n_books": int(len(books)),
        "n_rated_books": int(rated),
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings.user_idx.nunique()),
        "ratings_per_user": round(len(ratings) / max(ratings.user_idx.nunique(), 1), 2),
        "popularity_median": float(np.median(popularity)),
        "description_share": round(float(described) / max(len(books), 1), 4),
    }


def compare_inputs(current: dict, previous: dict) -> dict:
    """Relative change per training-input statistic, with a warning on anything that jumped."""
    changes, warnings = {}, []
    for key, new in current.items():
        old = previous.get(key)
        if not isinstance(old, int | float) or not old:
            continue
        change = (new - old) / old
        changes[key] = round(change, 4)
        if abs(change) > INPUT_WARN:
            warnings.append(f"{key} changed {change:+.0%} ({old:g} -> {new:g})")
    return {"changes": changes, "warnings": warnings}


def summarise(drift: dict | None) -> str:
    """One readable block for the retrain log and the workflow summary."""
    if not drift:
        return "drift: no previous fingerprint to compare against (first run)"
    lines = []
    if predictions := drift.get("predictions"):
        lines.append(
            f"  probe readers: {predictions['overlap_mean']:.0%} of top-20 kept "
            f"(worst reader {predictions['overlap_min']:.0%}), "
            f"popularity {predictions['popularity_pct_shift']:+.1%}, "
            f"genre mix {predictions['genre_shift']:.1%}"
        )
        lines += [f"  WARN {w}" for w in predictions["warnings"]]
    if inputs := drift.get("inputs"):
        lines += [f"  WARN {w}" for w in inputs["warnings"]]
    head = "drift: nothing unusual" if not any("WARN" in ln for ln in lines) else "drift: see warnings"
    return "\n".join([head, *lines])


def from_artifacts(artifact_dir: Path) -> dict:
    """Fingerprint a model that is already exported, without retraining it.

    Used to give an existing production model a fingerprint so the next retrain has something to
    compare against, instead of waiting two cycles for drift monitoring to mean anything.
    """
    import json

    import lightgbm as lgb

    from alexandria_core.recommender import CatalogArrays
    from alexandria_ml.evaluate import SERVED_AUTHOR_CAP, SERVED_DIVERSITY, SERVED_NEW_BOOK_SLOTS

    books = pd.DataFrame(json.loads((artifact_dir / "books.json").read_text(encoding="utf-8")))
    books["popularity"] = books.get("popularity", books.ratings_count)
    books["has_ratings"] = books.get("has_ratings", True)
    recommender = HybridRecommender(CatalogArrays(
        content=np.load(artifact_dir / "content_embeddings.npy"),
        popularity=books.popularity.to_numpy(),
        avg_rating=books.avg_rating.to_numpy(),
        genres=books.genres.tolist(),
        cf_factors=np.load(artifact_dir / "cf_factors.npy"),
        cf_bias=np.load(artifact_dir / "cf_bias.npy"),
        authors=books.authors.tolist(),
        titles=books.title.tolist(),
        cold_start=(~books.has_ratings).tolist(),
    ))

    reranker = None
    if (path := artifact_dir / "ranker.txt").exists():
        booster = lgb.Booster(model_str=path.read_text(encoding="utf-8").replace("\r\n", "\n"))
        reranker = Reranker(booster, booster.feature_name())

    return fingerprint(
        books, recommender, reranker,
        diversity=SERVED_DIVERSITY, max_per_author=SERVED_AUTHOR_CAP,
        new_book_slots=SERVED_NEW_BOOK_SLOTS,
    )


def main(argv=None) -> int:
    """Backfill the fingerprint of an exported model into its manifest.

        python -m alexandria_ml.drift --artifacts artifacts
    """
    import argparse
    import json

    from alexandria_ml.config import ARTIFACT_DIR
    from alexandria_ml.registry import REGISTRY_PATH

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    p.add_argument("--registry", default=str(REGISTRY_PATH),
                   help="also backfill this model's entry, so the next retrain has a baseline")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    artifact_dir = Path(args.artifacts)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"] = from_artifacts(artifact_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    probes = manifest["fingerprint"]
    registry_path = Path(args.registry)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for entry in registry.get("history", []):
            if entry["model_version"] == manifest["model_version"]:
                entry["fingerprint"] = probes
                registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
                print(f"backfilled {registry_path.name}")
                break

    print(f"fingerprinted {probes['n_readers']} probe readers for model {manifest['model_version']}")
    print(f"  mean popularity percentile {probes['popularity_pct']:.3f}, "
          f"new releases {probes['cold_start_share']:.1%}")
    for reader, ids in sorted(probes["readers"].items()):
        titles = ", ".join(str(t) for t in ids[:3])
        print(f"  {reader:<20} top ids {titles} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
