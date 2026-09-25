"""Write the model artifacts the API seeds its database from."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def export_artifacts(
    out_dir: Path,
    books: pd.DataFrame,
    content: np.ndarray,
    cf_factors: np.ndarray,
    cf_bias: np.ndarray,
    manifest: dict,
    ranker=None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in books.itertuples(index=False):
        records.append(
            {
                "book_id": int(row.book_id),
                "title": row.title,
                "authors": row.authors,
                "year": None if pd.isna(row.year) else int(row.year),
                "avg_rating": float(row.avg_rating),
                "ratings_count": int(row.ratings_count),
                "image_url": row.image_url or None,
                "genres": list(row.genres),
                "tags": list(row.tags),
                "description": row.description if isinstance(getattr(row, "description", None), str) else None,
                "popularity": int(getattr(row, "popularity", row.ratings_count)),
                "has_ratings": bool(getattr(row, "has_ratings", True)),
            }
        )
    (out_dir / "books.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    np.save(out_dir / "content_embeddings.npy", content.astype(np.float32))
    np.save(out_dir / "cf_factors.npy", cf_factors.astype(np.float32))
    np.save(out_dir / "cf_bias.npy", cf_bias.astype(np.float32))

    ranker_path = out_dir / "ranker.txt"
    if ranker is not None:
        # newline="\n": LightGBM cannot parse a model written with Windows CRLF endings.
        ranker_path.write_text(ranker.model_to_string(), encoding="utf-8", newline="\n")
    elif ranker_path.exists():
        ranker_path.unlink()  # stale ranker would not match these embeddings

    manifest = {
        **manifest,
        "created_at": datetime.now(UTC).isoformat(),
        "n_books": len(records),
        "content_dim": int(content.shape[1]),
        "cf_dim": int(cf_factors.shape[1]),
    }
    # Round-trip through JSON to turn numpy scalars into floats and NaN into null.
    clean = json.loads(json.dumps(manifest, default=float), parse_constant=lambda _: None)
    (out_dir / "manifest.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return out_dir
