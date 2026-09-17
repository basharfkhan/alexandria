"""Load ML pipeline artifacts into the database.

    python -m app.seed --artifacts ../ml/artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import delete, text

from app.db import SessionLocal, engine, init_db, is_postgres
from app.models import Book, ModelMeta
from app.services.recommender import reset_service

log = logging.getLogger("alexandria.seed")


def seed(artifact_dir: Path, batch_size: int = 1000) -> int:
    books = json.loads((artifact_dir / "books.json").read_text(encoding="utf-8"))
    content = np.load(artifact_dir / "content_embeddings.npy")
    factors = np.load(artifact_dir / "cf_factors.npy")
    bias = np.load(artifact_dir / "cf_bias.npy")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(books) == len(content) == len(factors) == len(bias), "artifact row counts differ"

    init_db()
    with SessionLocal() as db:
        # First load: plain bulk insert. Refresh: merge by primary key so user interactions
        # referencing existing books survive a model update.
        fresh = db.query(Book.id).first() is None
        for start in range(0, len(books), batch_size):
            for i in range(start, min(start + batch_size, len(books))):
                b = books[i]
                row = Book(
                    id=b["book_id"], title=b["title"][:500], authors=b["authors"][:500], year=b["year"],
                    avg_rating=b["avg_rating"], ratings_count=b["ratings_count"], image_url=b["image_url"],
                    genres=b["genres"], tags=b["tags"], content_embedding=content[i],
                    cf_factors=factors[i], cf_bias=float(bias[i]),
                )
                if fresh:
                    db.add(row)
                else:
                    db.merge(row)
            db.commit()
            log.info("seeded %d/%d books", min(start + batch_size, len(books)), len(books))

        db.execute(delete(ModelMeta).where(ModelMeta.key == "manifest"))
        db.add(ModelMeta(key="manifest", value=manifest))
        db.commit()

    if is_postgres():
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_books_content_hnsw "
                "ON books USING hnsw (content_embedding vector_cosine_ops)"
            ))
    reset_service()
    return len(books)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default=str(Path(__file__).resolve().parents[2] / "ml" / "artifacts"))
    n = seed(Path(p.parse_args().artifacts))
    log.info("✓ loaded %d books", n)


if __name__ == "__main__":
    main()
