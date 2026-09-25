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
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import SessionLocal, engine, init_db, is_postgres
from app.models import Book, ModelBlob, ModelMeta
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
    rows = [
        {
            "id": b["book_id"], "title": b["title"][:500], "authors": b["authors"][:500], "year": b["year"],
            "avg_rating": b["avg_rating"], "ratings_count": b["ratings_count"], "image_url": b["image_url"],
            "genres": b["genres"], "tags": b["tags"], "description": b.get("description"),
            "content_embedding": content[i], "cf_factors": factors[i], "cf_bias": float(bias[i]),
        }
        for i, b in enumerate(books)
    ]

    with SessionLocal() as db:
        if is_postgres():
            # INSERT ... ON CONFLICT DO UPDATE: one round trip per batch, and user interactions that
            # reference existing books survive a model refresh.
            table = Book.__table__
            stmt = pg_insert(table)
            stmt = stmt.on_conflict_do_update(
                index_elements=[table.c.id],
                set_={c.name: stmt.excluded[c.name] for c in table.columns if c.name != "id"},
            )
            for start in range(0, len(rows), batch_size):
                db.execute(stmt, rows[start : start + batch_size])
                db.commit()
                log.info("upserted %d/%d books", min(start + batch_size, len(rows)), len(rows))
        else:
            fresh = db.query(Book.id).first() is None
            for start in range(0, len(rows), batch_size):
                for row in rows[start : start + batch_size]:
                    if fresh:
                        db.add(Book(**row))
                    else:
                        db.merge(Book(**row))
                db.commit()
                log.info("seeded %d/%d books", min(start + batch_size, len(rows)), len(rows))

        ranker_path = artifact_dir / "ranker.txt"
        db.execute(delete(ModelBlob).where(ModelBlob.key == "ranker"))
        if ranker_path.exists():
            # read_text() normalises CRLF -> LF, which LightGBM requires to parse the model.
            db.add(ModelBlob(key="ranker", data=ranker_path.read_text(encoding="utf-8").encode("utf-8")))
            log.info("stored second-stage ranker (%.0f KB)", ranker_path.stat().st_size / 1024)
        else:
            log.info("no ranker.txt in artifacts - serving stage-1 ranking only")

        # Written last: the API hot-reloads when it sees a new model_version (services/recommender.py).
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
