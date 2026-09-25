"""Fold real app feedback back into training data.

Readers rate books in the app (loved / liked / disliked); those rows live in the API's
`interactions` table. Retraining reads them and appends them to the Goodbooks ratings as extra
users, so the collaborative model learns from actual usage - the loop the scheduled retraining
workflow closes.

App users are given ids above the Goodbooks range, so the two sources never collide.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# The app's signals mapped onto the 1-5 rating scale the models are trained on.
SIGNAL_TO_RATING = {"loved": 5, "liked": 4, "disliked": 2, "not_interested": 1}
IGNORED_SIGNALS = {"want_to_read"}  # intent, not an opinion


def load_app_ratings(database_url: str, books: pd.DataFrame, first_user_idx: int) -> pd.DataFrame:
    """Return app interactions as (user_idx, item_idx, rating) rows, ready to append to training data."""
    from sqlalchemy import bindparam, create_engine, text

    engine = create_engine(database_url.replace("postgres://", "postgresql+psycopg://", 1))
    query = text("SELECT user_id, book_id, signal FROM interactions WHERE signal IN :signals").bindparams(
        bindparam("signals", expanding=True)  # portable across Postgres and SQLite
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"signals": list(SIGNAL_TO_RATING)}).fetchall()

    if not rows:
        log.info("no app feedback found")
        return pd.DataFrame(columns=["user_idx", "item_idx", "rating"])

    item_of = dict(zip(books.book_id, books.item_idx, strict=True))
    user_codes: dict[int, int] = {}
    records = []
    for user_id, book_id, signal in rows:
        if book_id not in item_of or signal in IGNORED_SIGNALS:
            continue
        user_idx = user_codes.setdefault(user_id, first_user_idx + len(user_codes))
        records.append((user_idx, item_of[book_id], SIGNAL_TO_RATING[signal]))

    frame = pd.DataFrame(records, columns=["user_idx", "item_idx", "rating"]).astype(
        {"user_idx": np.int32, "item_idx": np.int32, "rating": np.int8}
    )
    log.info("app feedback: %d ratings from %d readers", len(frame), frame.user_idx.nunique())
    return frame
