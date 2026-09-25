import pandas as pd
from sqlalchemy import create_engine, text

from alexandria_ml.data.app_feedback import load_app_ratings

BOOKS = pd.DataFrame({"book_id": [10, 20, 30], "item_idx": [0, 1, 2]})


def _database(tmp_path, rows):
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE interactions (user_id INT, book_id INT, signal TEXT)"))
        for row in rows:
            conn.execute(text("INSERT INTO interactions VALUES (:u, :b, :s)"),
                         {"u": row[0], "b": row[1], "s": row[2]})
    return url


def test_signals_become_ratings_for_new_users(tmp_path):
    url = _database(tmp_path, [
        (7, 10, "loved"), (7, 20, "disliked"), (9, 30, "liked"),
        (9, 10, "want_to_read"),   # intent, not an opinion - ignored
        (9, 99, "loved"),          # book not in the catalog - ignored
    ])
    frame = load_app_ratings(url, BOOKS, first_user_idx=53424)

    assert len(frame) == 3
    assert sorted(frame.user_idx.unique()) == [53424, 53425], "app users are appended after Goodbooks users"
    loved = frame[(frame.user_idx == 53424) & (frame.item_idx == 0)]
    assert loved.rating.item() == 5
    assert frame[(frame.user_idx == 53424) & (frame.item_idx == 1)].rating.item() == 2
    assert frame.dtypes["item_idx"].kind == "i"


def test_no_feedback_returns_empty_frame(tmp_path):
    frame = load_app_ratings(_database(tmp_path, []), BOOKS, first_user_idx=100)
    assert frame.empty and list(frame.columns) == ["user_idx", "item_idx", "rating"]
