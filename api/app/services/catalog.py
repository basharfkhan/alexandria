from __future__ import annotations

import re

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import Book
from app.schemas import BookMention

_SERIES = re.compile(r"\s*\([^)]*#[^)]*\)\s*$")  # "Harry Potter ... (Harry Potter, #1)"


def search_books(db: Session, query: str, limit: int = 10) -> list[Book]:
    q = query.strip()
    if not q:
        return []
    pattern = f"%{q.lower()}%"
    stmt = (
        select(Book)
        .where(or_(func.lower(Book.title).like(pattern), func.lower(Book.authors).like(pattern)))
        # Prefer exact titles, then title matches, then the most-read books.
        .order_by(
            (func.lower(Book.title) == q.lower()).desc(),
            func.lower(Book.title).like(pattern).desc(),
            Book.ratings_count.desc(),
        )
        .limit(limit)
    )
    return list(db.scalars(stmt))


def match_mention(db: Session, mention: BookMention) -> Book | None:
    """Resolve a free-text book mention from the chatbot to a catalog entry."""
    title = _SERIES.sub("", mention.title).strip()
    if not title:
        return None
    candidates = search_books(db, title, limit=10)
    if not candidates:
        return None
    if mention.author:
        last_name = mention.author.split()[-1].lower()
        by_author = [b for b in candidates if last_name in b.authors.lower()]
        if by_author:
            return by_author[0]
    return candidates[0]


def match_mentions(db: Session, mentions: list[BookMention]) -> list[Book]:
    seen: set[int] = set()
    out: list[Book] = []
    for m in mentions:
        book = match_mention(db, m)
        if book and book.id not in seen:
            seen.add(book.id)
            out.append(book)
    return out
