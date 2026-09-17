from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Event, Interaction


def upsert_interaction(
    db: Session, user_id: int, book_id: int, signal: str, source: str, position: int | None = None
) -> Interaction:
    existing = db.scalar(select(Interaction).where(Interaction.user_id == user_id, Interaction.book_id == book_id))
    if existing:
        existing.signal = signal
        existing.source = source
    else:
        existing = Interaction(user_id=user_id, book_id=book_id, signal=signal, source=source)
        db.add(existing)
    db.add(Event(user_id=user_id, book_id=book_id, type=f"feedback:{signal}", position=position))
    return existing
