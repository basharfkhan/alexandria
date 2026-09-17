from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db import Base

settings = get_settings()


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    favorite_genres: Mapped[list[str]] = mapped_column(JSON, default=list)
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    interactions: Mapped[list["Interaction"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Book(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)  # Goodbooks book_id
    title: Mapped[str] = mapped_column(String(500), index=True)
    authors: Mapped[str] = mapped_column(String(500))
    year: Mapped[int | None] = mapped_column(Integer)
    avg_rating: Mapped[float] = mapped_column(Float)
    ratings_count: Mapped[int] = mapped_column(Integer, index=True)
    image_url: Mapped[str | None] = mapped_column(String(500))
    genres: Mapped[list[str]] = mapped_column(JSON, default=list)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    content_embedding = mapped_column(Vector(settings.content_dim), nullable=False)
    cf_factors = mapped_column(Vector(settings.cf_dim), nullable=True)
    cf_bias: Mapped[float | None] = mapped_column(Float)


class Interaction(Base):
    """Current state of a user's relationship with a book (one row per user/book)."""

    __tablename__ = "interactions"
    __table_args__ = (UniqueConstraint("user_id", "book_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id"))
    signal: Mapped[str] = mapped_column(String(20))  # loved | liked | want_to_read | disliked | not_interested
    source: Mapped[str] = mapped_column(String(20), default="app")  # onboarding | chat | recommendation | app
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="interactions")
    book: Mapped[Book] = relationship()


class Event(Base):
    """Append-only log of impressions and feedback.

    Kept separate from ``interactions`` so we can later measure click-through by
    recommendation reason / position and export training data for retraining.
    """

    __tablename__ = "events"
    __table_args__ = (Index("ix_events_user_time", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    book_id: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(30))  # impression | feedback:<signal> | removed
    position: Mapped[int | None] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(String(20))
    model_version: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatUsage(Base):
    """One row per chat-librarian turn, so LLM spend limits survive restarts and scale-to-zero."""

    __tablename__ = "chat_usage"
    __table_args__ = (Index("ix_chat_usage_user_time", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    ip: Mapped[str | None] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ModelMeta(Base):
    __tablename__ = "model_meta"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
