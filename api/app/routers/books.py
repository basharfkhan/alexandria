from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.db import is_postgres
from app.models import Book
from app.schemas import BookDetailOut, BookOut, GenreOut
from app.security import DB
from app.services.catalog import search_books
from app.services.recommender import get_service

router = APIRouter(tags=["books"])


@router.get("/books/search", response_model=list[BookOut])
def search(db: DB, q: str = Query(min_length=1, max_length=200), limit: int = Query(10, le=50)):
    return search_books(db, q, limit)


@router.get("/books/popular", response_model=list[BookOut])
def popular(db: DB, genre: str | None = None, limit: int = Query(24, le=100)):
    svc = get_service(db)
    rec = svc.recommender
    idx = rec.genre_members.get(genre) if genre else None
    pool = idx if idx is not None else range(rec.n_items)
    top = sorted(pool, key=lambda i: -rec.pop_z[i])[:limit]
    ids = [int(svc.book_ids[i]) for i in top]
    books = {b.id: b for b in db.scalars(select(Book).where(Book.id.in_(ids)))}
    return [books[i] for i in ids]


@router.get("/books/{book_id}", response_model=BookDetailOut)
def get_book(book_id: int, db: DB):
    book = db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    return book


@router.get("/books/{book_id}/similar", response_model=list[BookOut])
def similar(book_id: int, db: DB, limit: int = Query(10, le=50)):
    book = db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")

    if is_postgres():
        # Approximate nearest neighbours via the pgvector HNSW index.
        stmt = (
            select(Book)
            .where(Book.id != book_id)
            .order_by(Book.content_embedding.cosine_distance(book.content_embedding))
            .limit(limit)
        )
        return list(db.scalars(stmt))

    svc = get_service(db)
    ids = [int(svc.book_ids[j]) for j, _ in svc.recommender.similar(svc.index_of[book_id], limit)]
    books = {b.id: b for b in db.scalars(select(Book).where(Book.id.in_(ids)))}
    return [books[i] for i in ids]


@router.get("/genres", response_model=list[GenreOut])
def genres(db: DB):
    rec = get_service(db).recommender
    return [GenreOut(slug=g, count=len(rec.genre_members[g])) for g in rec.genres]
