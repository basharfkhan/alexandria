"""Everything scoped to the logged-in reader: onboarding, library, feedback, recommendations."""

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models import Book, Event, Interaction
from app.schemas import (
    FeedbackIn,
    LibraryItem,
    OnboardingIn,
    RecommendationOut,
    RecommendationsOut,
    UserOut,
)
from app.security import DB, CurrentUser
from app.services.interactions import upsert_interaction
from app.services.recommender import get_service

router = APIRouter(prefix="/me", tags=["me"])


def _genre_label(slug: str) -> str:
    return slug.replace("-", " ").title()


EXPLANATIONS = {
    "similar": lambda b, g: f"Because you enjoyed {b.title}" if b else "Matches your taste profile",
    "collaborative": lambda b, g: f"Readers who loved {b.title} also loved this" if b else "Loved by readers like you",
    "genre": lambda b, g: f"A standout in {_genre_label(g)}" if g else "A standout in your genres",
    "popular": lambda b, g: "A reader favorite",
    "explore": lambda b, g: "Something a little different to broaden your shelf",
}


@router.post("/onboarding", response_model=UserOut)
def onboarding(body: OnboardingIn, user: CurrentUser, db: DB):
    svc = get_service(db)
    unknown_genres = set(body.genres) - set(svc.recommender.genre_members)
    if unknown_genres:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown genres: {sorted(unknown_genres)}")

    all_ids = {*body.loved_book_ids, *body.liked_book_ids, *body.disliked_book_ids}
    missing = all_ids - set(svc.index_of)
    if missing:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown book ids: {sorted(missing)}")

    user.favorite_genres = sorted(set(user.favorite_genres or []) | set(body.genres))
    for signal, ids in (("liked", body.liked_book_ids), ("loved", body.loved_book_ids), ("disliked", body.disliked_book_ids)):
        for book_id in ids:
            upsert_interaction(db, user.id, book_id, signal, source=body.source)
    user.onboarded = True
    db.commit()
    return user


@router.put("/genres", response_model=UserOut)
def set_genres(genres: list[str], user: CurrentUser, db: DB):
    valid = set(get_service(db).recommender.genre_members)
    user.favorite_genres = sorted(g for g in set(genres) if g in valid)
    db.commit()
    return user


@router.get("/library", response_model=list[LibraryItem])
def library(user: CurrentUser, db: DB):
    rows = db.scalars(
        select(Interaction)
        .options(selectinload(Interaction.book))
        .where(Interaction.user_id == user.id)
        .order_by(Interaction.updated_at.desc())
    )
    return [LibraryItem(book=it.book, signal=it.signal) for it in rows]


@router.put("/books/{book_id}", response_model=LibraryItem)
def give_feedback(book_id: int, body: FeedbackIn, user: CurrentUser, db: DB):
    book = db.get(Book, book_id)
    if not book:
        raise HTTPException(404, "Book not found")
    upsert_interaction(db, user.id, book_id, body.signal, source=body.source, position=body.position)
    db.commit()
    return LibraryItem(book=book, signal=body.signal)


@router.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_feedback(book_id: int, user: CurrentUser, db: DB):
    it = db.scalar(select(Interaction).where(Interaction.user_id == user.id, Interaction.book_id == book_id))
    if it:
        db.delete(it)
        db.add(Event(user_id=user.id, book_id=book_id, type="removed"))
        db.commit()


@router.get("/recommendations", response_model=RecommendationsOut)
def recommendations(user: CurrentUser, db: DB, limit: int = Query(20, ge=1, le=60), explore: bool = True):
    svc = get_service(db)
    slots = get_settings().recommendation_explore_slots if explore else 0
    recs, personalization = svc.recommend_for(user, k=limit, explore_slots=slots)

    ids = {int(svc.book_ids[r.index]) for r in recs}
    ids |= {int(svc.book_ids[r.because_of]) for r in recs if r.because_of is not None}
    books = {b.id: b for b in db.scalars(select(Book).where(Book.id.in_(ids)))}

    items = []
    for pos, r in enumerate(recs):
        book = books[int(svc.book_ids[r.index])]
        because = books[int(svc.book_ids[r.because_of])] if r.because_of is not None else None
        items.append(
            RecommendationOut(
                book=book,
                score=round(r.score, 4),
                reason=r.reason,
                explanation=EXPLANATIONS[r.reason](because, r.genre),
                because_of=because,
            )
        )
        db.add(Event(user_id=user.id, book_id=book.id, type="impression", position=pos,
                     reason=r.reason, model_version=svc.model_version))
    db.commit()
    return RecommendationsOut(model_version=svc.model_version, personalization=personalization, items=items)
