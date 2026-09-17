from fastapi import APIRouter, HTTPException, Request, status

from app.llm import onboarding_agent
from app.schemas import ChatIn, ChatOut
from app.security import DB, CurrentUser
from app.services.catalog import match_mentions
from app.services.rate_limit import enforce_chat_limits
from app.services.recommender import get_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/onboarding", response_model=ChatOut)
async def onboarding_chat(body: ChatIn, request: Request, user: CurrentUser, db: DB):
    genres = get_service(db).recommender.genres
    # Count the turn before calling the LLM so failed or slow calls still consume the budget.
    if any(m.role == "user" for m in body.messages):
        enforce_chat_limits(db, user, request)
    try:
        reply, ready, prefs = await onboarding_agent.run_turn(body.messages, genres)
    except onboarding_agent.ChatUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return ChatOut(
        reply=reply,
        ready=ready,
        preferences=prefs,
        matched_loved=match_mentions(db, prefs.loved_books),
        matched_disliked=match_mentions(db, prefs.disliked_books),
    )
