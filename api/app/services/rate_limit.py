"""Spend guards for the LLM-backed chat endpoint.

Per-user and global limits are counted in Postgres, so they hold across restarts (the free
Render instance sleeps and wakes constantly). The per-IP limit is in memory: it's a cheap
first line of defence against one client creating many accounts, and the global daily cap
bounds worst-case cost regardless.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import ChatUsage, User


class SlidingWindow:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float) -> bool:
        """Record a hit; return False (and don't record) if the key is over its limit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


ip_window = SlidingWindow()


def client_ip(request: Request) -> str:
    # Render/Vercel proxies put the real client first in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _too_many(message: str, retry_after_s: int) -> HTTPException:
    return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, message, headers={"Retry-After": str(retry_after_s)})


def enforce_chat_limits(db: Session, user: User, request: Request) -> None:
    """Raise 429 if this chat turn would exceed a limit; otherwise record it."""
    s = get_settings()
    now = datetime.now(UTC)

    def count(since: datetime, user_id: int | None = None) -> int:
        stmt = select(func.count()).select_from(ChatUsage).where(ChatUsage.created_at >= since)
        if user_id is not None:
            stmt = stmt.where(ChatUsage.user_id == user_id)
        return db.scalar(stmt) or 0

    if count(now - timedelta(days=1)) >= s.chat_limit_global_per_day:
        raise _too_many("The librarian has had a busy day - please use Quick picks, or try the chat tomorrow.", 3600)
    if count(now - timedelta(days=1), user.id) >= s.chat_limit_user_per_day:
        raise _too_many("You've reached today's chat limit - Quick picks still work, or come back tomorrow.", 3600)
    if count(now - timedelta(hours=1), user.id) >= s.chat_limit_user_per_hour:
        raise _too_many("You're chatting fast! Please wait a bit before sending more messages.", 600)

    ip = client_ip(request)
    if not ip_window.hit(ip, s.chat_limit_ip_per_hour, 3600):
        raise _too_many("Too many chat messages from your network - please try again later.", 600)

    db.add(ChatUsage(user_id=user.id, ip=ip[:45], created_at=now))
    db.commit()
