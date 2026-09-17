import pytest

from app.config import get_settings
from app.db import SessionLocal
from app.llm import onboarding_agent
from app.models import ChatUsage
from app.schemas import ExtractedPreferences
from app.services.rate_limit import SlidingWindow, ip_window

HELLO = {"messages": [{"role": "user", "content": "I love fantasy"}]}


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def fake_turn(messages, genres):
        calls.append(messages)
        return "Tell me more!", False, ExtractedPreferences()

    monkeypatch.setattr(onboarding_agent, "run_turn", fake_turn)
    return calls


@pytest.fixture
def limits(monkeypatch):
    """Isolate each test: empty usage table, fresh IP window, and settings to override."""
    with SessionLocal() as db:
        db.query(ChatUsage).delete()
        db.commit()
    ip_window.reset()
    return lambda **kw: [monkeypatch.setattr(get_settings(), f"chat_limit_{k}", v) for k, v in kw.items()]


def test_user_hourly_limit_blocks_before_llm_call(client, auth, fake_llm, limits):
    limits(user_per_hour=2)
    assert [client.post("/chat/onboarding", json=HELLO, headers=auth).status_code for _ in range(3)] == [200, 200, 429]
    assert len(fake_llm) == 2, "the blocked request must not reach the LLM"
    blocked = client.post("/chat/onboarding", json=HELLO, headers=auth)
    assert "Retry-After" in blocked.headers and "wait" in blocked.json()["detail"]


def test_limits_are_per_user(client, auth, fake_llm, limits):
    limits(user_per_hour=1)
    assert client.post("/chat/onboarding", json=HELLO, headers=auth).status_code == 200
    assert client.post("/chat/onboarding", json=HELLO, headers=auth).status_code == 429

    other = client.post("/auth/register", json={"username": "second_reader", "password": "password123"})
    headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert client.post("/chat/onboarding", json=HELLO, headers=headers).status_code == 200


def test_global_daily_cap(client, auth, fake_llm, limits):
    limits(global_per_day=1)
    assert client.post("/chat/onboarding", json=HELLO, headers=auth).status_code == 200
    blocked = client.post("/chat/onboarding", json=HELLO, headers=auth)
    assert blocked.status_code == 429 and "busy day" in blocked.json()["detail"]


def test_ip_limit_uses_forwarded_for(client, auth, fake_llm, limits):
    limits(ip_per_hour=1)
    ip_a = {**auth, "X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
    ip_b = {**auth, "X-Forwarded-For": "198.51.100.9"}
    assert client.post("/chat/onboarding", json=HELLO, headers=ip_a).status_code == 200
    assert client.post("/chat/onboarding", json=HELLO, headers=ip_a).status_code == 429
    assert client.post("/chat/onboarding", json=HELLO, headers=ip_b).status_code == 200


def test_opening_request_without_user_message_is_free(client, auth, fake_llm, limits):
    limits(user_per_hour=0)
    assert client.post("/chat/onboarding", json={"messages": []}, headers=auth).status_code == 200


def test_sliding_window_expires():
    window = SlidingWindow()
    assert window.hit("k", 1, window_s=0.0)
    assert window.hit("k", 1, window_s=0.0), "hits older than the window no longer count"
