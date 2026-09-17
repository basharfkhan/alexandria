from app.llm import onboarding_agent
from app.schemas import BookMention, ExtractedPreferences


def genre_of(book):
    return book["genres"][0]


def test_health_and_genres(client):
    assert client.get("/health").json() == {"status": "ok", "books": 90}
    slugs = {g["slug"] for g in client.get("/genres").json()}
    assert slugs == {"fantasy", "romance", "mystery"}


def test_auth_flow(client):
    creds = {"username": "reader_one", "password": "password123"}
    assert client.post("/auth/register", json=creds).status_code == 201
    assert client.post("/auth/register", json=creds).status_code == 409
    assert client.post("/auth/login", json={**creds, "password": "wrong-password"}).status_code == 401
    token = client.post("/auth/login", json=creds).json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["username"] == "reader_one" and me["onboarded"] is False
    assert client.get("/me/recommendations").status_code == 401


def test_search_and_similar(client):
    results = client.get("/books/search", params={"q": "romance book"}).json()
    assert results and all("Romance" in b["title"] for b in results)
    similar = client.get(f"/books/{results[0]['id']}/similar", params={"limit": 5}).json()
    assert len(similar) == 5 and all(genre_of(b) == "romance" for b in similar)


def test_cold_start_uses_genres(client, auth):
    res = client.post("/me/onboarding", json={"genres": ["mystery"]}, headers=auth)
    assert res.status_code == 200 and res.json()["onboarded"]
    recs = client.get("/me/recommendations", params={"limit": 10, "explore": False}, headers=auth).json()
    assert recs["personalization"] == 0
    assert all(genre_of(r["book"]) == "mystery" for r in recs["items"])
    assert recs["items"][0]["reason"] == "genre"


def test_feedback_adapts_recommendations(client, auth):
    fantasy = client.get("/books/search", params={"q": "fantasy book", "limit": 3}).json()
    client.post("/me/onboarding", json={"genres": ["romance"], "loved_book_ids": [b["id"] for b in fantasy]}, headers=auth)

    romance = client.get("/books/search", params={"q": "romance book", "limit": 3}).json()
    for b in romance:
        r = client.put(f"/me/books/{b['id']}", json={"signal": "disliked"}, headers=auth)
        assert r.status_code == 200

    recs = client.get("/me/recommendations", params={"limit": 10, "explore": False}, headers=auth).json()
    rec_ids = {r["book"]["id"] for r in recs["items"]}
    assert not rec_ids & {b["id"] for b in fantasy + romance}, "rated books must not be recommended"
    assert sum(genre_of(r["book"]) == "fantasy" for r in recs["items"]) >= 7
    assert recs["personalization"] > 0
    assert any(r["because_of"] for r in recs["items"])

    library = client.get("/me/library", headers=auth).json()
    assert len(library) == 6
    assert client.delete(f"/me/books/{romance[0]['id']}", headers=auth).status_code == 204
    assert len(client.get("/me/library", headers=auth).json()) == 5


def test_onboarding_validation(client, auth):
    assert client.post("/me/onboarding", json={"genres": ["cookbooks"]}, headers=auth).status_code == 422
    assert client.post("/me/onboarding", json={"loved_book_ids": [99999]}, headers=auth).status_code == 422


def test_chat_without_api_key_is_503(client, auth):
    res = client.post("/chat/onboarding", json={"messages": [{"role": "user", "content": "I love fantasy"}]}, headers=auth)
    assert res.status_code == 503


def test_chat_matches_books(client, auth, monkeypatch):
    async def fake_turn(messages, genres):
        assert "fantasy" in genres
        prefs = ExtractedPreferences(
            genres=["fantasy"],
            loved_books=[BookMention(title="Fantasy Book 1", author="Writer 0")],
            disliked_books=[BookMention(title="Nonexistent Title")],
        )
        return "Great picks! Your recommendations are ready.", True, prefs

    monkeypatch.setattr(onboarding_agent, "run_turn", fake_turn)
    res = client.post("/chat/onboarding", json={"messages": [{"role": "user", "content": "hi"}]}, headers=auth)
    body = res.json()
    assert res.status_code == 200 and body["ready"]
    assert [b["title"] for b in body["matched_loved"]] == ["Fantasy Book 1"]
    assert body["matched_disliked"] == []
