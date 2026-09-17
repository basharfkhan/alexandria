from app.config import get_settings
from app.db import SessionLocal
from app.models import ModelMeta


def _set_version(version: str) -> None:
    with SessionLocal() as db:
        meta = db.get(ModelMeta, "manifest")
        meta.value = {**meta.value, "model_version": version}
        db.commit()


def test_new_model_version_is_hot_reloaded(client, auth, monkeypatch):
    client.post("/me/onboarding", json={"genres": ["fantasy"]}, headers=auth)
    assert client.get("/me/recommendations", headers=auth).json()["model_version"] == "test-1"

    monkeypatch.setattr(get_settings(), "model_reload_interval_s", 3600)
    _set_version("test-2")
    try:
        # Within the check interval the cached model keeps serving.
        assert client.get("/me/recommendations", headers=auth).json()["model_version"] == "test-1"

        monkeypatch.setattr(get_settings(), "model_reload_interval_s", 0)
        assert client.get("/me/recommendations", headers=auth).json()["model_version"] == "test-2"
    finally:
        _set_version("test-1")
        client.get("/me/recommendations", headers=auth)
