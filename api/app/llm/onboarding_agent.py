"""Conversational onboarding: an LLM-powered "librarian" that interviews a new reader.

Each turn is one stateless Messages API call. The client sends the whole transcript;
The model returns, via structured outputs, both its next message *and* the preferences
extracted so far. The API then resolves book mentions against the catalog and the
frontend lets the user confirm them before they become feedback.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from functools import lru_cache

import anthropic

from app.config import get_settings
from app.schemas import ChatMessage, ExtractedPreferences

log = logging.getLogger(__name__)

OPENING_MESSAGE = (
    "Welcome to Alexandria! I'm your librarian. 📚 Tell me a bit about what you like to read - "
    "a few genres you gravitate toward, or a book you couldn't put down?"
)

SYSTEM_PROMPT = """You are the onboarding librarian for Alexandria, a book recommendation app.
Your job is a short, friendly interview that captures a new reader's taste so the
recommender can get started. The app already greeted the user with: "{opening}"

Over roughly 3-5 exchanges, learn:
1. Genres they enjoy.
2. Two or more specific books they loved (title, and author when known).
3. Books or genres they disliked or want to avoid.
4. Optionally, what they're in the mood for right now.

Guidelines:
- Ask one or two short questions per message. Be warm and concise (under 60 words).
- React briefly to books they mention, but do not recommend books yourself - the app does that.
- If a title is ambiguous, you may ask which one they mean.
- Once you know at least one genre and two loved books - or the user wants to finish - set
  "ready" to true and tell them their recommendations are ready to generate.

Extraction rules for the "preferences" field (cumulative over the whole conversation):
- "genres": only slugs from this list: {genres}. Map what the user says onto the closest slugs.
- "loved_books": books the user said they loved or really liked.
- "disliked_books": books the user said they disliked or did not finish.
- Only include what the user actually said. Use the canonical book title and author's full name.
  Use "" for author when unknown."""


def _response_schema(genres: Sequence[str]) -> dict:
    mention = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "author": {"type": "string"}},
        "required": ["title", "author"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "ready": {"type": "boolean"},
            "preferences": {
                "type": "object",
                "properties": {
                    "genres": {"type": "array", "items": {"type": "string", "enum": list(genres)}},
                    "loved_books": {"type": "array", "items": mention},
                    "disliked_books": {"type": "array", "items": mention},
                },
                "required": ["genres", "loved_books", "disliked_books"],
                "additionalProperties": False,
            },
        },
        "required": ["reply", "ready", "preferences"],
        "additionalProperties": False,
    }


class ChatUnavailableError(RuntimeError):
    pass


@lru_cache
def _client() -> anthropic.AsyncAnthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ChatUnavailableError("The chat librarian is not configured (ANTHROPIC_API_KEY is unset).")
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def run_turn(
    messages: Sequence[ChatMessage], genres: Sequence[str]
) -> tuple[str, bool, ExtractedPreferences]:
    """Return (assistant reply, ready flag, cumulative preferences)."""
    # The opening message is generated locally, so the transcript sent to the model must
    # start at the user's first message.
    history = list(messages)
    while history and history[0].role == "assistant":
        history.pop(0)
    if not history:
        return OPENING_MESSAGE, False, ExtractedPreferences()

    settings = get_settings()
    client = _client()
    try:
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT.format(opening=OPENING_MESSAGE, genres=", ".join(genres)),
            messages=[{"role": m.role, "content": m.content} for m in history],
            output_config={
                "effort": settings.llm_effort,
                "format": {"type": "json_schema", "schema": _response_schema(genres)},
            },
            # If a safety classifier declines, retry server-side on Anthropic's recommended fallback model.
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
    except anthropic.RateLimitError as exc:
        raise ChatUnavailableError("The librarian is busy right now - please try again in a moment.") from exc
    except anthropic.APIConnectionError as exc:
        raise ChatUnavailableError("Could not reach the language model.") from exc
    except anthropic.APIStatusError as exc:
        log.exception("Anthropic API error %s", exc.status_code)
        raise ChatUnavailableError("The librarian hit an error - please try again.") from exc

    if response.stop_reason == "refusal":
        return "Sorry, I can't help with that. Tell me about some books you've enjoyed?", False, ExtractedPreferences()

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.error("Unparseable model output (stop_reason=%s): %r", response.stop_reason, text[:200])
        raise ChatUnavailableError("The librarian got tongue-tied - please try again.") from exc

    prefs = ExtractedPreferences.model_validate(data["preferences"])
    return data["reply"], bool(data["ready"]), prefs
