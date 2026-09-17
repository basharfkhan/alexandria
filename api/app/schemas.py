from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Signal = Literal["loved", "liked", "want_to_read", "disliked", "not_interested"]


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    favorite_genres: list[str]
    onboarded: bool


class BookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    authors: str
    year: int | None
    avg_rating: float
    ratings_count: int
    image_url: str | None
    genres: list[str]


class BookDetailOut(BookOut):
    description: str | None = None


class GenreOut(BaseModel):
    slug: str
    count: int


class OnboardingIn(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=15)
    loved_book_ids: list[int] = Field(default_factory=list, max_length=50)
    liked_book_ids: list[int] = Field(default_factory=list, max_length=50)
    disliked_book_ids: list[int] = Field(default_factory=list, max_length=50)
    source: Literal["onboarding", "chat"] = "onboarding"


class FeedbackIn(BaseModel):
    signal: Signal
    source: Literal["recommendation", "app", "search"] = "app"
    position: int | None = None


class LibraryItem(BaseModel):
    book: BookOut
    signal: Signal


class RecommendationOut(BaseModel):
    book: BookOut
    score: float
    reason: Literal["similar", "collaborative", "genre", "popular", "explore"]
    explanation: str
    because_of: BookOut | None = None


class RecommendationsOut(BaseModel):
    model_version: str | None
    personalization: float = Field(description="0 = cold start, 1 = fully driven by the user's history")
    items: list[RecommendationOut]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=40)


class BookMention(BaseModel):
    title: str
    author: str = ""


class ExtractedPreferences(BaseModel):
    genres: list[str] = Field(default_factory=list)
    loved_books: list[BookMention] = Field(default_factory=list)
    disliked_books: list[BookMention] = Field(default_factory=list)


class ChatOut(BaseModel):
    reply: str
    ready: bool
    preferences: ExtractedPreferences
    matched_loved: list[BookOut]
    matched_disliked: list[BookOut]
