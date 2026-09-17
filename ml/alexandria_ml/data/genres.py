"""Map noisy user-generated Goodreads shelf tags to a clean set of genres.

Goodreads tags are free text ("to-read", "fantasy-sci-fi", "ya", "owned-books"...).
We keep a curated genre vocabulary and match tag *tokens* against keywords.
"""

from __future__ import annotations

import re

GENRE_KEYWORDS: dict[str, list[str]] = {
    "fantasy": ["fantasy", "high-fantasy", "epic-fantasy", "urban-fantasy", "magic", "dragons"],
    "science-fiction": ["sci-fi", "scifi", "science-fiction", "sf", "space-opera", "cyberpunk"],
    "dystopian": ["dystopian", "dystopia", "post-apocalyptic", "apocalyptic"],
    "romance": ["romance", "love-story", "romantic", "chick-lit"],
    "mystery": ["mystery", "mysteries", "detective", "whodunit"],
    "thriller": ["thriller", "thrillers", "suspense", "psychological-thriller"],
    "crime": ["crime", "noir"],
    "horror": ["horror", "scary", "ghosts", "zombies"],
    "paranormal": ["paranormal", "vampires", "vampire", "werewolves", "supernatural"],
    "historical-fiction": ["historical-fiction", "historical", "hist-fic"],
    "literary-fiction": ["literary-fiction", "literary", "literature"],
    "classics": ["classics", "classic", "classic-literature"],
    "contemporary": ["contemporary", "contemporary-fiction", "realistic-fiction"],
    "young-adult": ["young-adult", "ya", "teen", "ya-fiction"],
    "childrens": ["childrens", "children", "kids", "picture-books", "middle-grade"],
    "adventure": ["adventure", "action"],
    "humor": ["humor", "humour", "funny", "comedy"],
    "graphic-novels": ["graphic-novels", "graphic-novel", "comics", "manga"],
    "poetry": ["poetry", "poems"],
    "non-fiction": ["non-fiction", "nonfiction"],
    "biography": ["biography", "biographies", "autobiography"],
    "memoir": ["memoir", "memoirs"],
    "history": ["history", "world-history"],
    "science": ["science", "popular-science", "physics", "biology"],
    "psychology": ["psychology", "neuroscience"],
    "philosophy": ["philosophy"],
    "self-help": ["self-help", "self-improvement", "personal-development", "productivity"],
    "business": ["business", "economics", "finance", "leadership"],
    "religion": ["religion", "spirituality", "christian", "theology"],
}

# Shelves that describe the reader, not the book.
NOISE_PATTERNS = re.compile(
    r"(to-read|currently|read-in|read-\d|\d{4}|favou?rite|(^|-)own(ed)?($|-)|kindle|ebook|e-book|"
    r"library|wish|audio|shelf|default|(^|-)buy|borrow|book-club|tbr|dnf|re-read|series|"
    r"(^|-)have($|-)|^my-|^i-)"
)


def _compile(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"(^|[-_]){re.escape(keyword)}($|[-_])")


_GENRE_PATTERNS = {g: [_compile(k) for k in kws] for g, kws in GENRE_KEYWORDS.items()}


def genres_for_tag(tag: str) -> list[str]:
    tag = tag.lower().strip()
    return [g for g, pats in _GENRE_PATTERNS.items() if any(p.search(tag) for p in pats)]


def is_noise_tag(tag: str) -> bool:
    return bool(NOISE_PATTERNS.search(tag.lower()))


def assign_genres(tag_counts: dict[str, int], max_genres: int = 3, min_share: float = 0.1) -> list[str]:
    """Pick the dominant genres for a book from its {tag: count} shelf statistics."""
    scores: dict[str, int] = {}
    for tag, count in tag_counts.items():
        for g in genres_for_tag(tag):
            scores[g] = scores.get(g, 0) + count
    if not scores:
        return []
    top = max(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [g for g, s in ranked[:max_genres] if s >= min_share * top]
