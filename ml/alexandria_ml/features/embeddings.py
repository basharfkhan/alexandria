"""Content embeddings for books.

* ``sbert``  - sentence-transformers MiniLM (384-d semantic embeddings). Best quality.
* ``tfidf``  - TF-IDF + truncated SVD, padded to the same width. No model download,
               used in CI and as a fallback when sentence-transformers is absent.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from alexandria_ml.config import CONTENT_DIM, SBERT_MODEL
from alexandria_ml.data.preprocess import book_text

log = logging.getLogger(__name__)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def embed_sbert(texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    assert emb.shape[1] == CONTENT_DIM, f"expected {CONTENT_DIM}-d embeddings, got {emb.shape[1]}"
    return emb.astype(np.float32)


def embed_tfidf(texts: Sequence[str], dim: int = CONTENT_DIM, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    n_comp = max(1, min(dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    reduced = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(tfidf)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    out[:, :n_comp] = reduced
    return _normalize(out)


def embed_books(texts: Sequence[str], method: str = "auto") -> tuple[np.ndarray, str]:
    if method in ("auto", "sbert"):
        try:
            return embed_sbert(texts), "sbert"
        except ImportError:
            if method == "sbert":
                raise
            log.warning("sentence-transformers not installed; falling back to TF-IDF embeddings")
    return embed_tfidf(texts), "tfidf"


def content_embeddings(books, cache_dir: Path, method: str = "auto") -> tuple[np.ndarray, str]:
    """Two-field book embedding: mean of a metadata embedding and a description embedding.

    Chosen on the validation split (see docs/ARCHITECTURE.md): description-only embeddings gave
    better "similar books" but slightly lower ranking accuracy; metadata-only embeddings matched
    titles word-for-word ("The Martian" -> "The Martian Chronicles", "The Humans"). The 50/50 mean
    kept validation accuracy equal to metadata-only while keeping most of the semantic gains.
    Books without descriptions fall back to the metadata embedding.
    """
    meta_texts = books.apply(lambda row: book_text(row, with_description=False), axis=1).tolist()
    meta, used = cached_embeddings(meta_texts, cache_dir, method)
    if "description" not in books or not books.description.map(lambda d: isinstance(d, str)).any():
        return meta, used
    full, _ = cached_embeddings(books.apply(book_text, axis=1).tolist(), cache_dir, used)
    return _normalize(_normalize(meta) + _normalize(full)), used


def cached_embeddings(texts: Sequence[str], cache_dir: Path, method: str = "auto") -> tuple[np.ndarray, str]:
    """Embed once per distinct set of book texts; MiniLM over 10k books takes minutes on CPU.

    The cache key hashes the texts, so changing how book text is built (e.g. genre mapping)
    automatically invalidates stale embeddings.
    """
    digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    for candidate in (["sbert", "tfidf"] if method == "auto" else [method]):
        path = cache_dir / f"content_{candidate}_{digest}.npy"
        if path.exists():
            log.info("loaded cached %s embeddings from %s", candidate, path)
            return np.load(path), candidate
    emb, used = embed_books(texts, method)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"content_{used}_{digest}.npy", emb)
    return emb, used
