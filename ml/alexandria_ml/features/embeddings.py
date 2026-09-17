"""Content embeddings for books.

* ``sbert``  - sentence-transformers MiniLM (384-d semantic embeddings). Best quality.
* ``tfidf``  - TF-IDF + truncated SVD, padded to the same width. No model download,
               used in CI and as a fallback when sentence-transformers is absent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from alexandria_ml.config import CONTENT_DIM, SBERT_MODEL

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


def cached_embeddings(texts: Sequence[str], cache_dir: Path, method: str = "auto") -> tuple[np.ndarray, str]:
    """Embed once per dataset; MiniLM over 10k books takes minutes on CPU."""
    for candidate in (["sbert", "tfidf"] if method == "auto" else [method]):
        path = cache_dir / f"content_{candidate}.npy"
        if path.exists():
            emb = np.load(path)
            if len(emb) == len(texts):
                log.info("loaded cached %s embeddings from %s", candidate, path)
                return emb, candidate
    emb, used = embed_books(texts, method)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"content_{used}.npy", emb)
    return emb, used
