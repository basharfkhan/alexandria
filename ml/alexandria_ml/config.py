from __future__ import annotations

import os
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("ALEXANDRIA_DATA_DIR", ML_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = Path(os.getenv("ALEXANDRIA_ARTIFACT_DIR", ML_ROOT / "artifacts"))

# Must match CONTENT_DIM / CF_DIM in the API (they size the pgvector columns).
CONTENT_DIM = 384
CF_DIM = 64

POSITIVE_RATING = 4  # ratings >= this count as "liked"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
