"""Download the Goodbooks-10k dataset (10k books, ~6M ratings from ~53k Goodreads users).

Source: https://github.com/zygmuntz/goodbooks-10k (CC BY-SA 4.0).
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

from alexandria_ml.config import RAW_DIR

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
FILES = ["books.csv", "ratings.csv", "book_tags.csv", "tags.csv"]


def download_goodbooks(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw_dir / name
        if target.exists() and not force:
            log.info("✓ %s already present", name)
            continue
        log.info("↓ downloading %s", name)
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(BASE_URL + name, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(target)
    return raw_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_goodbooks()
