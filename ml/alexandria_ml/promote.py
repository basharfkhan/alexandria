"""Decide whether a freshly trained model may go to production.

    python -m alexandria_ml.promote --artifacts artifacts            # decide and record
    python -m alexandria_ml.promote --artifacts artifacts --dry-run  # decide only

Exit code 0 = promote, 1 = rejected (the scheduled retraining workflow fails the run so the
regression is visible, and the production model stays untouched).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from alexandria_ml.config import ARTIFACT_DIR
from alexandria_ml.registry import REGISTRY_PATH, evaluate_candidate, load_registry, record

log = logging.getLogger("alexandria.promote")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    p.add_argument("--registry", default=str(REGISTRY_PATH))
    p.add_argument("--dry-run", action="store_true", help="decide without writing the registry")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = json.loads((Path(args.artifacts) / "manifest.json").read_text(encoding="utf-8"))
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    decision = evaluate_candidate(registry, manifest)

    print(f"candidate {manifest['model_version']} vs production {registry.get('production') or 'none'}")
    print(decision.summary())

    if not args.dry_run:
        record(registry, manifest, decision, promoted=decision.promote, path=registry_path)

    # Expose the decision to later workflow steps.
    if summary_path := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(f"### Model promotion\n\n```\n{decision.summary()}\n```\n")
    if output_path := os.getenv("GITHUB_OUTPUT"):
        with Path(output_path).open("a", encoding="utf-8") as fh:
            fh.write(f"promote={str(decision.promote).lower()}\nmodel_version={manifest['model_version']}\n")
    return 0 if decision.promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
