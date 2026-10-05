"""Decide whether a freshly trained model may go to production.

    python -m alexandria_ml.promote --artifacts artifacts            # decide and record
    python -m alexandria_ml.promote --artifacts artifacts --dry-run  # decide only

    python -m alexandria_ml.promote --set-reference 20260925-194856   # re-pin the baseline

Exit code 0 = promote, 1 = rejected (the scheduled retraining workflow fails the run so the
regression is visible, and the production model stays untouched).

Alongside the gate this prints drift against the live model: whether the same probe readers are
still shown the same books, and whether the training inputs changed shape. Those are warnings, not
blockers, because a better model is allowed to change its mind.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from alexandria_ml.config import ARTIFACT_DIR
from alexandria_ml.drift import compare, compare_inputs, summarise
from alexandria_ml.registry import (
    REGISTRY_PATH,
    evaluate_candidate,
    live_model,
    load_registry,
    record,
)

log = logging.getLogger("alexandria.promote")


def measure_drift(registry: dict, manifest: dict) -> dict | None:
    """Prediction and input drift against the live model, or None if there is nothing to compare."""
    live = live_model(registry)
    if live is None:
        return None
    drift = {}
    if (previous := live.get("fingerprint")) and (current := manifest.get("fingerprint")):
        drift["predictions"] = compare(current, previous)
    if (previous := live.get("inputs")) and (current := manifest.get("inputs")):
        drift["inputs"] = compare_inputs(current, previous)
    return drift or None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    p.add_argument("--registry", default=str(REGISTRY_PATH))
    p.add_argument("--dry-run", action="store_true", help="decide without writing the registry")
    p.add_argument("--set-reference", metavar="VERSION",
                   help="re-pin the baseline that cumulative drift is measured against, then exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    registry_path = Path(args.registry)
    registry = load_registry(registry_path)

    if args.set_reference:
        versions = {m["model_version"] for m in registry.get("history", [])}
        if args.set_reference not in versions:
            print(f"no such model in the registry: {args.set_reference}")
            return 1
        registry["reference"] = args.set_reference
        registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        print(f"reference model pinned to {args.set_reference}")
        return 0

    manifest = json.loads((Path(args.artifacts) / "manifest.json").read_text(encoding="utf-8"))
    decision = evaluate_candidate(registry, manifest)
    drift = measure_drift(registry, manifest)

    print(f"candidate {manifest['model_version']} vs production {registry.get('production') or 'none'}"
          f" (reference {registry.get('reference') or 'none'})")
    print(decision.summary())
    print(summarise(drift))

    if not args.dry_run:
        record(registry, manifest, decision, promoted=decision.promote, path=registry_path)

    # Expose the decision to later workflow steps.
    if summary_path := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(f"### Model promotion\n\n```\n{decision.summary()}\n\n{summarise(drift)}\n```\n")
    if output_path := os.getenv("GITHUB_OUTPUT"):
        with Path(output_path).open("a", encoding="utf-8") as fh:
            fh.write(f"promote={str(decision.promote).lower()}\nmodel_version={manifest['model_version']}\n")
    return 0 if decision.promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
