"""Model registry and promotion gate.

Retraining is only safe if a worse model cannot reach production. Every trained model is
recorded in `ml/model_registry.json` with its metrics; a new model is promoted only if it does
not regress against the one currently live, on the metrics that matter:

* `rerank_served.ndcg@20`    - ranking quality of exactly what the API returns
* `rerank_served.recall@20`  - did we surface the books they went on to love
* `rerank_cold5.ndcg@20`     - new readers, who see the app at its worst
* `rerank_served.coverage@20`- how much of the catalog is reachable (guards popularity drift)

Small run-to-run differences are expected (negative sampling, tie-breaking), so each metric may
fall by a tolerance before it counts as a regression.

That comparison alone cannot catch slow decay: ten retrains losing 1.9% each would all pass a 2%
gate. Every candidate is therefore *also* compared against a pinned reference model (the first
promoted one, re-pinnable with `promote --set-reference`) on a wider tolerance, which bounds how far
the model may wander from a known-good baseline no matter how many steps it takes to get there.
Prediction and input drift are measured separately in `drift.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from alexandria_ml.config import ML_ROOT

REGISTRY_PATH = ML_ROOT / "model_registry.json"

# metric path -> how much it may drop (relative) before the model is rejected
GATE_METRICS = {
    "rerank_rated_only.ndcg@20": 0.02,
    "rerank_rated_only.recall@20": 0.02,
    "rerank_cold5.ndcg@20": 0.05,
    "rerank_rated_only.coverage@20": 0.10,
}
# Cumulative budget against the pinned reference model. Wider than the per-run gate because a
# model is allowed to wander a little over many retrains, just not indefinitely.
REFERENCE_GATE = {
    "rerank_rated_only.ndcg@20": 0.05,
    "rerank_rated_only.recall@20": 0.05,
    "rerank_cold5.ndcg@20": 0.10,
    "rerank_rated_only.coverage@20": 0.20,
}
# Rows are looked up in order: the like-for-like row, then the served row, then the stage-1 row
# (so a run without a ranker, or from before these rows existed, is still comparable).
ROW_FALLBACKS = {
    "rerank_rated_only": ["rerank_served", "hybrid_served"],
    "rerank_served": ["hybrid_served"],
    "rerank_cold5": ["hybrid_cold5"],
}


def metric(metrics: dict, path: str) -> float | None:
    """Look up "row.metric", trying the fallback rows when that row is missing."""
    row, name = path.split(".")
    for candidate in [row, *ROW_FALLBACKS.get(row, [])]:
        if candidate in metrics:
            return metrics[candidate].get(name)
    return None


@dataclass
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else "REJECT"
        lines = [f"{head}: " + ("; ".join(self.reasons) if self.reasons else "no blocking regressions")]
        for c in self.comparisons:
            arrow = "ok  " if c["ok"] else "FAIL"  # ASCII: Windows consoles default to cp1252
            live = "n/a" if c["live"] is None else f"{c['live']:.4f}"
            against = "ref  " if c["metric"].startswith("ref ") else "live "
            lines.append(
                f"  {arrow} {c['metric']:<32} {against}{live} -> candidate {c['candidate']:.4f} ({c['change']:+.1%})"
            )
        return "\n".join(lines)


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"production": None, "history": []}


def live_model(registry: dict) -> dict | None:
    return _entry(registry, registry.get("production"))


def reference_model(registry: dict) -> dict | None:
    """The pinned known-good model every candidate is measured against, not just the last one."""
    return _entry(registry, registry.get("reference"))


def _entry(registry: dict, version: str | None) -> dict | None:
    return next((m for m in registry.get("history", []) if m["model_version"] == version), None)


def evaluate_candidate(
    registry: dict,
    manifest: dict,
    gate: dict[str, float] | None = None,
    reference_gate: dict[str, float] | None = None,
) -> Decision:
    """Compare a freshly trained model against the live one, and against the pinned reference."""
    live = live_model(registry)
    candidate_metrics = manifest.get("metrics", {})
    if live is None:
        return Decision(promote=True, reasons=["no model in production yet"])

    decision = Decision(promote=True)
    _check(decision, candidate_metrics, live, gate or GATE_METRICS, label="")
    reference = reference_model(registry)
    if reference is not None and reference["model_version"] != live["model_version"]:
        _check(decision, candidate_metrics, reference, reference_gate or REFERENCE_GATE, label="ref ")
    return decision


def _check(decision: Decision, candidate: dict, against: dict, gate: dict[str, float], label: str) -> None:
    """Apply one gate, recording a comparison per metric and a reason for each failure."""
    for path, tolerance in gate.items():
        new = metric(candidate, path)
        old = metric(against.get("metrics", {}), path)
        if new is None:
            decision.promote = False
            decision.reasons.append(f"candidate is missing {path}")
            continue
        if old is None:
            continue
        change = (new - old) / old if old else 0.0
        ok = new >= old * (1 - tolerance)
        decision.comparisons.append(
            {"metric": f"{label}{path}", "live": old, "candidate": new, "change": change, "ok": ok}
        )
        if not ok:
            decision.promote = False
            where = f"drifted {abs(change):.1%} from reference" if label else f"fell {abs(change):.1%}"
            decision.reasons.append(f"{path} {where} (limit {tolerance:.0%})")


def record(
    registry: dict, manifest: dict, decision: Decision, promoted: bool, path: Path = REGISTRY_PATH
) -> dict:
    """Append this run to the registry and, when promoted, make it the production model."""
    entry = {
        "model_version": manifest["model_version"],
        "trained_at": manifest.get("created_at", datetime.now(UTC).isoformat()),
        "dataset": manifest.get("dataset"),
        "embedder": manifest.get("embedder"),
        "n_books": manifest.get("n_books"),
        "ranker_trees": (manifest.get("ranker") or {}).get("best_iteration"),
        "metrics": manifest.get("metrics", {}),
        "fingerprint": manifest.get("fingerprint"),
        "inputs": manifest.get("inputs"),
        "promoted": promoted,
        "decision": {"promote": decision.promote, "reasons": decision.reasons},
    }
    registry.setdefault("history", []).append(entry)
    registry["history"] = registry["history"][-50:]
    if promoted:
        registry["production"] = entry["model_version"]
        registry["promoted_at"] = entry["trained_at"]
        # The first promoted model becomes the baseline every later candidate is measured against.
        registry.setdefault("reference", entry["model_version"])
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry
