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
    "rerank_served.ndcg@20": 0.02,
    "rerank_served.recall@20": 0.02,
    "rerank_cold5.ndcg@20": 0.05,
    "rerank_served.coverage@20": 0.10,
}
# A model without a ranker is compared on the stage-1 rows instead.
STAGE1_FALLBACK = {"rerank_served": "hybrid_served", "rerank_cold5": "hybrid_cold5"}


def metric(metrics: dict, path: str) -> float | None:
    """Look up "row.metric", falling back to the stage-1 row when there is no ranker."""
    row, name = path.split(".")
    if row not in metrics and row in STAGE1_FALLBACK:
        row = STAGE1_FALLBACK[row]
    return metrics.get(row, {}).get(name)


@dataclass
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else "REJECT"
        lines = [f"{head}: " + ("; ".join(self.reasons) if self.reasons else "no blocking regressions")]
        for c in self.comparisons:
            arrow = "✓" if c["ok"] else "✗"
            live = "n/a" if c["live"] is None else f"{c['live']:.4f}"
            lines.append(f"  {arrow} {c['metric']:<28} live {live} -> candidate {c['candidate']:.4f} ({c['change']:+.1%})")
        return "\n".join(lines)


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"production": None, "history": []}


def live_model(registry: dict) -> dict | None:
    version = registry.get("production")
    return next((m for m in registry.get("history", []) if m["model_version"] == version), None)


def evaluate_candidate(registry: dict, manifest: dict, gate: dict[str, float] | None = None) -> Decision:
    """Compare a freshly trained model against the live one."""
    gate = gate or GATE_METRICS
    live = live_model(registry)
    candidate_metrics = manifest.get("metrics", {})
    if live is None:
        return Decision(promote=True, reasons=["no model in production yet"])

    decision = Decision(promote=True)
    for path, tolerance in gate.items():
        new = metric(candidate_metrics, path)
        old = metric(live.get("metrics", {}), path)
        if new is None:
            decision.promote = False
            decision.reasons.append(f"candidate is missing {path}")
            continue
        change = (new - old) / old if old else 0.0
        ok = old is None or new >= old * (1 - tolerance)
        decision.comparisons.append(
            {"metric": path, "live": old, "candidate": new, "change": change, "ok": ok}
        )
        if not ok:
            decision.promote = False
            decision.reasons.append(f"{path} fell {abs(change):.1%} (limit {tolerance:.0%})")
    return decision


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
        "promoted": promoted,
        "decision": {"promote": decision.promote, "reasons": decision.reasons},
    }
    registry.setdefault("history", []).append(entry)
    registry["history"] = registry["history"][-50:]
    if promoted:
        registry["production"] = entry["model_version"]
        registry["promoted_at"] = entry["trained_at"]
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry
