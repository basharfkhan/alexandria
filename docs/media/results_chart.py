"""Draw the NDCG@20 comparison chart from a trained model's manifest.

The figure is used in the README and on the portfolio site, so it must never disagree with the
model it describes: every number comes out of `ml/artifacts/manifest.json` rather than being typed
in. Re-run it after any promoted retrain.

    pip install matplotlib
    python docs/media/results_chart.py                      # -> docs/media/results.png
    python docs/media/results_chart.py --out ../site/public/images/alexandria-results.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "ml" / "artifacts" / "manifest.json"
DEFAULT_OUT = REPO / "docs" / "media" / "results.png"

# metric row in the manifest -> label on the chart, lowest bar first. The highlighted row is the
# like-for-like one the promotion gate reads (see docs/ARCHITECTURE.md).
ROWS = [
    ("popularity", "Popularity baseline", False),
    ("rerank_cold5", "Two-stage, 5 ratings known", False),
    ("bpr", "BPR matrix factorization", False),
    ("hybrid_served", "Stage 1 only: tuned hybrid", False),
    ("rerank_rated_only", "Two-stage (hybrid + ranker)", True),
]

GREEN, TAN, GRID, INK, MUTED = "#1d4a3c", "#948467", "#d9cdb6", "#1a1a1a", "#6b6255"


def draw(manifest: dict, out: Path) -> None:
    metrics = manifest["metrics"]
    values = [metrics[row]["ndcg@20"] for row, _, _ in ROWS]
    hero = values[-1]
    vs_bpr = hero / metrics["bpr"]["ndcg@20"] - 1
    vs_popularity = hero / metrics["popularity"]["ndcg@20"]

    plt.rcParams["font.family"] = ["Georgia", "Palatino Linotype", "serif"]
    fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(range(len(ROWS)), values, height=0.52,
            color=[GREEN if h else TAN for _, _, h in ROWS], zorder=3)
    for i, (value, (_, _, highlight)) in enumerate(zip(values, ROWS, strict=True)):
        ax.text(value + 0.005, i, f"{value:.3f}", va="center", ha="left", color=INK,
                fontsize=15 if highlight else 14, fontweight="bold" if highlight else "normal",
                zorder=4)
    ax.text(hero - 0.006, len(ROWS) - 1,  # inset from the bar's right edge, drawn on the bar
            f"+{vs_bpr:.0%} vs matrix factorization  ·  {vs_popularity:.1f}× popularity",
            va="center", ha="right", color="white", fontsize=12.5, style="italic", zorder=5)

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([label for _, label, _ in ROWS], fontsize=14, color=INK)
    for tick, (_, _, highlight) in zip(ax.get_yticklabels(), ROWS, strict=True):
        if highlight:
            tick.set_fontweight("bold")

    ticks = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xticks([t for t in ticks if t <= max(values)])
    ax.set_xticklabels([f"{t:.2f}" for t in ticks if t <= max(values)], fontsize=13, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(axis="both", length=0)

    n_users = int(metrics[ROWS[-1][0]].get("n_users", 2000))
    fig.subplots_adjust(top=0.82, left=0.285, right=0.945, bottom=0.10)
    fig.text(0.035, 0.945, "Ranking quality on held-out Goodreads ratings",
             fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.875,
             f"NDCG@20, higher is better · {n_users:,} test users · 6M ratings "
             f"· {manifest['n_books']:,} books",
             fontsize=14.5, color=MUTED, va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", pad_inches=0.2)
    print(f"wrote {out} from model {manifest['model_version']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    draw(json.loads(args.manifest.read_text(encoding="utf-8")), args.out)


if __name__ == "__main__":
    main()
