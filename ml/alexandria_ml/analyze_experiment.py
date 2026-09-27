"""Analysis for experiment 001, following docs/experiments/001-ranker-ab.md.

    python -m alexandria_ml.analyze_experiment \
        --control artifacts/exp001_control.jsonl \
        --treatment artifacts/exp001_treatment.jsonl

Reports the effect size and its confidence interval as the headline, with the p-value secondary.
Runs the validity checks the pre-registration commits to first, because an imbalanced or
overlapping split would make the rest meaningless.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, fmean, stdev

_NORM = NormalDist()


@dataclass
class Arm:
    name: str
    rates: list[float]
    readers: list[dict]

    @property
    def n(self) -> int:
        return len(self.rates)

    @property
    def mean(self) -> float:
        return fmean(self.rates)

    @property
    def sd(self) -> float:
        return stdev(self.rates)

    @property
    def impressions(self) -> int:
        return sum(r["shown"] for r in self.readers)


def load_arm(path: str, name: str) -> Arm:
    with open(path, encoding="utf-8") as f:
        readers = [json.loads(line) for line in f]
    return Arm(name=name, rates=[r["positive_rate"] for r in readers], readers=readers)


def welch(a: Arm, b: Arm) -> tuple[float, float, float]:
    """Welch's t-test. Returns (t, dof, two-sided p). Does not assume equal variances."""
    va, vb = a.sd**2 / a.n, b.sd**2 / b.n
    se = sqrt(va + vb)
    t = (b.mean - a.mean) / se
    dof = (va + vb) ** 2 / (va**2 / (a.n - 1) + vb**2 / (b.n - 1))
    # Normal approximation to the t distribution; at n in the hundreds the difference is negligible.
    p = 2 * (1 - _NORM.cdf(abs(t)))
    return t, dof, p


def bootstrap_ci(a: Arm, b: Arm, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over readers, resampling each arm independently."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        ra = [rng.choice(a.rates) for _ in range(a.n)]
        rb = [rng.choice(b.rates) for _ in range(b.n)]
        diffs.append(fmean(rb) - fmean(ra))
    diffs.sort()
    lo = diffs[int(0.025 * resamples)]
    hi = diffs[int(0.975 * resamples)]
    return lo, hi


def check_validity(a: Arm, b: Arm) -> list[str]:
    """Pre-registered checks that must pass before the result means anything."""
    problems = []

    overlap = {r["user_idx"] for r in a.readers} & {r["user_idx"] for r in b.readers}
    if overlap:
        problems.append(f"FAIL: {len(overlap)} readers appear in BOTH arms; arms must be disjoint")

    if abs(a.n - b.n) > 0.05 * max(a.n, b.n):
        problems.append(f"WARN: arm sizes differ by more than 5% ({a.n} vs {b.n})")

    for arm in (a, b):
        shown = {r["shown"] for r in arm.readers}
        if len(shown) > 1:
            problems.append(f"WARN: {arm.name} readers saw differing impression counts {sorted(shown)}")

    return problems


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", required=True)
    p.add_argument("--treatment", required=True)
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    a = load_arm(args.control, "control")
    b = load_arm(args.treatment, "treatment")

    print("=" * 62)
    print("EXPERIMENT 001: second-stage ranker vs stage-1 retrieval")
    print("=" * 62)

    problems = check_validity(a, b)
    print("\nVALIDITY CHECKS")
    if problems:
        for line in problems:
            print(f"  {line}")
    else:
        print("  arms disjoint, balanced, equal exposure per reader: OK")

    print("\nARMS")
    print(f"{'':>12} {'readers':>9} {'impressions':>13} {'mean rate':>11} {'sd':>8}")
    for arm in (a, b):
        print(f"{arm.name:>12} {arm.n:>9,} {arm.impressions:>13,} {arm.mean:>11.4f} {arm.sd:>8.4f}")

    diff = b.mean - a.mean
    lo, hi = bootstrap_ci(a, b, args.resamples, args.seed)
    t, dof, pval = welch(a, b)

    print("\nEFFECT (treatment minus control)")
    print(f"  absolute difference   {diff:+.4f}")
    print(f"  relative to control   {diff / a.mean:+.1%}")
    print(f"  95% bootstrap CI      [{lo:+.4f}, {hi:+.4f}]")
    print(f"                        [{lo / a.mean:+.1%}, {hi / a.mean:+.1%}] relative")
    print(f"  Welch t               {t:.3f}  (dof {dof:.0f})")
    print(f"  two-sided p           {pval:.4f}")

    crosses_zero = lo <= 0 <= hi
    print("\nREADING")
    if crosses_zero:
        print("  The interval includes zero, so no effect is demonstrated. Given the experiment")
        print("  was powered for a 16% relative lift, this rules out effects of that size, not")
        print("  small ones. Report as 'no effect of 16% or larger detected'.")
    else:
        direction = "higher" if diff > 0 else "LOWER"
        print(f"  Treatment is {direction} than control and the interval excludes zero.")
        print("  Check the guardrails before concluding anything about shipping.")


if __name__ == "__main__":
    main()
