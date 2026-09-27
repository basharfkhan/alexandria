"""Sample-size and minimum-detectable-effect calculations for the recommender experiments.

The unit of analysis is a *reader*, not an impression: impressions within one reader are
correlated, so treating them as independent observations inflates significance. Each reader
contributes one number (their positive-feedback rate), and the two arms are compared as two
samples of reader-level rates.

    python -m alexandria_ml.power --baseline 0.39 --sd 0.18
    python -m alexandria_ml.power --baseline 0.39 --sd 0.18 --n-per-arm 100

Feed it pilot estimates from `estimate_from_pilot` once a pilot run exists; the grid it prints
without them is only as good as the numbers you pass in.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist

_NORM = NormalDist()


def _z(p: float) -> float:
    return _NORM.inv_cdf(p)


def required_n(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Readers per arm needed to detect `mde_abs` at the given alpha and power (two-sided).

    Standard two-sample comparison of means. Uses the normal approximation, then adds a small
    correction for using a t distribution, which matters at the sample sizes we can afford.
    """
    if mde_abs <= 0:
        raise ValueError("mde_abs must be positive")
    n = 2 * ((_z(1 - alpha / 2) + _z(power)) ** 2) * (sd**2) / (mde_abs**2)
    return ceil(n) + 2  # +2 approximates the t-vs-z penalty for small samples


def detectable_effect(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """The smallest absolute difference detectable with `n_per_arm` readers per arm."""
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    return (_z(1 - alpha / 2) + _z(power)) * sqrt(2 / n_per_arm) * sd


def binomial_sd(p: float, impressions_per_reader: int) -> float:
    """Within-reader sampling SD only, assuming every reader behaves identically.

    This is a *floor*, not an estimate. Real readers differ from each other far more than
    coin flips differ, so the true between-reader SD is larger and the real sample size is
    bigger than what this implies. Useful only as a sanity bound.
    """
    return sqrt(p * (1 - p) / impressions_per_reader)


@dataclass
class PilotEstimate:
    n_readers: int
    mean_rate: float
    sd_rate: float

    def __str__(self) -> str:
        return f"{self.n_readers} readers, mean rate {self.mean_rate:.3f}, sd {self.sd_rate:.3f}"


def estimate_from_pilot(per_reader_rates: list[float]) -> PilotEstimate:
    """Mean and sample SD of reader-level rates from a pilot run."""
    n = len(per_reader_rates)
    if n < 2:
        raise ValueError("need at least 2 readers to estimate a standard deviation")
    mean = sum(per_reader_rates) / n
    var = sum((r - mean) ** 2 for r in per_reader_rates) / (n - 1)
    return PilotEstimate(n_readers=n, mean_rate=mean, sd_rate=sqrt(var))


def _grid(baseline: float, sd: float, alpha: float, power: float) -> None:
    print(f"\nbaseline positive rate {baseline:.0%}, reader-level sd {sd:.3f}, "
          f"alpha {alpha}, power {power:.0%}\n")
    print(f"{'relative lift':>14} {'absolute':>10} {'readers/arm':>13} {'total':>8}")
    print("-" * 48)
    for rel in (0.05, 0.10, 0.15, 0.20, 0.30):
        abs_mde = baseline * rel
        n = required_n(sd, abs_mde, alpha, power)
        print(f"{rel:>13.0%} {abs_mde:>10.3f} {n:>13,} {2 * n:>8,}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=float, required=True, help="control arm positive-feedback rate")
    p.add_argument("--sd", type=float, help="reader-level sd of that rate (from a pilot)")
    p.add_argument("--impressions-per-reader", type=int, default=60,
                   help="used only for the binomial floor when --sd is omitted")
    p.add_argument("--n-per-arm", type=int, help="if given, report the detectable effect at this n instead")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    args = p.parse_args(argv)

    sd = args.sd
    if sd is None:
        sd = binomial_sd(args.baseline, args.impressions_per_reader)
        print(f"\nno --sd given; using the binomial floor {sd:.3f} "
              f"({args.impressions_per_reader} impressions/reader).")
        print("This UNDERSTATES the real sample size. Run a pilot and pass --sd.")

    if args.n_per_arm:
        mde = detectable_effect(sd, args.n_per_arm, args.alpha, args.power)
        print(f"\nwith {args.n_per_arm} readers/arm (sd {sd:.3f}): detectable absolute "
              f"difference {mde:.3f} ({mde / args.baseline:.0%} relative)\n")
    else:
        _grid(args.baseline, sd, args.alpha, args.power)


if __name__ == "__main__":
    main()
