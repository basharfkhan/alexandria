import pytest

from alexandria_ml.power import (
    binomial_sd,
    detectable_effect,
    estimate_from_pilot,
    required_n,
)


def test_required_n_matches_the_textbook_formula():
    # Two-sample, alpha 0.05, power 0.80: n = 2 * (1.96 + 0.8416)^2 * sd^2 / delta^2
    # sd 0.15, delta 0.058 -> ~105, plus the small-sample correction.
    assert required_n(sd=0.15, mde_abs=0.058) == pytest.approx(107, abs=2)


def test_smaller_effects_need_more_readers():
    big = required_n(sd=0.15, mde_abs=0.078)
    small = required_n(sd=0.15, mde_abs=0.039)
    assert small > big
    # Halving the effect quadruples the sample, since n scales with 1/delta^2.
    assert small == pytest.approx(4 * big, rel=0.1)


def test_more_variance_needs_more_readers():
    assert required_n(sd=0.25, mde_abs=0.058) > required_n(sd=0.15, mde_abs=0.058)


def test_higher_power_needs_more_readers():
    assert required_n(sd=0.15, mde_abs=0.058, power=0.95) > required_n(sd=0.15, mde_abs=0.058, power=0.80)


def test_detectable_effect_inverts_required_n():
    n = required_n(sd=0.15, mde_abs=0.058)
    assert detectable_effect(sd=0.15, n_per_arm=n) == pytest.approx(0.058, rel=0.05)


def test_binomial_sd_is_below_realistic_between_reader_spread():
    """The floor must be optimistic, or it would not be a floor."""
    assert binomial_sd(0.39, impressions_per_reader=60) < 0.15


def test_estimate_from_pilot_computes_sample_sd():
    est = estimate_from_pilot([0.2, 0.4, 0.6])
    assert est.n_readers == 3
    assert est.mean_rate == pytest.approx(0.4)
    assert est.sd_rate == pytest.approx(0.2)  # sample sd, n-1 denominator


def test_estimate_from_pilot_rejects_a_single_reader():
    with pytest.raises(ValueError, match="at least 2 readers"):
        estimate_from_pilot([0.4])


def test_zero_or_negative_effect_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        required_n(sd=0.15, mde_abs=0.0)
