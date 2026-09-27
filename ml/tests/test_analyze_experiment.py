"""Tests for the experiment analysis.

The statistics are checked against cases with a known answer, and the validity checks are
checked by feeding them data that should fail.
"""

import json

import pytest

from alexandria_ml.analyze_experiment import Arm, bootstrap_ci, check_validity, load_arm, welch


def _arm(name: str, rates: list[float], user_idx: list[int] | None = None) -> Arm:
    idx = user_idx if user_idx is not None else list(range(len(rates)))
    readers = [{"shown": 60, "positive_rate": r, "user_idx": i} for r, i in zip(rates, idx, strict=True)]
    return Arm(name=name, rates=rates, readers=readers)


def test_identical_arms_show_no_effect():
    rates = [0.1, 0.2, 0.3, 0.4] * 10
    a, b = _arm("control", rates), _arm("treatment", list(rates), user_idx=list(range(100, 140)))
    t, _, p = welch(a, b)
    assert t == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_a_clear_shift_is_detected():
    a = _arm("control", [0.20] * 50 + [0.25] * 50)
    b = _arm("treatment", [0.40] * 50 + [0.45] * 50, user_idx=list(range(200, 300)))
    _, _, p = welch(a, b)
    assert p < 0.001


def test_bootstrap_interval_brackets_the_observed_difference():
    a = _arm("control", [0.1, 0.2, 0.3] * 20)
    b = _arm("treatment", [0.3, 0.4, 0.5] * 20, user_idx=list(range(500, 560)))
    lo, hi = bootstrap_ci(a, b, resamples=2000, seed=1)
    observed = b.mean - a.mean
    assert lo < observed < hi
    assert lo > 0  # a real shift should not straddle zero


def test_bootstrap_interval_straddles_zero_when_arms_match():
    rates = [0.1, 0.2, 0.3, 0.4, 0.5] * 12
    a = _arm("control", rates)
    b = _arm("treatment", list(rates), user_idx=list(range(900, 960)))
    lo, hi = bootstrap_ci(a, b, resamples=2000, seed=1)
    assert lo <= 0 <= hi


def test_validity_check_catches_overlapping_arms():
    a = _arm("control", [0.2, 0.3], user_idx=[1, 2])
    b = _arm("treatment", [0.4, 0.5], user_idx=[2, 3])  # reader 2 in both
    problems = check_validity(a, b)
    assert any("BOTH arms" in p for p in problems)


def test_validity_check_passes_for_a_clean_split():
    a = _arm("control", [0.2, 0.3], user_idx=[1, 2])
    b = _arm("treatment", [0.4, 0.5], user_idx=[3, 4])
    assert check_validity(a, b) == []


def test_validity_check_flags_imbalanced_arms():
    a = _arm("control", [0.2] * 100, user_idx=list(range(100)))
    b = _arm("treatment", [0.3] * 50, user_idx=list(range(100, 150)))
    assert any("arm sizes differ" in p for p in check_validity(a, b))


def test_load_arm_reads_jsonl(tmp_path):
    path = tmp_path / "arm.jsonl"
    rows = [{"shown": 60, "positive_rate": 0.25, "user_idx": 1},
            {"shown": 60, "positive_rate": 0.35, "user_idx": 2}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    arm = load_arm(str(path), "control")
    assert arm.n == 2
    assert arm.impressions == 120
    assert arm.mean == pytest.approx(0.30)
