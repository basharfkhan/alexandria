"""The reader pool must split into arms that never share a reader.

Overlap between arms would break the independence assumption behind the between-arm
comparison, so this is checked rather than assumed.
"""

import numpy as np


def _arm_slice(candidates, seed: int, readers: int, arms: int, arm_index: int):
    """Mirrors the pool split in simulate_traffic.main."""
    pool = np.random.default_rng(seed).choice(candidates, size=readers * arms, replace=False)
    return pool[arm_index * readers : (arm_index + 1) * readers]


CANDIDATES = np.arange(5000)


def test_arms_never_share_a_reader():
    a = _arm_slice(CANDIDATES, seed=7, readers=300, arms=2, arm_index=0)
    b = _arm_slice(CANDIDATES, seed=7, readers=300, arms=2, arm_index=1)
    assert len(a) == len(b) == 300
    assert not set(a.tolist()) & set(b.tolist())


def test_each_arm_is_internally_unique():
    a = _arm_slice(CANDIDATES, seed=7, readers=300, arms=2, arm_index=0)
    assert len(set(a.tolist())) == 300


def test_split_is_reproducible_across_runs():
    """Both arms run as separate processes, so the same seed must rebuild the same pool."""
    first = _arm_slice(CANDIDATES, seed=7, readers=300, arms=2, arm_index=1)
    second = _arm_slice(CANDIDATES, seed=7, readers=300, arms=2, arm_index=1)
    assert first.tolist() == second.tolist()


def test_a_different_seed_draws_a_different_pool():
    a = _arm_slice(CANDIDATES, seed=7, readers=50, arms=2, arm_index=0)
    b = _arm_slice(CANDIDATES, seed=8, readers=50, arms=2, arm_index=0)
    assert a.tolist() != b.tolist()


def test_single_arm_default_matches_a_plain_draw():
    """--arms 1 must behave exactly as the script did before arms existed."""
    split = _arm_slice(CANDIDATES, seed=3, readers=30, arms=1, arm_index=0)
    plain = np.random.default_rng(3).choice(CANDIDATES, size=30, replace=False)
    assert split.tolist() == plain.tolist()
