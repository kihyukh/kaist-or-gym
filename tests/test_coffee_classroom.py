"""Varied student starts are safe; policy comparisons keep one canonical pose."""

from itertools import product

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_classroom import (
    CUP_POSITION_JITTER,
    INITIAL_LAYOUT,
    POLICY_START_SEED,
    POT_POSITION_JITTER,
    classroom_layout,
    fixed_policy_layout,
    fresh_classroom_seed,
)
from kaist_rl_lab.envs import CoffeePouringEnv


def test_seed_reproduces_independent_varied_coordinates_without_mutating_defaults():
    original = {key: value.copy() for key, value in INITIAL_LAYOUT.items()}
    starts = []
    for seed in range(100):
        actual = classroom_layout(seed)
        assert actual == classroom_layout(seed)
        for name, jitter in (("cup_center", CUP_POSITION_JITTER), ("pot_center", POT_POSITION_JITTER)):
            for axis in range(2):
                assert abs(actual[name][axis] - INITIAL_LAYOUT[name][axis]) <= jitter[axis]
        starts.append(actual["cup_center"] + actual["pot_center"])
    assert classroom_layout(123) != classroom_layout(456)
    starts = np.asarray(starts)
    assert np.linalg.matrix_rank(starts - starts.mean(axis=0)) == 4
    assert np.all(np.ptp(starts, axis=0) > 1.5 * np.asarray(CUP_POSITION_JITTER + POT_POSITION_JITTER))
    actual["cup_center"][0] = 100
    assert INITIAL_LAYOUT == original


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "12", None])
def test_seed_must_be_an_unsigned_32_bit_integer(seed):
    with pytest.raises(ValueError, match="Classroom seed"):
        classroom_layout(seed)


def test_fresh_seed_can_be_saved_and_replayed():
    seed = fresh_classroom_seed()
    assert type(seed) is int and 0 <= seed < 2**32
    assert classroom_layout(seed) == classroom_layout(seed)


def test_policy_start_remains_canonical_and_returns_fresh_lists():
    expected = {"cup_center": [-0.28, 0.28], "pot_center": [0.26, 0.62]}
    assert POLICY_START_SEED == 7001
    assert fixed_policy_layout() == expected
    layout = fixed_policy_layout()
    layout["cup_center"][0] = 100
    assert fixed_policy_layout() == expected == INITIAL_LAYOUT


def test_edges_and_random_starts_are_reachable_upright_and_collision_free():
    layouts = [classroom_layout(seed) for seed in range(32)]
    for offsets in product((-1, 1), repeat=4):
        layouts.append({
            "cup_center": [INITIAL_LAYOUT["cup_center"][axis] + offsets[axis] * CUP_POSITION_JITTER[axis]
                           for axis in range(2)],
            "pot_center": [INITIAL_LAYOUT["pot_center"][axis] + offsets[axis + 2] * POT_POSITION_JITTER[axis]
                           for axis in range(2)],
        })
    env = CoffeePouringEnv(horizon=1, dt=1 / 32)
    try:
        for layout in layouts:
            observation, _ = env.reset(seed=0, options={**layout, "target_fill": 0.7})
            assert np.isfinite(observation).all()
            assert env.fill == env.spill == 0
            assert sum(env.joint_angles[:3]) == pytest.approx(0, abs=1e-12)
            assert sum(env.joint_angles[3:]) == pytest.approx(0, abs=1e-12)
    finally:
        env.close()
