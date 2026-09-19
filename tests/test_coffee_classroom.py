"""The small classroom start distribution is repeatable and physically valid."""

from itertools import product

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_classroom import (
    HORIZONTAL_JITTER,
    INITIAL_LAYOUT,
    POT_HEIGHT_JITTER,
    classroom_layout,
    fresh_classroom_seed,
)
from kaist_rl_lab.envs import CoffeePouringEnv


def test_seed_reproduces_a_bounded_two_coordinate_layout_without_mutating_defaults():
    original = {key: value.copy() for key, value in INITIAL_LAYOUT.items()}
    for seed in range(100):
        actual = classroom_layout(seed)
        assert actual == classroom_layout(seed)
        horizontal = actual["cup_center"][0] - INITIAL_LAYOUT["cup_center"][0]
        assert abs(horizontal) <= HORIZONTAL_JITTER
        assert actual["cup_center"][1] == INITIAL_LAYOUT["cup_center"][1]
        assert actual["pot_center"][0] - INITIAL_LAYOUT["pot_center"][0] == pytest.approx(horizontal)
        assert abs(actual["pot_center"][1] - INITIAL_LAYOUT["pot_center"][1]) <= POT_HEIGHT_JITTER
    assert classroom_layout(123) != classroom_layout(456)
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


def test_edges_and_random_starts_are_reachable_upright_and_collision_free():
    layouts = [classroom_layout(seed) for seed in range(32)]
    for horizontal, pot_height in product((-HORIZONTAL_JITTER, HORIZONTAL_JITTER), (-POT_HEIGHT_JITTER, POT_HEIGHT_JITTER)):
        layouts.append({
            "cup_center": [INITIAL_LAYOUT["cup_center"][0] + horizontal, INITIAL_LAYOUT["cup_center"][1]],
            "pot_center": [INITIAL_LAYOUT["pot_center"][0] + horizontal, INITIAL_LAYOUT["pot_center"][1] + pot_height],
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
