"""The fallback teaching examples must be real, reproducible successful attempts."""

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import (
    CUP_POSITION_JITTER,
    POT_POSITION_JITTER,
    classroom_layout,
)
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import (
    EXAMPLE_COUNT,
    MAX_EXAMPLE_STEPS,
    generate_example,
    load_examples,
)
from kaist_rl_lab.envs import CoffeePouringEnv


@pytest.mark.parametrize("index", range(EXAMPLE_COUNT))
def test_packaged_example_replays_to_success_from_the_unmodified_student_start(index):
    arrays, metadata = read_demonstration(load_examples()[index])
    assert metadata["participant"] == f"Generated example {index + 1}"
    assert metadata["dt"] == BROWSER_DT
    assert metadata["target_fill_l"] == 0.700
    assert metadata["success"] is True
    assert metadata["manual_finish"] is False
    assert arrays["terminated"][-1]
    assert not arrays["truncated"].any()
    env = CoffeePouringEnv(horizon=None, dt=BROWSER_DT)
    observation, info = env.reset(
        seed=metadata["seed"],
        options={**classroom_layout(metadata["seed"]), "target_fill": 0.700},
    )
    try:
        assert env.fill == env.spill == 0
        np.testing.assert_array_equal(env.joint_angles, metadata["initial_joint_angles_rad"])
        for step, action in enumerate(arrays["actions"]):
            np.testing.assert_array_equal(observation, arrays["observations"][step])
            observation, reward, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(observation, arrays["next_observations"][step])
            assert reward == pytest.approx(arrays["rewards"][step], abs=1e-6)
            assert terminated == arrays["terminated"][step]
            assert truncated == arrays["truncated"][step]
        assert info["is_success"]
        assert env.fill == pytest.approx(0.700, abs=0.005)
        assert env.spill < 0.001
        assert env.fill == metadata["fill_l"]
        assert env.spill == metadata["spill_l"]
        assert env.elapsed_steps <= MAX_EXAMPLE_STEPS
    finally:
        env.close()


def test_generator_reproduces_the_first_packaged_example():
    expected, metadata = read_demonstration(load_examples()[0])
    actual, generated = read_demonstration(generate_example(0))
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert generated["fill_l"] == metadata["fill_l"]
    assert generated["spill_l"] == metadata["spill_l"]
    assert generated["success"]


def test_packaged_examples_cover_distinct_classroom_starting_poses():
    examples = [read_demonstration(data) for data in load_examples()]
    assert len(examples) == 15
    assert len({metadata["seed"] for _, metadata in examples}) == 15
    starts = np.asarray([metadata["initial_joint_angles_rad"] for _, metadata in examples])
    assert len(np.unique(starts, axis=0)) == 15
    layouts = [classroom_layout(metadata["seed"]) for _, metadata in examples]
    centers = np.asarray([layout["cup_center"] + layout["pot_center"] for layout in layouts])
    assert np.all(np.ptp(centers, axis=0) > 1.5 * np.asarray(CUP_POSITION_JITTER + POT_POSITION_JITTER))
