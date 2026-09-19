"""Exact geometry caching and opt-in diagnostics must not alter the task."""

from copy import copy
from dataclasses import replace

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import load_examples
from kaist_rl_lab.envs import CoffeePouringEnv

RENDER_ONLY_INFO = {
    "stable_cup_capacity", "cup_surface_y", "pot_surface_y", "stream_path",
    "spill_path", "direct_spill_path", "cup_runoff_path",
}


def assert_feedback_equal(full, lightweight):
    assert set(full) - set(lightweight) == RENDER_ONLY_INFO
    for key, value in lightweight.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(full[key], value)
        else:
            assert full[key] == value


def test_lightweight_info_preserves_every_transition_reward_and_rendered_scene():
    arrays, metadata = read_demonstration(load_examples()[0])
    full = CoffeePouringEnv(dt=metadata["dt"], horizon=None)
    fast = CoffeePouringEnv(dt=metadata["dt"], horizon=None, include_render_info=False)
    options = {"joint_angles": metadata["initial_joint_angles_rad"],
               "target_fill": metadata["target_fill_l"]}
    try:
        first = full.reset(seed=metadata["seed"], options=options)
        second = fast.reset(seed=metadata["seed"], options=options)
        np.testing.assert_array_equal(first[0], second[0])
        assert_feedback_equal(first[1], second[1])
        for index, action in enumerate(arrays["actions"]):
            first, second = full.step(action), fast.step(action)
            np.testing.assert_array_equal(first[0], second[0])
            assert first[1:4] == second[1:4]
            assert_feedback_equal(first[4], second[4])
            if index % 127 == 0 or first[2] or first[3]:
                assert full.render_snapshot() == fast.render_snapshot()
        assert first[4]["is_success"]
    finally:
        full.close()
        fast.close()


def test_lightweight_info_computes_surfaces_only_when_render_requested(monkeypatch):
    env = CoffeePouringEnv(include_render_info=False)
    calls = []
    original = env._cup_surface_world_y

    def surface(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "_cup_surface_world_y", surface)
    try:
        env.reset(seed=7001)
        env.step(np.zeros(6))
        assert calls == []
        scene = env.render_snapshot()
        assert len(calls) == 2
        assert "cup_surface_y_m" in scene["state"]["liquid"]
    finally:
        env.close()
    with pytest.raises(TypeError, match="include_render_info"):
        CoffeePouringEnv(include_render_info=1)


def test_polygon_area_shift_keeps_the_original_dot_product_order():
    rng = np.random.default_rng(2026)
    for size in range(3, 15):
        for _ in range(10):
            points = rng.uniform(-1, 1, (size, 2))
            x, y = points[:, 0], points[:, 1]
            previous = float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            assert CoffeePouringEnv._polygon_area(points) == previous


def test_runoff_cache_uses_exact_geometry_and_returns_independent_arrays():
    env = CoffeePouringEnv()
    try:
        env.reset(seed=7001)
        tools = env.tool_positions()
        first = env._cup_runoff_path(tools)
        expected = first.copy()
        first[:] = 999
        np.testing.assert_array_equal(env._cup_runoff_path(tools), expected)
        for preferred in (None, -1.0, 1.0):
            actual = env._cup_runoff_path(tools, preferred_x=preferred)
            env._cup_runoff_cache.clear()
            np.testing.assert_array_equal(actual, env._cup_runoff_path(tools, preferred_x=preferred))
        env.geometry = replace(env.geometry, table_y=env.geometry.table_y + .01)
        changed = env._cup_runoff_path(tools)
        env._cup_runoff_cache.clear()
        np.testing.assert_array_equal(changed, env._cup_runoff_path(tools))
        assert not np.array_equal(changed, expected)
        for index in range(100):
            env._cup_runoff_path(tools, preferred_x=index * .001)
        assert len(env._cup_runoff_cache) <= 32
    finally:
        env.close()


def test_surface_cache_rechecks_volume_pose_and_capacity_and_stays_bounded():
    env = CoffeePouringEnv()
    try:
        env.reset(seed=7001)
        for index in range(70):
            volume = index / 100
            env.CUP_CAPACITY = 1.02 if index % 2 else 1.05
            tools = env.tool_positions()
            tools["cup_center"] = tools["cup_center"] + np.array([index * .001, index * .002])
            first = env._cup_surface_world_y(tools, volume)
            assert env._cup_surface_world_y(tools, volume) == first
            saved = env._cup_surface_cache
            env._cup_surface_cache = {}
            assert env._cup_surface_world_y(tools, volume) == first
            env._cup_surface_cache = saved
        assert len(env._cup_surface_cache) <= 32
    finally:
        env.close()


def test_shallow_forecast_copy_cannot_poison_original_geometry_cache():
    env = CoffeePouringEnv()
    try:
        env.reset(seed=7001)
        tools = env.tool_positions()
        original_path = env._cup_runoff_path(tools)
        original_surface = env._cup_surface_world_y(tools, .4)
        forecast = copy(env)
        forecast.joint_angles = env.joint_angles.copy()
        forecast.joint_angles[0] += .04
        forecast.fill = .6
        forecast_tools = forecast.tool_positions()
        forecast._cup_runoff_path(forecast_tools)
        forecast._cup_surface_world_y(forecast_tools)
        # copy(env), as used by the example generator, shares cache storage;
        # the complete immutable keys keep its hypothetical pose independent.
        np.testing.assert_array_equal(env._cup_runoff_path(tools), original_path)
        assert env._cup_surface_world_y(tools, .4) == original_surface
    finally:
        env.close()
