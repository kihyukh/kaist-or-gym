import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import kaist_rl_lab  # noqa: F401 - import registers package environments
from kaist_rl_lab.envs import LaundryFoldingEnv


def _assert_rigid_links(env: LaundryFoldingEnv) -> None:
    geometry = env.geometry
    for arm_index in range(2):
        kin = env.arm_kinematics(arm_index)
        assert np.linalg.norm(kin["elbow"] - kin["base"]) == pytest.approx(
            geometry.upper_length, abs=1e-12
        )
        assert np.linalg.norm(kin["wrist"] - kin["elbow"]) == pytest.approx(
            geometry.fore_length, abs=1e-12
        )
        assert np.linalg.norm(kin["palm"] - kin["wrist"]) == pytest.approx(
            geometry.hand_length, abs=1e-12
        )
        for base, tip in zip(kin["finger_bases"], kin["finger_tips"], strict=True):
            assert np.linalg.norm(tip - base) == pytest.approx(geometry.finger_length, abs=1e-12)


def test_registration_and_gymnasium_contract():
    registered = gym.make("kaist-or/LaundryFoldingEnv-v0")
    assert isinstance(registered.unwrapped, LaundryFoldingEnv)
    check_env(registered.unwrapped, skip_render_check=True)
    registered.close()


def test_action_observation_and_info_contract():
    env = LaundryFoldingEnv(horizon=4)
    observation, info = env.reset(seed=9)
    assert env.action_space.shape == (8,)
    assert observation.dtype == np.float32
    assert observation.shape == env.observation_space.shape
    assert len(env.OBSERVATION_NAMES) == observation.size
    assert env.observation_space.contains(observation)
    assert info["stage"] == "straighten"
    assert info["physics_model"] == "xpbd_cloth_v1"

    observation, reward, terminated, truncated, info = env.step(np.zeros(8))
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["joint_angles"].shape == (2, 3)
    assert info["gripper_openings"].shape == (2,)
    env.close()


def test_seeded_reset_and_dynamics_are_reproducible():
    first = LaundryFoldingEnv(horizon=None)
    second = LaundryFoldingEnv(horizon=None)
    first_observation, _ = first.reset(seed=31)
    second_observation, _ = second.reset(seed=31)
    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(first.cloth_positions, second.cloth_positions)

    action = np.array([0.2, -0.1, 0.3, -1.0, -0.2, -0.1, 0.3, -1.0])
    for _ in range(3):
        first_result = first.step(action)
        second_result = second.step(action)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        assert first_result[1:4] == second_result[1:4]
        np.testing.assert_array_equal(first.cloth_positions, second.cloth_positions)
    first.close()
    second.close()


def test_different_seeds_change_the_random_towel_pose():
    env = LaundryFoldingEnv()
    env.reset(seed=1)
    first = env.cloth_positions.copy()
    env.reset(seed=2)
    assert not np.array_equal(first, env.cloth_positions)
    env.close()


def test_arm_segments_remain_fixed_and_mechanical_limits_hold():
    env = LaundryFoldingEnv(horizon=None)
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(12):
        _assert_rigid_links(env)
        env.step(rng.uniform(-1.0, 1.0, size=8))
        assert np.all(env.joint_angles >= env.joint_low - 1e-12)
        assert np.all(env.joint_angles <= env.joint_high + 1e-12)
        assert env._robot_pose_valid(env.joint_angles, env.gripper_openings)
    _assert_rigid_links(env)
    env.close()


def test_cloth_stays_finite_and_supported_during_idle_settling():
    env = LaundryFoldingEnv(horizon=None)
    env.reset(seed=12, options={"wrinkle_amplitude": 0.09})
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(np.zeros(8))
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    assert np.min(env.cloth_positions[:, 2]) >= env.geometry.table_height - 1e-9
    assert info["strain_rms"] < 0.05
    # Settling by itself is not accepted as the required bimanual stretch.
    assert info["stage"] == "straighten"
    assert not info["straightened_once"]
    assert info["bimanual_tension"] == 0.0
    env.close()


def test_closing_fingers_grasps_a_patch_and_opening_releases_it():
    env = LaundryFoldingEnv(horizon=None)
    env.reset(seed=1, options={"wrinkle_amplitude": 0.06})
    close = np.zeros(8)
    close[[3, 7]] = -1.0
    for _ in range(16):
        env.step(close)
        if all(len(vertices) == 3 for vertices in env.grasped_vertices):
            break
    assert all(len(vertices) == 3 for vertices in env.grasped_vertices)
    before = [vertices.copy() for vertices in env.grasped_vertices]
    for arm_index in range(2):
        np.testing.assert_allclose(
            env.cloth_positions[before[arm_index]],
            env._grasp_anchors(arm_index),
            atol=2e-4,
        )

    open_action = np.zeros(8)
    open_action[[3, 7]] = 1.0
    for _ in range(12):
        env.step(open_action)
    assert all(len(vertices) == 0 for vertices in env.grasped_vertices)
    env.close()


def test_fold_metric_uses_material_pairing_and_recognizes_an_ideal_fold():
    env = LaundryFoldingEnv(horizon=None)
    env.reset(seed=0, options={"wrinkle_amplitude": 0.0, "settle_steps": 0})
    unfolded_score = env.cloth_metrics()["fold_score"]

    folded = env.rest_positions.reshape(env.mesh_rows, env.mesh_cols, 3).copy()
    folded[:, :, 1] = -np.abs(folded[:, :, 1])
    middle = env.mesh_rows // 2
    folded[:middle, :, 2] = 0.5 * env.geometry.towel_thickness
    folded[middle + 1 :, :, 2] = 1.75 * env.geometry.towel_thickness
    folded[middle, :, 2] = 0.5 * env.geometry.towel_thickness
    env.cloth_positions = folded.reshape(-1, 3)
    env.cloth_velocities.fill(0.0)
    folded_metrics = env.cloth_metrics()
    assert folded_metrics["fold_score"] > 0.75
    assert folded_metrics["fold_score"] > unfolded_score + 0.70
    env.straightened_once = True
    env.straight_streak = 5
    env._previous_metrics = folded_metrics
    _, _, terminated, truncated, info = env.step(np.zeros(8))
    assert terminated and not truncated
    assert info["is_success"]
    assert info["termination_reason"] == "success"
    env.close()


def test_snapshot_is_json_safe_deep_copied_and_side_effect_free():
    env = LaundryFoldingEnv()
    observation, _ = env.reset(seed=5)
    positions = env.cloth_positions.copy()
    velocities = env.cloth_velocities.copy()
    rng_state = env.np_random.bit_generator.state.copy()

    snapshot = env.render_snapshot()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["schema_version"] == 1
    assert snapshot["geometry"]["towel"]["mesh_rows"] == env.mesh_rows
    assert len(snapshot["state"]["cloth_vertices_m"]) == env.vertex_count
    snapshot["state"]["cloth_vertices_m"][0][0] = 99.0
    assert env.render_snapshot()["state"]["cloth_vertices_m"][0][0] != 99.0
    np.testing.assert_array_equal(observation, env._get_obs())
    np.testing.assert_array_equal(positions, env.cloth_positions)
    np.testing.assert_array_equal(velocities, env.cloth_velocities)
    assert env.np_random.bit_generator.state == rng_state
    env.close()


def test_horizon_and_invalid_inputs():
    with pytest.raises(ValueError):
        LaundryFoldingEnv(dt=0.0)
    with pytest.raises(TypeError):
        LaundryFoldingEnv(horizon=True)
    with pytest.raises(ValueError):
        LaundryFoldingEnv(mesh_rows=10)

    env = LaundryFoldingEnv(horizon=2)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(7))
    _, _, terminated, truncated, _ = env.step(np.zeros(8))
    assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(np.zeros(8))
    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit"
    with pytest.raises(RuntimeError):
        env.step(np.zeros(8))
    env.close()
