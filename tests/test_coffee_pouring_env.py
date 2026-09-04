import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import kaist_or_gym  # noqa: F401 - importing registers package environments
from kaist_or_gym.envs import CoffeePouringEnv


def _assert_rigid_lengths(env):
    points = env.joint_positions()
    geometry = env.geometry
    expected = {
        "cup": (geometry.cup_upper, geometry.cup_fore),
        "pot": (geometry.pot_upper, geometry.pot_fore),
    }
    for arm_name, (upper, fore) in expected.items():
        base, elbow, wrist = points[arm_name]
        np.testing.assert_allclose(np.linalg.norm(elbow - base), upper, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(wrist - elbow), fore, atol=1e-12)


def _joints_for_geometry(env, cup_center, cup_angle, spout, pot_angle):
    geometry = env.geometry
    cup_center = np.asarray(cup_center, dtype=np.float64)
    spout = np.asarray(spout, dtype=np.float64)

    cup_wrist = cup_center + env._rotation(cup_angle) @ np.asarray(geometry.cup_grip)
    cup_q1, cup_q2 = env._inverse_kinematics(
        np.asarray(geometry.cup_base),
        cup_wrist,
        geometry.cup_upper,
        geometry.cup_fore,
        elbow_sign=-1.0,
    )
    pot_center = spout - env._rotation(pot_angle) @ np.asarray(geometry.pot_spout)
    pot_wrist = pot_center + env._rotation(pot_angle) @ np.asarray(geometry.pot_grip)
    pot_q1, pot_q2 = env._inverse_kinematics(
        np.asarray(geometry.pot_base),
        pot_wrist,
        geometry.pot_upper,
        geometry.pot_fore,
        elbow_sign=1.0,
    )
    return np.array(
        [
            cup_q1,
            cup_q2,
            cup_angle - cup_q1 - cup_q2,
            pot_q1,
            pot_q2,
            pot_angle - pot_q1 - pot_q2,
        ]
    )


def _reference_action(env, desired_pot_angle):
    geometry = env.geometry
    desired_cup = np.array([-0.05, 0.27])
    desired_cup_angle = 0.0
    cup_wrist = desired_cup + np.asarray(geometry.cup_grip)
    cup_q1, cup_q2 = env._inverse_kinematics(
        np.asarray(geometry.cup_base),
        cup_wrist,
        geometry.cup_upper,
        geometry.cup_fore,
        elbow_sign=-1.0,
    )

    # Recompute shoulder and elbow IK throughout the tilt so the rigid pot
    # follows a constant-spout path over the cup.
    desired_spout = desired_cup + np.asarray(geometry.cup_mouth) + np.array([0.0, 0.20])
    pot_center = desired_spout - env._rotation(desired_pot_angle) @ np.asarray(
        geometry.pot_spout
    )
    pot_wrist = pot_center + env._rotation(desired_pot_angle) @ np.asarray(geometry.pot_grip)
    pot_q1, pot_q2 = env._inverse_kinematics(
        np.asarray(geometry.pot_base),
        pot_wrist,
        geometry.pot_upper,
        geometry.pot_fore,
        elbow_sign=1.0,
    )
    target = np.array(
        [
            cup_q1,
            cup_q2,
            desired_cup_angle - cup_q1 - cup_q2,
            pot_q1,
            pot_q2,
            desired_pot_angle - pot_q1 - pot_q2,
        ]
    )
    angle_error = np.arctan2(np.sin(target - env.joint_angles), np.cos(target - env.joint_angles))
    action = np.clip(angle_error / (env.dt * env.max_joint_speeds), -1.0, 1.0)
    return action, angle_error


def _predicted_return_volume(env, pot_angle, pot_path_rate):
    """Approximate coffee released while the constant-spout path returns upright."""

    angle_grid = np.linspace(0.0, pot_angle, 257)
    spout_offset = np.asarray(env.geometry.pot_spout)
    rotated_spout_y = (
        np.sin(angle_grid) * spout_offset[0] + np.cos(angle_grid) * spout_offset[1]
    )
    intensity = np.clip(
        (-rotated_spout_y - env.FLOW_START_SPOUT_DROP)
        / (env.FLOW_FULL_SPOUT_DROP - env.FLOW_START_SPOUT_DROP),
        0.0,
        1.0,
    )
    integral = np.sum(0.5 * (intensity[:-1] + intensity[1:]) * np.diff(angle_grid))
    return env.max_flow_rate * integral / pot_path_rate


def test_environment_passes_gymnasium_checker_and_registration():
    registered = gym.make("kaist-or/CoffeePouringEnv-v0")
    assert isinstance(registered.unwrapped, CoffeePouringEnv)
    # RGB rendering is covered below. Skip Gymnasium's attempt to instantiate
    # optional desktop "human" rendering because pygame is an extra dependency.
    check_env(registered.unwrapped, skip_render_check=True)
    registered.close()


def test_observation_and_action_contract():
    env = CoffeePouringEnv()
    observation, info = env.reset(seed=7)
    assert observation.dtype == np.float32
    assert observation.shape == (16,)
    assert len(env.OBSERVATION_NAMES) == observation.shape[0]
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (6,)
    assert info["joint_angles"].shape == (6,)
    observation, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_full_scale_joint_command_takes_ten_seconds_for_a_quarter_turn():
    env = CoffeePouringEnv(horizon=None)
    initial_joints = env.joint_low + 0.05
    env.reset(seed=7, options={"joint_angles": initial_joints})
    env.max_flow_rate = 0.0
    env.max_leak_rate = 0.0

    step_count = round(env.FULL_SCALE_QUARTER_TURN_SECONDS / env.dt)
    assert step_count == 80
    assert step_count * env.dt == pytest.approx(env.FULL_SCALE_QUARTER_TURN_SECONDS)
    for _ in range(step_count):
        _, _, terminated, truncated, _ = env.step(np.ones(6, dtype=np.float32))
        assert not terminated
        assert not truncated

    np.testing.assert_allclose(env.joint_angles - initial_joints, np.pi / 2.0, atol=1e-12)
    env.close()


def test_continuous_motor_path_depends_on_simulated_time_not_decision_rate():
    action = np.array([0.4, -0.3, 0.2, -0.4, 0.3, -0.2])
    final_joints = []
    for dt in (0.05, 0.125, 0.25):
        env = CoffeePouringEnv(horizon=None, dt=dt)
        initial = 0.5 * (env.joint_low + env.joint_high)
        env.reset(seed=7, options={"joint_angles": initial})
        env.max_flow_rate = 0.0
        env.max_leak_rate = 0.0
        for _ in range(round(2.0 / dt)):
            env.step(action)
        final_joints.append(env.joint_angles.copy())
        env.close()
    for joints in final_joints[1:]:
        np.testing.assert_allclose(joints, final_joints[0], rtol=0.0, atol=1e-12)


def test_default_horizon_preserves_the_slow_motor_motion_budget():
    env = CoffeePouringEnv()
    assert env.DEFAULT_HORIZON == 330
    assert env.DEFAULT_HORIZON * env.dt == pytest.approx(41.25)
    env.reset(seed=7)
    for expected_step in range(1, env.DEFAULT_HORIZON + 1):
        _, _, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated
        assert truncated is (expected_step == env.DEFAULT_HORIZON)
    assert info["termination_reason"] == "time_limit"
    env.close()


def test_distinct_legal_joint_states_have_distinct_observations():
    env = CoffeePouringEnv()
    env.reset(seed=7)
    first = env.joint_angles.copy()
    second = first.copy()
    first[5] = 1.4
    second[5] = 2.0
    env.joint_angles = first
    first_observation = env._get_obs()
    env.joint_angles = second
    second_observation = env._get_obs()
    assert not np.array_equal(first_observation, second_observation)
    env.close()


def test_links_never_change_length_under_random_joint_commands():
    env = CoffeePouringEnv()
    rng = np.random.default_rng(123)
    env.reset(seed=123)
    for _ in range(500):
        _assert_rigid_lengths(env)
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1.0, 1.0, size=6))
        if terminated or truncated:
            env.reset(seed=123)
    _assert_rigid_lengths(env)
    env.close()


def test_same_seed_and_actions_are_exactly_reproducible():
    first = CoffeePouringEnv(render_mode="rgb_array")
    second = CoffeePouringEnv(render_mode="rgb_array")
    obs_first, info_first = first.reset(seed=99)
    obs_second, info_second = second.reset(seed=99)
    np.testing.assert_array_equal(obs_first, obs_second)
    np.testing.assert_array_equal(first.render(), second.render())
    assert info_first["target_fill"] == info_second["target_fill"]

    actions = np.random.default_rng(4).uniform(-0.8, 0.8, size=(40, 6))
    for action in actions:
        result_first = first.step(action)
        result_second = second.step(action)
        np.testing.assert_array_equal(result_first[0], result_second[0])
        assert result_first[1:4] == result_second[1:4]
        assert result_first[4]["fill"] == result_second[4]["fill"]
        assert result_first[4]["spill"] == result_second[4]["spill"]
        np.testing.assert_array_equal(first.render(), second.render())
        if result_first[2] or result_first[3]:
            break
    first.close()
    second.close()


def test_rgb_render_is_owned_by_environment_and_does_not_mutate_state():
    env = CoffeePouringEnv(render_mode="rgb_array", width=640, height=400)
    observation, _ = env.reset(seed=5)
    joints_before = env.joint_angles.copy()
    frame_first = env.render()
    frame_second = env.render()
    assert frame_first.shape == (400, 640, 3)
    assert frame_first.dtype == np.uint8
    assert np.unique(frame_first.reshape(-1, 3), axis=0).shape[0] > 20
    np.testing.assert_array_equal(frame_first, frame_second)
    np.testing.assert_array_equal(joints_before, env.joint_angles)
    np.testing.assert_array_equal(observation, env._get_obs())
    env.close()


def test_render_snapshot_is_json_safe_deep_copied_and_side_effect_free():
    env = CoffeePouringEnv(render_mode="rgb_array")
    observation, _ = env.reset(seed=5)
    joints_before = env.joint_angles.copy()
    frame_before = env.render()
    rng_before = env.np_random.bit_generator.state.copy()

    first = env.render_snapshot()
    second = env.render_snapshot()
    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["schema_version"] == 1
    assert first["state"]["step"] == env.elapsed_steps
    assert first["state"]["elapsed_time_s"] == pytest.approx(env.elapsed_steps * env.dt)
    assert first["geometry"]["table_y_m"] == env.geometry.table_y
    np.testing.assert_allclose(first["state"]["joint_angles_rad"], env.joint_angles)

    tools = env.tool_positions()
    for name in ("cup_center", "cup_mouth", "pot_center", "pot_spout"):
        np.testing.assert_allclose(first["state"]["landmarks_m"][name], tools[name])

    first["geometry"]["arms"]["cup"]["base_m"][0] = 999.0
    assert env.render_snapshot()["geometry"]["arms"]["cup"]["base_m"][0] != 999.0
    np.testing.assert_array_equal(env.joint_angles, joints_before)
    np.testing.assert_array_equal(env._get_obs(), observation)
    np.testing.assert_array_equal(env.render(), frame_before)
    assert env.np_random.bit_generator.state == rng_before
    env.close()


def test_interpolated_joint_keyframes_keep_all_arm_segments_rigid():
    env = CoffeePouringEnv(horizon=None)
    env.reset(seed=12)
    before = env.render_snapshot()
    env.step(np.array([0.8, -0.7, 0.6, -0.5, 0.4, -0.3], dtype=np.float32))
    after = env.render_snapshot()

    for alpha in np.linspace(0.0, 1.0, 5):
        first = np.asarray(before["state"]["joint_angles_rad"])
        second = np.asarray(after["state"]["joint_angles_rad"])
        joints = first + alpha * (second - first)
        for arm_name, offset in (("cup", 0), ("pot", 3)):
            arm = before["geometry"]["arms"][arm_name]
            points = env._arm_points(
                np.asarray(arm["base_m"]),
                joints[offset],
                joints[offset + 1],
                *arm["link_lengths_m"],
            )
            lengths = [
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[1]),
            ]
            np.testing.assert_allclose(lengths, arm["link_lengths_m"], atol=1e-12)
    env.close()


def test_renderer_handles_vessels_partly_outside_the_canvas():
    env = CoffeePouringEnv(render_mode="rgb_array", width=480, height=300)
    env.reset(seed=5)
    rng = np.random.default_rng(55)
    for _ in range(60):
        env.joint_angles = rng.uniform(env.joint_low, env.joint_high)
        env.fill = float(rng.uniform(0.0, 1.02))
        frame = env.render()
        assert frame.shape == (300, 480, 3)
        assert frame.dtype == np.uint8
    env.close()


def test_joint_angles_are_clipped_to_mechanical_limits():
    env = CoffeePouringEnv(horizon=300)
    env.reset(seed=3)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(np.ones(6))
        if terminated or truncated:
            break
    assert np.all(env.joint_angles <= env.joint_high + 1e-12)
    assert np.all(env.joint_angles >= env.joint_low - 1e-12)
    env.close()


def test_upright_pot_has_no_flow():
    env = CoffeePouringEnv()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(6))
    assert info["flow"] == 0.0
    assert info["fill"] == 0.0
    assert info["spill"] == 0.0
    env.close()


def test_liquid_dynamics_scale_with_simulated_time_not_step_count():
    cup_center = np.array([-0.05, 0.27])
    cup_mouth = cup_center + np.array([0.0, 0.13])
    pot_angle = 1.05
    fills = []
    for dt in (0.0625, 0.125, 0.25):
        env = CoffeePouringEnv(dt=dt, horizon=100)
        joints = _joints_for_geometry(
            env,
            cup_center,
            0.0,
            cup_mouth + np.array([0.0, 0.20]),
            pot_angle,
        )
        env.reset(seed=1, options={"joint_angles": joints, "target_fill": 0.90})
        for _ in range(round(1.0 / dt)):
            env.step(np.zeros(6))
        fills.append(env.fill)
        env.close()
    np.testing.assert_allclose(fills, fills[0], rtol=0.0, atol=1e-12)


def test_pour_rule_is_continuous_across_angle_wrap_and_upside_down_is_not_success():
    env = CoffeePouringEnv()
    base_joints = env.reset(seed=2)[1]["joint_angles"].astype(np.float64)
    intensities = []
    for unwrapped_angle in (np.pi - 0.01, np.pi + 0.01):
        joints = base_joints.copy()
        joints[3:5] = [0.35, 0.10]
        joints[5] = unwrapped_angle - np.sum(joints[3:5])
        env.reset(
            seed=2,
            options={"joint_angles": joints, "target_fill": 0.70, "fill": 0.70},
        )
        _, _, _, _, info = env.step(np.zeros(6))
        intensities.append(info["flow_rate"])
        assert info["flow_rate"] > 0.0
        assert not info["is_success"]
    assert abs(intensities[0] - intensities[1]) < 0.006
    env.close()


def test_aligned_tilted_pot_fills_cup_and_misaligned_pot_spills():
    cup_center = np.array([-0.05, 0.27])
    cup_mouth = cup_center + np.array([0.0, 0.13])
    pot_angle = 1.05

    aligned = CoffeePouringEnv()
    aligned_joints = _joints_for_geometry(
        aligned,
        cup_center,
        0.0,
        cup_mouth + np.array([0.0, 0.20]),
        pot_angle,
    )
    aligned.reset(
        seed=1,
        options={"joint_angles": aligned_joints, "target_fill": 0.70},
    )
    _, _, _, _, aligned_info = aligned.step(np.zeros(6))
    assert aligned_info["flow"] > 0.0
    assert aligned_info["fill"] > 0.0
    assert aligned_info["spill"] < aligned_info["flow"]

    misaligned = CoffeePouringEnv()
    misaligned_joints = _joints_for_geometry(
        misaligned,
        cup_center,
        0.0,
        cup_mouth + np.array([0.25, 0.20]),
        pot_angle,
    )
    misaligned.reset(
        seed=1,
        options={"joint_angles": misaligned_joints, "target_fill": 0.70},
    )
    _, _, _, _, misaligned_info = misaligned.step(np.zeros(6))
    assert misaligned_info["flow"] > 0.0
    assert misaligned_info["fill"] == 0.0
    assert misaligned_info["spill"] == misaligned_info["flow"]
    assert misaligned_info["stream_end"][1] <= misaligned_info["pot_spout"][1]
    aligned.close()
    misaligned.close()


def test_reset_rejects_invalid_or_unreachable_state_options():
    env = CoffeePouringEnv()
    with pytest.raises(ValueError, match="fill"):
        env.reset(options={"fill": -0.1})
    with pytest.raises(ValueError, match="spill"):
        env.reset(options={"spill": np.nan})
    with pytest.raises(ValueError, match="unreachable"):
        env.reset(options={"cup_center": [10.0, 10.0]})
    with pytest.raises(ValueError, match="mechanical limits"):
        env.reset(options={"joint_angles": np.full(6, 100.0)})
    with pytest.raises(ValueError, match="cannot be combined"):
        env.reset(options={"joint_angles": np.zeros(6), "cup_center": [0.0, 0.0]})
    env.close()


def test_success_failure_and_time_limit_have_distinct_endings():
    success_env = CoffeePouringEnv()
    success_env.reset(seed=0, options={"target_fill": 0.70, "fill": 0.68})
    _, _, terminated, truncated, info = success_env.step(np.zeros(6))
    assert terminated and not truncated
    assert info["is_success"]
    assert info["termination_reason"] == "success"

    failure_env = CoffeePouringEnv()
    failure_env.reset(seed=0, options={"spill": 0.40})
    _, _, terminated, truncated, info = failure_env.step(np.zeros(6))
    assert terminated and not truncated
    assert not info["is_success"]
    assert info["termination_reason"] == "spill_or_overflow"

    time_env = CoffeePouringEnv(horizon=3)
    time_env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, info = time_env.step(np.zeros(6))
    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit"
    success_env.close()
    failure_env.close()
    time_env.close()


def test_unlimited_horizon_has_no_time_truncation_but_keeps_normal_endings():
    env = CoffeePouringEnv(horizon=None, render_mode="rgb_array")
    observation, info = env.reset(seed=0)
    assert observation[-1] == -1.0
    assert info["time_remaining"] is None
    assert not info["has_time_limit"]

    for expected_step in range(1, 201):
        observation, _, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated
        assert not truncated
        assert observation[-1] == -1.0
        assert env.observation_space.contains(observation)
        assert info["elapsed_steps"] == expected_step
        assert info["elapsed_time"] == pytest.approx(expected_step * env.dt)
    assert env.render().shape == (560, 960, 3)

    env.reset(seed=0, options={"target_fill": 0.70, "fill": 0.68})
    _, _, terminated, truncated, info = env.step(np.zeros(6))
    assert terminated and not truncated
    assert info["is_success"]
    env.close()


@pytest.mark.parametrize(
    ("horizon", "error_type"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_invalid_horizons_are_rejected(horizon, error_type):
    with pytest.raises(error_type):
        CoffeePouringEnv(horizon=horizon)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_invalid_decision_intervals_are_rejected(dt):
    with pytest.raises(ValueError, match="finite positive"):
        CoffeePouringEnv(dt=dt)


def test_environment_is_solvable_by_a_simple_joint_space_reference_controller():
    pot_path_rate = 0.070
    maximum_pot_angle = 1.05
    for seed in range(7001, 7006):
        env = CoffeePouringEnv()
        env.reset(seed=seed)
        phase = "approach"
        pot_reference_angle = 0.0
        for _ in range(env.horizon):
            action, angle_error = _reference_action(env, pot_reference_angle)
            if phase == "approach" and np.max(np.abs(angle_error)) < 0.002:
                phase = "pour"

            if phase == "pour":
                remaining = env.target_fill - env.fill
                expected_return = _predicted_return_volume(
                    env, pot_reference_angle, pot_path_rate
                )
                if remaining <= expected_return + 0.010 or pot_reference_angle >= maximum_pot_angle:
                    phase = "return"
                else:
                    pot_reference_angle = min(
                        maximum_pot_angle,
                        pot_reference_angle + pot_path_rate * env.dt,
                    )
                action, _ = _reference_action(env, pot_reference_angle)
            elif phase == "return":
                pot_reference_angle = max(0.0, pot_reference_angle - pot_path_rate * env.dt)
                action, _ = _reference_action(env, pot_reference_angle)

            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated and not truncated, (seed, info)
        assert info["is_success"], (seed, info)
        assert info["termination_reason"] == "success"
        assert info["elapsed_steps"] <= 260
        env.close()
