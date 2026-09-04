import json
from itertools import pairwise

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


def _rendered_cup_opening(env, cup_center, cup_angle):
    """Return the world-space endpoints of the cup's rendered top edge."""

    geometry = env.geometry
    cup_center = np.asarray(cup_center, dtype=np.float64)
    local_opening = np.array(
        [
            [-0.5 * geometry.cup_width, 0.5 * geometry.cup_height],
            [0.5 * geometry.cup_width, 0.5 * geometry.cup_height],
        ]
    )
    return cup_center + local_opening @ env._rotation(cup_angle).T


def _opening_coordinates(opening, point):
    """Return along-opening and normal coordinates of a world-space point."""

    opening = np.asarray(opening, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    center = np.mean(opening, axis=0)
    tangent = opening[1] - opening[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    return float(np.dot(point - center, tangent)), float(np.dot(point - center, normal))


def _strictly_inside_convex_polygon(point, polygon, tolerance=1e-8):
    """Whether ``point`` lies away from every edge of a convex polygon."""

    polygon = np.asarray(polygon, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = point - polygon
    crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(crosses > tolerance) or np.all(crosses < -tolerance))


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
    # Aim right of the mouth so the gravity-driven ballistic arc lands near
    # the centre instead of pretending that the stream is vertical.
    desired_spout = desired_cup + np.asarray(geometry.cup_mouth) + np.array([0.10, 0.20])
    pot_center = desired_spout - env._rotation(desired_pot_angle) @ np.asarray(geometry.pot_spout)
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

    if pot_angle <= 0.0:
        return 0.0
    angle_grid = np.linspace(pot_angle, 0.0, 257)
    remaining = env.source_remaining
    released = 0.0
    saved_joints = env.joint_angles.copy()
    try:
        for start, end in pairwise(angle_grid):
            middle = 0.5 * (start + end)
            env.joint_angles[5] = middle - np.sum(env.joint_angles[3:5])
            rate, _, _ = env._flow_state(remaining)
            interval = abs(end - start) / pot_path_rate
            amount = min(rate * interval, remaining)
            released += amount
            remaining -= amount
    finally:
        env.joint_angles = saved_joints
    return float(released)


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
    # A collision-free configuration whose full 90-degree all-joint sweep
    # remains clear of the tabletop and mechanical upper limits.
    initial_joints = np.array(
        [1.06569964, -1.67585190, -2.03996040, 0.48670250, 0.18905989, -1.52938532]
    )
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
    assert first["schema_version"] == 4
    assert first["state"]["step"] == env.elapsed_steps
    assert first["state"]["elapsed_time_s"] == pytest.approx(env.elapsed_steps * env.dt)
    assert first["geometry"]["table_y_m"] == env.geometry.table_y
    np.testing.assert_allclose(first["state"]["joint_angles_rad"], env.joint_angles)
    assert len(first["state"]["liquid"]["stream_path_m"]) == env.STREAM_PATH_SAMPLES
    assert len(first["state"]["liquid"]["spill_path_m"]) == env.STREAM_PATH_SAMPLES
    assert len(first["state"]["liquid"]["direct_spill_path_m"]) == env.STREAM_PATH_SAMPLES
    assert len(first["state"]["liquid"]["cup_runoff_path_m"]) == env.STREAM_PATH_SAMPLES
    assert first["state"]["liquid"]["last_jet_radius_m"] == 0.0
    assert first["state"]["liquid"]["spill_impact_x_m"] == 0.0
    assert first["state"]["liquid"]["source_remaining_l"] == env.source_remaining

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


def test_table_contact_stops_the_arm_without_allowing_penetration():
    env = CoffeePouringEnv(horizon=None)
    env.reset(seed=7)
    env.max_flow_rate = 0.0
    action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for _ in range(300):
        env.step(action)

    contact_angles = env.joint_angles.copy()
    assert env._arm_table_clearance("cup", contact_angles[:3]) >= -1e-9
    assert contact_angles[0] > env.joint_low[0] + 0.5
    for _ in range(20):
        env.step(action)
    np.testing.assert_allclose(env.joint_angles, contact_angles, atol=1e-10)
    env.close()


def test_cross_robot_contact_stops_inward_motion_and_allows_separation():
    env = CoffeePouringEnv(horizon=None)
    default_joints = env.reset(seed=0)[1]["joint_angles"].astype(np.float64)
    colliding_joints = np.array(
        [0.69334231, -0.85815422, -1.90451004, 1.50150783, 1.31243359, -1.46690584]
    )
    with pytest.raises(ValueError, match="robot configuration"):
        env.reset(options={"joint_angles": colliding_joints})
    handle_collision = np.array(
        [1.347713392, -0.942228745, -1.693278731, 2.375357950, 0.307520056, -0.707350249]
    )
    assert env._cross_robot_collision(handle_collision)
    with pytest.raises(ValueError, match="robot configuration"):
        env.reset(options={"joint_angles": handle_collision})

    direction = colliding_joints - default_joints
    start = default_joints + 0.89 * direction
    action = direction / np.max(np.abs(direction))
    env.reset(options={"joint_angles": start})
    env.max_flow_rate = 0.0

    unconstrained = np.clip(
        start + env.dt * env.max_joint_speeds * action,
        env.joint_low,
        env.joint_high,
    )
    env.step(action)
    contact = env.joint_angles.copy()

    assert not env._cross_robot_collision(contact)
    assert np.linalg.norm(contact - start) < np.linalg.norm(unconstrained - start)

    env.step(action)
    np.testing.assert_allclose(env.joint_angles, contact, atol=1e-8)

    env.step(-action)
    assert np.linalg.norm(env.joint_angles - contact) > 1e-4
    assert not env._cross_robot_collision(env.joint_angles)
    env.close()


def test_upright_pot_has_no_flow():
    env = CoffeePouringEnv()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(6))
    assert info["flow"] == 0.0
    assert info["fill"] == 0.0
    assert info["spill"] == 0.0
    env.close()


def test_pour_rate_increases_with_tilt_and_decreases_as_pot_empties():
    env = CoffeePouringEnv(horizon=None)
    env.reset(seed=1)
    rates = []
    for angle_degrees in (20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0):
        env.joint_angles[5] = np.deg2rad(angle_degrees) - np.sum(env.joint_angles[3:5])
        rates.append(env._flow_state(env.INITIAL_POT_VOLUME)[0])

    assert all(first < second for first, second in pairwise(rates))
    assert 0.015 < rates[1] < 0.035
    assert 0.095 < rates[-2] < 0.120
    assert rates[-1] > rates[-2]
    full_rate = env._flow_state(env.INITIAL_POT_VOLUME)[0]
    low_head_rate = env._flow_state(0.30)[0]
    assert 0.0 < low_head_rate < full_rate
    env.close()


def test_tilt_flow_curve_is_smooth_and_stream_width_follows_flow_continuity():
    env = CoffeePouringEnv(horizon=None)
    env.reset(seed=1)
    rates = []
    radii = []
    for angle_degrees in np.arange(15.0, 80.01, 0.25):
        env.joint_angles[5] = np.deg2rad(angle_degrees) - np.sum(env.joint_angles[3:5])
        rate, exit_speed, _ = env._flow_state(env.INITIAL_POT_VOLUME)
        rates.append(rate)
        radii.append(env._jet_radius(rate, exit_speed))

    rate_changes = np.diff(rates)
    assert np.all(rate_changes >= -1e-12)
    assert np.max(rate_changes) < 0.003
    assert np.all(np.isfinite(radii))
    assert np.all(np.asarray(radii) >= 0.0)
    assert max(radii) <= env.JET_RADIUS
    env.close()


def test_liquid_landmarks_match_the_rendered_rim_and_spout_tip():
    env = CoffeePouringEnv()
    geometry = env.geometry
    assert geometry.cup_mouth == pytest.approx((0.0, 0.5 * geometry.cup_height))
    assert geometry.pot_spout == pytest.approx(
        (-0.78 * geometry.pot_width, 0.22 * geometry.pot_height)
    )
    env.close()


def test_tilted_cup_retains_only_volume_below_its_rendered_lower_rim():
    env = CoffeePouringEnv(horizon=None)
    base_joints = env.reset(seed=3)[1]["joint_angles"].astype(np.float64)
    cup_center = np.array([-0.05, 0.27])
    cup_angle = np.deg2rad(60.0)
    geometry = env.geometry
    cup_wrist = cup_center + env._rotation(cup_angle) @ np.asarray(geometry.cup_grip)
    cup_q1, cup_q2 = env._inverse_kinematics(
        np.asarray(geometry.cup_base),
        cup_wrist,
        geometry.cup_upper,
        geometry.cup_fore,
        elbow_sign=-1.0,
    )
    base_joints[:3] = [cup_q1, cup_q2, cup_angle - cup_q1 - cup_q2]
    env.reset(
        seed=3,
        options={"joint_angles": base_joints, "target_fill": 0.90, "fill": 0.80},
    )
    stable_capacity = env._stable_cup_capacity()
    env.max_flow_rate = 0.0

    _, _, _, _, info = env.step(np.zeros(6))

    assert stable_capacity < 0.80
    assert info["fill"] == pytest.approx(stable_capacity)
    assert info["spill"] == pytest.approx(0.80 - stable_capacity)
    assert info["fill"] + info["spill"] == pytest.approx(0.80)
    assert info["cup_runoff"] == pytest.approx(info["spill"])
    cup_rims = env._cup_polygon()[:2]
    lower_rim = cup_rims[np.argmin(cup_rims[:, 1])]
    np.testing.assert_allclose(info["cup_runoff_path"][0], lower_rim, atol=1e-7)
    assert info["cup_runoff_path"][-1, 1] == pytest.approx(env.geometry.table_y)
    assert info["spill_impact_x"] == pytest.approx(info["cup_runoff_path"][-1, 0])
    env.close()


def test_large_tilt_dump_stays_inside_observation_space_and_conserves_volume():
    env = CoffeePouringEnv(horizon=None)
    base_joints = env.reset(seed=3)[1]["joint_angles"].astype(np.float64)
    cup_center = np.array([-0.05, 0.27])
    cup_angle = np.deg2rad(89.0)
    geometry = env.geometry
    cup_wrist = cup_center + env._rotation(cup_angle) @ np.asarray(geometry.cup_grip)
    cup_q1, cup_q2 = env._inverse_kinematics(
        np.asarray(geometry.cup_base),
        cup_wrist,
        geometry.cup_upper,
        geometry.cup_fore,
        elbow_sign=-1.0,
    )
    base_joints[:3] = [cup_q1, cup_q2, cup_angle - cup_q1 - cup_q2]
    env.reset(
        seed=3,
        options={"joint_angles": base_joints, "target_fill": 0.90, "fill": 0.80},
    )
    env.max_flow_rate = 0.0

    observation, _, _, _, info = env.step(np.zeros(6))

    assert info["spill"] > 0.60
    assert info["fill"] + info["spill"] == pytest.approx(0.80)
    assert env.observation_space.contains(observation)
    env.close()


def test_liquid_dynamics_scale_with_simulated_time_not_step_count():
    cup_center = np.array([-0.05, 0.27])
    pot_angle = 1.05
    fills = []
    for dt in (0.0625, 0.125, 0.25):
        env = CoffeePouringEnv(dt=dt, horizon=100)
        cup_mouth = cup_center + np.asarray(env.geometry.cup_mouth)
        joints = _joints_for_geometry(
            env,
            cup_center,
            0.0,
            cup_mouth + np.array([0.10, 0.20]),
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
            options={"joint_angles": joints, "target_fill": 0.70},
        )
        _, _, _, _, info = env.step(np.zeros(6))
        intensities.append(info["flow_rate"])
        assert info["flow_rate"] > 0.0
        assert not info["is_success"]
    assert abs(intensities[0] - intensities[1]) < 0.006
    env.close()


def test_safe_off_center_stream_is_fully_captured_at_rendered_opening():
    env = CoffeePouringEnv(horizon=None)
    cup_center = np.array([-0.05, 0.27])
    cup_angle = 0.0
    pot_angle = 1.05
    opening = _rendered_cup_opening(env, cup_center, cup_angle)
    opening_center = np.mean(opening, axis=0)
    # Aim right of centre so the ballistic stream lands visibly off-centre
    # while retaining ample clearance from both rendered rims.
    spout = np.array([opening_center[0] + 0.100, np.max(opening[:, 1]) + 0.180])
    joints = _joints_for_geometry(env, cup_center, cup_angle, spout, pot_angle)
    env.reset(seed=1, options={"joint_angles": joints, "target_fill": 0.90})

    _, _, _, _, info = env.step(np.zeros(6))

    rendered_opening = _rendered_cup_opening(env, env.tool_positions()["cup_center"], env.cup_angle)
    along, normal_distance = _opening_coordinates(rendered_opening, info["stream_end"])
    rim_clearance = 0.5 * env.geometry.cup_width - abs(along)
    assert rim_clearance > 0.035
    assert info["flow"] > 0.0
    assert info["captured"] == pytest.approx(info["flow"])
    assert info["fill"] == pytest.approx(info["flow"])
    assert info["spill"] == pytest.approx(0.0)
    assert normal_distance == pytest.approx(0.0, abs=1e-6)
    env.close()


def test_tilted_cup_fully_captures_stream_on_rotated_rendered_opening():
    env = CoffeePouringEnv(horizon=None)
    cup_center = np.array([-0.05, 0.27])
    cup_angle = np.deg2rad(12.0)
    pot_angle = 1.05
    opening = _rendered_cup_opening(env, cup_center, cup_angle)
    opening_center = np.mean(opening, axis=0)
    spout = np.array([opening_center[0] + 0.100, np.max(opening[:, 1]) + 0.180])
    joints = _joints_for_geometry(env, cup_center, cup_angle, spout, pot_angle)
    env.reset(seed=1, options={"joint_angles": joints, "target_fill": 0.90})

    _, _, _, _, info = env.step(np.zeros(6))

    rendered_opening = _rendered_cup_opening(env, env.tool_positions()["cup_center"], env.cup_angle)
    along, normal_distance = _opening_coordinates(rendered_opening, info["stream_end"])
    assert abs(along) < 0.050
    assert info["flow"] > 0.0
    assert info["captured"] == pytest.approx(info["flow"])
    assert info["fill"] == pytest.approx(info["flow"])
    assert info["spill"] == pytest.approx(0.0)
    assert normal_distance == pytest.approx(0.0, abs=1e-6)
    env.close()


def test_jet_straddling_rendered_rim_splits_and_draws_exterior_runoff():
    env = CoffeePouringEnv(horizon=None)
    cup_center = np.array([-0.05, 0.27])
    cup_angle = 0.0
    pot_angle = 1.05
    opening = _rendered_cup_opening(env, cup_center, cup_angle)
    opening_center = np.mean(opening, axis=0)
    spout = np.array([opening_center[0] + 0.140, opening_center[1] + 0.180])
    joints = _joints_for_geometry(env, cup_center, cup_angle, spout, pot_angle)
    env.reset(seed=1, options={"joint_angles": joints, "target_fill": 0.90})

    _, _, _, _, info = env.step(np.zeros(6))

    assert 0.0 < info["capture_fraction"] < 1.0
    assert info["fill"] == pytest.approx(info["captured"])
    assert info["fill"] + info["spill"] == pytest.approx(info["flow"])
    np.testing.assert_allclose(info["spill_path"][0], opening[1], atol=1e-6)
    assert info["spill_path"][-1, 1] == pytest.approx(env.geometry.table_y)
    assert info["spill_impact_x"] == pytest.approx(info["spill_path"][-1, 0], abs=5e-4)
    cup_polygon = env._cup_polygon()
    assert not any(
        _strictly_inside_convex_polygon(point, cup_polygon)
        for point in np.vstack([info["stream_path"], info["spill_path"]])
    )
    env.close()


def test_stream_outside_rotated_rendered_opening_spills():
    env = CoffeePouringEnv(horizon=None)
    cup_center = np.array([-0.05, 0.27])
    cup_angle = np.deg2rad(12.0)
    pot_angle = 1.05
    opening = _rendered_cup_opening(env, cup_center, cup_angle)
    opening_center = np.mean(opening, axis=0)
    spout = np.array([opening_center[0] + 0.180, np.max(opening[:, 1]) + 0.180])
    joints = _joints_for_geometry(env, cup_center, cup_angle, spout, pot_angle)
    env.reset(seed=1, options={"joint_angles": joints, "target_fill": 0.90})

    _, _, _, _, info = env.step(np.zeros(6))

    assert info["flow"] > 0.0
    assert info["captured"] == pytest.approx(0.0)
    assert info["fill"] == pytest.approx(0.0)
    assert info["spill"] == pytest.approx(info["flow"])
    np.testing.assert_allclose(info["spill_path"][0], info["stream_end"], atol=1e-6)
    assert info["spill_path"][-1, 1] == pytest.approx(env.geometry.table_y)
    assert info["spill_impact_x"] == pytest.approx(info["spill_path"][-1, 0], abs=5e-4)
    cup_polygon = env._cup_polygon()
    assert not any(
        _strictly_inside_convex_polygon(point, cup_polygon)
        for point in np.vstack([info["stream_path"], info["spill_path"]])
    )
    env.close()


def test_inverted_cup_impacts_follow_solid_boundary_without_crossing_cup():
    configurations = (
        np.array([0.50096306, -0.73832156, -1.71726589, 1.41459615, 0.87317342, -1.04224464]),
        np.array([0.67480088, -0.67781375, -3.14148182, 1.49979242, 0.65396784, -1.01210920]),
        np.array([0.58788369, -0.31576590, 2.84017674, 2.07837846, 1.00368681, -0.47051473]),
        np.array([1.23095226, -1.64949960, 1.65107044, 2.06157912, 0.36348720, -1.04090282]),
    )
    for joints in configurations:
        env = CoffeePouringEnv(horizon=None)
        env.reset(seed=0)
        # These deliberately pathological legacy poses are injected directly;
        # current resets reject their cross-robot collisions before simulation.
        env.joint_angles = joints.copy()
        tools = env.tool_positions()
        flow_rate, exit_speed, _ = env._flow_state()
        stream_path, capture_fraction, spill_path = env._ballistic_stream(
            tools, flow_rate, exit_speed
        )
        cup_polygon = env._cup_polygon(tools)

        assert capture_fraction == 0.0
        if not _strictly_inside_convex_polygon(stream_path[0], cup_polygon):
            assert not env._path_enters_convex_polygon(stream_path[1:], cup_polygon)
        assert not env._path_enters_convex_polygon(spill_path, cup_polygon)
        assert spill_path[-1, 1] == pytest.approx(env.geometry.table_y)
        env.close()


def test_substep_spill_keeps_its_causal_path_when_endpoint_capture_is_clean():
    env = CoffeePouringEnv(horizon=None)
    joints = np.array([1.2031278, -1.67021086, 0.54789723, 1.60754743, 0.88192041, -1.81671864])
    action = np.array([0.26200419, -0.0901046, 0.53661246, 0.91232352, -0.40584921, -0.06192144])
    env.reset(options={"joint_angles": joints})

    _, _, _, _, info = env.step(action)

    assert info["fill"] > 0.0
    assert info["spill"] > 0.0
    assert info["capture_fraction"] == pytest.approx(1.0)
    assert info["direct_spill"] == pytest.approx(info["spill"])
    assert info["direct_spill_rate"] == pytest.approx(info["direct_spill"] / env.dt)
    assert np.linalg.norm(info["direct_spill_path"][-1] - info["direct_spill_path"][0]) > 0.1
    assert info["direct_spill_path"][-1, 1] == pytest.approx(env.geometry.table_y)
    snapshot = env.render_snapshot()
    assert snapshot["state"]["liquid"]["direct_spill_l"] == pytest.approx(info["spill"])
    env.close()


def test_aligned_tilted_pot_fills_cup_and_misaligned_pot_spills():
    cup_center = np.array([-0.05, 0.27])
    pot_angle = 1.05

    aligned = CoffeePouringEnv()
    cup_mouth = cup_center + np.asarray(aligned.geometry.cup_mouth)
    aligned_joints = _joints_for_geometry(
        aligned,
        cup_center,
        0.0,
        cup_mouth + np.array([0.10, 0.20]),
        pot_angle,
    )
    aligned.reset(
        seed=1,
        options={"joint_angles": aligned_joints, "target_fill": 0.70},
    )
    _, _, _, _, aligned_info = aligned.step(np.zeros(6))
    assert aligned_info["flow"] > 0.0
    assert aligned_info["fill"] > 0.0
    assert aligned_info["captured"] == pytest.approx(aligned_info["flow"])
    assert aligned_info["spill"] == pytest.approx(0.0)

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


def test_depleted_source_has_no_stale_stream_at_decision_endpoint():
    env = CoffeePouringEnv(horizon=None)
    joints = env.reset(seed=1)[1]["joint_angles"].astype(np.float64)
    joints[5] = np.deg2rad(80.0) - np.sum(joints[3:5])
    env.reset(
        seed=1,
        options={
            "joint_angles": joints,
            "target_fill": 0.90,
            "fill": 0.70,
            "spill": 0.4995,
        },
    )

    _, _, _, _, info = env.step(np.zeros(6))

    assert info["flow"] == pytest.approx(0.0005)
    assert info["source_remaining"] == 0.0
    assert info["flow_rate"] == 0.0
    np.testing.assert_allclose(
        info["stream_path"],
        np.repeat(info["pot_spout"][None, :], env.STREAM_PATH_SAMPLES, axis=0),
        atol=1e-7,
    )
    env.close()


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
    with pytest.raises(ValueError, match="intersects the table"):
        env.reset(options={"joint_angles": env.joint_low + 0.05})
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
                expected_return = _predicted_return_volume(env, pot_reference_angle, pot_path_rate)
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
