import json

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_pouring_app import (
    InteractiveSession,
    _configured_horizon,
    _metrics,
    _motor_text,
    _time_display,
    build_app,
)


def test_latched_vector_moves_multiple_joints_in_one_environment_step():
    session = InteractiveSession(seed=7001, target_ml=700)
    before = session.env.joint_angles.copy()
    session.motors[[0, 3]] = [1.0, -1.0]
    session.env.step(session.motors)
    after = session.env.joint_angles
    assert after[0] > before[0]
    assert after[3] < before[3]
    np.testing.assert_array_equal(after[[1, 2, 4, 5]], before[[1, 2, 4, 5]])
    assert "cup shoulder ↺" in _motor_text(session.motors)
    assert "pot shoulder ↻" in _motor_text(session.motors)
    session.close()


def test_manual_finish_stops_motors_and_marks_last_transition_truncated():
    session = InteractiveSession(seed=7001, target_ml=700)
    observation = session.observation.copy()
    session.trajectory.append(
        {
            "observation": observation,
            "action": np.ones(6, dtype=np.float32),
            "reward": 0.0,
            "next_observation": observation,
            "terminated": False,
            "truncated": False,
        }
    )
    session.motors.fill(1.0)
    session.finish()
    assert not session.running
    assert session.manual_finish
    assert session.trajectory[-1]["truncated"]
    np.testing.assert_array_equal(session.motors, np.zeros(6))
    session.close()


def test_pause_freezes_time_state_and_demonstration_until_resumed():
    session = InteractiveSession(seed=7001, target_ml=700)
    session.motors[[0, 3]] = [1.0, -1.0]
    joints_before = session.env.joint_angles.copy()
    motors_before = session.motors.copy()

    assert session.toggle_pause()
    assert not session.advance()
    np.testing.assert_array_equal(session.env.joint_angles, joints_before)
    np.testing.assert_array_equal(session.motors, motors_before)
    assert session.env.elapsed_steps == 0
    assert not session.trajectory

    assert not session.toggle_pause()
    assert session.advance()
    assert session.env.elapsed_steps == 1
    assert len(session.trajectory) == 1
    session.close()


def test_wall_clock_speed_changes_timer_interval_not_environment_dynamics():
    slow = InteractiveSession(seed=7001, target_ml=700, speed=0.5)
    fast = InteractiveSession(seed=7001, target_ml=700, speed=2.0)
    action = np.array([0.4, -0.2, 0.1, -0.3, 0.5, -0.4], dtype=np.float32)
    slow.motors[:] = action
    fast.motors[:] = action
    assert slow.advance() and fast.advance()
    np.testing.assert_array_equal(slow.observation, fast.observation)
    assert slow.cumulative_reward == fast.cumulative_reward
    assert slow.timer_interval == pytest.approx(0.25)
    assert fast.timer_interval == pytest.approx(0.0625)
    assert slow.env.FULL_SCALE_QUARTER_TURN_SECONDS / slow.speed == pytest.approx(20.0)
    assert fast.env.FULL_SCALE_QUARTER_TURN_SECONDS / fast.speed == pytest.approx(5.0)
    slow.close()
    fast.close()


def test_timeline_and_metrics_make_finite_and_unlimited_time_clear():
    unlimited = InteractiveSession(seed=7001, target_ml=700)
    assert "Step 0 · no limit" in _time_display(unlimited)
    assert _metrics(unlimited)["time remaining (s)"] == "no limit"

    finite = InteractiveSession(seed=7001, target_ml=700, horizon=10)
    assert "Step 0 / 10" in _time_display(finite)
    assert _metrics(finite)["time remaining (s)"] == pytest.approx(1.25)
    unlimited.close()
    finite.close()


def test_episode_cap_configuration_is_explicit():
    assert _configured_horizon(False, 330) is None
    assert _configured_horizon(True, 330) == 330
    with pytest.raises(ValueError):
        _configured_horizon(True, 0)


def test_animation_snapshots_do_not_create_extra_decisions_or_demonstration_rows():
    session = InteractiveSession(seed=7001, target_ml=700)
    initial = session.animation_snapshot()
    assert initial["schema_version"] == 4
    assert "stream_path_m" in initial["state"]["liquid"]
    assert "spill_path_m" in initial["state"]["liquid"]
    assert "direct_spill_path_m" in initial["state"]["liquid"]
    assert "cup_runoff_path_m" in initial["state"]["liquid"]
    for _ in range(100):
        snapshot = session.animation_snapshot()
        json.dumps(snapshot, allow_nan=False)
        assert snapshot == initial
    assert session.env.elapsed_steps == 0
    assert not session.trajectory

    session.motors[0] = 1.0
    assert session.advance()
    stepped = session.animation_snapshot()
    assert stepped["state"]["step"] == 1
    assert stepped["playback"]["kind"] == "step"
    assert len(session.trajectory) == 1
    for _ in range(100):
        session.animation_snapshot()
    assert session.env.elapsed_steps == 1
    assert len(session.trajectory) == 1
    session.close()


def test_finished_session_reports_zero_live_pour_rate():
    session = InteractiveSession(seed=7001, target_ml=700)
    session.info["flow_rate"] = 0.1
    session.finish()
    assert _metrics(session)["pour rate (mL/s)"] == 0.0
    session.close()


def test_restart_generation_and_revision_reject_stale_animation_payloads():
    session = InteractiveSession(seed=7001, target_ml=700)
    first = session.animation_snapshot()
    session.set_motor(0, 1)
    controlled = session.animation_snapshot()
    assert controlled["playback"]["revision"] > first["playback"]["revision"]
    session.restart(seed=7002, target_ml=710, speed=1.0, horizon=None)
    restarted = session.animation_snapshot()
    assert restarted["playback"]["generation"] > controlled["playback"]["generation"]
    assert restarted["playback"]["revision"] > controlled["playback"]["revision"]
    assert restarted["state"]["step"] == 0
    session.close()


def test_gradio_app_builds_when_interactive_extra_is_installed():
    pytest.importorskip("gradio")
    app = build_app()
    assert app.config["title"] == "KAIST OR Gym — Coffee Pouring"
    assert len(app.config["dependencies"]) >= 20
    canvas = next(
        component
        for component in app.config["components"]
        if component["type"] == "html"
        and "coffee-stage" in component["props"].get("html_template", "")
    )
    assert "requestAnimationFrame" in canvas["props"]["js_on_load"]
    assert "joint_angles_rad" in canvas["props"]["js_on_load"]
    assert "raw.schema_version !== 4" in canvas["props"]["js_on_load"]
    assert "direct_spill_path_m" in canvas["props"]["js_on_load"]
    assert "cup_runoff_path_m" in canvas["props"]["js_on_load"]
    assert canvas["props"].get("min_height") is None
    app.close()
