import json

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_pouring_app import (
    InteractiveSession,
    _configured_horizon,
    _metrics,
    _motor_text,
    _time_display,
    _validated_motor_command,
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


def test_each_joint_button_sets_an_explicit_latched_command():
    session = InteractiveSession(seed=7001, target_ml=700)
    assert session.set_motor(2, 1) == 1
    assert session.set_motor(2, 1) == 1
    assert session.motors[2] == 1
    assert session.set_motor(2, -1) == -1
    assert session.motors[2] == -1
    assert session.set_motor(2, 0) == 0
    assert session.motors[2] == 0
    session.close()


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_canvas_motor_command_validation_accepts_all_button_commands(index, direction):
    assert _validated_motor_command(index, direction) == (index, direction)


@pytest.mark.parametrize(
    ("index", "direction"),
    [(-1, 0), (6, 0), (0, -2), (0, 2), (True, 0), (0, False), ("0", 0)],
)
def test_canvas_motor_command_validation_rejects_malformed_payloads(index, direction):
    with pytest.raises((TypeError, ValueError)):
        _validated_motor_command(index, direction)


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
    assert slow.timer_interval == pytest.approx(1.0)
    assert fast.timer_interval == pytest.approx(0.25)
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
    assert initial["playback"]["motors"] == [0.0] * 6
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
    assert stepped["playback"]["motors"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
    assert canvas["props"]["html_template"].count("coffee-joint-control ") == 6
    assert canvas["props"]["html_template"].count('class="coffee-joint-button"') == 18
    assert canvas["props"]["html_template"].count('data-joint-index="') == 6
    assert 'sequence: inputSequence' in canvas["props"]["js_on_load"]
    assert "playback.motors" in canvas["props"]["js_on_load"]
    assert canvas["props"].get("min_height") is None
    standalone_motor_buttons = {
        component["props"].get("value")
        for component in app.config["components"]
        if component["type"] == "button"
    }
    assert standalone_motor_buttons == {
        "Resume time", "Reset + start", "Stop all motors", "Save trajectory"
    }
    canvas_click_dependencies = [
        dependency
        for dependency in app.config["dependencies"]
        if (canvas["id"], "click") in dependency["targets"]
    ]
    assert len(canvas_click_dependencies) == 1
    assert canvas_click_dependencies[0]["collects_event_data"]
    assert canvas_click_dependencies[0]["trigger_mode"] == "multiple"
    app.close()


def test_ordered_controls_keep_pause_motors_and_trajectory_in_sync():
    gr = pytest.importorskip("gradio")
    app = build_app()
    handlers = {h.fn.__name__: h.fn for h in app.fns.values() if h.fn}
    request = gr.Request(session_hash="coffee-ordered-controls")

    def control(sequence, kind="motor", motors=None, paused=False, generation=1):
        event = gr.EventData(None, {
            "sequence": sequence, "generation": generation, "kind": kind,
            "motors": motors or [0] * 6, "paused": paused,
        })
        return handlers["canvas_joint_control"](request, event)

    try:
        initial = handlers["initialize"](request)
        assert json.loads(initial[0])["playback"]["paused"]
        assert initial[3].value == "Resume time"
        control(1, motors=[1, 0, 0, 0, 0, 0], paused=True)
        assert json.loads(handlers["tick"](request)[0])["state"]["step"] == 0
        control(2, kind="pause", motors=[1, 0, 0, 0, 0, 0])
        assert json.loads(handlers["tick"](request)[0])["state"]["step"] == 4

        # Requests can arrive in reverse order. The complete desired vector and
        # clock state from the newest request must win, including after Stop.
        control(5, motors=[-1, 0, 1, 0, 0, 0], paused=True)
        assert control(4, motors=[1, 0, 0, 0, 0, 0]) == ({"__type__": "update"},) * 5
        state = json.loads(handlers["tick"](request)[0])
        assert state["state"]["step"] == 4
        assert state["playback"]["motors"] == [-1, 0, 1, 0, 0, 0]
        assert state["playback"]["input_sequence"] == 5
        stopped = control(6, kind="stop", paused=True)
        assert stopped[3].value == "Resume time" and stopped[4] == {"__type__": "update"}
        control(5, motors=[1, 0, 0, 0, 0, 0])
        assert json.loads(handlers["tick"](request)[0])["playback"]["motors"] == [0] * 6

        resumed = control(7, kind="pause")
        assert resumed[3].value == "Pause time" and resumed[4].active
        assert json.loads(handlers["tick"](request)[0])["state"]["step"] == 8
        reset = control(8, kind="reset")
        fresh = json.loads(reset[0])
        assert fresh["state"]["step"] == 0
        assert fresh["playback"]["generation"] == 2
        assert fresh["playback"]["motors"] == [0] * 6
        assert not fresh["playback"]["paused"] and reset[4].active
        control(9, motors=[1] * 6, generation=1)
        assert json.loads(handlers["tick"](request)[0])["playback"]["motors"] == [0] * 6
    finally:
        handlers["cleanup"](request)
        app.close()
