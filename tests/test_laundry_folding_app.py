import numpy as np
import pytest

from kaist_rl_lab.apps.laundry_folding_app import (
    InteractiveSession,
    _command_text,
    _configured_horizon,
    _metrics,
    _time_display,
    _write_demonstration,
    build_app,
)


def _session(**kwargs):
    return InteractiveSession(
        seed=7001,
        wrinkle_amplitude=0.075,
        env_kwargs={"mesh_rows": 5, "mesh_cols": 5},
        **kwargs,
    )


def test_latched_eight_dimensional_action_advances_exactly_one_decision():
    session = _session()
    session.set_command(0, 1)
    session.set_command(4, -1)
    session.set_command(3, -1)
    expected_action = session.commands.copy()

    assert session.advance()
    assert session.env.elapsed_steps == 1
    assert len(session.trajectory) == 1
    np.testing.assert_array_equal(session.trajectory[0]["action"], expected_action)
    np.testing.assert_array_equal(session.env.last_action, expected_action)
    text = _command_text(expected_action)
    assert "left shoulder yaw: positive" in text
    assert "left gripper: close" in text
    assert "right shoulder yaw: negative" in text

    frame = session.frame()
    assert frame.shape == (session.env.height, session.env.width, 3)
    assert session.env.elapsed_steps == 1
    assert len(session.trajectory) == 1
    session.close()


def test_pause_freezes_simulation_and_recording_until_resumed():
    session = _session()
    session.commands[[0, 4]] = [1.0, -1.0]
    observation = session.observation.copy()

    assert session.toggle_pause()
    assert not session.advance()
    np.testing.assert_array_equal(session.observation, observation)
    assert session.env.elapsed_steps == 0
    assert not session.trajectory
    np.testing.assert_array_equal(session.commands[[0, 4]], [1.0, -1.0])

    assert not session.toggle_pause()
    assert session.advance()
    assert session.env.elapsed_steps == 1
    assert len(session.trajectory) == 1
    session.close()


def test_wall_speed_and_camera_change_playback_not_environment_time():
    session = _session(speed=0.5)
    assert session.timer_interval == pytest.approx(0.2)
    session.set_speed(2.0)
    assert session.timer_interval == pytest.approx(0.05)
    assert session.env.dt == pytest.approx(0.1)

    before = session.observation.copy()
    step_before = session.env.elapsed_steps
    session.set_camera("top")
    assert session.camera == "top"
    assert session.env.camera_elevation == pytest.approx(88.0)
    assert session.env.elapsed_steps == step_before
    np.testing.assert_array_equal(session.observation, before)
    assert not session.trajectory

    with pytest.raises(ValueError):
        session.set_speed(2.01)
    with pytest.raises(ValueError):
        session.set_camera("orbit")
    session.close()


def test_time_display_optional_cap_and_manual_finish_are_explicit():
    unlimited = _session()
    assert "Step 0 · no limit" in _time_display(unlimited)
    assert _configured_horizon(False, 500) is None

    finite = _session(horizon=3)
    assert "Step 0 / 3" in _time_display(finite)
    assert _configured_horizon(True, 3) == 3
    with pytest.raises(ValueError):
        _configured_horizon(True, 0)

    finite.advance()
    finite.finish()
    assert not finite.running
    assert finite.manual_finish
    assert finite.trajectory[-1]["truncated"]
    np.testing.assert_array_equal(finite.commands, np.zeros(8, dtype=np.float32))
    assert _metrics(finite)["status"] == "finished"
    unlimited.close()
    finite.close()


def test_bc_export_contains_transitions_and_reproducibility_metadata():
    session = _session(camera="side")
    session.commands[[1, 3, 5, 7]] = [0.5, -1.0, -0.5, -1.0]
    session.advance()
    session.finish()
    path = _write_demonstration(session)

    with np.load(path) as data:
        expected = {
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
            "physics_model",
            "action_names",
            "observation_names",
            "mesh_rows",
            "mesh_cols",
            "mesh_faces",
            "mesh_rest_positions_m",
            "solver_iterations",
            "max_physics_step_s",
        }
        assert expected <= set(data.files)
        assert data["observations"].shape == (1, session.env.observation_space.shape[0])
        assert data["actions"].shape == (1, 8)
        assert data["next_observations"].shape == data["observations"].shape
        assert data["truncated"].tolist() == [True]
        assert data["manual_finish"].item()
        assert data["horizon"].item() == -1
        assert data["physics_model"].item() == session.env.PHYSICS_MODEL
        assert data["action_names"].tolist() == list(session.env.ACTION_NAMES)
        assert data["mesh_rows"].item() == session.env.mesh_rows
        assert data["mesh_cols"].item() == session.env.mesh_cols
        np.testing.assert_array_equal(data["mesh_faces"], session.env.faces)
        np.testing.assert_allclose(data["mesh_rest_positions_m"], session.env.rest_positions)

    assert path.exists()
    session.close()
    assert not path.exists()


def test_gradio_app_builds_with_environment_image_and_all_controls():
    pytest.importorskip("gradio")
    app = build_app()
    assert app.config["title"] == "KAIST RL Lab — Laundry Folding"

    components = app.config["components"]
    assert any(component["type"] == "image" for component in components)
    control_labels = {
        component["props"].get("label")
        for component in components
        if component["type"] in {"textbox", "radio"}
    }
    assert "Latched 8-D action" in control_labels
    assert "Camera preset" in control_labels
    assert "Time speed" in control_labels
    assert len(app.config["dependencies"]) >= 30
    app.close()
