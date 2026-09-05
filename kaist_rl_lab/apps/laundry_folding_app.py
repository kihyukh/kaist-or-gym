"""Colab-compatible controller for :class:`LaundryFoldingEnv`.

The app is intentionally a thin client.  It latches an eight-dimensional
action, advances the Gymnasium environment once per timer event, and displays
the RGB frame returned by ``env.render()``.  Cloth physics, robot kinematics,
task metrics, and camera projection all remain owned by the environment.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any

import numpy as np

from kaist_rl_lab.envs import LaundryFoldingEnv

DEFAULT_HORIZON = LaundryFoldingEnv.DEFAULT_HORIZON
MIN_TIME_SPEED = 0.25
MAX_TIME_SPEED = 2.0
CAMERA_PRESETS = ("perspective", "top", "front", "side")


def _validated_speed(speed: float) -> float:
    value = float(speed)
    if not np.isfinite(value) or not MIN_TIME_SPEED <= value <= MAX_TIME_SPEED:
        raise ValueError(f"time speed must be between {MIN_TIME_SPEED:g} and {MAX_TIME_SPEED:g}")
    return value


def _validated_camera(preset: str) -> str:
    value = str(preset)
    if value not in CAMERA_PRESETS:
        choices = ", ".join(CAMERA_PRESETS)
        raise ValueError(f"camera preset must be one of: {choices}")
    return value


def _configured_horizon(cap_time: bool, max_steps: float) -> int | None:
    if not cap_time:
        return None
    value = float(max_steps)
    if not np.isfinite(value) or value < 1:
        raise ValueError("maximum steps must be a positive number")
    return round(value)


class InteractiveSession:
    """One isolated environment and demonstration buffer per browser tab."""

    def __init__(
        self,
        seed: int,
        wrinkle_amplitude: float,
        *,
        speed: float = 1.0,
        horizon: int | None = None,
        camera: str = "perspective",
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.lock = RLock()
        self.seed = int(seed)
        self.wrinkle_amplitude = float(wrinkle_amplitude)
        self.speed = _validated_speed(speed)
        self.camera = _validated_camera(camera)
        self._env_kwargs = dict(env_kwargs or {})
        self.env = self._make_env(horizon)
        self.commands = np.zeros(8, dtype=np.float32)
        self.trajectory: list[dict[str, Any]] = []
        self.running = True
        self.paused = False
        self.manual_finish = False
        self.message = "New episode"
        self.cumulative_reward = 0.0
        self.generation = 1
        self.revision = 0
        self.event_kind = "reset"
        self.observation, self.info = self.env.reset(
            seed=self.seed,
            options={"wrinkle_amplitude": self.wrinkle_amplitude},
        )
        self.exported_files: list[Path] = []

    def _make_env(self, horizon: int | None) -> LaundryFoldingEnv:
        env = LaundryFoldingEnv(
            render_mode="rgb_array",
            horizon=horizon,
            **self._env_kwargs,
        )
        env.set_camera(preset=self.camera)
        return env

    def close(self) -> None:
        self.env.close()
        for path in self.exported_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.exported_files.clear()

    def restart(
        self,
        seed: int,
        wrinkle_amplitude: float,
        *,
        speed: float,
        horizon: int | None,
        camera: str,
    ) -> None:
        """Start a fresh episode without replacing this session object."""

        self.close()
        self.seed = int(seed)
        self.wrinkle_amplitude = float(wrinkle_amplitude)
        self.speed = _validated_speed(speed)
        self.camera = _validated_camera(camera)
        self.env = self._make_env(horizon)
        self.commands = np.zeros(8, dtype=np.float32)
        self.trajectory = []
        self.running = True
        self.paused = False
        self.manual_finish = False
        self.message = "New episode"
        self.cumulative_reward = 0.0
        self.generation += 1
        self.revision += 1
        self.event_kind = "reset"
        self.observation, self.info = self.env.reset(
            seed=self.seed,
            options={"wrinkle_amplitude": self.wrinkle_amplitude},
        )

    @property
    def timer_interval(self) -> float:
        """Wall-clock seconds per fixed-duration environment decision."""

        return self.env.dt / self.speed

    def set_speed(self, speed: float) -> None:
        value = _validated_speed(speed)
        if value != self.speed:
            self.speed = value
            self.revision += 1
            self.event_kind = "speed"

    def set_camera(self, preset: str) -> None:
        value = _validated_camera(preset)
        self.env.set_camera(preset=value)
        if value != self.camera:
            self.camera = value
            self.revision += 1
            self.event_kind = "camera"

    def toggle_pause(self) -> bool:
        if self.running:
            self.paused = not self.paused
            self.revision += 1
            self.event_kind = "pause" if self.paused else "resume"
        return self.paused

    def set_command(self, index: int, direction: int) -> float:
        """Latch or toggle one normalized joint/gripper velocity command."""

        if not 0 <= int(index) < len(self.commands):
            raise IndexError("control index is outside the eight-dimensional action")
        if direction not in (-1, 0, 1):
            raise ValueError("control direction must be -1, 0, or 1")
        value = float(direction)
        new_value = 0.0 if direction and self.commands[index] == value else value
        if new_value != self.commands[index]:
            self.commands[index] = new_value
            self.revision += 1
            self.event_kind = "control"
        return new_value

    def stop_all(self) -> None:
        if np.any(self.commands):
            self.commands.fill(0.0)
            self.revision += 1
            self.event_kind = "control"

    def finish(self) -> None:
        """Stop the run and mark its final recorded transition as truncated."""

        self.commands.fill(0.0)
        self.running = False
        self.paused = False
        self.manual_finish = True
        self.revision += 1
        self.event_kind = "finish"
        if self.trajectory:
            final = self.trajectory[-1]
            if not final["terminated"] and not final["truncated"]:
                final["truncated"] = True

    def advance(self) -> bool:
        """Advance exactly one Gymnasium decision unless paused or finished."""

        if not self.running or self.paused:
            return False
        before = self.observation.copy()
        observation, reward, terminated, truncated, info = self.env.step(self.commands)
        self.trajectory.append(
            {
                "observation": before,
                "action": self.commands.copy(),
                "reward": float(reward),
                "next_observation": observation.copy(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        self.observation = observation
        self.info = info
        self.cumulative_reward += reward
        self.revision += 1
        self.event_kind = "step"
        if terminated or truncated:
            self.running = False
            self.commands.fill(0.0)
        return True

    def frame(self) -> np.ndarray:
        """Return the environment-owned camera view without advancing time."""

        frame = self.env.render()
        if frame is None:  # pragma: no cover - guarded by rgb_array construction
            raise RuntimeError("LaundryFoldingEnv did not return an RGB frame")
        return frame


_sessions: dict[str, InteractiveSession] = {}
_sessions_lock = RLock()


def _session_key(request: Any) -> str:
    session_hash = getattr(request, "session_hash", None)
    return str(session_hash or "local-session")


def _new_session(
    seed: float,
    wrinkle_amplitude: float,
    speed: float,
    camera: str,
    cap_time: bool,
    max_steps: float,
) -> InteractiveSession:
    return InteractiveSession(
        int(seed),
        float(wrinkle_amplitude),
        speed=float(speed),
        horizon=_configured_horizon(cap_time, max_steps),
        camera=camera,
    )


def _create_session(
    request: Any,
    seed: float,
    wrinkle_amplitude: float,
    speed: float,
    camera: str,
    cap_time: bool,
    max_steps: float,
) -> InteractiveSession:
    key = _session_key(request)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _new_session(seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
            _sessions[key] = session
            return session

    with session.lock:
        session.restart(
            int(seed),
            float(wrinkle_amplitude),
            speed=float(speed),
            horizon=_configured_horizon(cap_time, max_steps),
            camera=camera,
        )
    return session


def _get_session(
    request: Any,
    seed: float,
    wrinkle_amplitude: float,
    speed: float,
    camera: str,
    cap_time: bool,
    max_steps: float,
) -> InteractiveSession:
    key = _session_key(request)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _new_session(seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
            _sessions[key] = session
        return session


def _cleanup_session(request: Any) -> None:
    key = _session_key(request)
    with _sessions_lock:
        session = _sessions.pop(key, None)
    if session is not None:
        with session.lock:
            session.running = False
            session.close()


def _command_text(commands: np.ndarray) -> str:
    parts: list[str] = []
    for name, value in zip(LaundryFoldingEnv.ACTION_NAMES, commands, strict=True):
        integer = int(value)
        if name.endswith("gripper"):
            word = {-1: "close", 0: "stop", 1: "open"}[integer]
        else:
            word = {-1: "negative", 0: "stop", 1: "positive"}[integer]
        parts.append(f"{name.replace('_', ' ')}: {word}")
    return "  ·  ".join(parts)


def _metrics(session: InteractiveSession) -> dict[str, Any]:
    info = session.info
    if not session.running:
        status = info["termination_reason"] or "finished"
    elif session.paused:
        status = "paused"
    else:
        status = "running"
    grasps = info["grasped_vertices"]
    return {
        "stage": info["stage"],
        "straightness": round(float(info["straightness"]), 3),
        "fold score": round(float(info["fold_score"]), 3),
        "coverage": round(float(info["coverage"]), 3),
        "bimanual tension": round(float(info["bimanual_tension"]), 3),
        "grasped vertices": f"left {len(grasps[0])} · right {len(grasps[1])}",
        "dropped fraction": round(float(info["dropped_fraction"]), 3),
        "episode reward": round(float(session.cumulative_reward), 3),
        "recorded transitions": len(session.trajectory),
        "success": bool(info["is_success"]),
        "status": status,
    }


def _status(session: InteractiveSession, message: str) -> str:
    if not session.running:
        clock = "stopped"
    elif session.paused:
        clock = "paused"
    else:
        clock = f"running at {session.speed:g}×"
    return f"**{message}**  ·  clock {clock}  ·  eight controls latch independently."


def _time_display(session: InteractiveSession) -> str:
    step = int(session.info["elapsed_steps"])
    if session.env.horizon is None:
        step_label = f"{step} · no limit"
    else:
        step_label = f"{step} / {session.env.horizon}"
    if not session.running:
        clock = "stopped"
    elif session.paused:
        clock = "paused"
    else:
        clock = "running"
    return (
        f"## Step {step_label}\n"
        f"{session.info['elapsed_time']:.2f} s simulated · **{clock}** · "
        f"{session.speed:g}× wall speed"
    )


def _view(session: InteractiveSession, message: str | None = None):
    if message is not None:
        session.message = message
    return (
        session.frame(),
        _time_display(session),
        _status(session, session.message),
        _command_text(session.commands),
        _metrics(session),
    )


def _write_demonstration(session: InteractiveSession) -> Path:
    """Write one behavior-cloning trajectory and its simulator metadata."""

    if not session.trajectory:
        raise ValueError("no transitions have been recorded")
    with NamedTemporaryFile(
        prefix="laundry_folding_demonstration_", suffix=".npz", delete=False
    ) as handle:
        path = Path(handle.name)

    env = session.env
    np.savez_compressed(
        path,
        observations=np.stack([row["observation"] for row in session.trajectory]),
        actions=np.stack([row["action"] for row in session.trajectory]),
        rewards=np.asarray([row["reward"] for row in session.trajectory], dtype=np.float64),
        next_observations=np.stack([row["next_observation"] for row in session.trajectory]),
        terminated=np.asarray([row["terminated"] for row in session.trajectory], dtype=np.bool_),
        truncated=np.asarray([row["truncated"] for row in session.trajectory], dtype=np.bool_),
        seed=np.asarray(session.seed),
        wrinkle_amplitude_m=np.asarray(session.wrinkle_amplitude),
        manual_finish=np.asarray(session.manual_finish),
        dt=np.asarray(env.dt),
        horizon=np.asarray(-1 if env.horizon is None else env.horizon),
        physics_model=np.asarray(env.PHYSICS_MODEL),
        action_names=np.asarray(env.ACTION_NAMES),
        joint_names=np.asarray(env.JOINT_NAMES),
        observation_names=np.asarray(env.OBSERVATION_NAMES),
        max_joint_speeds_rad_s=env.max_joint_speeds.copy(),
        max_gripper_speed_m_s=np.asarray(env.max_gripper_speed),
        joint_low_rad=env.joint_low.copy(),
        joint_high_rad=env.joint_high.copy(),
        mesh_rows=np.asarray(env.mesh_rows),
        mesh_cols=np.asarray(env.mesh_cols),
        mesh_faces=env.faces.copy(),
        mesh_rest_positions_m=env.rest_positions.copy(),
        towel_size_m=np.asarray(
            [env.geometry.towel_width, env.geometry.towel_depth, env.geometry.towel_thickness]
        ),
        solver_iterations=np.asarray(env.solver_iterations),
        max_physics_step_s=np.asarray(env.MAX_PHYSICS_STEP),
    )
    session.exported_files.append(path)
    return path


def build_app():
    """Build and return the Colab-compatible Gradio application."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The interactive app needs Gradio. Install with: "
            "pip install 'kaist-rl-lab[interactive]'"
        ) from exc

    def timer_update(session: InteractiveSession, active: bool):
        return gr.Timer(value=session.timer_interval, active=active)

    def initialize(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _create_session(
            request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps
        )
        return _view(session, "New randomized towel")

    def reset(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _create_session(
            request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps
        )
        return (
            *_view(session, "Episode reset; the timer has started"),
            timer_update(session, True),
        )

    def tick(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            if not session.running or session.paused:
                return (*_view(session), timer_update(session, False))
            session.advance()
            if not session.running:
                message = "Fold complete" if session.info["is_success"] else "Episode complete"
                return (*_view(session, message), timer_update(session, False))
            # An ordinary tick must not reconfigure the timer: a late response
            # could otherwise undo a newer pause or speed selection.
            return (*_view(session), gr.skip())

    def set_control(
        index: int,
        direction: int,
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            value = session.set_command(index, direction)
            name = LaundryFoldingEnv.ACTION_NAMES[index].replace("_", " ")
            if name.endswith("gripper"):
                word = {-1: "closing", 0: "stopped", 1: "opening"}[int(value)]
            else:
                word = {-1: "negative", 0: "stopped", 1: "positive"}[int(value)]
            return _view(session, f"{name.capitalize()} {word}")

    def stop_all(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            session.stop_all()
            return _view(session, "All controls stopped")

    def toggle_pause(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            if not session.running:
                message = "Episode is complete; reset to start again"
            elif session.toggle_pause():
                message = "Time paused; latched controls are preserved"
            else:
                message = "Time resumed"
            active = session.running and not session.paused
            return (*_view(session, message), timer_update(session, active))

    def change_speed(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            session.set_speed(speed)
            active = session.running and not session.paused
            return (
                *_view(session, f"Time speed set to {session.speed:g}×"),
                timer_update(session, active),
            )

    def change_camera(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            session.set_camera(camera)
            return _view(session, f"Camera set to {session.camera}")

    def finish(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            session.finish()
            return (
                *_view(session, "Demonstration finished"),
                timer_update(session, False),
            )

    def export_demonstration(
        seed: float,
        wrinkle_amplitude: float,
        speed: float,
        camera: str,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, wrinkle_amplitude, speed, camera, cap_time, max_steps)
        with session.lock:
            if not session.trajectory:
                return None, _status(session, "No transitions recorded yet")
            path = _write_demonstration(session)
            session.message = "Demonstration file ready"
            return str(path), _status(session, session.message)

    def control_handler(index: int, direction: int):
        def handler(
            seed_value: float,
            wrinkle_value: float,
            speed_value: float,
            camera_value: str,
            cap_value: bool,
            steps_value: float,
            request: gr.Request,
        ):
            return set_control(
                index,
                direction,
                seed_value,
                wrinkle_value,
                speed_value,
                camera_value,
                cap_value,
                steps_value,
                request,
            )

        return handler

    def cleanup(request: gr.Request):
        _cleanup_session(request)

    with gr.Blocks(title="KAIST RL Lab — Laundry Folding") as demo:
        gr.Markdown(
            "# Two-arm laundry folding\n"
            "First straighten the randomized towel, then fold it across the marked crease. "
            "Each command stays latched, so both arms and grippers can move together. "
            "The displayed image comes directly from the Gymnasium environment; every timer "
            "event is exactly one recorded decision."
        )
        with gr.Row():
            with gr.Column(scale=3):
                time_display = gr.Markdown(elem_classes=["laundry-time"])
                frame = gr.Image(
                    type="numpy",
                    format="png",
                    image_mode="RGB",
                    label="Environment camera",
                    interactive=False,
                    buttons=["fullscreen"],
                )
                status = gr.Markdown()
                command_state = gr.Textbox(
                    label="Latched 8-D action",
                    interactive=False,
                    lines=3,
                    elem_classes=["laundry-actions"],
                )
            with gr.Column(scale=2):
                with gr.Row():
                    seed = gr.Number(value=7001, precision=0, label="Seed")
                    wrinkle = gr.Slider(
                        0.0,
                        0.12,
                        value=0.075,
                        step=0.005,
                        label="Initial wrinkle amplitude (m)",
                    )
                camera = gr.Radio(
                    choices=[
                        ("Perspective", "perspective"),
                        ("Top", "top"),
                        ("Front", "front"),
                        ("Side", "side"),
                    ],
                    value="perspective",
                    label="Camera preset",
                )
                with gr.Row():
                    speed = gr.Radio(
                        choices=[("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)],
                        value=1.0,
                        label="Time speed",
                    )
                    pause_button = gr.Button("Pause / resume time")
                with gr.Row():
                    cap_time = gr.Checkbox(value=False, label="Cap episode on reset")
                    max_steps = gr.Number(
                        value=DEFAULT_HORIZON,
                        precision=0,
                        label="Maximum steps if capped",
                    )
                gr.Markdown(
                    "Seed, wrinkle, and episode-length settings apply on **Reset + start**."
                )
                with gr.Row():
                    reset_button = gr.Button("Reset + start", variant="primary")
                    stop_button = gr.Button("Stop all controls")
                    finish_button = gr.Button("Finish")

                control_buttons: list[tuple[Any, Any, Any]] = []
                for action_name in LaundryFoldingEnv.ACTION_NAMES:
                    with gr.Row(equal_height=True):
                        gr.Markdown(
                            action_name.replace("_", " ").title(),
                            elem_classes=["laundry-control-name"],
                        )
                        if action_name.endswith("gripper"):
                            negative = gr.Button("Close", min_width=58)
                            zero = gr.Button("■", min_width=44)
                            positive = gr.Button("Open", min_width=58)
                        else:
                            negative = gr.Button("−", min_width=50)
                            zero = gr.Button("■", min_width=50)
                            positive = gr.Button("+", min_width=50)
                        control_buttons.append((negative, zero, positive))

                metrics = gr.JSON(label="Task metrics")
                export_button = gr.Button("Export human demonstration (.npz)")
                export_file = gr.File(label="Recorded demonstration", interactive=False)

        timer = gr.Timer(value=LaundryFoldingEnv.DEFAULT_DT, active=True)
        outputs = [frame, time_display, status, command_state, metrics]
        config_inputs = [seed, wrinkle, speed, camera, cap_time, max_steps]
        demo.load(initialize, inputs=config_inputs, outputs=outputs, queue=False)
        reset_button.click(
            reset,
            inputs=config_inputs,
            outputs=[*outputs, timer],
            queue=False,
        )
        pause_button.click(
            toggle_pause,
            inputs=config_inputs,
            outputs=[*outputs, timer],
            queue=False,
        )
        speed.change(
            change_speed,
            inputs=config_inputs,
            outputs=[*outputs, timer],
            queue=False,
        )
        camera.change(
            change_camera,
            inputs=config_inputs,
            outputs=outputs,
            queue=False,
        )
        stop_button.click(stop_all, inputs=config_inputs, outputs=outputs, queue=False)
        finish_button.click(
            finish,
            inputs=config_inputs,
            outputs=[*outputs, timer],
            queue=False,
        )
        timer.tick(
            tick,
            inputs=config_inputs,
            outputs=[*outputs, timer],
            queue=False,
            trigger_mode="once",
        )

        for index, (negative, zero, positive) in enumerate(control_buttons):
            for button, direction in ((negative, -1), (zero, 0), (positive, 1)):
                button.click(
                    control_handler(index, direction),
                    inputs=config_inputs,
                    outputs=outputs,
                    queue=False,
                )

        export_button.click(
            export_demonstration,
            inputs=config_inputs,
            outputs=[export_file, status],
            queue=False,
        )
        demo.unload(cleanup)
    return demo


def main() -> None:
    """Launch locally; Colab notebooks should call :func:`build_app` directly."""

    build_app().launch()


if __name__ == "__main__":  # pragma: no cover
    main()
