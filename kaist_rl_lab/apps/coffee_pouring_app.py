"""Gradio controller for the coffee-pouring Gymnasium environment.

This module intentionally contains no robot geometry or dynamics.  It latches
joint commands, calls ``env.step()``, and forwards environment-owned render
snapshots to a smooth browser canvas.  ``env.render()`` remains the canonical
RGB renderer for training, tests, and video export.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any

import numpy as np

from kaist_rl_lab.envs import CoffeePouringEnv
from kaist_rl_lab.envs.coffee_pouring_canvas import (
    CANVAS_CSS,
    CANVAS_HTML,
    CANVAS_JAVASCRIPT,
)

DEFAULT_HORIZON = CoffeePouringEnv.DEFAULT_HORIZON
MIN_TIME_SPEED = 0.25
MAX_TIME_SPEED = 2.0


def _validated_speed(speed: float) -> float:
    value = float(speed)
    if not np.isfinite(value) or not MIN_TIME_SPEED <= value <= MAX_TIME_SPEED:
        raise ValueError(f"time speed must be between {MIN_TIME_SPEED:g} and {MAX_TIME_SPEED:g}")
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
        target_ml: float,
        *,
        speed: float = 1.0,
        horizon: int | None = None,
    ) -> None:
        self.lock = RLock()
        self.env = CoffeePouringEnv(render_mode="rgb_array", horizon=horizon)
        self.seed = int(seed)
        self.speed = _validated_speed(speed)
        self.motors = np.zeros(6, dtype=np.float32)
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
            seed=self.seed, options={"target_fill": float(target_ml) / 1000.0}
        )
        self.exported_files: list[Path] = []

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
        target_ml: float,
        *,
        speed: float,
        horizon: int | None,
    ) -> None:
        """Start a fresh episode without replacing this session object."""

        self.close()
        self.env = CoffeePouringEnv(render_mode="rgb_array", horizon=horizon)
        self.seed = int(seed)
        self.speed = _validated_speed(speed)
        self.motors = np.zeros(6, dtype=np.float32)
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
            seed=self.seed, options={"target_fill": float(target_ml) / 1000.0}
        )

    def finish(self) -> None:
        """Stop this run and mark its final recorded transition as truncated."""

        self.motors.fill(0.0)
        self.running = False
        self.paused = False
        self.manual_finish = True
        self.revision += 1
        self.event_kind = "finish"
        if self.trajectory:
            final = self.trajectory[-1]
            if not final["terminated"] and not final["truncated"]:
                final["truncated"] = True

    @property
    def timer_interval(self) -> float:
        """Wall-clock seconds per fixed-duration environment step."""

        return self.env.dt / self.speed

    def set_speed(self, speed: float) -> None:
        value = _validated_speed(speed)
        if value != self.speed:
            self.speed = value
            self.revision += 1
            self.event_kind = "speed"

    def toggle_pause(self) -> bool:
        if self.running:
            self.paused = not self.paused
            self.revision += 1
            self.event_kind = "pause" if self.paused else "resume"
        return self.paused

    def set_motor(self, index: int, direction: int) -> float:
        """Latch or toggle one normalized motor command."""

        value = float(direction)
        new_value = 0.0 if self.motors[index] == value else value
        if new_value != self.motors[index]:
            self.motors[index] = new_value
            self.revision += 1
            self.event_kind = "control"
        return new_value

    def stop_all_motors(self) -> None:
        if np.any(self.motors):
            self.motors.fill(0.0)
            self.revision += 1
            self.event_kind = "control"

    def animation_snapshot(self) -> dict[str, Any]:
        """Combine an environment scene keyframe with wall-clock playback state."""

        snapshot = self.env.render_snapshot()
        snapshot["playback"] = {
            "generation": int(self.generation),
            "revision": int(self.revision),
            "kind": self.event_kind,
            "speed": float(self.speed),
            "paused": bool(self.paused),
            "running": bool(self.running),
            "decision_interval_wall_ms": float(1000.0 * self.timer_interval),
        }
        return snapshot

    def advance(self) -> bool:
        """Advance one simulator step, unless this session is paused or finished."""

        if not self.running or self.paused:
            return False
        before = self.observation.copy()
        observation, reward, terminated, truncated, info = self.env.step(self.motors)
        self.trajectory.append(
            {
                "observation": before,
                "action": self.motors.copy(),
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
            self.motors.fill(0.0)
        return True


_sessions: dict[str, InteractiveSession] = {}
_sessions_lock = RLock()


def _session_key(request) -> str:
    session_hash = getattr(request, "session_hash", None)
    return str(session_hash or "local-session")


def _new_session(
    seed: float,
    target_ml: float,
    speed: float,
    cap_time: bool,
    max_steps: float,
) -> InteractiveSession:
    return InteractiveSession(
        int(seed),
        float(target_ml),
        speed=float(speed),
        horizon=_configured_horizon(cap_time, max_steps),
    )


def _create_session(
    request,
    seed: float,
    target_ml: float,
    speed: float,
    cap_time: bool,
    max_steps: float,
) -> InteractiveSession:
    key = _session_key(request)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _new_session(seed, target_ml, speed, cap_time, max_steps)
            _sessions[key] = session
            return session

    with session.lock:
        session.restart(
            int(seed),
            float(target_ml),
            speed=float(speed),
            horizon=_configured_horizon(cap_time, max_steps),
        )
    return session


def _get_session(
    request,
    seed: float,
    target_ml: float,
    speed: float,
    cap_time: bool,
    max_steps: float,
) -> InteractiveSession:
    key = _session_key(request)
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _new_session(seed, target_ml, speed, cap_time, max_steps)
            _sessions[key] = session
        return session


def _cleanup_session(request) -> None:
    key = _session_key(request)
    with _sessions_lock:
        session = _sessions.pop(key, None)
    if session is not None:
        with session.lock:
            session.running = False
            session.close()


def _motor_text(motors: np.ndarray) -> str:
    symbols = {-1: "↻", 0: "■", 1: "↺"}
    parts = []
    for name, value in zip(CoffeePouringEnv.JOINT_NAMES, motors):
        parts.append("{} {}".format(name.replace("_", " "), symbols[int(value)]))
    return "  ·  ".join(parts)


def _metrics(session: InteractiveSession) -> dict[str, Any]:
    info = session.info
    time_remaining = info["time_remaining"]
    displayed_flow_rate = float(info["flow_rate"]) if session.running else 0.0
    if not session.running:
        episode_status = "finished"
    elif session.paused:
        episode_status = "paused"
    else:
        episode_status = "running"
    return {
        "current step": int(info["elapsed_steps"]),
        "simulated time (s)": round(float(info["elapsed_time"]), 3),
        "time remaining (s)": (
            "no limit" if time_remaining is None else round(float(time_remaining), 3)
        ),
        "time speed": f"{session.speed:g}×",
        "target (mL)": round(float(info["target_fill"]) * 1000.0, 1),
        "in pot (mL)": round(float(info["source_remaining"]) * 1000.0, 1),
        "in cup (mL)": round(float(info["fill"]) * 1000.0, 1),
        "fill error (mL)": round(float(info["fill_error"]) * 1000.0, 1),
        "spilled (mL)": round(float(info["spill"]) * 1000.0, 1),
        "pour rate (mL/s)": round(displayed_flow_rate * 1000.0, 1),
        "episode reward": round(float(session.cumulative_reward), 3),
        "recorded transitions": len(session.trajectory),
        "success": bool(info["is_success"]),
        "status": info["termination_reason"] or episode_status,
    }


def _status(session: InteractiveSession, message: str) -> str:
    if not session.running:
        clock = "stopped"
    elif session.paused:
        clock = "paused"
    else:
        clock = f"running at {session.speed:g}×"
    return f"**{message}**  ·  clock {clock}  ·  commands latch until changed or stopped."


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
        f"{session.info['elapsed_time']:.3f} s simulated · **{clock}** · "
        f"{session.speed:g}× speed"
    )


def _view(session: InteractiveSession, message: str | None = None):
    if message is not None:
        session.message = message
    return (
        json.dumps(session.animation_snapshot(), separators=(",", ":"), allow_nan=False),
        _time_display(session),
        _status(session, session.message),
        _motor_text(session.motors),
        _metrics(session),
    )


def build_app():
    """Build and return the Colab-compatible Gradio application."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The interactive app needs Gradio. Install with: pip install 'kaist-rl-lab[interactive]'"
        ) from exc

    def timer_update(session: InteractiveSession, active: bool):
        return gr.Timer(value=session.timer_interval, active=active)

    def initialize(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _create_session(request, seed, target_ml, speed, cap_time, max_steps)
        return _view(session, "New episode")

    def reset(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _create_session(request, seed, target_ml, speed, cap_time, max_steps)
        return (
            *_view(session, "Episode reset; the timer has started"),
            timer_update(session, True),
        )

    def tick(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            if not session.running or session.paused:
                return (*_view(session), timer_update(session, False))
            session.advance()
            if not session.running:
                message = "Success" if session.info["is_success"] else "Episode complete"
                return (*_view(session, message), timer_update(session, False))
            # Do not reconfigure the timer on an ordinary tick: an older tick
            # response must never undo a newer pause or speed change.
            return (*_view(session), gr.skip())

    def set_motor(
        index: int,
        direction: int,
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            new_value = session.set_motor(index, direction)
            label = CoffeePouringEnv.JOINT_NAMES[index].replace("_", " ")
            words = {-1: "clockwise", 0: "stopped", 1: "counterclockwise"}
            return _view(session, f"{label.capitalize()} {words[int(new_value)]}")

    def stop_all(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            session.stop_all_motors()
            return _view(session, "All motors stopped")

    def toggle_pause(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            if not session.running:
                message = "Episode is already complete; reset to start again"
            elif session.toggle_pause():
                message = "Time paused; latched motor commands are preserved"
            else:
                message = "Time resumed"
            timer_active = session.running and not session.paused
            return (*_view(session, message), timer_update(session, timer_active))

    def change_speed(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            session.set_speed(speed)
            timer_active = session.running and not session.paused
            return (
                *_view(session, f"Time speed set to {session.speed:g}×"),
                timer_update(session, timer_active),
            )

    def finish(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            session.finish()
            return (
                *_view(session, "Demonstration finished"),
                timer_update(session, False),
            )

    def export_demonstration(
        seed: float,
        target_ml: float,
        speed: float,
        cap_time: bool,
        max_steps: float,
        request: gr.Request,
    ):
        session = _get_session(request, seed, target_ml, speed, cap_time, max_steps)
        with session.lock:
            if not session.trajectory:
                return None, _status(session, "No transitions recorded yet")
            with NamedTemporaryFile(
                prefix="coffee_demonstration_", suffix=".npz", delete=False
            ) as handle:
                path = Path(handle.name)
            np.savez_compressed(
                path,
                observations=np.stack([row["observation"] for row in session.trajectory]),
                actions=np.stack([row["action"] for row in session.trajectory]),
                rewards=np.asarray([row["reward"] for row in session.trajectory]),
                next_observations=np.stack([row["next_observation"] for row in session.trajectory]),
                terminated=np.asarray([row["terminated"] for row in session.trajectory]),
                truncated=np.asarray([row["truncated"] for row in session.trajectory]),
                seed=np.asarray(session.seed),
                target_fill=np.asarray(session.env.target_fill),
                manual_finish=np.asarray(session.manual_finish),
                dt=np.asarray(session.env.dt),
                horizon=np.asarray(-1 if session.env.horizon is None else session.env.horizon),
                max_joint_speeds=session.env.max_joint_speeds.copy(),
                full_scale_quarter_turn_seconds=np.asarray(
                    session.env.FULL_SCALE_QUARTER_TURN_SECONDS
                ),
                cup_capacity_l=np.asarray(session.env.CUP_CAPACITY),
                pot_capacity_l=np.asarray(session.env.POT_CAPACITY),
                initial_pot_volume_l=np.asarray(session.env.INITIAL_POT_VOLUME),
                physics_model=np.asarray("torricelli_ballistic_v3"),
                joint_names=np.asarray(CoffeePouringEnv.JOINT_NAMES),
                observation_names=np.asarray(CoffeePouringEnv.OBSERVATION_NAMES),
            )
            session.exported_files.append(path)
            session.message = "Demonstration file ready"
            return str(path), _status(session, session.message)

    def motor_handler(index: int, direction: int):
        def handler(
            seed_value: float,
            target_value: float,
            speed_value: float,
            cap_value: bool,
            steps_value: float,
            request: gr.Request,
        ):
            return set_motor(
                index,
                direction,
                seed_value,
                target_value,
                speed_value,
                cap_value,
                steps_value,
                request,
            )

        return handler

    def cleanup(request: gr.Request):
        _cleanup_session(request)

    with gr.Blocks(title="KAIST OR Gym — Coffee Pouring") as demo:
        gr.Markdown(
            "# Fixed-link coffee pouring\n"
            "Each click latches one joint **counterclockwise**, **stopped**, or "
            "**clockwise**. Several joints can run together while the timer keeps stepping "
            "the Gymnasium environment. At full command, a 90° joint rotation takes about "
            "ten simulated seconds. Motor commands are held between discrete decisions; "
            "the scene animates smoothly between them. The rigid bodies stop at table or "
            "robot contact. Coffee follows a gravity-driven "
            "trajectory, and its flow rate changes with pot tilt and the amount remaining."
        )
        with gr.Row():
            with gr.Column(scale=3):
                time_display = gr.Markdown()
                frame = gr.HTML(
                    value="{}",
                    html_template=CANVAS_HTML,
                    css_template=CANVAS_CSS,
                    js_on_load=CANVAS_JAVASCRIPT,
                    apply_default_css=False,
                    container=False,
                )
                status = gr.Markdown()
                motor_state = gr.Textbox(
                    label="Latched joint commands",
                    interactive=False,
                    elem_classes=["motor-state"],
                )
            with gr.Column(scale=2):
                with gr.Row():
                    seed = gr.Number(value=7001, precision=0, label="Seed")
                    target = gr.Slider(500, 900, value=700, step=10, label="Target (mL)")
                with gr.Row():
                    speed = gr.Radio(
                        choices=[("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)],
                        value=1.0,
                        label="Time speed",
                    )
                    pause_button = gr.Button("Pause / resume time")
                with gr.Row():
                    cap_time = gr.Checkbox(
                        value=False,
                        label="Cap episode on reset",
                    )
                    max_steps = gr.Number(
                        value=DEFAULT_HORIZON,
                        precision=0,
                        label="Maximum steps if capped",
                    )
                gr.Markdown("Episode-length settings apply when **Reset + start** is pressed.")
                with gr.Row():
                    reset_button = gr.Button("Reset + start", variant="primary")
                    stop_button = gr.Button("Stop all motors")
                    finish_button = gr.Button("Finish")

                joint_buttons: list[tuple[Any, Any, Any]] = []
                for joint_name in CoffeePouringEnv.JOINT_NAMES:
                    with gr.Row(equal_height=True):
                        gr.Markdown(
                            joint_name.replace("_", " ").title(),
                            elem_classes=["joint-name"],
                        )
                        ccw = gr.Button("↺", min_width=50)
                        halt = gr.Button("■", min_width=50)
                        cw = gr.Button("↻", min_width=50)
                        joint_buttons.append((ccw, halt, cw))

                metrics = gr.JSON(label="Episode metrics")
                export_button = gr.Button("Export human demonstration (.npz)")
                export_file = gr.File(label="Recorded demonstration", interactive=False)

        timer = gr.Timer(value=CoffeePouringEnv.DEFAULT_DT, active=True)
        outputs = [frame, time_display, status, motor_state, metrics]
        config_inputs = [seed, target, speed, cap_time, max_steps]
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

        for index, (ccw, halt, cw) in enumerate(joint_buttons):
            for button, direction in ((ccw, 1), (halt, 0), (cw, -1)):
                button.click(
                    motor_handler(index, direction),
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
    """Launch the app locally; Colab should call :func:`build_app` directly."""

    build_app().launch()


if __name__ == "__main__":  # pragma: no cover
    main()
