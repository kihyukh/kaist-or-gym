"""Gradio controller for the coffee-pouring Gymnasium environment.

This module intentionally contains no robot geometry or dynamics.  It latches
joint commands, calls ``env.step()``, and forwards environment-owned render
snapshots to a smooth browser canvas.  ``env.render()`` remains the canonical
RGB renderer for training, tests, and video export.
"""

import json
from pathlib import Path
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


def _validated_motor_command(index: Any, direction: Any) -> tuple[int, int]:
    """Validate the untrusted payload emitted by the HTML joint controls."""

    integer_types = (int, np.integer)
    if isinstance(index, bool) or not isinstance(index, integer_types):
        raise TypeError("joint index must be an integer")
    if isinstance(direction, bool) or not isinstance(direction, integer_types):
        raise TypeError("motor direction must be an integer")
    index_value = int(index)
    direction_value = int(direction)
    if index_value not in range(6):
        raise ValueError("joint index must be between 0 and 5")
    if direction_value not in {-1, 0, 1}:
        raise ValueError("motor direction must be -1, 0, or 1")
    return index_value, direction_value


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
        """Latch one normalized motor command until explicitly changed."""

        value = float(direction)
        if value != self.motors[index]:
            self.motors[index] = value
            self.revision += 1
            self.event_kind = "control"
        return value

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
            "motors": self.motors.astype(float).tolist(),
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
        f"**{clock.capitalize()}** · Step {step_label}  \n"
        f"{session.info['elapsed_time']:.2f} s simulated"
    )


def _view(session: InteractiveSession, message: str | None = None):
    if message is not None:
        session.message = message
    return (
        json.dumps(session.animation_snapshot(), separators=(",", ":"), allow_nan=False),
        _time_display(session),
        session.message,
    )


def build_app():
    """Build and return the Colab-compatible Gradio application."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The interactive app needs Gradio. Install with: pip install 'kaist-rl-lab[interactive]'"
        ) from exc

    # The demo uses the original defaults: 700 mL, normal speed, unlimited practice.
    config = (7001, 700.0, 1.0, False, DEFAULT_HORIZON)

    def timer_update(session: InteractiveSession, active: bool):
        return gr.Timer(value=session.timer_interval, active=active)

    def view(session: InteractiveSession, message: str | None = None):
        return (
            *_view(session, message),
            gr.Button(
                "Resume time" if session.paused else "Pause time",
                interactive=session.running,
            ),
        )

    def initialize(request: gr.Request):
        session = _create_session(request, *config)
        return view(session, "Choose a joint direction below to begin.")

    def reset(request: gr.Request):
        session = _create_session(request, *config)
        return (
            *view(session, "New episode. All joints held; time is running."),
            timer_update(session, True),
        )

    def tick(request: gr.Request):
        session = _get_session(request, *config)
        with session.lock:
            if not session.running or session.paused:
                return (*view(session), timer_update(session, False))
            session.advance()
            if not session.running:
                message = "Success! Reset to pour again." if session.info["is_success"] else (
                    "Episode complete. Reset to try again."
                )
                return (*view(session, message), timer_update(session, False))
            # Ordinary ticks must not undo a more recent pause or button update.
            return (*_view(session), gr.skip(), gr.skip())

    def stop_all(request: gr.Request):
        session = _get_session(request, *config)
        with session.lock:
            session.stop_all_motors()
            clock = "Time remains paused." if session.paused else "Time keeps running."
            if not session.running:
                clock = "Reset to start again."
            return view(session, f"All motors stopped. {clock}")

    def toggle_pause(request: gr.Request):
        session = _get_session(request, *config)
        with session.lock:
            if not session.running:
                message = "Episode complete. Reset to start again."
            elif session.toggle_pause():
                message = "Time paused. Joint commands are preserved for resume."
            else:
                message = "Time resumed. Selected joint commands are active."
            timer_active = session.running and not session.paused
            return (*view(session, message), timer_update(session, timer_active))

    def canvas_joint_control(request: gr.Request, event: gr.EventData):
        try:
            index, direction = _validated_motor_command(event.joint_index, event.direction)
        except (AttributeError, TypeError, ValueError) as exc:
            raise gr.Error("Invalid joint control command") from exc
        session = _get_session(request, *config)
        with session.lock:
            if not session.running:
                return view(session, "Episode complete. Reset to start again.")
            new_value = session.set_motor(index, direction)
            label = CoffeePouringEnv.JOINT_NAMES[index].replace("_", " ").capitalize()
            words = {-1: "clockwise", 0: "held", 1: "counterclockwise"}
            suffix = " Time is paused." if session.paused else ""
            return view(session, f"{label} {words[int(new_value)]}.{suffix}")

    def cleanup(request: gr.Request):
        _cleanup_session(request)

    with gr.Blocks(title="KAIST OR Gym — Coffee Pouring") as demo:
        with gr.Column(min_width=0, elem_id="coffee-demo"):
            gr.Markdown(
                "Guide the pot toward the cup and tilt to pour. "
                "Use the joint controls below the scene; several joints can move together."
            )
            with gr.Row(equal_height=True, elem_id="coffee-toolbar"):
                with gr.Column(scale=1, min_width=180):
                    time_display = gr.Markdown()
                pause_button = gr.Button(
                    "Pause time", variant="primary", scale=0, min_width=140, size="md"
                )
                reset_button = gr.Button("Reset + start", scale=0, min_width=140, size="md")
                stop_button = gr.Button(
                    "Stop all motors", variant="stop", scale=0, min_width=150, size="md"
                )
            frame = gr.HTML(
                value="{}",
                html_template=CANVAS_HTML,
                css_template=CANVAS_CSS,
                js_on_load=CANVAS_JAVASCRIPT,
                apply_default_css=False,
                container=False,
            )
            status = gr.Markdown()

        timer = gr.Timer(value=CoffeePouringEnv.DEFAULT_DT, active=True)
        outputs = [frame, time_display, status, pause_button]
        demo.load(initialize, outputs=outputs, queue=False)
        reset_button.click(reset, outputs=[*outputs, timer], queue=False)
        pause_button.click(toggle_pause, outputs=[*outputs, timer], queue=False)
        stop_button.click(stop_all, outputs=outputs, queue=False)
        timer.tick(
            tick,
            outputs=[*outputs, timer],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
        frame.click(
            canvas_joint_control,
            outputs=outputs,
            queue=False,
            show_progress="hidden",
            trigger_mode="multiple",
        )
        demo.unload(cleanup)
    return demo


def main() -> None:
    """Launch the app locally; Colab should call :func:`build_app` directly."""

    build_app().launch()


if __name__ == "__main__":  # pragma: no cover
    main()
