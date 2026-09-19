"""The same Python environment and recorder, hosted in a browser worker."""

import base64
import json

from kaist_rl_lab.apps.coffee_classroom import (
    INITIAL_LAYOUT,
    classroom_layout,
    fresh_classroom_seed,
)
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession, _validated_motor_command

__all__ = ["BROWSER_DT", "INITIAL_LAYOUT", "BrowserRuntime"]

BROWSER_DT = 1 / 32


class BrowserRuntime:
    """One authoritative timeline; display frames are actual recorded states."""

    def __init__(self, *, seed=None):
        seed = fresh_classroom_seed() if seed is None else seed
        self.session = InteractiveSession(
            seed, 700, start_paused=True, dt=BROWSER_DT, steps_per_update=1,
            reset_options=classroom_layout(seed),
        )

    def dispatch(self, encoded: str) -> str:
        command = json.loads(encoded)
        session = self.session
        kind = command["kind"]
        result = {}
        if kind == "tick":
            # Bound memory even if a student leaves the demo running all lecture.
            from kaist_rl_lab.apps.coffee_demonstrations import MAX_TRANSITIONS

            if len(session.trajectory) >= MAX_TRANSITIONS:
                session.finish()
            else:
                session.advance()
        elif kind == "save":
            path = session.save_demonstration(command.get("participant", ""))
            result["archive"] = base64.b64encode(path.read_bytes()).decode("ascii")
        elif kind in {"motor", "pause", "stop", "reset"}:
            sequence = command["sequence"]
            if type(sequence) is not int or sequence <= session.input_sequence:
                return self.snapshot()
            if command["generation"] != session.generation:
                return self.snapshot()
            motors, paused = command["motors"], command["paused"]
            if not isinstance(motors, list) or len(motors) != 6 or type(paused) is not bool:
                raise ValueError("Invalid controls")
            for i, motor in enumerate(motors):
                _validated_motor_command(i, motor)
            if kind == "reset":
                seed = fresh_classroom_seed()
                session.reset_options = classroom_layout(seed)
                session.restart(seed=seed, target_ml=700, speed=1, horizon=None)
            elif session.running:
                for i, motor in enumerate(motors):
                    session.set_motor(i, motor)
                if session.paused != paused:
                    session.toggle_pause()
            session.input_sequence = sequence
            session.revision += 1
        elif kind != "snapshot":
            raise ValueError("Unknown simulation command")
        return self.snapshot(result)

    def snapshot(self, extra=None):
        return json.dumps({
            "snapshot": self.session.animation_snapshot(),
            "episode_id": self.session.episode_id,
            **(extra or {}),
        }, allow_nan=False, separators=(",", ":"))
