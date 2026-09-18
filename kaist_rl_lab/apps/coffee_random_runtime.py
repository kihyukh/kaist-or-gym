"""A fixed random policy in the same coffee simulation used by students.

Each decision samples six independent motor commands, then holds them for a
uniformly sampled 1–32 simulation steps. Observations and rewards never affect
those choices, and nothing is learned between trials.
"""

import json

import numpy as np

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT, INITIAL_LAYOUT
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession

MAX_DURATION_SECONDS = 1
TRIAL_SECONDS = 30
MAX_HOLD_STEPS = round(MAX_DURATION_SECONDS / BROWSER_DT)
TRIAL_STEPS = round(TRIAL_SECONDS / BROWSER_DT)
ENVIRONMENT_SEED = 7001


class RandomAgentRuntime:
    """An independent instructor experiment; it cannot save student recordings."""

    def __init__(self):
        self.session = InteractiveSession(
            ENVIRONMENT_SEED, 700, start_paused=True, dt=BROWSER_DT,
            steps_per_update=1, horizon=TRIAL_STEPS, reset_options=INITIAL_LAYOUT,
        )
        self.rng = np.random.default_rng()
        self.attempt = 0
        self.decisions = 0
        self.hold_steps = 0
        self.remaining_steps = 0
        self.done = False
        self.outcome = None

    def _restart(self, *, reset_experiment: bool) -> None:
        self.session.restart(
            seed=ENVIRONMENT_SEED, target_ml=700, speed=1, horizon=TRIAL_STEPS,
        )
        self.session.paused = reset_experiment
        self.attempt = 0 if reset_experiment else self.attempt + 1
        self.decisions = 0
        self.hold_steps = 0
        self.remaining_steps = 0
        self.done = False
        self.outcome = None

    def _choose_control(self) -> None:
        # This policy deliberately has no observation or reward input.
        motors = self.rng.integers(-1, 2, size=6)
        self.hold_steps = int(self.rng.integers(1, MAX_HOLD_STEPS + 1))
        self.remaining_steps = self.hold_steps
        self.decisions += 1
        for index, direction in enumerate(motors):
            self.session.set_motor(index, int(direction))

    def dispatch(self, encoded: str) -> str:
        command = json.loads(encoded)
        if not isinstance(command, dict):
            raise TypeError("Invalid random-agent command")
        kind = command.get("kind")
        if kind == "random-start":
            if "seed" in command:
                seed = command["seed"]
                if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                    raise ValueError("Seed must be an integer between 0 and 4294967295")
                self.rng = np.random.default_rng(seed)
            self._restart(reset_experiment=False)
            self._choose_control()
        elif kind == "random-reset":
            self.rng = np.random.default_rng()
            self._restart(reset_experiment=True)
        elif kind == "random-pause":
            paused = command.get("paused")
            if type(paused) is not bool:
                raise ValueError("Pause must be true or false")
            if self.attempt and self.session.running and self.session.paused != paused:
                self.session.toggle_pause()
        elif kind == "tick":
            if self.attempt and self.session.running and not self.session.paused:
                if self.remaining_steps == 0:
                    self._choose_control()
                self.session.advance()
                self.remaining_steps -= 1
                if not self.session.running:
                    self.done = True
                    self.outcome = self.session.info["termination_reason"]
                    self.remaining_steps = 0
                    self.session.paused = True
        elif kind != "snapshot":
            raise ValueError("Unknown random-agent command")
        return self.snapshot()

    def snapshot(self) -> str:
        return json.dumps({
            "snapshot": self.session.animation_snapshot(),
            "episode_id": self.session.episode_id,
            "random_agent": {
                "attempt": self.attempt,
                "decisions": self.decisions,
                "hold_seconds": self.hold_steps * BROWSER_DT,
                "remaining_seconds": self.remaining_steps * BROWSER_DT,
                "elapsed_seconds": self.session.env.elapsed_steps * BROWSER_DT,
                "limit_seconds": TRIAL_SECONDS,
                "max_duration_seconds": MAX_DURATION_SECONDS,
                "done": self.done,
                "outcome": self.outcome,
            },
        }, allow_nan=False, separators=(",", ":"))
