"""Run a fitted behavior-cloning policy in the real student coffee environment."""

import json

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT, INITIAL_LAYOUT
from kaist_rl_lab.apps.coffee_cloning import NearestNeighborPolicy
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession

TRIAL_SECONDS = 60
TRIAL_STEPS = round(TRIAL_SECONDS / BROWSER_DT)
ENVIRONMENT_SEED = 7001


class CloningAgentRuntime:
    """Independent instructor rollout with no recording upload or expert fallback."""

    def __init__(self):
        self.session = InteractiveSession(
            ENVIRONMENT_SEED, 700, start_paused=True, dt=BROWSER_DT,
            steps_per_update=1, horizon=TRIAL_STEPS, reset_options=INITIAL_LAYOUT,
        )
        self.policy = None
        self.attempt = 0
        self.done = False
        self.outcome = None

    def _restart(self, *, reset_experiment: bool) -> None:
        self.session.restart(
            seed=ENVIRONMENT_SEED, target_ml=700, speed=1, horizon=TRIAL_STEPS,
        )
        self.session.paused = reset_experiment
        self.attempt = 0 if reset_experiment else self.attempt + 1
        self.done = False
        self.outcome = None

    def dispatch(self, encoded: str) -> str:
        command = json.loads(encoded)
        if not isinstance(command, dict):
            raise TypeError("Invalid behavior-cloning command")
        kind = command.get("kind")
        if kind == "cloning-load":
            policy = NearestNeighborPolicy(command.get("model"))
            self.policy = policy
            self._restart(reset_experiment=True)
        elif kind == "cloning-start":
            if self.policy is None:
                raise ValueError("Train and load a behavior-cloning policy first.")
            self._restart(reset_experiment=False)
        elif kind == "cloning-reset":
            self._restart(reset_experiment=True)
        elif kind == "cloning-pause":
            paused = command.get("paused")
            if type(paused) is not bool:
                raise ValueError("Pause must be true or false")
            if self.attempt and self.session.running and self.session.paused != paused:
                self.session.toggle_pause()
        elif kind == "tick":
            if (
                self.policy is not None and self.attempt
                and self.session.running and not self.session.paused
            ):
                # Every command is selected afresh from the current physical
                # state. Rewards, time, and the previous action are not inputs.
                action = self.policy.predict(self.session.observation)
                for index, direction in enumerate(action):
                    self.session.set_motor(index, float(direction))
                self.session.advance()
                if not self.session.running:
                    self.done = True
                    self.outcome = self.session.info["termination_reason"]
                    self.session.paused = True
        elif kind != "snapshot":
            raise ValueError("Unknown behavior-cloning command")
        return self.snapshot()

    def snapshot(self) -> str:
        return json.dumps({
            "snapshot": self.session.animation_snapshot(),
            "episode_id": self.session.episode_id,
            "cloning_agent": {
                "model_loaded": self.policy is not None,
                "attempt": self.attempt,
                "step": self.session.env.elapsed_steps,
                "elapsed_seconds": self.session.env.elapsed_steps * BROWSER_DT,
                "limit_seconds": TRIAL_SECONDS,
                "done": self.done,
                "outcome": self.outcome,
            },
        }, allow_nan=False, separators=(",", ":"))
