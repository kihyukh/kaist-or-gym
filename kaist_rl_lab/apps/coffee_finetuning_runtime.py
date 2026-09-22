"""Cooperative browser training and evaluation for the instructor's RL demo.

Training advances only in explicit, bounded chunks, allowing the worker to process
pause/stop messages between chunks. Displayed trials batch unchanged 32 Hz physics
steps for accelerated playback and cannot export or submit classroom demonstrations.
"""

import json
from copy import deepcopy

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import (
    ARM_BASE_DISTANCE_M,
    POLICY_START_SEED,
    fixed_policy_layout,
    fresh_classroom_seed,
)
from kaist_rl_lab.apps.coffee_cloning import NearestNeighborPolicy
from kaist_rl_lab.apps.coffee_finetuning import (
    DEFAULT_EPISODES,
    MAX_EPISODES,
    STEP_DISCOUNT,
    STRATEGIES,
    FineTuningTrainer,
)
from kaist_rl_lab.apps.coffee_finetuning_reward import fine_tuning_reward
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession

TRIAL_SECONDS = 60
TRIAL_STEPS = round(TRIAL_SECONDS / BROWSER_DT)
DEFAULT_TRAINING_EPISODES = DEFAULT_EPISODES
MAX_TRAINING_EPISODES = MAX_EPISODES
DEFAULT_CHUNK_STEPS = 32
MAX_CHUNK_STEPS = 32


def _bounded_integer(value, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}.")
    return value


def _new_session(seed=None) -> InteractiveSession:
    seed = POLICY_START_SEED if seed is None else seed
    return InteractiveSession(
        seed, 700, arm_base_distance=ARM_BASE_DISTANCE_M, start_paused=True, dt=BROWSER_DT,
        steps_per_update=1, horizon=TRIAL_STEPS, reset_options=fixed_policy_layout(),
        include_render_info=False,
    )


def _playback_speed(value):
    if type(value) is not int or value not in (0, 4, 8):
        raise ValueError("Speed must be 4, 8, or 0 for fastest.")
    return value


class FineTuningRuntime:
    """Own one BC model, a cooperative trainer, and independent policy trials."""

    def __init__(self):
        self.session = _new_session()
        self.model = None
        self.base_policy = None
        self._reference_policy = None
        self.best_policy = None
        self.trainer = None
        self.training_active = False
        self.training_paused = False
        self.training_stopped = False
        self.progress = None
        self.result = None
        self.rollout_active = False
        self.rollout_policy = "base"
        self.rollout_done = False
        self.rollout_outcome = None
        self.playback_speed = 0
        self.discounted_return = 0.0
        self._discount_weight = 1.0

    def _dispose_training(self) -> None:
        if self.trainer is not None:
            self.trainer.close()
            self.trainer = None

    def close(self) -> None:
        """Release any session owned by this worker."""
        training_session = self.trainer.session if self.trainer is not None else None
        self._dispose_training()
        if self.session is not training_session:
            self.session.close()
        self.training_active = False
        self.rollout_active = False

    def _replace_scene(self, seed=None) -> None:
        self.close()
        self.session = _new_session(seed)
        self.discounted_return = 0.0
        self._discount_weight = 1.0
        self.rollout_active = False
        self.rollout_done = False
        self.rollout_outcome = None
        self.training_paused = False

    def _capture_training(self, *, checkpoint: bool = False) -> None:
        self.session = self.trainer.session
        self.result = self.trainer.result()
        self.progress = {
            key: self.result[key]
            for key in (
                "phase", "episode", "episodes", "completed_episodes", "total_steps", "episode_steps",
            )
        }
        self.progress["elapsed_seconds"] = self.result["episode_steps"] * BROWSER_DT
        self.discounted_return = float(self.trainer.discounted_return)
        self.progress["reward"] = self.discounted_return
        self.progress["raw_return"] = float(self.session.cumulative_reward)
        self.result["improved"] = bool(
            self.result["baseline"] is not None and self.result["best"] is not None
            and self.result["best"]["return"] > self.result["baseline"]["return"]
        )
        self.result["best_episode"] = (
            None if self.result["best"] is None else self.result["best"]["episode"]
        )
        # A policy becomes available only after an entire deterministic
        # evaluation. A partially evaluated candidate is never advertised as
        # better than BC, including when the instructor stops early.
        if checkpoint:
            self.best_policy = (
                deepcopy(self.trainer.best_policy) if self.result["baseline"] is not None else None
            )

    def _stop_training(self) -> None:
        if self.training_active:
            if not self.session.running and not self.trainer.done:
                # Finalize a fully executed trial before checkpointing. The
                # trainer yields at this boundary without another physics step.
                self.trainer.step_chunk(max_steps=1)
            self._capture_training(checkpoint=True)
            self.training_active = False
            self.training_paused = False
            self.training_stopped = True
            self.session.paused = True

    def dispatch(self, encoded: str) -> str:
        command = json.loads(encoded)
        if not isinstance(command, dict):
            raise TypeError("Invalid fine-tuning command.")
        kind = command.get("kind")
        if kind == "ft-load":
            # Validate before replacing a working experiment.
            policy = NearestNeighborPolicy(command.get("model"))
            if policy.arm_base_distance != ARM_BASE_DISTANCE_M:
                raise ValueError("This policy uses different arm spacing. Retrain behavior cloning for this classroom.")
            self._replace_scene()
            self.model = command["model"]
            self.base_policy = policy
            self._reference_policy = NearestNeighborPolicy(command["model"])
            self.best_policy = None
            self.progress = None
            self.result = None
            self.training_stopped = False
            self.rollout_policy = "base"
        elif kind == "ft-train":
            if self.model is None:
                raise ValueError("Train and load a behavior-cloning policy first.")
            episodes = _bounded_integer(
                command.get("episodes", DEFAULT_TRAINING_EPISODES),
                1, MAX_TRAINING_EPISODES, "Training episodes",
            )
            speed = _playback_speed(command.get("speed", self.playback_speed))
            seed = _bounded_integer(command["seed"] if "seed" in command else fresh_classroom_seed(),
                                    0, 2**32 - 1, "Seed")
            if self.training_active:
                raise ValueError("Stop the current training run before starting another.")
            strategy = command.get("strategy", "policy_search")
            if strategy not in STRATEGIES:
                raise ValueError("Choose residual_search, policy_search, or ppo for fine-tuning.")
            trainer = FineTuningTrainer(self.model, seed=seed, episodes=episodes, strategy=strategy)
            self.close()
            self.trainer = trainer
            self.playback_speed = speed
            self.best_policy = None
            self.training_active = True
            self.training_paused = False
            self.training_stopped = False
            self.rollout_active = False
            self.rollout_done = False
            self.rollout_outcome = None
            self._capture_training()
        elif kind == "ft-step":
            steps = _bounded_integer(
                command.get("max_steps", DEFAULT_CHUNK_STEPS),
                1, MAX_CHUNK_STEPS, "Training chunk size",
            )
            if self.training_active and not self.training_paused:
                self.trainer.step_chunk(max_steps=steps)
                self._capture_training()
                if self.trainer.done:
                    self._capture_training(checkpoint=True)
                    self.training_active = False
                    self.training_paused = False
                    self.session.paused = True
        elif kind == "ft-stop":
            self._stop_training()
        elif kind == "ft-speed":
            self.playback_speed = _playback_speed(command.get("speed"))
        elif kind == "ft-run":
            policy = command.get("policy", "best")
            if not isinstance(policy, str) or policy not in {"base", "best"}:
                raise ValueError("Choose the base or best policy.")
            if self.base_policy is None:
                raise ValueError("Train and load a behavior-cloning policy first.")
            if self.training_active:
                raise ValueError("Stop or finish training before running a comparison.")
            if policy == "best" and self.best_policy is None:
                raise ValueError("Finish a policy evaluation before running the best policy.")
            speed = _playback_speed(command.get("speed", 4))
            self._replace_scene((self.result or {}).get("evaluation_seed"))
            self.playback_speed = speed
            self.rollout_policy = policy
            self.rollout_active = True
            self.session.paused = False
        elif kind == "ft-pause":
            paused = command.get("paused")
            if type(paused) is not bool:
                raise ValueError("Pause must be true or false.")
            if self.training_active:
                self.training_paused = paused
                self.session.paused = paused
            elif self.rollout_active and self.session.running:
                if self.session.paused != paused:
                    self.session.toggle_pause()
        elif kind == "ft-reset":
            self._stop_training()
            self._replace_scene()
        elif kind == "tick":
            steps = _bounded_integer(command.get("max_steps", 1), 1, MAX_CHUNK_STEPS, "Trial chunk size")
            for _ in range(steps):
                if not (self.rollout_active and self.session.running and not self.session.paused):
                    break
                policy = self.base_policy if self.rollout_policy == "base" else self.best_policy
                action = policy.predict(self.session.observation)
                for index, direction in enumerate(action):
                    self.session.set_motor(index, float(direction))
                self.session.advance()
                self.discounted_return += self._discount_weight * fine_tuning_reward(self.session.info, BROWSER_DT)
                self._discount_weight *= STEP_DISCOUNT
                if not self.session.running:
                    self.rollout_active = False
                    self.rollout_done = True
                    self.rollout_outcome = self.session.info["termination_reason"]
                    self.session.paused = True
        elif kind != "snapshot":
            raise ValueError("Unknown fine-tuning command.")
        return self.snapshot()

    def snapshot(self) -> str:
        best_available = self.best_policy is not None or (
            self.training_active and self.result["baseline"] is not None
        )
        return json.dumps({
            "snapshot": self.session.animation_snapshot(),
            "episode_id": self.session.episode_id,
            "finetuning": {
                "model_loaded": self.base_policy is not None,
                "training_active": self.training_active,
                "training_paused": self.training_paused,
                "training_running": self.training_active and not self.training_paused,
                "training_stopped": self.training_stopped,
                "has_result": self.result is not None and self.result.get("baseline") is not None,
                "best_available": best_available,
                "progress": self.progress,
                "result": self.result,
                "live_action": self._live_action(),
                "paused": self.session.paused,
                "playback_speed": self.playback_speed,
                "rollout": {
                    "active": self.rollout_active,
                    "policy": self.rollout_policy,
                    "elapsed_seconds": self.session.env.elapsed_steps * BROWSER_DT,
                    "limit_seconds": TRIAL_SECONDS,
                    "done": self.rollout_done,
                    "outcome": self.rollout_outcome,
                    "reward": float(self.discounted_return),
                    "raw_return": float(self.session.cumulative_reward),
                },
            },
        }, allow_nan=False, separators=(",", ":"))

    def _live_action(self) -> dict:
        """Describe the command that produced this scene, never a best-policy replay.

        A terminal scene has stopped motors, so read the recorded transition
        rather than the current motor latch. Compare BC at that same pre-action
        state; querying it at the displayed next state can pick another label.
        """
        training_scene = self.trainer is not None and self.session is self.trainer.session
        if training_scene:
            phase = (self.result or {}).get("phase", "baseline")
            phase = "evaluation" if phase == "complete" else phase
            source = {
                "baseline": "original_clone",
                "training": "exploration",
                "evaluation": "deterministic_evaluation",
            }[phase]
            episode = (self.result or {}).get("episode", 0)
        else:
            phase = "replay" if self.rollout_active or self.rollout_done else "idle"
            source = f"{self.rollout_policy}_replay" if phase == "replay" else "idle"
            episode = None
        transition = self.session.trajectory[-1] if self.session.trajectory else None
        action = transition["action"].astype(float) if transition is not None else None
        cloned = self._reference_policy.predict(transition["observation"]).astype(float) if (
            transition is not None and self._reference_policy is not None
        ) else None
        delta = action - cloned if cloned is not None else None
        return {
            "phase": phase,
            "source": source,
            "episode": episode,
            "episode_id": self.session.episode_id,
            "step_index": len(self.session.trajectory),
            "executed_action": None if action is None else action.tolist(),
            "cloned_action": None if cloned is None else cloned.tolist(),
            "action_delta": None if delta is None else delta.tolist(),
            "max_action_change": None if delta is None else float(abs(delta).max()),
        }
