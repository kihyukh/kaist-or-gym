"""Small NumPy actor-critic that refines a frozen behavior-cloned policy.

The actor chooses one shared speed multiplier, not six unconstrained controls.
Every motor command stays within 15% of its BC command and zero BC commands stay
zero. A Gaussian latent action is squashed with tanh; its fixed transform is
part of the environment, so PPO ratios use the retained Gaussian latent value.

The objective is the environment's original, undiscounted 60-second episodic
return. The time limit is part of this task and receives zero value bootstrap,
just like success/failure. A learned linear critic supplies GAE advantages.
Checkpoints are evaluated without exploration and the unchanged BC checkpoint
is retained whenever training does not improve the actual evaluation return.
"""

from typing import Any

import numpy as np

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import classroom_layout
from kaist_rl_lab.apps.coffee_cloning import NearestNeighborPolicy
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession

SPEED_BOUND = 0.15
LATENT_STD = 0.30
MAX_UPDATE_KL = 0.01
MAX_MEAN_CHANGE = 0.10
DECISION_STEPS = 8
TRIAL_STEPS = 60 * 32
DEFAULT_EPISODES = 8
MAX_EPISODES = 12
ACTOR_FEATURES = 5
GAE_LAMBDA = 0.95


def actor_features(observation: np.ndarray) -> np.ndarray:
    """Bounded physical features, excluding elapsed time and future states."""
    observation = np.asarray(observation)
    return np.array([
        1.0,
        np.clip(observation[12] / max(float(observation[14]), 0.5), 0, 1.5),
        np.clip(observation[8], -1, 1),
        np.clip((observation[10] - 0.10) * 2, -1, 1),
        np.clip((observation[11] - 0.20) * 2, -1, 1),
    ], dtype=np.float64)


def critic_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features)
    return np.concatenate((x, x[..., 1:] ** 2, (x[..., 1] * x[..., 2])[..., None]), axis=-1)


def generalized_advantages(rewards, values, *, trace_decay=GAE_LAMBDA):
    """GAE for one complete finite-horizon episode, gamma=1, bootstrap=0."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or rewards.shape != values.shape or not len(rewards):
        raise ValueError("Rewards and values must be equal nonempty vectors.")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("Rewards and values must be finite.")
    advantages = np.zeros_like(rewards)
    carry = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        carry = rewards[index] + next_value - values[index] + trace_decay * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages


def ppo_actor_update(weights, features, latent, advantages, *, epochs=4):
    """Clipped policy gradient with explicit sampled-state Gaussian KL backtracking.

    Fixed variance makes KL analytic. The L1 parameter-change limit additionally
    bounds every possible feature-vector's mean change (features <=1.5), not just
    the collected states. Returned diagnostics describe the complete update.
    """
    original = np.asarray(weights, dtype=np.float64).copy()
    current = original.copy()
    features = np.asarray(features, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
    old_mean = features @ original
    old_logp = -0.5 * ((latent - old_mean) / LATENT_STD) ** 2
    last_kl = 0.0
    for _ in range(epochs):
        mean = features @ current
        log_ratio = -0.5 * ((latent - mean) / LATENT_STD) ** 2 - old_logp
        ratio = np.exp(np.clip(log_ratio, -30, 30))
        active = ((advantages >= 0) & (ratio <= 1.2)) | ((advantages < 0) & (ratio >= 0.8))
        score = ratio * advantages * active * (latent - mean) / LATENT_STD**2
        gradient = np.mean(score[:, None] * features, axis=0)
        length = float(np.linalg.norm(gradient))
        if not np.isfinite(length) or length < 1e-12:
            break
        step = 0.025 * gradient / max(length, 1.0)
        for _ in range(16):
            candidate = current + step
            delta = candidate - original
            # All actor features are <=1.5 in absolute value.
            max_change = 1.5 * float(np.abs(delta).sum())
            candidate_kl = float(np.mean((features @ delta) ** 2) / (2 * LATENT_STD**2))
            if max_change <= MAX_MEAN_CHANGE and candidate_kl <= MAX_UPDATE_KL:
                current = candidate
                last_kl = candidate_kl
                break
            step *= 0.5
    return current, {
        "mean_kl": last_kl,
        "mean_change_bound": 1.5 * float(np.abs(current - original).sum()),
        "actor_change": float(np.linalg.norm(current - original)),
    }


class FineTunedPolicy:
    """The immutable BC policy plus one learned, bounded speed actor."""

    def __init__(self, base: NearestNeighborPolicy, weights=None):
        self.base = base
        self.weights = np.zeros(ACTOR_FEATURES) if weights is None else np.asarray(
            weights, dtype=np.float64,
        ).copy()
        if self.weights.shape != (ACTOR_FEATURES,) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid fine-tuned actor weights.")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        latent_mean = float(actor_features(observation) @ self.weights)
        return self.action_with_latent(observation, latent_mean)

    def action_with_latent(self, observation: np.ndarray, latent: float) -> np.ndarray:
        if not np.isfinite(latent):
            raise ValueError("Latent speed must be finite.")
        action = self.base.predict(observation)
        gain = SPEED_BOUND * np.tanh(latent)
        return np.clip(action * (1 + gain), -1, 1).astype(np.float32)


class FineTuningTrainer:
    """Incremental training: a browser worker can yield between bounded chunks."""

    def __init__(self, model: dict[str, Any], *, seed=2026, episodes=DEFAULT_EPISODES):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Training seed must be a 32-bit nonnegative integer.")
        if type(episodes) is not int or not 1 <= episodes <= MAX_EPISODES:
            raise ValueError(f"Choose between 1 and {MAX_EPISODES} training episodes.")
        self.base = NearestNeighborPolicy(model)
        # Defensive copies prevent training or callers from changing the BC labels.
        self.base.states = self.base.states.copy()
        self.base.actions = self.base.actions.copy()
        self.base.states.flags.writeable = False
        self.base.actions.flags.writeable = False
        self.policy = FineTunedPolicy(self.base)
        self.best_policy = FineTunedPolicy(self.base)
        self.rng = np.random.default_rng(seed)
        # Keep exploration noise independent of layout sampling. All policy
        # checkpoints share one sampled evaluation pose for a fair comparison;
        # exploratory rollouts get fresh poses from the classroom distribution.
        self.layout_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.evaluation_seed = int(self.layout_rng.integers(0, 2**32))
        self.seed = seed
        self.episodes = episodes
        self.episode = 0
        self.phase = "baseline"
        self.done = False
        self.total_steps = 0
        self.baseline = None
        self.best = None
        self.history = []
        self.critic = np.zeros(10)
        self.critic_states = []
        self.critic_returns = []
        self.update = {"mean_kl": 0.0, "mean_change_bound": 0.0, "actor_change": 0.0}
        self.session = InteractiveSession(
            self.evaluation_seed, 700, dt=BROWSER_DT, steps_per_update=1, horizon=TRIAL_STEPS,
            reset_options=classroom_layout(self.evaluation_seed),
        )
        self._reset_rollout()

    def _reset_rollout(self):
        seed = (int(self.layout_rng.integers(0, 2**32))
                if self.phase == "training" else self.evaluation_seed)
        self.session.reset_options = classroom_layout(seed)
        self.session.restart(seed=seed, target_ml=700, speed=1, horizon=TRIAL_STEPS)
        self.session.paused = False
        self.rollout_features = []
        self.rollout_latent = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.decision_remaining = 0
        self.latent = 0.0

    def _begin_decision(self):
        features = actor_features(self.session.observation)
        mean = float(features @ self.policy.weights) if self.phase != "baseline" else 0.0
        self.latent = float(self.rng.normal(mean, LATENT_STD)) if self.phase == "training" else mean
        self.rollout_features.append(features)
        self.rollout_latent.append(self.latent)
        self.rollout_rewards.append(0.0)
        self.rollout_values.append(float(critic_features(features) @ self.critic))
        self.decision_remaining = DECISION_STEPS

    def _fit_critic(self):
        rewards = np.asarray(self.rollout_rewards)
        returns = np.cumsum(rewards[::-1])[::-1]
        self.critic_states.extend(critic_features(np.asarray(self.rollout_features)))
        self.critic_returns.extend(returns)
        x = np.asarray(self.critic_states)
        y = np.asarray(self.critic_returns)
        # Reward-to-go regression is critic learning; demonstrations are never a
        # critic target. Ridge regularization keeps a tiny linear model stable.
        self.critic = np.linalg.solve(x.T @ x + 0.05 * np.eye(x.shape[1]), x.T @ y)

    def _metrics(self):
        return {
            "return": float(self.session.cumulative_reward),
            "fill_ml": float(self.session.env.fill * 1000),
            "spill_ml": float(self.session.env.spill * 1000),
            "seconds": float(self.session.env.elapsed_steps * BROWSER_DT),
            "success": bool(self.session.info["is_success"]),
            "outcome": self.session.info["termination_reason"],
            "initial_seed": self.session.seed,
        }

    def _finish_rollout(self):
        metrics = self._metrics()
        if self.phase == "baseline":
            self.baseline = metrics
            self.best = dict(metrics)
            self.best["episode"] = 0
            self._fit_critic()
            self.episode = 1
            self.phase = "training"
        elif self.phase == "training":
            advantages = generalized_advantages(self.rollout_rewards, self.rollout_values)
            self.policy.weights, self.update = ppo_actor_update(
                self.policy.weights, self.rollout_features, self.rollout_latent, advantages,
            )
            self._fit_critic()
            self.history.append({
                "episode": self.episode, "training": metrics,
                "evaluation": None, "update": dict(self.update),
            })
            self.phase = "evaluation"
        else:
            self.history[-1]["evaluation"] = metrics
            if metrics["return"] > self.best["return"]:
                self.best = {**metrics, "episode": self.episode}
                self.best_policy = FineTunedPolicy(self.base, self.policy.weights)
            if self.episode >= self.episodes:
                self.done = True
                self.phase = "complete"
                self.session.paused = True
                return
            self.episode += 1
            self.phase = "training"
        self._reset_rollout()

    def step_chunk(self, max_steps=32):
        if type(max_steps) is not int or not 1 <= max_steps <= 32:
            raise ValueError("A training chunk must contain between 1 and 32 steps.")
        for _ in range(max_steps):
            if self.done:
                break
            if not self.decision_remaining:
                self._begin_decision()
            # Requery BC at every new physical state even while holding a
            # sampled speed: no prerecorded action timeline is replayed.
            latent = self.latent
            if self.phase == "evaluation":
                latent = float(actor_features(self.session.observation) @ self.policy.weights)
            action = self.policy.action_with_latent(self.session.observation, latent)
            for index, value in enumerate(action):
                self.session.set_motor(index, float(value))
            self.session.advance()
            self.rollout_rewards[-1] += self.session.trajectory[-1]["reward"]
            self.decision_remaining -= 1
            self.total_steps += 1
            if not self.session.running:
                self._finish_rollout()
        return self.result()

    def result(self):
        return {
            "algorithm": "bounded_speed_ppo_actor_critic",
            "phase": self.phase,
            "done": self.done,
            "episode": self.episode,
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "episode_steps": self.session.env.elapsed_steps,
            "seed": self.seed,
            "evaluation_seed": self.evaluation_seed,
            "speed_bound": SPEED_BOUND,
            "latent_std": LATENT_STD,
            "decision_seconds": DECISION_STEPS * BROWSER_DT,
            "limit_seconds": TRIAL_STEPS * BROWSER_DT,
            "max_update_kl": MAX_UPDATE_KL,
            "baseline": self.baseline,
            "best": self.best,
            "history": self.history,
            "latest_update": self.update,
            "improved": bool(self.best and self.best["episode"] > 0),
            "completed_episodes": sum(row["evaluation"] is not None for row in self.history),
            "actor_weights": self.policy.weights.tolist(),
            "best_weights": self.best_policy.weights.tolist(),
        }

    def close(self):
        self.session.close()
