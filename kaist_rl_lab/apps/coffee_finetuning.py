"""Two small reward learners that refine a frozen behavior-cloned policy.

Paired policy search tests coherent faster/slower versions of the current
policy. PPO instead learns a state-dependent speed actor with a linear critic.
Both query BC at each physical step, preserve motor directions, and use the same
time-focused discounted objective. Every exploration trial is followed by a
separate noise-free evaluation; only evaluated checkpoints are offered for replay.
"""

from copy import deepcopy
from typing import Any

import numpy as np

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import (
    ARM_BASE_DISTANCE_M,
    POLICY_START_SEED,
    fixed_policy_layout,
)
from kaist_rl_lab.apps.coffee_cloning import NearestNeighborPolicy
from kaist_rl_lab.apps.coffee_finetuning_reward import DISCOUNT_PER_SECOND, fine_tuning_reward
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession

SPEED_BOUND = 0.5
LATENT_STD = 0.5
MAX_UPDATE_KL = 0.08
MAX_MEAN_CHANGE = 0.6
MAX_LATENT_MEAN = 1.25
DECISION_STEPS = 16
PPO_EPOCHS = 12
ACTOR_LEARNING_RATE = 0.15
TRIAL_STEPS = 60 * 32
DEFAULT_EPISODES = 100
MAX_EPISODES = 100
ACTOR_FEATURES = 5
GAE_LAMBDA = 0.95
STEP_DISCOUNT = DISCOUNT_PER_SECOND**BROWSER_DT
CRITIC_RETENTION = 0.95
STRATEGIES = ("policy_search", "ppo")
SEARCH_SPEED_MIN = 0.7
SEARCH_SPEED_MAX = 1.4
SEARCH_RADIUS_MIN = 0.08
SEARCH_RADIUS_MAX = 0.12


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


def generalized_advantages(rewards, values, *, discounts=None, trace_decay=GAE_LAMBDA):
    """GAE over variable-duration decisions, with zero terminal bootstrap.

    Each decision reward already discounts its constituent physics steps.
    Its continuation discount is gamma ** actual_steps, including a short final
    decision. A missing discount vector represents unit-length decisions.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or rewards.shape != values.shape or not len(rewards):
        raise ValueError("Rewards and values must be equal nonempty vectors.")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("Rewards and values must be finite.")
    discounts = np.full_like(rewards, STEP_DISCOUNT) if discounts is None else np.asarray(discounts)
    if discounts.shape != rewards.shape or not np.isfinite(discounts).all() or np.any(
        (discounts < 0) | (discounts > 1)
    ):
        raise ValueError("Discounts must match rewards and lie between zero and one.")
    advantages = np.zeros_like(rewards)
    carry = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        carry = rewards[index] + discounts[index] * (
            next_value + trace_decay * carry
        ) - values[index]
        advantages[index] = carry
        next_value = values[index]
    return advantages


def ppo_actor_update(weights, features, latent, advantages, *, epochs=PPO_EPOCHS, discount_weights=None):
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
    if discount_weights is not None:
        # Discounted state occupancy: early decisions contribute more to the
        # episode-start objective, rather than optimizing an undiscounted sum.
        advantages *= np.asarray(discount_weights, dtype=np.float64)
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
        step = ACTOR_LEARNING_RATE * gradient / max(length, 1.0)
        for _ in range(16):
            candidate = current + step
            delta = candidate - original
            # All actor features are <=1.5 in absolute value.
            max_change = 1.5 * float(np.abs(delta).sum())
            candidate_kl = float(np.mean((features @ delta) ** 2) / (2 * LATENT_STD**2))
            # Keep tanh away from saturation so fixed latent variance continues
            # to produce meaningful control exploration even after 100 updates.
            mean_bound = 1.5 * float(np.abs(candidate).sum())
            allowed_bound = max(MAX_LATENT_MEAN, 1.5 * float(np.abs(original).sum()))
            if (
                max_change <= MAX_MEAN_CHANGE and candidate_kl <= MAX_UPDATE_KL
                and mean_bound <= allowed_bound
            ):
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


class ScaledClonedPolicy:
    """A state-feedback clone with one learned global speed parameter."""

    def __init__(self, base: NearestNeighborPolicy, multiplier=1.0):
        if not np.isfinite(multiplier) or not SEARCH_SPEED_MIN <= multiplier <= SEARCH_SPEED_MAX:
            raise ValueError("Invalid policy-search speed multiplier.")
        self.base = base
        self.multiplier = float(multiplier)
        self.weights = np.array([self.multiplier])

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return np.clip(self.base.predict(observation) * self.multiplier, -1, 1).astype(np.float32)


class FineTuningTrainer:
    """Incremental training: a browser worker can yield between bounded chunks."""

    def __init__(self, model: dict[str, Any], *, seed=2026, episodes=DEFAULT_EPISODES, strategy="ppo"):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Training seed must be a 32-bit nonnegative integer.")
        if type(episodes) is not int or not 1 <= episodes <= MAX_EPISODES:
            raise ValueError(f"Choose between 1 and {MAX_EPISODES} training episodes.")
        if strategy not in STRATEGIES:
            raise ValueError("Choose policy_search or ppo for fine-tuning.")
        self.strategy = strategy
        self.base = NearestNeighborPolicy(model)
        if self.base.arm_base_distance != ARM_BASE_DISTANCE_M:
            raise ValueError("This policy uses different arm spacing. Retrain behavior cloning for this classroom.")
        # Defensive copies prevent training or callers from changing the BC labels.
        self.base.states = self.base.states.copy()
        self.base.actions = self.base.actions.copy()
        self.base.states.flags.writeable = False
        self.base.actions.flags.writeable = False
        self.policy = (ScaledClonedPolicy(self.base) if strategy == "policy_search"
                       else FineTunedPolicy(self.base))
        self.best_policy = deepcopy(self.policy)
        self.search_proposals = []
        self.search_center = 1.0
        self.candidate_policy = None
        self.rng = np.random.default_rng(seed)
        # Demonstrations cover varied poses; this experiment changes only the
        # policy. Every exploration/evaluation starts from the same fixed pose.
        self.evaluation_seed = POLICY_START_SEED
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
        self.critic_gram = np.zeros((10, 10))
        self.critic_rhs = np.zeros(10)
        self.update = {"mean_kl": 0.0, "mean_change_bound": 0.0, "actor_change": 0.0}
        self.session = InteractiveSession(
            self.evaluation_seed, 700, arm_base_distance=ARM_BASE_DISTANCE_M, dt=BROWSER_DT, steps_per_update=1, horizon=TRIAL_STEPS,
            reset_options=fixed_policy_layout(), include_render_info=False,
        )
        self._reset_rollout()

    def _reset_rollout(self):
        seed = self.evaluation_seed
        self.session.reset_options = fixed_policy_layout()
        self.session.restart(seed=seed, target_ml=700, speed=1, horizon=TRIAL_STEPS)
        self.session.paused = False
        self.rollout_features = []
        self.rollout_latent = []
        self.rollout_means = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_discounts = []
        self.rollout_discount_weights = []
        self.discounted_return = 0.0
        self.discount_factor = 1.0
        self.decision_remaining = 0
        self.latent = 0.0
        if self.strategy == "policy_search" and self.phase == "training":
            if not self.search_proposals:
                self.search_center = self.policy.multiplier
                radius = float(self.rng.uniform(SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX))
                self.search_proposals = [
                    float(np.clip(self.search_center + sign * radius, SEARCH_SPEED_MIN, SEARCH_SPEED_MAX))
                    for sign in self.rng.permutation([-1, 1])
                ]
            self.candidate_policy = ScaledClonedPolicy(self.base, self.search_proposals.pop(0))

    def _begin_decision(self):
        if self.strategy == "policy_search":
            self.decision_remaining = DECISION_STEPS
            return
        features = actor_features(self.session.observation)
        mean = float(features @ self.policy.weights) if self.phase != "baseline" else 0.0
        self.latent = float(self.rng.normal(mean, LATENT_STD)) if self.phase == "training" else mean
        self.rollout_features.append(features)
        self.rollout_latent.append(self.latent)
        self.rollout_means.append(mean)
        self.rollout_rewards.append(0.0)
        self.rollout_values.append(float(critic_features(features) @ self.critic))
        self.rollout_discounts.append(1.0)
        self.rollout_discount_weights.append(self.discount_factor)
        self.decision_remaining = DECISION_STEPS

    def _fit_critic(self):
        y = generalized_advantages(
            self.rollout_rewards, np.zeros(len(self.rollout_rewards)),
            discounts=self.rollout_discounts, trace_decay=1.0,
        )
        x = critic_features(np.asarray(self.rollout_features))
        # Reward-to-go regression is critic learning; demonstrations are never a
        # critic target. Ridge regularization keeps a tiny linear model stable.
        # Constant-size sufficient statistics avoid refitting an ever-growing
        # history at iteration 100. Forget stale policies gradually.
        self.critic_gram *= CRITIC_RETENTION
        self.critic_rhs *= CRITIC_RETENTION
        self.critic_gram += x.T @ x
        self.critic_rhs += x.T @ y
        self.critic = np.linalg.solve(self.critic_gram + 0.05 * np.eye(x.shape[1]), self.critic_rhs)

    def _metrics(self):
        return {
            "return": float(self.discounted_return),
            "raw_return": float(self.session.cumulative_reward),
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
            if self.strategy == "ppo":
                self._fit_critic()
            self.episode = 1
            self.phase = "training"
        elif self.phase == "training":
            if self.strategy == "policy_search":
                old_speed = self.policy.multiplier
                candidate_speed = self.candidate_policy.multiplier
                accepted = metrics["return"] > self.best["return"]
                if accepted:
                    self.policy = self.candidate_policy
                self.update = {
                    "actor_change": abs(self.policy.multiplier - old_speed),
                    "accepted": accepted, "pair_center": self.search_center,
                    "candidate_speed": candidate_speed,
                    "exploration": {
                        "kind": "paired_parameter", "decisions": 1,
                        "speed_min": candidate_speed, "speed_max": candidate_speed,
                        "slower_decisions": int(candidate_speed < self.search_center),
                        "faster_decisions": int(candidate_speed > self.search_center),
                    },
                }
            else:
                advantages = generalized_advantages(
                    self.rollout_rewards, self.rollout_values, discounts=self.rollout_discounts,
                )
                self.policy.weights, self.update = ppo_actor_update(
                    self.policy.weights, self.rollout_features, self.rollout_latent, advantages,
                    epochs=PPO_EPOCHS, discount_weights=self.rollout_discount_weights,
                )
                self.update["exploration"] = self._exploration_metrics()
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
                self.best_policy = deepcopy(self.policy)
            if self.episode >= self.episodes:
                self.done = True
                self.phase = "complete"
                self.session.paused = True
                return
            self.episode += 1
            self.phase = "training"
        self._reset_rollout()

    def _exploration_metrics(self):
        """Observed perturbation coverage, distinct from the evaluation score."""
        latents = np.asarray(self.rollout_latent)
        means = np.asarray(self.rollout_means)
        gains = SPEED_BOUND * np.tanh(latents)
        return {
            "decisions": len(latents),
            "speed_min": float(1 + gains.min()),
            "speed_max": float(1 + gains.max()),
            "speed_std": float(gains.std()),
            "noise_std": float((latents - means).std()),
            "slower_decisions": int(np.sum(latents < means)),
            "faster_decisions": int(np.sum(latents > means)),
        }

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
            if self.strategy == "policy_search":
                active_policy = self.candidate_policy if self.phase == "training" else self.policy
                action = active_policy.predict(self.session.observation)
            else:
                latent = self.latent
                if self.phase == "evaluation":
                    latent = float(actor_features(self.session.observation) @ self.policy.weights)
                action = self.policy.action_with_latent(self.session.observation, latent)
            for index, value in enumerate(action):
                self.session.set_motor(index, float(value))
            self.session.advance()
            reward = fine_tuning_reward(self.session.info, BROWSER_DT)
            self.discounted_return += self.discount_factor * reward
            self.discount_factor *= STEP_DISCOUNT
            if self.strategy == "ppo":
                self.rollout_rewards[-1] += self.rollout_discounts[-1] * reward
                self.rollout_discounts[-1] *= STEP_DISCOUNT
            self.decision_remaining -= 1
            self.total_steps += 1
            if not self.session.running:
                self._finish_rollout()
        return self.result()

    def result(self):
        return {
            "algorithm": ("bounded_paired_policy_search" if self.strategy == "policy_search"
                          else "bounded_speed_ppo_actor_critic"),
            "strategy": self.strategy,
            "speed_multiplier": getattr(self.policy, "multiplier", None),
            "best_speed_multiplier": getattr(self.best_policy, "multiplier", None),
            "search_speed_range": [SEARCH_SPEED_MIN, SEARCH_SPEED_MAX],
            "search_radius_range": [SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX],
            "phase": self.phase,
            "done": self.done,
            "episode": self.episode,
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "episode_steps": self.session.env.elapsed_steps,
            "seed": self.seed,
            "evaluation_seed": self.evaluation_seed,
            "arm_base_distance_m": ARM_BASE_DISTANCE_M,
            "speed_bound": SPEED_BOUND,
            "latent_std": LATENT_STD,
            "max_latent_mean": MAX_LATENT_MEAN,
            "discount_per_second": DISCOUNT_PER_SECOND,
            "step_discount": STEP_DISCOUNT,
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
