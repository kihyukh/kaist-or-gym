"""Reward learning, trust bounds, and real-physics checkpoint validation."""

import copy

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M, fixed_policy_layout
from kaist_rl_lab.apps.coffee_cloning import NearestNeighborPolicy, train_behavior_cloning
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import load_examples
from kaist_rl_lab.apps.coffee_finetuning import (
    ACTOR_FEATURES,
    DEFAULT_EPISODES,
    LATENT_STD,
    MAX_EPISODES,
    MAX_LATENT_MEAN,
    MAX_MEAN_CHANGE,
    MAX_UPDATE_KL,
    SPEED_BOUND,
    STEP_DISCOUNT,
    FineTunedPolicy,
    FineTuningTrainer,
    actor_features,
    generalized_advantages,
    ppo_actor_update,
)
from kaist_rl_lab.envs import CoffeePouringEnv


@pytest.fixture(scope="module")
def example_model():
    return train_behavior_cloning([read_demonstration(data) for data in load_examples()])


def test_finite_horizon_gae_discounts_variable_duration_decisions_and_zero_bootstrap():
    rewards = np.array([0.2, -0.1, 15.0])
    values = np.array([10.0, 12.0, 14.0])
    discounts = STEP_DISCOUNT ** np.array([8, 8, 3])
    returns = np.array([rewards[0] + discounts[0] * (rewards[1] + discounts[1] * rewards[2]),
                        rewards[1] + discounts[1] * rewards[2], rewards[2]])
    np.testing.assert_allclose(
        generalized_advantages(rewards, values, discounts=discounts, trace_decay=1),
        returns - values,
    )
    np.testing.assert_allclose(
        generalized_advantages(rewards, values, discounts=discounts, trace_decay=0),
        rewards + discounts * np.append(values[1:], 0) - values,
    )
    # Separate calls are episode boundaries; the second trajectory never enters
    # the first return, including the environment's task-terminal time limit.
    np.testing.assert_allclose(generalized_advantages([1], [5]), [-4])


def test_ppo_gradient_uses_rewards_and_respects_analytic_trust_region():
    features = np.zeros((100, ACTOR_FEATURES))
    features[:, 0] = 1
    latent = np.linspace(-0.6, 0.6, len(features))
    initial = np.zeros(ACTOR_FEATURES)
    positive, stats = ppo_actor_update(initial, features, latent, latent)
    negative, _ = ppo_actor_update(initial, features, latent, -latent)
    stationary, _ = ppo_actor_update(initial, features, latent, np.ones(len(features)))
    assert positive[0] > 0 > negative[0]
    np.testing.assert_array_equal(initial, 0)
    np.testing.assert_array_equal(stationary, initial)
    actual_kl = np.mean((features @ (positive - initial)) ** 2) / (2 * LATENT_STD**2)
    assert actual_kl == pytest.approx(stats["mean_kl"])
    assert actual_kl <= MAX_UPDATE_KL
    assert 1.5 * abs(positive - initial).sum() <= MAX_MEAN_CHANGE
    assert stats["actor_change"] > 0


def test_ppo_trust_bound_survives_large_advantages_and_repeated_epochs():
    rng = np.random.default_rng(7)
    for _ in range(20):
        features = rng.uniform(-1, 1, (75, ACTOR_FEATURES))
        features[:, 0] = 1
        initial = rng.normal(size=ACTOR_FEATURES)
        latent = features @ initial + rng.normal(0, LATENT_STD, len(features))
        updated, stats = ppo_actor_update(
            initial, features, latent, rng.normal(size=len(features)) * 1e5, epochs=10,
        )
        assert stats["mean_kl"] <= MAX_UPDATE_KL + 1e-12
        assert 1.5 * abs(updated - initial).sum() <= MAX_MEAN_CHANGE + 1e-12


def test_speed_refinement_bounds_all_motors_and_preserves_stopped_joints():
    action = np.array([0, 0.3, -0.8, 1, -1, 0.02], dtype=np.float32)

    class Base:
        def predict(self, observation):
            return action.copy()

    observation = np.zeros(16)
    observation[14] = 0.7
    policy = FineTunedPolicy(Base())
    np.testing.assert_array_equal(policy.predict(observation), action)
    for latent in [-1e6, -2, 0, 2, 1e6]:
        refined = policy.action_with_latent(observation, latent)
        assert np.all(abs(refined - action) <= SPEED_BOUND * abs(action) + 1e-7)
        assert np.all(abs(refined) <= 1)
        assert refined[0] == 0
        np.testing.assert_array_equal(np.sign(refined), np.sign(action))
    observation[:] = 1e4
    assert np.max(abs(actor_features(observation))) <= 1.5


def test_incremental_training_preserves_bc_and_never_exceeds_chunk(example_model):
    original = copy.deepcopy(example_model)
    trainer = FineTuningTrainer(example_model, episodes=1)
    try:
        assert trainer.result()["baseline"] is None
        for count in [1, 8, 32]:
            before = trainer.total_steps
            trainer.step_chunk(count)
            assert trainer.total_steps - before == count
        assert example_model == original
        assert not trainer.base.states.flags.writeable
        assert not trainer.base.actions.flags.writeable
        np.testing.assert_array_equal(trainer.policy.weights, 0)
        for invalid in [0, 33, True, 1.5]:
            with pytest.raises(ValueError):
                trainer.step_chunk(invalid)
    finally:
        trainer.close()


def test_real_training_uses_rewards_retains_best_and_replays_exactly(example_model):
    original = copy.deepcopy(example_model)
    trainer = FineTuningTrainer(example_model, seed=2027, episodes=1)
    try:
        while not trainer.done:
            trainer.step_chunk()
        result = trainer.result()
        assert result["phase"] == "complete"
        assert result["baseline"]["success"]
        assert result["best"]["return"] >= result["baseline"]["return"]
        assert len(result["history"]) == 1
        assert all(row["evaluation"] is not None for row in result["history"])
        evaluation_seed = result["evaluation_seed"]
        assert result["baseline"]["initial_seed"] == evaluation_seed
        assert result["history"][0]["evaluation"]["initial_seed"] == evaluation_seed
        assert result["history"][0]["training"]["initial_seed"] == evaluation_seed
        assert any(row["update"]["actor_change"] > 0 for row in result["history"])
        assert np.linalg.norm(trainer.critic) > 0
        assert example_model == original
        assert trainer.step_chunk() == result
        env = CoffeePouringEnv(arm_base_distance=ARM_BASE_DISTANCE_M, dt=BROWSER_DT, horizon=60 * 32)
        observation, _ = env.reset(
            seed=evaluation_seed, options={**fixed_policy_layout(), "target_fill": 0.7},
        )
        reward = 0.0
        raw_reward = 0.0
        discount = 1.0
        for _ in range(60 * 32):
            observation, current, terminal, truncated, info = env.step(
                trainer.best_policy.predict(observation),
            )
            reward += discount * current
            raw_reward += current
            discount *= STEP_DISCOUNT
            if terminal or truncated:
                break
        assert reward == pytest.approx(result["best"]["return"], abs=1e-9)
        assert raw_reward == pytest.approx(result["best"]["raw_return"], abs=1e-9)
        assert reward < raw_reward
        assert env.fill * 1000 == pytest.approx(result["best"]["fill_ml"], abs=1e-9)
        assert info["is_success"] == result["best"]["success"]
        env.close()
    finally:
        trainer.close()


@pytest.mark.parametrize("options", [{"seed": -1}, {"seed": True}, {"episodes": 0},
                                     {"episodes": 101}, {"episodes": True}])
def test_training_budget_is_bounded(example_model, options):
    with pytest.raises(ValueError):
        FineTuningTrainer(example_model, **options)


def test_new_actor_is_exactly_the_cloned_policy(example_model):
    base = NearestNeighborPolicy(example_model)
    refined = FineTunedPolicy(base)
    observation = np.zeros(16)
    observation[14] = 0.7
    np.testing.assert_array_equal(refined.predict(observation), base.predict(observation))


def test_ppo_update_matches_finite_difference_of_clipped_objective():
    rng = np.random.default_rng(1024)
    for _ in range(3):
        x = rng.uniform(-1, 1, (22, ACTOR_FEATURES))
        x[:, 0] = 1
        weights = rng.normal(0, 0.03, ACTOR_FEATURES)
        latent = x @ weights + rng.normal(0, LATENT_STD, len(x))
        advantages = rng.normal(size=len(x))
        advantages = (advantages - advantages.mean()) / advantages.std()
        old_logp = -0.5 * ((latent - x @ weights) / LATENT_STD) ** 2

        def objective(candidate, latent=latent, x=x, advantages=advantages, old_logp=old_logp):
            ratio = np.exp(-0.5 * ((latent - x @ candidate) / LATENT_STD) ** 2 - old_logp)
            return np.minimum(
                ratio * advantages, np.clip(ratio, 0.8, 1.2) * advantages,
            ).mean()

        expected = weights.copy()
        for _ in range(4):
            gradient = np.array([
                (objective(expected + 1e-6 * basis) - objective(expected - 1e-6 * basis)) / 2e-6
                for basis in np.eye(ACTOR_FEATURES)
            ])
            step = 0.025 * gradient / max(np.linalg.norm(gradient), 1.0)
            for _ in range(16):
                candidate = expected + step
                delta = candidate - weights
                kl = np.mean((x @ delta) ** 2) / (2 * LATENT_STD**2)
                if 1.5 * abs(delta).sum() <= MAX_MEAN_CHANGE and kl <= MAX_UPDATE_KL:
                    expected = candidate
                    break
                step *= 0.5
        actual, _ = ppo_actor_update(weights, x, latent, advantages)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-6)


def test_discount_prefers_earlier_equivalent_success_and_is_physics_time_based():
    # A fixed success reward is worth more at 20 seconds than at 30 seconds.
    assert STEP_DISCOUNT**32 == pytest.approx(0.99)
    assert 15 * STEP_DISCOUNT**(20 * 32) > 15 * STEP_DISCOUNT**(30 * 32)
    # 19 physics rewards, including a final partial decision, must give the
    # same return whether discounted directly or grouped into actor decisions.
    rewards = np.linspace(-0.2, 1.2, 19)
    groups = [rewards[:8], rewards[8:16], rewards[16:]]
    grouped = [np.sum(group * STEP_DISCOUNT ** np.arange(len(group))) for group in groups]
    discounts = [STEP_DISCOUNT ** len(group) for group in groups]
    actual = generalized_advantages(grouped, np.zeros(3), discounts=discounts, trace_decay=1)[0]
    assert actual == pytest.approx(np.sum(rewards * STEP_DISCOUNT ** np.arange(19)))


def test_hundred_iterations_default_and_critic_memory_is_constant(example_model):
    trainer = FineTuningTrainer(example_model)
    try:
        assert trainer.episodes == DEFAULT_EPISODES == MAX_EPISODES == 100
        features = np.array([[1, 0, 0, 0, 0], [1, 0.5, 0.2, 0, 0]])
        trainer.rollout_features = features
        trainer.rollout_rewards = [1.0, 15.0]
        trainer.rollout_discounts = [STEP_DISCOUNT**8, STEP_DISCOUNT**3]
        for _ in range(100):
            trainer._fit_critic()
        assert trainer.critic_gram.shape == (10, 10)
        assert trainer.critic_rhs.shape == (10,)
        assert np.isfinite(trainer.critic).all()
        assert not hasattr(trainer, 'critic_states')
    finally:
        trainer.close()


def test_exploration_covers_slower_and_faster_controls_without_saturating():
    rng = np.random.default_rng(17)
    noise = rng.normal(0, LATENT_STD, 50_000)
    for mean in [-MAX_LATENT_MEAN, 0, MAX_LATENT_MEAN]:
        gain = SPEED_BOUND * np.tanh(mean + noise)
        assert gain.std() > 0.035
        assert np.mean(gain < SPEED_BOUND * np.tanh(mean)) == pytest.approx(0.5, abs=0.015)
        assert np.mean(gain > SPEED_BOUND * np.tanh(mean)) == pytest.approx(0.5, abs=0.015)
    # Deliberately pressure the actor toward saturation for 100 updates.
    features = np.zeros((100, ACTOR_FEATURES))
    features[:, 0] = 1
    weights = np.zeros(ACTOR_FEATURES)
    for _ in range(100):
        latent = features @ weights + rng.normal(0, LATENT_STD, len(features))
        weights, _ = ppo_actor_update(weights, features, latent, latent)
        assert 1.5 * abs(weights).sum() <= MAX_LATENT_MEAN + 1e-12


def test_evaluation_is_greedy_current_policy_and_never_samples_noise(example_model):
    class NoNoise:
        def normal(self, *args, **kwargs):
            raise AssertionError('Evaluation must never sample exploration noise')

    trainer = FineTuningTrainer(example_model, episodes=1)
    try:
        trainer.phase = 'evaluation'
        trainer.history = [{'episode': 1, 'evaluation': None}]
        trainer.policy.weights[:] = [0.12, 0.02, -0.01, 0.03, 0]
        trainer.rng = NoNoise()
        expected_env = CoffeePouringEnv(arm_base_distance=ARM_BASE_DISTANCE_M, dt=BROWSER_DT, horizon=60 * 32)
        observation, _ = expected_env.reset(seed=trainer.evaluation_seed,
                                            options={**fixed_policy_layout(), 'target_fill': 0.7})
        discounted = 0.0
        for step in range(20):
            action = trainer.policy.predict(observation)
            observation, reward, _, _, _ = expected_env.step(action)
            discounted += STEP_DISCOUNT**step * reward
            trainer.step_chunk(1)
            np.testing.assert_array_equal(trainer.session.observation, observation)
            np.testing.assert_array_equal(trainer.session.trajectory[-1]['action'], action)
        assert trainer.discounted_return == pytest.approx(discounted, abs=1e-12)
        assert trainer.history[0]['evaluation'] is None  # Never publish a partial evaluation.
        expected_env.close()
    finally:
        trainer.close()
