"""Reward-selected coherent exploration and independent policy evaluation."""

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_cloning import train_behavior_cloning
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import load_examples
from kaist_rl_lab.apps.coffee_finetuning import (
    SEARCH_RADIUS_MAX,
    SEARCH_RADIUS_MIN,
    SEARCH_SPEED_MAX,
    SEARCH_SPEED_MIN,
    FineTuningTrainer,
    ScaledClonedPolicy,
)


@pytest.fixture(scope="module")
def model():
    return train_behavior_cloning([read_demonstration(data) for data in load_examples()])


def test_paired_proposals_stay_anchored_while_incumbent_changes(model):
    trainer = FineTuningTrainer(model, strategy="policy_search", episodes=2, seed=2026)
    try:
        for center in (SEARCH_SPEED_MIN, 1.0, SEARCH_SPEED_MAX):
            trainer.search_proposals.clear()
            trainer.search_pair_started = False
            trainer.policy = ScaledClonedPolicy(trainer.base, center, 0.93)
            trainer.phase = "training"
            trainer._reset_rollout()
            first = trainer.candidate_policy.multiplier
            trainer.policy = trainer.candidate_policy  # Pretend the first trial improved reward.
            trainer._reset_rollout()
            second = trainer.candidate_policy.multiplier
            np.testing.assert_array_equal(trainer.search_center, [center, 0.93])
            assert trainer.candidate_policy.return_multiplier == 0.93
            assert SEARCH_SPEED_MIN <= min(first, second) <= center <= max(first, second) <= SEARCH_SPEED_MAX
            assert SEARCH_RADIUS_MIN <= max(abs(first - center), abs(second - center)) <= SEARCH_RADIUS_MAX
            assert abs(first - second) <= 2 * SEARCH_RADIUS_MAX
    finally:
        trainer.close()


def test_failed_pair_rotates_and_shrinks_only_after_both_proposals(model):
    trainer = FineTuningTrainer(model, strategy="policy_search", episodes=2, seed=2026)
    try:
        trainer.phase = "training"
        trainer._reset_rollout()
        assert trainer.search_coordinate == 0
        trainer._reset_rollout()
        assert trainer.search_coordinate == 0
        trainer._reset_rollout()  # Both rejected: switch to the other gain.
        assert trainer.search_coordinate == 1
        np.testing.assert_array_equal(trainer.search_radius_scales, [0.5, 1])
        trainer.search_pair_improved = True
        trainer._reset_rollout()
        trainer._reset_rollout()  # Either proposal improved: keep this gain.
        assert trainer.search_coordinate == 1
        np.testing.assert_array_equal(trainer.search_radius_scales, [0.5, 1])
    finally:
        trainer.close()


def test_search_changes_speed_but_preserves_stop_direction_and_control_limits():
    action = np.array([0, .3, -.8, 1, -1, .02], dtype=np.float32)

    class Base:
        def predict(self, observation):
            return action.copy()

    for multiplier in (SEARCH_SPEED_MIN, 1.0, SEARCH_SPEED_MAX):
        policy = ScaledClonedPolicy(Base(), multiplier)
        actual = policy.predict(np.zeros(16))
        np.testing.assert_array_equal(actual, np.clip(action * multiplier, -1, 1))
        np.testing.assert_array_equal(np.sign(actual), np.sign(action))
        assert actual[0] == 0


def test_search_uses_live_cloned_pot_command_for_phase_without_modifying_observation():
    observation = np.arange(16, dtype=np.float32)

    class Base:
        def predict(self, actual_observation):
            assert actual_observation is observation
            return self.action.copy()

    base = Base()
    policy = ScaledClonedPolicy(base, 1.3, 0.8)
    for pot_action, expected_gain in (([.2, .3, -.1], 1.3), ([-.2, -.3, .1], .8), ([0, 0, 0], 1.3)):
        base.action = np.array([0, .3, -.8, *pot_action], dtype=np.float32)
        np.testing.assert_array_equal(policy.predict(observation), np.clip(base.action * expected_gain, -1, 1))
        np.testing.assert_array_equal(observation, np.arange(16, dtype=np.float32))


@pytest.mark.parametrize("gains", [(float("nan"), 1), (1, float("inf")), (.69, 1), (1, 1.41)])
def test_search_rejects_invalid_phase_gains(gains):
    with pytest.raises(ValueError, match="speed multiplier"):
        ScaledClonedPolicy(None, *gains)


def test_real_search_evaluates_separately_and_learns_from_reward(model):
    trainer = FineTuningTrainer(model, strategy="policy_search", episodes=2, seed=2026)
    try:
        while not trainer.done:
            trainer.step_chunk()
        result = trainer.result()
        assert result["algorithm"] == "bounded_paired_policy_search"
        assert result["best"]["return"] > result["baseline"]["return"]
        assert result["best"]["seconds"] < result["baseline"]["seconds"]
        assert result["best"]["success"]
        rows = result["history"]
        assert len(rows) == 2
        total_seconds = result["baseline"]["seconds"]
        for row in rows:
            total_seconds += row["training"]["seconds"] + row["evaluation"]["seconds"]
            assert row["evaluation"]["success"]
            assert row["evaluation"]["return"] >= result["baseline"]["return"]
            exploration = row["update"]["exploration"]
            assert exploration["decisions"] == 1
            assert exploration["kind"] == "paired_phase_parameter"
            if row["update"]["accepted"]:
                # Repeating the learned policy in this deterministic environment
                # agrees exactly, but the transitions must still be recomputed.
                assert row["evaluation"] == row["training"]
        assert trainer.total_steps == round(total_seconds * 32)
        assert sum(row["update"]["exploration"]["faster_decisions"] for row in rows) == 1
        assert sum(row["update"]["exploration"]["slower_decisions"] for row in rows) == 1
        assert not np.any(trainer.critic)  # Search is not mislabeled actor–critic.
    finally:
        trainer.close()


def test_real_ten_trial_search_improves_precision_and_speed(model):
    trainer = FineTuningTrainer(model, strategy="policy_search", episodes=10, seed=2026)
    try:
        while not trainer.done:
            trainer.step_chunk()
        result = trainer.result()
        assert result["parameterization"] == "phase_speeds"
        assert result["best"]["success"]
        assert abs(result["best"]["fill_ml"] - 700) <= 5
        assert result["best"]["seconds"] < result["baseline"]["seconds"]
        assert result["best"]["return"] > result["baseline"]["return"]
        assert result["actor_weights"] == [result["speed_multiplier"], result["return_speed_multiplier"]]
        assert all(row["evaluation"] is not None for row in result["history"])
        assert not np.any(trainer.critic)
    finally:
        trainer.close()
