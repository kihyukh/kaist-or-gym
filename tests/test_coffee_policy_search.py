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
            trainer.policy = ScaledClonedPolicy(trainer.base, center)
            trainer.phase = "training"
            trainer._reset_rollout()
            first = trainer.candidate_policy.multiplier
            trainer.policy = trainer.candidate_policy  # Pretend the first trial improved reward.
            trainer._reset_rollout()
            second = trainer.candidate_policy.multiplier
            assert trainer.search_center == center
            assert SEARCH_SPEED_MIN <= min(first, second) <= center <= max(first, second) <= SEARCH_SPEED_MAX
            assert SEARCH_RADIUS_MIN <= max(abs(first - center), abs(second - center)) <= SEARCH_RADIUS_MAX
            assert abs(first - second) <= 2 * SEARCH_RADIUS_MAX
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
            assert exploration["kind"] == "paired_parameter"
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
