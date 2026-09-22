"""Bounded direction-capable exploration and independent evaluation."""

import json
from copy import deepcopy

import numpy as np
import pytest

from kaist_rl_lab.apps import coffee_finetuning as core
from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M
from kaist_rl_lab.apps.coffee_finetuning_runtime import FineTuningRuntime


def stationary_model():
    return {
        'schema_version': 2, 'algorithm': 'nearest_neighbor',
        'arm_base_distance_m': ARM_BASE_DISTANCE_M,
        'feature_indices': list(range(15)),
        'feature_scales': [1.] * 6 + [.5] * 6 + [1.] * 3,
        'states': [[0.] * 15], 'actions': [[0.] * 6], 'metrics': {},
    }


def test_residual_can_move_a_stalled_joint_and_reverse_a_small_wrong_command():
    class Base:
        def predict(self, observation):
            return np.array([0, -.001, .99, -.9, .3, .8], dtype=np.float32)

    base = Base()
    original = base.predict(None)
    policy = core.ResidualClonedPolicy(base)
    np.testing.assert_array_equal(policy.predict(None), original)
    policy.weights[:] = 2
    changed = policy.predict(None)
    assert changed[0] > 0 and changed[1] > 0
    assert np.all(abs(changed - original) <= core.RESIDUAL_BOUND + 1e-7)
    assert np.max(abs(changed)) <= 1
    policy.weights[:] = -2
    assert policy.predict(None)[0] < 0


@pytest.mark.parametrize('weights', [np.zeros(6), np.full((2, 6), np.nan), np.full((2, 6), 3)])
def test_invalid_residual_parameters_are_rejected(weights):
    with pytest.raises(ValueError):
        core.ResidualClonedPolicy(None, weights)


def test_actual_motor_exploration_is_displayed_and_failed_candidates_are_rejected(monkeypatch):
    monkeypatch.setattr(core, 'TRIAL_STEPS', 3)
    model = stationary_model()
    original = deepcopy(model)
    runtime = FineTuningRuntime()

    def call(kind, **kwargs):
        return json.loads(runtime.dispatch(json.dumps({'kind': kind, **kwargs})))

    try:
        call('ft-load', model=model)
        call('ft-train', strategy='residual_search', episodes=2, seed=2026)
        call('ft-step')  # Original clone's terminal scene.
        call('ft-step')  # Compute baseline, choose first candidate.
        frame = call('ft-step')
        live = frame['finetuning']['live_action']
        assert live['source'] == 'exploration'
        assert live['max_action_change'] > 0
        assert live['max_action_change'] < core.RESIDUAL_BOUND
        np.testing.assert_array_equal(live['cloned_action'], np.zeros(6))
        np.testing.assert_array_equal(live['executed_action'], runtime.session.trajectory[-1]['action'])
        assert frame['snapshot'] == runtime.session.animation_snapshot()
        call('ft-step')  # Candidate rejected by actual final tilt cost.
        first = runtime.result['history'][0]
        assert not first['update']['accepted']
        assert first['training']['return'] < runtime.result['baseline']['return']
        evaluation = call('ft-step')['finetuning']['live_action']
        assert evaluation['source'] == 'deterministic_evaluation'
        np.testing.assert_array_equal(evaluation['executed_action'], np.zeros(6))
        while runtime.training_active:
            call('ft-step')
        assert runtime.result['completed_episodes'] == 2
        assert runtime.result['best']['episode'] == 0
        history = runtime.result['history']
        assert all(row['evaluation'] is not None for row in history)
        np.testing.assert_allclose(history[0]['update']['candidate_offsets'],
                                   -np.array(history[1]['update']['candidate_offsets']))
        assert model == original
    finally:
        runtime.close()
