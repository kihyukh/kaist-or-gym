"""The cloned policy learns demonstrated controls and runs the actual dynamics."""

import copy
import json

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M
from kaist_rl_lab.apps.coffee_cloning import (
    MAX_MODEL_SAMPLES,
    NearestNeighborPolicy,
    train_behavior_cloning,
)
from kaist_rl_lab.apps.coffee_cloning_runtime import CloningAgentRuntime


def demonstration(actions, *, positions=None):
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 6)
    observations = np.zeros((len(actions), 16), dtype=np.float32)
    observations[:, 0] = positions if positions is not None else np.arange(len(actions)) * 0.01
    observations[:, 14] = 0.7
    observations[:, 15] = -1
    return {"observations": observations, "actions": actions}, {
        "dt": BROWSER_DT, "target_fill_l": 0.7, "arm_base_distance_m": ARM_BASE_DISTANCE_M,
    }


def call(runtime, kind, **kwargs):
    return json.loads(runtime.dispatch(json.dumps({"kind": kind, **kwargs})))


@pytest.fixture
def runtime():
    value = CloningAgentRuntime()
    yield value
    value.session.close()


def test_actions_are_supervised_labels_and_time_rewards_future_are_unused():
    pair = demonstration([[1, 0, -1, 0, 0.25, 0], [-1, 0, 1, 0, 0.5, 0]])
    model = train_behavior_cloning([pair])
    policy = NearestNeighborPolicy(json.loads(json.dumps(model)))
    for index, expected in enumerate(pair[0]["actions"]):
        observed = pair[0]["observations"][index].copy()
        observed[15] = 0.5
        np.testing.assert_array_equal(policy.predict(observed), expected)
    assert model["metrics"]["heldout_action_mae"] is None
    changed = copy.deepcopy(pair)
    changed[0]["rewards"] = np.full(2, 1e10)
    changed[0]["next_observations"] = np.full((2, 16), -1e10)
    changed[1]["success"] = False
    assert train_behavior_cloning([changed]) == model


def test_holdout_excludes_an_entire_trajectory_and_final_fit_uses_it():
    first = demonstration(np.zeros((4, 6)))
    heldout = demonstration(np.ones((4, 6)))
    model = train_behavior_cloning([first, heldout])
    assert model["metrics"] == {
        "demonstrations": 2, "total_steps": 8, "training_samples": 4,
        "validation_trajectories": 1, "validation_steps": 4,
        "validation_total_steps": 4, "heldout_action_mae": 1.0,
    }
    np.testing.assert_array_equal(NearestNeighborPolicy(model).predict(first[0]["observations"][0]), 1)


def test_exact_stationary_duplicates_keep_last_demonstrated_action():
    pair = demonstration(np.vstack([np.zeros(6), np.ones(6)]), positions=[0, 0])
    model = train_behavior_cloning([pair])
    assert model["metrics"]["training_samples"] == 1
    np.testing.assert_array_equal(NearestNeighborPolicy(model).predict(pair[0]["observations"][0]), 1)


def test_budget_is_balanced_per_trajectory_and_redistributed():
    short = demonstration(np.zeros((2, 6)), positions=[-1, -0.99])
    long = demonstration(np.ones((20, 6)))
    model = train_behavior_cloning([short, long], max_samples=8)
    assert model["metrics"]["training_samples"] == 8
    assert sum(row[0] == 0 for row in model["actions"]) == 2
    assert sum(row[0] == 1 for row in model["actions"]) == 6


@pytest.mark.parametrize("mutation", [
    lambda arrays, metadata: metadata.update(dt=0.05),
    lambda arrays, metadata: metadata.update(target_fill_l=0.5),
    lambda arrays, metadata: metadata.update(target_fill_l=float("nan")),
    lambda arrays, metadata: arrays.update(actions=np.zeros((2, 5))),
    lambda arrays, metadata: arrays.update(actions=np.full((2, 6), 1.1)),
    lambda arrays, metadata: arrays.update(observations=np.full((2, 16), float("nan"))),
    lambda arrays, metadata: arrays.update(observations=np.zeros((2, 16))),
])
def test_rejects_incompatible_or_invalid_training_data(mutation):
    pair = demonstration(np.zeros((2, 6)))
    mutation(*pair)
    with pytest.raises(ValueError):
        train_behavior_cloning([pair])


@pytest.mark.parametrize("max_samples", [0, True, MAX_MODEL_SAMPLES + 1])
def test_rejects_unbounded_or_invalid_training_budget(max_samples):
    with pytest.raises(ValueError):
        train_behavior_cloning([demonstration(np.zeros((2, 6)))], max_samples=max_samples)


@pytest.mark.parametrize("mutation", [
    lambda model: model.update(algorithm="expert_controller"),
    lambda model: model.update(feature_indices=list(range(16))),
    lambda model: model.update(feature_scales=[0] * 15),
    lambda model: model.update(states=[]),
    lambda model: model.update(actions=[[float("nan")] * 6]),
    lambda model: model.update(actions=[[1.1] * 6]),
])
def test_browser_model_rejects_invalid_payloads(mutation):
    model = train_behavior_cloning([demonstration(np.zeros((2, 6)))])
    mutation(model)
    with pytest.raises(ValueError):
        NearestNeighborPolicy(model)


def test_runtime_requires_model_and_pause_reset_preserve_it(runtime):
    initial = call(runtime, "snapshot")
    assert not initial["cloning_agent"]["model_loaded"]
    assert call(runtime, "tick") == initial
    with pytest.raises(ValueError, match="Train and load"):
        call(runtime, "cloning-start")
    loaded = call(runtime, "cloning-load", model=train_behavior_cloning([
        demonstration(np.zeros((2, 6))),
    ]))
    assert loaded["cloning_agent"]["model_loaded"]
    assert loaded["snapshot"]["playback"]["paused"]
    assert call(runtime, "tick") == loaded
    first = call(runtime, "cloning-start")
    assert first["cloning_agent"]["attempt"] == 1
    assert call(runtime, "tick")["cloning_agent"]["step"] == 1
    paused = call(runtime, "cloning-pause", paused=True)
    assert call(runtime, "tick") == paused
    call(runtime, "cloning-pause", paused=False)
    assert call(runtime, "tick")["cloning_agent"]["step"] == 2
    second = call(runtime, "cloning-start")
    assert second["cloning_agent"]["attempt"] == 2
    assert second["cloning_agent"]["step"] == 0
    reset = call(runtime, "cloning-reset")
    assert reset["cloning_agent"]["model_loaded"]
    assert reset["cloning_agent"]["attempt"] == 0
    assert reset["snapshot"]["playback"]["paused"]
    assert call(runtime, "tick") == reset


def test_runtime_queries_current_observation_for_every_action(runtime):
    observed = []

    class RecordingPolicy:
        def predict(self, observation):
            observed.append(observation.copy())
            return np.asarray([0, 0, 0, 0, 0, 0.25], dtype=np.float32)

    runtime.policy = RecordingPolicy()
    call(runtime, "cloning-start")
    for _ in range(3):
        before = runtime.session.observation.copy()
        call(runtime, "tick")
        np.testing.assert_array_equal(observed[-1], before)
        np.testing.assert_array_equal(runtime.session.trajectory[-1]["action"], [0, 0, 0, 0, 0, 0.25])
    assert len(observed) == 3
    assert not np.array_equal(observed[0], observed[-1])


def test_varied_generated_examples_train_a_successful_policy_at_the_canonical_pose(runtime):
    from kaist_rl_lab.apps.coffee_classroom import POLICY_START_SEED
    from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
    from kaist_rl_lab.apps.coffee_expert import (
        EXAMPLE_COUNT,
        EXAMPLE_FILL_TOLERANCE,
        load_examples,
    )

    demonstrations = [read_demonstration(data) for data in load_examples()]
    model = train_behavior_cloning(demonstrations)
    assert model["metrics"]["demonstrations"] == EXAMPLE_COUNT
    assert model["metrics"]["heldout_action_mae"] is not None
    call(runtime, "cloning-load", model=model)
    call(runtime, "cloning-start", seed=POLICY_START_SEED)
    for _ in range(60 * 32):
        final = call(runtime, "tick")
        if final["cloning_agent"]["done"]:
            break
    assert final["cloning_agent"]["outcome"] == "success"
    assert runtime.session.info["is_success"]
    assert abs(runtime.session.info["fill"] - 0.7) < EXAMPLE_FILL_TOLERANCE
    assert runtime.session.info["spill"] < 0.01
    assert final["snapshot"]["playback"]["paused"]
    assert call(runtime, "tick") == final
    assert call(runtime, "cloning-pause", paused=False) == final


def test_fitted_models_record_geometry_and_legacy_models_default_to_original_spacing():
    pair = demonstration(np.zeros((2, 6)))
    model = train_behavior_cloning([pair])
    assert model["schema_version"] == 2
    assert model["arm_base_distance_m"] == ARM_BASE_DISTANCE_M
    assert NearestNeighborPolicy(model).arm_base_distance == ARM_BASE_DISTANCE_M
    legacy = copy.deepcopy(pair)
    legacy[1].pop("arm_base_distance_m")
    legacy_model = train_behavior_cloning([legacy])
    assert legacy_model["schema_version"] == 1
    assert legacy_model["arm_base_distance_m"] == 1.16
    legacy_model.pop("arm_base_distance_m")
    assert NearestNeighborPolicy(legacy_model).arm_base_distance == 1.16


def test_new_model_schema_requires_spacing_but_legacy_explicit_spacing_is_supported():
    model = train_behavior_cloning([demonstration(np.zeros((2, 6)))])
    missing_spacing = copy.deepcopy(model)
    missing_spacing.pop("arm_base_distance_m")
    with pytest.raises(ValueError, match="schema 2 requires arm_base_distance_m"):
        NearestNeighborPolicy(missing_spacing)
    model["schema_version"] = 1
    assert NearestNeighborPolicy(model).arm_base_distance == ARM_BASE_DISTANCE_M


@pytest.mark.parametrize("version", [None, True, 0, 3, "2"])
def test_model_schema_rejects_invalid_or_unknown_versions(version):
    model = train_behavior_cloning([demonstration(np.zeros((2, 6)))])
    model["schema_version"] = version
    with pytest.raises(ValueError, match="Unsupported behavior-cloning model"):
        NearestNeighborPolicy(model)


def test_behavior_cloning_rejects_mixed_arm_spacing_including_legacy_archives():
    original = demonstration(np.zeros((2, 6)))
    original[1].pop("arm_base_distance_m")
    wider = demonstration(np.ones((2, 6)))
    wider[1]["arm_base_distance_m"] = 1.28
    with pytest.raises(ValueError, match="same arm spacing"):
        train_behavior_cloning([original, wider])
    original[1]["arm_base_distance_m"] = 1.28
    model = train_behavior_cloning([original, wider])
    assert model["arm_base_distance_m"] == 1.28


@pytest.mark.parametrize("distance", [None, True, "1.28", float("nan"), float("inf"), .7, 1.5])
def test_model_and_training_reject_invalid_arm_spacing(distance):
    pair = demonstration(np.zeros((2, 6)))
    model = train_behavior_cloning([pair])
    pair[1]["arm_base_distance_m"] = distance
    model["arm_base_distance_m"] = distance
    with pytest.raises(ValueError, match="arm spacing"):
        train_behavior_cloning([pair])
    with pytest.raises(ValueError, match="arm spacing"):
        NearestNeighborPolicy(model)


def test_loading_a_model_from_another_arm_spacing_keeps_the_current_model(runtime):
    pair = demonstration(np.zeros((2, 6)))
    call(runtime, "cloning-load", model=train_behavior_cloning([pair]))
    call(runtime, "cloning-start")
    before = call(runtime, "snapshot")
    pair[1]["arm_base_distance_m"] = 1.16
    other = train_behavior_cloning([pair])
    with pytest.raises(ValueError, match="arm spacing"):
        call(runtime, "cloning-load", model=other)
    assert call(runtime, "snapshot") == before


def test_model_missing_required_spacing_keeps_the_running_policy(runtime):
    model = train_behavior_cloning([demonstration(np.zeros((2, 6)))])
    call(runtime, "cloning-load", model=model)
    call(runtime, "cloning-start")
    call(runtime, "tick")
    before = call(runtime, "snapshot")
    model.pop("arm_base_distance_m")
    with pytest.raises(ValueError, match="schema 2 requires arm_base_distance_m"):
        call(runtime, "cloning-load", model=model)
    assert call(runtime, "snapshot") == before
