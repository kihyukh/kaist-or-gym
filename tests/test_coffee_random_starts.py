"""Demonstrations vary their starts; BC and RL trials share a fixed comparison pose."""

import base64
import json

import numpy as np
import pytest

from kaist_rl_lab.apps import coffee_browser_runtime as browser_module
from kaist_rl_lab.apps import coffee_cloning_runtime as cloning_module
from kaist_rl_lab.apps import coffee_finetuning as trainer_module
from kaist_rl_lab.apps import coffee_finetuning_runtime as finetuning_module
from kaist_rl_lab.apps import coffee_random_runtime as random_module
from kaist_rl_lab.apps.coffee_classroom import (
    ARM_BASE_DISTANCE_M,
    POLICY_START_SEED,
    classroom_layout,
    fixed_policy_layout,
)
from kaist_rl_lab.apps.coffee_cloning import FEATURE_INDICES, FEATURE_SCALES
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration


def call(runtime, kind, **kwargs):
    return json.loads(runtime.dispatch(json.dumps({"kind": kind, **kwargs})))


@pytest.fixture
def model():
    return {
        "schema_version": 1, "algorithm": "nearest_neighbor",
        "arm_base_distance_m": ARM_BASE_DISTANCE_M,
        "feature_indices": list(FEATURE_INDICES), "feature_scales": list(FEATURE_SCALES),
        "states": [[0.0] * 15], "actions": [[0.0] * 6],
    }


def assert_classroom_pose(session, *, fixed=False):
    assert session.env.arm_base_distance == ARM_BASE_DISTANCE_M
    layout = fixed_policy_layout() if fixed else classroom_layout(session.seed)
    tools = session.env.tool_positions()
    for vessel in ("cup", "pot"):
        np.testing.assert_allclose(tools[vessel + "_center"], layout[vessel + "_center"], atol=1e-12)
    assert session.env.fill == 0
    assert session.env.spill == 0
    assert session.env.target_fill == pytest.approx(.7)
    assert session.env.source_remaining == session.env.INITIAL_POT_VOLUME


def test_student_resets_use_fresh_poses_and_record_the_actual_start(monkeypatch):
    seeds = iter([11, 22, 33])
    monkeypatch.setattr(browser_module, "fresh_classroom_seed", lambda: next(seeds))
    runtime = browser_module.BrowserRuntime()
    try:
        starts = []
        for sequence, expected_seed in enumerate([11, 22, 33]):
            if sequence:
                call(runtime, "reset", sequence=sequence, generation=runtime.session.generation,
                     motors=[0] * 6, paused=False)
            assert runtime.session.seed == expected_seed
            assert_classroom_pose(runtime.session)
            starts.append(runtime.session.initial_joint_angles.copy())
        assert not np.array_equal(starts[0], starts[1])
        assert not np.array_equal(starts[1], starts[2])
        initial_observation = runtime.session.observation.copy()
        call(runtime, "tick")
        archive = call(runtime, "save", participant="test")
        arrays, metadata = read_demonstration(base64.b64decode(archive["archive"]))
        assert metadata["seed"] == 33
        np.testing.assert_array_equal(metadata["initial_joint_angles_rad"], starts[-1])
        np.testing.assert_array_equal(arrays["observations"][0], initial_observation)
    finally:
        runtime.session.close()


def test_explicit_student_seed_reproduces_the_initial_pose():
    first = browser_module.BrowserRuntime(seed=123)
    second = browser_module.BrowserRuntime(seed=123)
    try:
        np.testing.assert_array_equal(first.session.observation, second.session.observation)
        assert_classroom_pose(first.session)
    finally:
        first.session.close()
        second.session.close()


def test_random_agent_varies_by_default_and_repeats_with_an_explicit_seed(monkeypatch):
    seeds = iter(range(100, 110))
    monkeypatch.setattr(random_module, "fresh_classroom_seed", lambda: next(seeds))
    runtime = random_module.RandomAgentRuntime()
    try:
        sampled = []
        for _ in range(2):
            call(runtime, "random-start")
            assert_classroom_pose(runtime.session)
            sampled.append((runtime.session.seed, runtime.session.observation.copy()))
        assert sampled[0][0] != sampled[1][0]
        assert not np.array_equal(sampled[0][1], sampled[1][1])
        call(runtime, "random-start", seed=345)
        initial = runtime.session.observation.copy()
        motors = runtime.session.motors.copy()
        call(runtime, "tick")
        next_observation = runtime.session.observation.copy()
        call(runtime, "random-start", seed=345)
        np.testing.assert_array_equal(runtime.session.observation, initial)
        np.testing.assert_array_equal(runtime.session.motors, motors)
        call(runtime, "tick")
        np.testing.assert_array_equal(runtime.session.observation, next_observation)
        call(runtime, "random-reset")
        assert runtime.session.seed != 345
        assert runtime.session.paused
        assert_classroom_pose(runtime.session)
    finally:
        runtime.session.close()


def test_behavior_cloning_always_restarts_from_the_fixed_policy_pose(model):
    runtime = cloning_module.CloningAgentRuntime()
    try:
        call(runtime, "cloning-load", model=model)
        initial = runtime.session.observation.copy()
        for seed in (None, 12, 999, None):
            call(runtime, "cloning-start", **({"seed": seed} if seed is not None else {}))
            assert_classroom_pose(runtime.session, fixed=True)
            np.testing.assert_array_equal(runtime.session.observation, initial)
            call(runtime, "tick")
        call(runtime, "cloning-reset")
        assert runtime.session.seed == POLICY_START_SEED
        assert runtime.session.paused
        np.testing.assert_array_equal(runtime.session.observation, initial)
    finally:
        runtime.session.close()


@pytest.mark.parametrize("invalid", [True, -1, 2**32, 1.5, "12"])
def test_invalid_seed_does_not_replace_a_working_policy_trial(model, invalid):
    for runtime_class, prefix in [
        (cloning_module.CloningAgentRuntime, "cloning"),
        (random_module.RandomAgentRuntime, "random"),
    ]:
        runtime = runtime_class()
        try:
            if prefix == "cloning":
                call(runtime, "cloning-load", model=model)
            call(runtime, prefix + "-start", seed=22)
            prior = call(runtime, "snapshot")
            with pytest.raises(ValueError, match="[Ss]eed"):
                call(runtime, prefix + "-start", seed=invalid)
            assert call(runtime, "snapshot") == prior
        finally:
            runtime.session.close()
    with pytest.raises(ValueError, match="[Ss]eed"):
        browser_module.BrowserRuntime(seed=invalid)
    with pytest.raises(ValueError, match="[Ss]eed"):
        trainer_module.FineTuningTrainer(model, seed=invalid)


@pytest.mark.parametrize("strategy", trainer_module.STRATEGIES)
def test_exploration_and_all_checkpoints_share_the_fixed_policy_pose(monkeypatch, model, strategy):
    monkeypatch.setattr(trainer_module, "TRIAL_STEPS", 2)
    trainers = [trainer_module.FineTuningTrainer(model, seed=45, episodes=3, strategy=strategy) for _ in range(2)]
    results = []
    try:
        for trainer in trainers:
            poses = []
            while not trainer.done:
                assert trainer.session.env.elapsed_steps == 0
                assert_classroom_pose(trainer.session, fixed=True)
                poses.append((trainer.phase, trainer.session.seed, trainer.session.initial_joint_angles.copy()))
                trainer.step_chunk(max_steps=2)
                # Publish the terminal scene before advancing to the next pose.
                assert trainer.session.env.elapsed_steps == 2
                assert not trainer.session.running
                trainer.step_chunk(max_steps=1)
            evaluation = [item for item in poses if item[0] != "training"]
            exploration = [item for item in poses if item[0] == "training"]
            assert len(evaluation) == 4
            assert len(exploration) == 3
            assert {item[1] for item in evaluation} == {trainer.evaluation_seed}
            for _, _, joints in poses[1:]:
                np.testing.assert_array_equal(joints, evaluation[0][2])
            assert {item[1] for item in poses} == {POLICY_START_SEED}
            result = trainer.result()
            assert result["baseline"]["initial_seed"] == trainer.evaluation_seed
            assert result["best"]["initial_seed"] == trainer.evaluation_seed
            assert [row["training"]["initial_seed"] for row in result["history"]] == [item[1] for item in exploration]
            assert {row["evaluation"]["initial_seed"] for row in result["history"]} == {trainer.evaluation_seed}
            results.append(result)
        assert results[0] == results[1]
    finally:
        for trainer in trainers:
            trainer.close()


def test_finetuning_watch_and_reset_preserve_pose_while_experiments_refresh_noise_seed(monkeypatch, model):
    monkeypatch.setattr(trainer_module, "TRIAL_STEPS", 2)
    seeds = iter(range(200, 220))
    monkeypatch.setattr(finetuning_module, "fresh_classroom_seed", lambda: next(seeds))
    runtime = finetuning_module.FineTuningRuntime()
    try:
        call(runtime, "ft-load", model=model)
        call(runtime, "ft-train", episodes=1)
        first_seed = runtime.result["seed"]
        first_evaluation_seed = runtime.result["evaluation_seed"]
        # Each rollout publishes its terminal frame before a separate phase change.
        for _ in range(6):
            call(runtime, "ft-step", max_steps=6)
        assert not runtime.training_active
        assert runtime.best_policy is not None
        call(runtime, "ft-run", policy="base")
        assert runtime.session.seed == first_evaluation_seed
        original_pose = runtime.session.observation.copy()
        call(runtime, "tick")
        call(runtime, "ft-run", policy="best")
        assert runtime.session.seed == first_evaluation_seed
        np.testing.assert_array_equal(runtime.session.observation, original_pose)
        call(runtime, "ft-reset")
        assert runtime.session.seed == first_evaluation_seed
        assert_classroom_pose(runtime.session, fixed=True)
        call(runtime, "ft-run", policy="base")
        np.testing.assert_array_equal(runtime.session.observation, original_pose)
        call(runtime, "ft-train", episodes=1)
        assert runtime.result["seed"] != first_seed
        assert runtime.result["evaluation_seed"] == first_evaluation_seed
        call(runtime, "ft-stop")
        call(runtime, "ft-train", episodes=1, seed=first_seed)
        assert runtime.result["evaluation_seed"] == first_evaluation_seed
        np.testing.assert_array_equal(runtime.session.observation, original_pose)
    finally:
        runtime.close()


@pytest.mark.parametrize('runtime_class,command', [
    (cloning_module.CloningAgentRuntime, 'cloning-load'),
    (finetuning_module.FineTuningRuntime, 'ft-load'),
])
def test_policy_from_older_arm_spacing_is_rejected_without_replacing_scene(model, runtime_class, command):
    runtime = runtime_class()
    try:
        call(runtime, command, model=model)
        session = runtime.session
        observation = session.observation.copy()
        legacy = {key: value for key, value in model.items() if key != 'arm_base_distance_m'}
        with pytest.raises(ValueError, match='different arm spacing'):
            call(runtime, command, model=legacy)
        assert runtime.session is session
        np.testing.assert_array_equal(runtime.session.observation, observation)
        assert runtime.session.env.arm_base_distance == ARM_BASE_DISTANCE_M
    finally:
        runtime.session.close()


def test_trainer_rejects_model_for_another_arm_spacing(model):
    legacy = {key: value for key, value in model.items() if key != 'arm_base_distance_m'}
    with pytest.raises(ValueError, match='different arm spacing'):
        trainer_module.FineTuningTrainer(legacy, episodes=1)
