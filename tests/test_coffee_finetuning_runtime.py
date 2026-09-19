"""Fine-tuning worker commands cooperate with pause, stop, and BC replacement."""

import json
from copy import deepcopy
from typing import ClassVar

import numpy as np
import pytest

from kaist_rl_lab.apps import coffee_finetuning_runtime as module
from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import classroom_layout
from kaist_rl_lab.apps.coffee_cloning import FEATURE_INDICES, FEATURE_SCALES, NearestNeighborPolicy
from kaist_rl_lab.apps.coffee_finetuning_runtime import FineTuningRuntime


def call(runtime, kind, **kwargs):
    return json.loads(runtime.dispatch(json.dumps({"kind": kind, **kwargs})))


@pytest.fixture
def model():
    return {
        "schema_version": 1,
        "algorithm": "nearest_neighbor",
        "feature_indices": list(FEATURE_INDICES),
        "feature_scales": list(FEATURE_SCALES),
        "states": [[0.0] * 15],
        "actions": [[0.0] * 6],
    }


@pytest.fixture
def runtime():
    value = FineTuningRuntime()
    yield value
    value.close()


class FakeTrainer:
    """Real scene, short staged training to exercise command ordering cheaply."""

    instances: ClassVar[list] = []

    def __init__(self, model, *, seed, episodes):
        self.model = deepcopy(model)
        self.seed = seed
        self.episodes = episodes
        self.session = module._new_session()
        self.session.paused = False
        self.best_policy = NearestNeighborPolicy(model)
        self.calls = []
        self.done = False
        self.closed = False
        self.instances.append(self)

    def step_chunk(self, max_steps):
        self.calls.append(max_steps)
        for _ in range(max_steps):
            self.session.advance()
        self.done = len(self.calls) >= 4
        if self.done:
            improved = deepcopy(self.model)
            improved["actions"][0][0] = 0.2
            self.best_policy = NearestNeighborPolicy(improved)

    def result(self):
        phase = ["baseline", "baseline", "training", "evaluation", "complete"][len(self.calls)]
        baseline = None if len(self.calls) < 2 else {
            "return": -10.0, "fill_ml": 0.0, "spill_ml": 0.0,
            "seconds": 60.0, "success": False, "outcome": "time_limit",
        }
        best = None if baseline is None else {
            **baseline, "return": -5.0 if self.done else -10.0,
            "episode": 1 if self.done else 0,
        }
        return {
            "phase": phase, "done": self.done, "episode": int(baseline is not None),
            "episodes": self.episodes, "total_steps": sum(self.calls),
            "completed_episodes": int(self.done),
            "episode_steps": self.session.env.elapsed_steps,
            "evaluation_seed": self.session.seed,
            "baseline": baseline, "best": best,
            "history": [] if len(self.calls) < 3 else [{
                "episode": 1, "training": baseline,
                "evaluation": best if self.done else None,
                "update": {"mean_kl": 0.001, "mean_change_bound": 0.01, "actor_change": 0.005},
            }],
        }

    def close(self):
        self.closed = True
        self.session.close()


@pytest.fixture
def fake_trainer(monkeypatch):
    FakeTrainer.instances = []
    monkeypatch.setattr(module, "FineTuningTrainer", FakeTrainer)
    return FakeTrainer


def test_initial_state_and_load_are_idle(runtime, model, fake_trainer):
    initial = call(runtime, "snapshot")
    state = initial["finetuning"]
    assert not state["model_loaded"]
    assert not state["training_running"]
    assert not state["best_available"]
    assert not state["has_result"]
    assert state["paused"]
    for kind in ("tick", "ft-step", "ft-stop"):
        assert call(runtime, kind) == initial
    with pytest.raises(ValueError, match="behavior-cloning"):
        call(runtime, "ft-train")
    with pytest.raises(ValueError, match="behavior-cloning"):
        call(runtime, "ft-run", policy="base")

    loaded = call(runtime, "ft-load", model=model)
    assert loaded["finetuning"]["model_loaded"]
    assert loaded["snapshot"]["playback"]["paused"]
    assert not fake_trainer.instances
    assert runtime.session.trajectory == []
    assert runtime.session.env.dt == BROWSER_DT
    for name, position in classroom_layout(runtime.session.seed).items():
        np.testing.assert_allclose(runtime.session.env.tool_positions()[name], position)
    with pytest.raises(ValueError, match="evaluation"):
        call(runtime, "ft-run", policy="best")
    with pytest.raises(ValueError):
        call(runtime, "ft-load", model={})
    assert call(runtime, "snapshot") == loaded


def test_training_runs_only_in_bounded_chunks_and_pause_freezes_it(runtime, model, fake_trainer):
    call(runtime, "ft-load", model=model)
    training = call(runtime, "ft-train", episodes=4, seed=12)
    trainer = fake_trainer.instances[-1]
    assert (trainer.episodes, trainer.seed) == (4, 12)
    assert training["finetuning"]["training_running"]
    assert trainer.calls == []
    assert call(runtime, "tick") == training
    first = call(runtime, "ft-step", max_steps=1)
    assert first["finetuning"]["progress"]["total_steps"] == 1
    assert not first["finetuning"]["best_available"]
    paused = call(runtime, "ft-pause", paused=True)
    assert paused["finetuning"]["training_active"]
    assert paused["finetuning"]["training_paused"]
    assert not paused["finetuning"]["training_running"]
    for _ in range(3):
        assert call(runtime, "ft-step") == paused
        assert call(runtime, "tick") == paused
    call(runtime, "ft-pause", paused=False)
    second = call(runtime, "ft-step")
    assert trainer.calls == [1, 16]
    assert second["finetuning"]["has_result"]
    assert second["finetuning"]["best_available"]
    assert second["finetuning"]["progress"]["elapsed_seconds"] == 17 * BROWSER_DT
    assert second["finetuning"]["progress"]["reward"] == runtime.session.cumulative_reward
    assert second["finetuning"]["progress"]["completed_episodes"] == 0
    with pytest.raises(ValueError, match="Stop"):
        call(runtime, "ft-train")
    with pytest.raises(ValueError, match="Stop"):
        call(runtime, "ft-run", policy="base")


@pytest.mark.parametrize("value", [True, False, 0, -1, 13, 1.5, "4", None])
def test_invalid_episode_bounds_do_not_replace_the_loaded_model(runtime, model, value):
    before = call(runtime, "ft-load", model=model)
    with pytest.raises(ValueError, match="episodes"):
        call(runtime, "ft-train", episodes=value)
    assert call(runtime, "snapshot") == before


@pytest.mark.parametrize("value", [True, False, 0, -1, 33, 1.5, "4", None])
def test_invalid_chunk_size_is_rejected_without_advancement(runtime, model, fake_trainer, value):
    call(runtime, "ft-load", model=model)
    before = call(runtime, "ft-train")
    with pytest.raises(ValueError, match="chunk"):
        call(runtime, "ft-step", max_steps=value)
    assert call(runtime, "snapshot") == before


@pytest.mark.parametrize("value", [True, -1, 2**32, 1.5, "2026", None])
def test_invalid_seed_is_rejected(runtime, model, value):
    before = call(runtime, "ft-load", model=model)
    with pytest.raises(ValueError, match="Seed"):
        call(runtime, "ft-train", seed=value)
    assert call(runtime, "snapshot") == before


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_stop_retains_only_fully_evaluated_checkpoint(runtime, model, fake_trainer, steps):
    call(runtime, "ft-load", model=model)
    call(runtime, "ft-train")
    for _ in range(steps):
        call(runtime, "ft-step", max_steps=1)
    stopped = call(runtime, "ft-stop")
    state = stopped["finetuning"]
    assert state["training_stopped"]
    assert not state["training_active"]
    assert not state["training_running"]
    assert state["paused"]
    assert state["best_available"] == (steps >= 2)
    assert state["has_result"] == (steps >= 2)
    for kind in ("ft-stop", "ft-step", "tick"):
        assert call(runtime, kind) == stopped
    if steps < 2:
        with pytest.raises(ValueError, match="evaluation"):
            call(runtime, "ft-run", policy="best")
    else:
        assert state["result"]["best_episode"] == 0
        assert not state["result"]["improved"]
        call(runtime, "ft-run", policy="best")
        call(runtime, "tick")
        np.testing.assert_array_equal(runtime.session.trajectory[-1]["action"], np.zeros(6))


def test_completed_checkpoint_survives_reset_and_comparisons(runtime, model, fake_trainer):
    call(runtime, "ft-load", model=model)
    call(runtime, "ft-train")
    trainer = fake_trainer.instances[-1]
    for _ in range(4):
        completed = call(runtime, "ft-step", max_steps=1)
    assert completed["finetuning"]["best_available"]
    assert completed["finetuning"]["result"]["improved"]
    assert completed["finetuning"]["result"]["best_episode"] == 1
    assert completed["finetuning"]["progress"]["completed_episodes"] == 1
    assert not completed["finetuning"]["training_running"]
    assert call(runtime, "ft-step") == completed
    best = call(runtime, "ft-run", policy="best")
    assert trainer.closed
    assert best["finetuning"]["rollout"]["active"]
    call(runtime, "tick")
    np.testing.assert_allclose(runtime.session.trajectory[-1]["action"], [0.2, 0, 0, 0, 0, 0])
    paused = call(runtime, "ft-pause", paused=True)
    assert call(runtime, "tick") == paused
    call(runtime, "ft-pause", paused=False)
    assert call(runtime, "tick")["finetuning"]["rollout"]["elapsed_seconds"] == 2 * BROWSER_DT
    reset = call(runtime, "ft-reset")
    assert reset["finetuning"]["result"] == completed["finetuning"]["result"]
    assert reset["finetuning"]["best_available"]
    assert not reset["finetuning"]["rollout"]["active"]
    assert reset["finetuning"]["paused"]
    assert reset["finetuning"]["rollout"]["elapsed_seconds"] == 0
    base = call(runtime, "ft-run", policy="base")
    assert base["snapshot"]["state"] == best["snapshot"]["state"]
    call(runtime, "tick")
    np.testing.assert_array_equal(runtime.session.trajectory[-1]["action"], np.zeros(6))


def test_replacing_model_closes_old_training_and_clears_checkpoint(runtime, model, fake_trainer):
    call(runtime, "ft-load", model=model)
    call(runtime, "ft-train")
    old = fake_trainer.instances[-1]
    for _ in range(3):
        call(runtime, "ft-step")
    replacement = deepcopy(model)
    replacement["actions"][0][1] = 0.1
    loaded = call(runtime, "ft-load", model=replacement)
    assert old.closed
    assert loaded["finetuning"]["progress"] is None
    assert loaded["finetuning"]["result"] is None
    assert not loaded["finetuning"]["best_available"]
    assert call(runtime, "ft-step") == loaded
    call(runtime, "ft-run", policy="base")
    call(runtime, "tick")
    np.testing.assert_allclose(runtime.session.trajectory[-1]["action"], [0, 0.1, 0, 0, 0, 0])


def test_reset_stops_active_training_and_retains_evaluated_baseline(runtime, model, fake_trainer):
    call(runtime, "ft-load", model=model)
    call(runtime, "ft-train")
    trainer = fake_trainer.instances[-1]
    for _ in range(2):
        call(runtime, "ft-step")
    reset = call(runtime, "ft-reset")
    assert trainer.closed
    assert reset["finetuning"]["best_available"]
    assert not reset["finetuning"]["training_active"]
    assert call(runtime, "ft-step") == reset


def test_invalid_commands(runtime):
    with pytest.raises(TypeError):
        runtime.dispatch("[]")
    with pytest.raises(ValueError, match="Unknown"):
        call(runtime, "save")
    with pytest.raises(ValueError, match="Pause"):
        call(runtime, "ft-pause", paused=1)
    with pytest.raises(ValueError, match="base or best"):
        call(runtime, "ft-run", policy="expert")


def test_actual_training_and_best_rollout_report_the_same_physics(runtime, model):
    """One real actor-critic run, including its held-out deterministic comparison."""
    call(runtime, "ft-load", model=model)
    call(runtime, "ft-train", episodes=1)
    for _ in range(200):
        state = call(runtime, "ft-step", max_steps=32)["finetuning"]
        if not state["training_active"]:
            break
    assert not state["training_active"]
    assert state["has_result"] and state["best_available"]
    assert len(state["result"]["history"]) == 1
    assert state["result"]["history"][0]["evaluation"] is not None
    assert state["result"]["best"]["return"] >= state["result"]["baseline"]["return"]
    call(runtime, "ft-run", policy="best")
    for _ in range(module.TRIAL_STEPS):
        final = call(runtime, "tick")
        if final["finetuning"]["rollout"]["done"]:
            break
    rollout = final["finetuning"]["rollout"]
    assert rollout["done"]
    assert rollout["outcome"] == state["result"]["best"]["outcome"]
    assert rollout["reward"] == pytest.approx(state["result"]["best"]["return"])
    assert rollout["elapsed_seconds"] == state["result"]["best"]["seconds"]
    assert call(runtime, "tick") == final
    assert call(runtime, "ft-pause", paused=False) == final
