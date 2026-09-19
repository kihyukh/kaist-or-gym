"""Random controls share student physics but never adapt to feedback."""

import base64
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_browser import browser_bundle
from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M, classroom_layout
from kaist_rl_lab.apps.coffee_random_runtime import TRIAL_STEPS, RandomAgentRuntime
from kaist_rl_lab.envs import CoffeePouringEnv


def call(runtime, kind, **kwargs):
    return json.loads(runtime.dispatch(json.dumps({"kind": kind, **kwargs})))


@pytest.fixture
def runtime():
    value = RandomAgentRuntime()
    yield value
    value.session.close()


class FixedDurationRng:
    """Force either endpoint of the supported holding-time distribution."""

    def __init__(self, duration, *, moving=True):
        self.duration = duration
        self.moving = moving
        self.choices = 0

    def integers(self, low, high, *, size=None):
        if size is None:
            assert (low, high) == (1, 33)
            return self.duration
        assert (low, high, size) == (-1, 2, 6)
        self.choices += 1
        direction = (1 if self.choices % 2 else -1) if self.moving else 0
        return np.array([direction, 0, 0, 0, 0, 0])


def test_initial_state_cannot_run_until_started(runtime):
    initial = call(runtime, "snapshot")
    assert initial["random_agent"] == {
        "attempt": 0, "decisions": 0, "hold_seconds": 0,
        "remaining_seconds": 0, "elapsed_seconds": 0,
        "limit_seconds": 30, "max_duration_seconds": 1,
        "done": False, "outcome": None,
    }
    assert initial["snapshot"]["playback"]["paused"]
    call(runtime, "random-pause", paused=False)
    assert call(runtime, "tick") == initial
    assert runtime.session.env.dt == BROWSER_DT
    for name, center in classroom_layout(runtime.session.seed).items():
        np.testing.assert_allclose(runtime.session.env.tool_positions()[name], center)


@pytest.mark.parametrize("duration", [1, 32])
def test_controls_are_held_for_exact_sampled_step_count(runtime, duration):
    runtime.rng = FixedDurationRng(duration)
    started = call(runtime, "random-start")
    assert started["random_agent"]["hold_seconds"] == duration * BROWSER_DT
    assert started["random_agent"]["remaining_seconds"] == duration * BROWSER_DT
    for index in range(duration):
        tick = call(runtime, "tick")
        assert tick["random_agent"]["decisions"] == 1
        assert tick["random_agent"]["remaining_seconds"] == (duration - index - 1) * BROWSER_DT
        np.testing.assert_array_equal(runtime.session.trajectory[-1]["action"], [1, 0, 0, 0, 0, 0])
    tick = call(runtime, "tick")
    assert tick["random_agent"]["decisions"] == 2
    np.testing.assert_array_equal(runtime.session.trajectory[-1]["action"], [-1, 0, 0, 0, 0, 0])


def test_pause_freezes_physics_and_current_decision(runtime):
    call(runtime, "random-start", seed=42)
    call(runtime, "tick")
    paused = call(runtime, "random-pause", paused=True)
    for _ in range(5):
        assert call(runtime, "tick") == paused
    resumed = call(runtime, "random-pause", paused=False)
    assert resumed["random_agent"] == paused["random_agent"]
    assert not resumed["snapshot"]["playback"]["paused"]
    next_tick = call(runtime, "tick")
    assert next_tick["random_agent"]["elapsed_seconds"] == 2 * BROWSER_DT


def test_repeated_start_resets_physics_and_reset_clears_experiment(runtime):
    first = call(runtime, "random-start", seed=51)
    for _ in range(40):
        call(runtime, "tick")
    second = call(runtime, "random-start", seed=51)
    assert second["episode_id"] != first["episode_id"]
    assert second["random_agent"]["attempt"] == 2
    assert second["random_agent"]["decisions"] == 1
    assert second["random_agent"]["elapsed_seconds"] == 0
    assert second["snapshot"]["state"] == first["snapshot"]["state"]
    assert second["snapshot"]["playback"]["motors"] == first["snapshot"]["playback"]["motors"]
    assert runtime.session.trajectory == []
    assert not runtime.session.paused
    reset = call(runtime, "random-reset")
    assert reset["random_agent"]["attempt"] == 0
    assert reset["random_agent"]["decisions"] == 0
    assert reset["snapshot"]["playback"]["paused"]
    assert reset["snapshot"]["playback"]["motors"] == [0] * 6
    assert call(runtime, "tick") == reset


def test_trial_ends_after_30_simulated_seconds(runtime):
    runtime.rng = FixedDurationRng(32, moving=False)
    call(runtime, "random-start")
    for _ in range(TRIAL_STEPS):
        final = call(runtime, "tick")
    assert final["random_agent"]["done"]
    assert final["random_agent"]["outcome"] == "time_limit"
    assert final["random_agent"]["elapsed_seconds"] == 30
    assert final["random_agent"]["decisions"] == 30
    assert not final["snapshot"]["playback"]["running"]
    assert final["snapshot"]["playback"]["paused"]
    assert final["snapshot"]["playback"]["motors"] == [0] * 6
    assert len(runtime.session.trajectory) == TRIAL_STEPS
    assert runtime.session.trajectory[-1]["truncated"]
    assert call(runtime, "tick") == final
    assert call(runtime, "random-pause", paused=False) == final


def test_random_rollout_matches_the_actual_student_environment(runtime):
    reference = CoffeePouringEnv(arm_base_distance=ARM_BASE_DISTANCE_M, dt=BROWSER_DT, horizon=TRIAL_STEPS)
    try:
        call(runtime, "random-start", seed=191)
        reference.reset(seed=191, options={**classroom_layout(191), "target_fill": 0.7})
        for _ in range(TRIAL_STEPS):
            tick = call(runtime, "tick")
            transition = runtime.session.trajectory[-1]
            observation, reward, terminated, truncated, info = reference.step(transition["action"])
            np.testing.assert_array_equal(runtime.session.observation, observation)
            assert transition["reward"] == reward
            assert transition["terminated"] == terminated
            assert transition["truncated"] == truncated
            assert set(transition["action"]).issubset({-1, 0, 1})
            assert BROWSER_DT <= tick["random_agent"]["hold_seconds"] <= 1
            if terminated or truncated:
                assert tick["random_agent"]["done"]
                assert tick["random_agent"]["outcome"] == info["termination_reason"]
                break
        assert tick["random_agent"]["done"]
        assert call(runtime, "tick") == tick
    finally:
        reference.close()


def test_policy_is_deterministic_and_ignores_observations_and_rewards(runtime, monkeypatch):
    other = RandomAgentRuntime()
    try:
        call(runtime, "random-start", seed=37)
        call(other, "random-start", seed=37)
        actual_step = other.session.env.step

        def misleading_feedback(action):
            observation, _, terminated, truncated, info = actual_step(action)
            return np.full_like(observation, -987), 1e9, terminated, truncated, info

        monkeypatch.setattr(other.session.env, "step", misleading_feedback)
        for _ in range(100):
            first = call(runtime, "tick")
            second = call(other, "tick")
            assert first["random_agent"] == second["random_agent"]
            assert first["snapshot"]["playback"]["motors"] == second["snapshot"]["playback"]["motors"]
            np.testing.assert_array_equal(
                runtime.session.trajectory[-1]["action"], other.session.trajectory[-1]["action"],
            )
            if first["random_agent"]["done"]:
                break
        assert first["random_agent"]["decisions"] > 1
        assert runtime.session.cumulative_reward != other.session.cumulative_reward
        assert not np.array_equal(runtime.session.observation, other.session.observation)
    finally:
        other.session.close()


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True, "42", None, [], {}])
def test_invalid_seed_does_not_interrupt_an_existing_trial(runtime, seed):
    started = call(runtime, "random-start", seed=7)
    with pytest.raises(ValueError, match="Seed"):
        call(runtime, "random-start", seed=seed)
    assert call(runtime, "snapshot") == started


@pytest.mark.parametrize("paused", [None, 0, 1, "false"])
def test_pause_requires_a_boolean(runtime, paused):
    with pytest.raises(ValueError, match="Pause"):
        call(runtime, "random-pause", paused=paused)


@pytest.mark.parametrize("kind", ["save", "motor", "pause", "stop", "reset", "unknown"])
def test_random_experiment_cannot_save_or_accept_student_controls(runtime, kind):
    with pytest.raises(ValueError, match="Unknown"):
        call(runtime, kind)


def test_random_runtime_is_shipped_with_the_existing_browser_bundle():
    with ZipFile(BytesIO(base64.b64decode(browser_bundle()))) as archive:
        assert "kaist_rl_lab/apps/coffee_random_runtime.py" in archive.namelist()
