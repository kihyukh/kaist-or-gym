"""The RL objective prioritizes accurate completion, then rewards speed."""

from itertools import pairwise

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_finetuning_reward import (
    DISCOUNT_PER_SECOND,
    fine_tuning_reward,
)


def score_path(errors, *, success, dt=1 / 32, spill=0.0, control=0.0, tilt=0.0):
    result = 0.0
    for index, (before, after) in enumerate(pairwise(errors)):
        terminal = index == len(errors) - 2
        info = {
            "fill_error": after, "spill": spill, "is_success": success and terminal,
            "termination_reason": ("success" if success else "time_limit") if terminal else None,
            "reward_terms": {"fill_progress": 20 * (before - after), "spill": 0,
                             "control": -control * dt, "cup_tilt": -tilt * dt},
        }
        result += DISCOUNT_PER_SECOND**(index * dt) * fine_tuning_reward(info, dt)
    return result


def test_potential_shaping_does_not_change_ranking_or_reward_waiting_cycles():
    steps = 32 * 30
    linear = np.linspace(.7, .02, steps + 1)
    late = np.concatenate((np.full(steps - 100, .7), np.linspace(.7, .02, 101)))
    oscillating = linear.copy()
    oscillating[1:-1] += .05 * np.sin(np.arange(steps - 1))
    expected = score_path(linear, success=True)
    assert score_path(late, success=True) == pytest.approx(expected, abs=1e-10)
    assert score_path(oscillating, success=True) == pytest.approx(expected, abs=1e-10)


def test_faster_equally_accurate_success_scores_higher():
    scores = [score_path(np.linspace(.7, .02, 32 * seconds + 1), success=True)
              for seconds in (20, 30, 40, 60)]
    assert all(earlier > later for earlier, later in pairwise(scores))
    assert scores[0] - scores[1] > 10


def test_even_slow_costly_success_beats_early_or_late_failure():
    # Conservative maximum costs within successful-task limits: six motors at
    # full control, cup tilt at its wrapped-angle upper bound, 20mL spill.
    slow_success = score_path(np.linspace(.7, .04, 60 * 32 + 1), success=True,
                              spill=.02, control=.024 * 6, tilt=.032 * np.pi)
    for seconds in (1 / 32, 1, 20, 60):
        failure = score_path(np.linspace(.7, 0, round(seconds * 32) + 1), success=False)
        assert slow_success > failure


def test_final_accuracy_and_spill_are_penalized():
    accurate = score_path(np.linspace(.7, 0, 30 * 32 + 1), success=True)
    underfilled = score_path(np.linspace(.7, .04, 30 * 32 + 1), success=True)
    spilled = score_path(np.linspace(.7, 0, 30 * 32 + 1), success=True, spill=.02)
    assert accurate > underfilled
    assert accurate > spilled


def test_precision_peak_is_strictly_centered_on_exact_target():
    errors_ml = (0, 1, 2, 5, 10, 20, 40)
    scores = [score_path(np.linspace(.7, error / 1000, 30 * 32 + 1), success=True)
              for error in errors_ml]
    assert all(closer > farther for closer, farther in pairwise(scores))
    # The 5 mL band earns a substantial bonus even at its edge.
    assert scores[3] - scores[4] > 300
    assert scores[0] - scores[3] > 250


def test_slow_precision_band_beats_a_fast_pour_that_misses_by_ten_ml():
    slow_precise = score_path(np.linspace(.7, .005, 60 * 32 + 1), success=True,
                              spill=.02, control=.024 * 6, tilt=.032 * np.pi)
    # Even a physically implausible one-step completion cannot make 690 mL
    # preferable to a safely completed, costly 695 mL pour at the time limit.
    fastest_miss = score_path([.7, .01], success=True)
    assert slow_precise > fastest_miss


def test_timeout_at_exact_target_does_not_collect_precision_bonus():
    exact_timeout = score_path(np.linspace(.7, 0, 60 * 32 + 1), success=False)
    imprecise_success = score_path(np.linspace(.7, .04, 60 * 32 + 1), success=True)
    assert imprecise_success > exact_timeout


def test_precision_reward_still_prefers_faster_equally_accurate_completion():
    scores = [score_path(np.linspace(.7, .001, seconds * 32 + 1), success=True)
              for seconds in (20, 30, 40, 60)]
    assert all(earlier > later for earlier, later in pairwise(scores))
