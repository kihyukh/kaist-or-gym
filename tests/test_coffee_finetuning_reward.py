"""The additive coffee score agrees with measurements and complete trajectories."""

from itertools import pairwise
from math import radians

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_finetuning_reward import (
    DISCOUNT_PER_SECOND,
    fine_tuning_reward,
)
from kaist_rl_lab.envs import CoffeePouringEnv
from kaist_rl_lab.envs.coffee_reward import (
    PRECISION_BONUS,
    REWARD_MODEL,
    TIME_COST_PER_SECOND,
    fill_score,
    finish_reward_terms,
    transition_reward_terms,
)


def info_at(fill, *, target=.7, spill=0.0, success=False, terminal=None, tilt=0.0, flow=0.0):
    return {
        "fill": fill, "target_fill": target, "fill_error": abs(fill - target),
        "spill": spill, "is_success": success, "termination_reason": terminal,
        "pot_angle": radians(tilt), "flow_rate": flow,
    }


def score_path(fills, *, seconds, target=.7, success=True, spill=0.0, tilt=0.0, flow=0.0):
    """Evaluate measured state changes; finish terms apply on the last step only."""
    dt = seconds / (len(fills) - 1)
    total = 0.0
    details = []
    for index, (before, after) in enumerate(pairwise(fills)):
        final = index == len(fills) - 2
        current_spill = spill * (index + 1) / (len(fills) - 1)
        previous_spill = spill * index / (len(fills) - 1)
        info = info_at(after, target=target, spill=current_spill, success=success and final,
                       terminal=("success" if success else "time_limit") if final else None,
                       tilt=tilt, flow=flow)
        terms = transition_reward_terms(abs(before - target), previous_spill, info, dt)
        info["reward_terms"] = terms
        total += DISCOUNT_PER_SECOND ** (index * dt) * fine_tuning_reward(info, dt)
        details.append(terms)
    return total, details


def test_objective_constants_and_exact_700ml_30_second_example():
    assert REWARD_MODEL == "additive_v1"
    assert TIME_COST_PER_SECOND == 10
    assert PRECISION_BONUS == 100
    assert DISCOUNT_PER_SECOND == 1
    score, terms = score_path(np.linspace(0, .7, 30 * 32 + 1), seconds=30)
    assert fill_score(.7, .7) == pytest.approx(700)
    assert sum(row["fill_progress"] for row in terms) == pytest.approx(700)
    assert sum(row["time"] for row in terms) == pytest.approx(-300)
    assert sum(row["precision_bonus"] for row in terms) == pytest.approx(100)
    assert score == pytest.approx(500, abs=1e-9)


@pytest.mark.parametrize("fill,target,expected", [
    (0, .7, 0), (.5, .7, 500), (.69, .7, 690), (.7, .7, 700),
    (.71, .7, 690), (.9, .7, 500), (1.5, .7, -100),
    (.45, .5, 450), (.5, .5, 500), (.55, .5, 450),
])
def test_fill_score_is_target_minus_absolute_error_in_millilitres(fill, target, expected):
    assert fill_score(fill, target) == pytest.approx(expected)


@pytest.mark.parametrize("error,bonus", [
    (0, 100), (.001, 100), (.004999, 100), (.005, 100),
    (.005 + 5e-13, 100), (.005 + 2e-12, 0), (.005001, 0), (.04, 0),
])
@pytest.mark.parametrize("direction", [-1, 1])
def test_precision_band_is_inclusive_symmetric_and_numerically_tolerant(error, bonus, direction):
    info = info_at(.7 + direction * error, success=True, terminal="success")
    assert finish_reward_terms(info)["precision_bonus"] == bonus


def test_score_remains_finely_graded_inside_the_precision_band():
    fills = (.695, .698, .699, .6999, .7)
    scores = [score_path([0, fill], seconds=30)[0] for fill in fills]
    assert scores == pytest.approx([495, 498, 499, 499.9, 500])
    assert all(closer > farther for farther, closer in pairwise(scores))
    for under in fills:
        assert score_path([0, 1.4 - under], seconds=30)[0] == pytest.approx(
            score_path([0, under], seconds=30)[0], abs=1e-9,
        )


def test_bonus_discontinuity_at_band_edge_does_not_hide_fill_progress():
    edge, _ = score_path([0, .695], seconds=30)
    outside, _ = score_path([0, .694999], seconds=30)
    assert edge == pytest.approx(495)
    assert outside == pytest.approx(394.999)
    assert edge - outside == pytest.approx(100.001)


def test_ten_points_per_second_is_independent_of_accuracy_and_step_size():
    for fill in (.69, .699, .7, .71):
        earlier, _ = score_path(np.linspace(0, fill, 641), seconds=20)
        later, _ = score_path(np.linspace(0, fill, 961), seconds=30)
        assert earlier - later == pytest.approx(100, abs=1e-9)
    for dt in (1 / 64, 1 / 32, .125, 1):
        terms = transition_reward_terms(.7, 0, info_at(0), dt)
        assert terms["time"] == pytest.approx(-10 * dt)
        assert sum(terms.values()) == pytest.approx(-10 * dt)


@pytest.mark.parametrize("tilt", [-12, -3.5, 0, 3.5, 12])
def test_finish_penalizes_absolute_pot_tilt_in_degrees(tilt):
    terms = finish_reward_terms(info_at(.7, success=True, terminal="success", tilt=tilt))
    assert terms["pot_level"] == pytest.approx(-2 * abs(tilt))
    assert terms["precision_bonus"] == 100


@pytest.mark.parametrize("flow,bonus", [(0, 100), (.0004, 100), (.001, 100),
                                       (.001 + 5e-13, 100), (.001 + 2e-12, 0),
                                       (.005, 0), (.008, 0)])
def test_finish_flow_cost_and_settled_bonus_threshold(flow, bonus):
    terms = finish_reward_terms(info_at(.7, success=True, terminal="success", flow=flow))
    assert terms["flow_at_finish"] == pytest.approx(-5 * flow * 1000)
    assert terms["precision_bonus"] == bonus


@pytest.mark.parametrize("reason", ["time_limit", "spill_or_overflow", "manual_finish"])
def test_unsuccessful_finish_never_collects_precision_bonus(reason):
    info = info_at(.7, success=False, terminal=reason, tilt=-3, flow=.0004)
    terms = transition_reward_terms(.002, .004, {**info, "spill": .005}, 1 / 32)
    assert terms["precision_bonus"] == 0
    assert terms["pot_level"] == pytest.approx(-6)
    assert terms["flow_at_finish"] == pytest.approx(-2)
    assert terms["fill_progress"] == pytest.approx(2)
    assert terms["spill"] == pytest.approx(-1)
    assert sum(terms.values()) == pytest.approx(2 - 1 - 10 / 32 - 6 - 2)


def test_transient_precision_and_tilt_or_flow_do_not_award_or_charge_finish_terms():
    # Even a state flagged successful must actually finish before receiving the bonus.
    for success in (False, True):
        info = info_at(.7, success=success, tilt=11, flow=.0005)
        terms = transition_reward_terms(.004, .0, info, 1 / 32)
        assert terms["precision_bonus"] == 0
        assert terms["pot_level"] == 0
        assert terms["flow_at_finish"] == 0
        assert sum(terms.values()) == pytest.approx(4 - 10 / 32)


def test_spill_penalty_counts_each_millilitre_once():
    clean, _ = score_path(np.linspace(0, .7, 961), seconds=30)
    spilled, details = score_path(np.linspace(0, .7, 961), seconds=30, spill=.017)
    assert sum(row["spill"] for row in details) == pytest.approx(-17)
    assert clean - spilled == pytest.approx(17, abs=1e-9)
    no_new_spill = transition_reward_terms(.7, .017, info_at(0, spill=.017), 1 / 32)
    assert no_new_spill["spill"] == 0


@pytest.mark.parametrize("hz", [8, 32, 64])
def test_same_timed_path_has_same_score_across_step_sizes(hz):
    score, _ = score_path(np.linspace(0, .699, 30 * hz + 1), seconds=30,
                          spill=.004, tilt=-2.5, flow=.0004)
    # 699 fill + 100 precision - 300 time - 4 spill - 5 tilt - 2 flow.
    assert score == pytest.approx(488, abs=1e-9)


def test_fill_progress_telescopes_without_reward_for_waiting_or_overshoot_cycles():
    smooth = np.linspace(0, .698, 961)
    late = np.concatenate((np.zeros(860), np.linspace(0, .698, 101)))
    # A closed loop around the target contributes no net fill score. These are
    # mathematical state paths testing the shaping identity, not generated demos.
    cyclic = smooth.copy()
    cyclic[100:105] = [.690, .700, .710, .700, .690]
    expected = 698 + 100 - 300
    for fills in (smooth, late, cyclic):
        score, details = score_path(fills, seconds=30)
        assert sum(row["fill_progress"] for row in details) == pytest.approx(698, abs=1e-9)
        assert score == pytest.approx(expected, abs=1e-9)
        assert sum(row["precision_bonus"] > 0 for row in details) == 1
    first = transition_reward_terms(.010, 0, info_at(.7), 1 / 32)
    second = transition_reward_terms(0, 0, info_at(.710), 1 / 32)
    assert first["fill_progress"] + second["fill_progress"] == pytest.approx(0, abs=1e-9)


def test_fine_tuning_consumes_the_exact_environment_terms_without_extra_bonus_or_discount():
    info = info_at(.7, success=True, terminal="success", tilt=3, flow=.0004)
    info["reward_terms"] = transition_reward_terms(.01, .004, {**info, "spill": .006}, .125)
    original = dict(info["reward_terms"])
    expected = 10 - 2 - 1.25 + 100 - 6 - 2
    assert fine_tuning_reward(info, .125) == pytest.approx(expected)
    # dt has already contributed to the emitted transition terms; RL must not
    # add another time penalty or transform terminal reward based on dt.
    assert fine_tuning_reward(info, 1 / 32) == pytest.approx(expected)
    assert info["reward_terms"] == original


def test_terminal_environment_reward_cannot_be_repeated_without_reset():
    env = CoffeePouringEnv(dt=1 / 32, horizon=32, include_render_info=False)
    try:
        env.reset(seed=5, options={"target_fill": .7})
        # Initialize a settled successful state to exercise termination directly,
        # without depending on a trained policy or a particular demonstration.
        env.fill = .7
        _, reward, terminated, truncated, info = env.step(np.zeros(6))
        assert terminated and not truncated
        assert info["termination_reason"] == "success"
        assert info["reward_terms"]["precision_bonus"] == 100
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
        assert fine_tuning_reward(info, env.dt) == pytest.approx(reward)
        with pytest.raises(RuntimeError, match="episode ended"):
            env.step(np.zeros(6))
        env.reset(seed=5, options={"target_fill": .7})
        _, reward, terminated, truncated, info = env.step(np.zeros(6))
        assert not terminated and not truncated
        assert info["reward_terms"]["precision_bonus"] == 0
        assert reward == pytest.approx(-10 / 32)
    finally:
        env.close()
