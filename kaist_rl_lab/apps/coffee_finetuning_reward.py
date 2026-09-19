"""An accuracy-first RL objective, separate from archived environment rewards.

Safe completion is required to collect a narrow bonus centered on 700 mL. The
potential term supplies dense feedback without changing the discounted ranking
of completed trajectories: its sum is the same 14 points from the empty cup.
"""

from math import exp

DISCOUNT_PER_SECOND = 0.99
TIME_COST_PER_SECOND = 1.0
SUCCESS_BONUS = 100.0
PRECISION_BONUS = 1000.0
PRECISION_WIDTH_LITRES = 0.005
FAILURE_PENALTY = 100.0
FINAL_ERROR_COST_PER_LITRE = 100.0
FILL_POTENTIAL_SCALE = 20.0


def fine_tuning_reward(info: dict, dt: float) -> float:
    """Score one actual physics transition, including terminal/timeout steps."""
    terms = info["reward_terms"]
    error = info["fill_error"]
    previous_error = error + terms["fill_progress"] / FILL_POTENTIAL_SCALE
    terminal = info["termination_reason"] is not None
    previous_potential = -FILL_POTENTIAL_SCALE * previous_error
    next_potential = 0.0 if terminal else -FILL_POTENTIAL_SCALE * error
    shaping = DISCOUNT_PER_SECOND**dt * next_potential - previous_potential
    reward = (
        shaping - TIME_COST_PER_SECOND * dt
        + terms["spill"] + terms["control"] + terms["cup_tilt"]
    )
    if terminal:
        # A smooth peak avoids a flat reward plateau inside the 5 mL band. Only
        # an upright, settled, safely completed pour can earn this bonus;
        # merely passing through 700 mL or timing out there cannot collect it.
        completion_bonus = (
            SUCCESS_BONUS
            + PRECISION_BONUS * exp(-0.5 * (error / PRECISION_WIDTH_LITRES) ** 2)
            if info["is_success"] else -FAILURE_PENALTY
        )
        reward += (
            completion_bonus
            - FINAL_ERROR_COST_PER_LITRE * error - 14.0 * info["spill"]
        )
    return float(reward)
