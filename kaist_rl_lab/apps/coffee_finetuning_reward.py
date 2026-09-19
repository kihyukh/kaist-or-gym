"""A time-focused RL objective, separate from archived environment rewards.

Successful completion is required to collect the large terminal bonus. The
potential term supplies dense feedback without changing the discounted ranking
of completed trajectories: its sum is the same 14 points from the empty cup.
"""

DISCOUNT_PER_SECOND = 0.99
TIME_COST_PER_SECOND = 1.0
SUCCESS_BONUS = 100.0
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
        reward += (
            (SUCCESS_BONUS if info["is_success"] else -FAILURE_PENALTY)
            - FINAL_ERROR_COST_PER_LITRE * error - 14.0 * info["spill"]
        )
    return float(reward)
