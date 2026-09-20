"""Additive classroom points shared by the environment and policy learners.

From an empty cup, the undiscounted sum is target_ml - abs(fill_ml-target_ml),
plus a settled precision bonus, minus elapsed-time, spill, and final-state costs.
Physics and the environment's broader success tolerance are unchanged.
"""

from math import atan2, cos, degrees, sin

REWARD_MODEL = "additive_v1"
DISCOUNT_PER_SECOND = 1.0
TIME_COST_PER_SECOND = 10.0
PRECISION_BONUS = 100.0
PRECISION_TOLERANCE_LITRES = 0.005
BONUS_MAX_FLOW_LITRES_PER_SECOND = 0.001
SPILL_COST_PER_ML = 1.0
POT_TILT_COST_PER_DEGREE = 2.0
FLOW_COST_PER_ML_PER_SECOND = 5.0


def fill_score(fill_l: float, target_l: float) -> float:
    """One point per mL toward the target; lose one per mL beyond it."""
    return 1000.0 * (target_l - abs(fill_l - target_l))


def finish_reward_terms(info: dict) -> dict[str, float]:
    """Apply once when physics ends or the student manually saves an attempt.

    Flow is the actual final physical flow, not the renderer's stopped animation.
    The success flag additionally requires pot/cup tilt <=12/8 degrees and <=20mL
    spill. The bonus has the stricter <=1mL/s flow limit and <=5mL target error.
    """
    pot_angle = atan2(sin(info["pot_angle"]), cos(info["pot_angle"]))
    precise_and_settled = (
        info["is_success"]
        and info["fill_error"] <= PRECISION_TOLERANCE_LITRES + 1e-12
        and info["flow_rate"] <= BONUS_MAX_FLOW_LITRES_PER_SECOND + 1e-12
    )
    return {
        "precision_bonus": PRECISION_BONUS if precise_and_settled else 0.0,
        "pot_level": -POT_TILT_COST_PER_DEGREE * abs(degrees(pot_angle)),
        "flow_at_finish": -FLOW_COST_PER_ML_PER_SECOND * 1000.0 * info["flow_rate"],
    }


def transition_reward_terms(previous_error: float, previous_spill: float,
                            info: dict, dt: float) -> dict[str, float]:
    """Dense fill/spill increments telescope; terminal costs are never repeated."""
    terms = {
        "fill_progress": 1000.0 * (previous_error - info["fill_error"]),
        "spill": -SPILL_COST_PER_ML * 1000.0 * (info["spill"] - previous_spill),
        "time": -TIME_COST_PER_SECOND * dt,
        "precision_bonus": 0.0,
        "pot_level": 0.0,
        "flow_at_finish": 0.0,
    }
    if info["termination_reason"] is not None:
        terms.update(finish_reward_terms(info))
    return terms
