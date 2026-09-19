"""Varied demonstration starts and a stable policy-comparison starting pose."""

import secrets

import numpy as np

# Vessels start upright, empty cup and full pot, well apart from each other.
# Student demonstrations vary each vessel's horizontal and vertical position
# independently. Policy comparisons use the unchanged canonical pose instead.
INITIAL_LAYOUT = {"cup_center": [-0.28, 0.28], "pot_center": [0.26, 0.62]}
# Classroom arms are farther apart than the original 1.16 m environment.
ARM_BASE_DISTANCE_M = 1.28
POLICY_START_SEED = 7001
CUP_POSITION_JITTER = (0.09, 0.05)
POT_POSITION_JITTER = (0.11, 0.07)


def fresh_classroom_seed() -> int:
    """Choose a new seed while retaining exact reproducibility in recordings."""
    return secrets.randbits(32)


def classroom_layout(seed: int) -> dict[str, list[float]]:
    """Return a fresh layout sampled from the same bounded classroom task."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Classroom seed must be an integer between 0 and 2**32 - 1.")
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-1.0, 1.0, 4)
    return {
        "cup_center": [INITIAL_LAYOUT["cup_center"][axis] + float(offsets[axis]) * CUP_POSITION_JITTER[axis]
                       for axis in range(2)],
        "pot_center": [INITIAL_LAYOUT["pot_center"][axis] + float(offsets[axis + 2]) * POT_POSITION_JITTER[axis]
                       for axis in range(2)],
    }


def fixed_policy_layout() -> dict[str, list[float]]:
    """Give each policy comparison the same pose without sharing mutable lists."""
    return {name: center.copy() for name, center in INITIAL_LAYOUT.items()}
