"""Modest, reproducible starting-pose variation for classroom demonstrations."""

import secrets

import numpy as np

# Vessels start upright, empty cup and full pot, well apart from each other.
# Two independent coordinates keep this task learnable from a small class's
# demonstrations: a common left/right shift and the pot's starting height.
INITIAL_LAYOUT = {"cup_center": [-0.28, 0.28], "pot_center": [0.26, 0.62]}
HORIZONTAL_JITTER = 0.025
POT_HEIGHT_JITTER = 0.015


def fresh_classroom_seed() -> int:
    """Choose a new seed while retaining exact reproducibility in recordings."""
    return secrets.randbits(32)


def classroom_layout(seed: int) -> dict[str, list[float]]:
    """Return a fresh layout sampled from the same bounded classroom task."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Classroom seed must be an integer between 0 and 2**32 - 1.")
    rng = np.random.default_rng(seed)
    horizontal = float(rng.uniform(-HORIZONTAL_JITTER, HORIZONTAL_JITTER))
    pot_height = float(rng.uniform(-POT_HEIGHT_JITTER, POT_HEIGHT_JITTER))
    return {
        "cup_center": [INITIAL_LAYOUT["cup_center"][0] + horizontal, INITIAL_LAYOUT["cup_center"][1]],
        "pot_center": [INITIAL_LAYOUT["pot_center"][0] + horizontal, INITIAL_LAYOUT["pot_center"][1] + pot_height],
    }
