"""Small, inspectable behavior cloning using demonstrated state/action pairs.

The policy copies the action at the closest demonstrated physical state. It does
not use rewards, a trajectory's future states, a prerecorded action timeline, or
an expert controller at inference time. This local method can imitate a familiar
route, but does not establish generalization beyond the demonstrated range.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from kaist_rl_lab.envs.coffee_pouring import CoffeePouringEnv

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
MAX_MODEL_SAMPLES = 50_000
MAX_DEMONSTRATIONS = 100
MAX_DEMONSTRATION_STEPS = 30_000
MAX_VALIDATION_STEPS = 500
FEATURE_INDICES = tuple(range(15))
# Joint positions are already normalized. Orientations, relative vessel
# positions (metres), and liquid amounts (litres) use fixed physical scales.
# Fixed scales avoid amplifying tiny numerical variation in a nearly constant
# feature. Time is excluded: student recordings have no time-limit fraction.
FEATURE_SCALES = (1.0,) * 6 + (0.5,) * 4 + (0.5, 0.5, 1.0, 1.0, 1.0)
ALGORITHM = "nearest_neighbor"


def _arm_base_distance(metadata: dict[str, Any]) -> float:
    try:
        return CoffeePouringEnv.validate_arm_base_distance(
            metadata.get("arm_base_distance_m", CoffeePouringEnv.DEFAULT_ARM_BASE_DISTANCE),
        )
    except (TypeError, ValueError):
        raise ValueError("Invalid behavior-cloning arm spacing.") from None


def _matrix(value: Any, width: int, maximum: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != width
        or not 1 <= len(array) <= maximum
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
        or np.any(np.abs(array) > 10_000)
    ):
        raise ValueError(f"Invalid {label} matrix.")
    return array.astype(np.float32, copy=False)


def _features(observations: np.ndarray) -> np.ndarray:
    return observations[..., FEATURE_INDICES] / np.asarray(FEATURE_SCALES, dtype=np.float32)


def _fit_pairs(
    demonstrations: list[tuple[np.ndarray, np.ndarray]], max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Assign an equal budget per trajectory, then redistribute unused slots.
    counts = np.asarray([len(observations) for observations, _ in demonstrations])
    budgets = np.minimum(counts, max_samples // len(counts))
    remaining = int(max_samples - np.sum(budgets))
    while remaining and np.any(budgets < counts):
        available = np.flatnonzero(budgets < counts)
        increment = max(1, remaining // len(available))
        for index in available:
            add = min(increment, remaining, int(counts[index] - budgets[index]))
            budgets[index] += add
            remaining -= add
            if not remaining:
                break
    selected_states, selected_actions = [], []
    for (observations, actions), budget in zip(demonstrations, budgets):
        indices = np.linspace(0, len(observations) - 1, int(budget), dtype=np.int64)
        selected_states.append(_features(observations[indices]))
        selected_actions.append(actions[indices])
    states = np.concatenate(selected_states)
    actions = np.concatenate(selected_actions)
    # At a perfectly unchanged state, waiting and later moving are contradictory
    # labels for a memoryless policy. Keep the last demonstrated label; otherwise
    # an initial wait can trap the learner forever. This uses state/action pairs
    # only, and never consults the next observation or outcome.
    _, reversed_indices = np.unique(states[::-1], axis=0, return_index=True)
    indices = np.sort(len(states) - 1 - reversed_indices)
    return states[indices], actions[indices]


def _nearest_action(states: np.ndarray, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    differences = states - state
    distances = np.einsum("ij,ij->i", differences, differences)
    return actions[int(np.argmin(distances))].copy()


def train_behavior_cloning(
    demonstrations: Sequence[tuple[dict[str, np.ndarray], dict[str, Any]]],
    *, max_samples: int = MAX_MODEL_SAMPLES,
) -> dict[str, Any]:
    """Fit 1-nearest-neighbor BC and evaluate action error on a held-out trajectory.

    Inputs should first pass the archive reader's integrity checks. This second
    validation enforces this demo's physical observation/action and timing needs.
    The final model fits all supplied demonstrations. Reported validation, when
    available, excludes the entire final trajectory from a separate fitted model.
    """
    if not 1 <= len(demonstrations) <= MAX_DEMONSTRATIONS:
        raise ValueError(f"Choose between 1 and {MAX_DEMONSTRATIONS} demonstrations.")
    if type(max_samples) is not int or not len(demonstrations) <= max_samples <= MAX_MODEL_SAMPLES:
        raise ValueError(f"Training sample limit must be between the demo count and {MAX_MODEL_SAMPLES}.")
    prepared = []
    arm_base_distance = None
    for arrays, metadata in demonstrations:
        distance = _arm_base_distance(metadata)
        if arm_base_distance is not None and distance != arm_base_distance:
            raise ValueError("Behavior cloning requires trajectories with the same arm spacing.")
        arm_base_distance = distance
        if metadata.get("dt") != 1 / 32:
            raise ValueError("Behavior cloning requires browser recordings made at 32 steps/second.")
        target = metadata.get("target_fill_l")
        if (
            not isinstance(target, (int, float))
            or not np.isfinite(target) or abs(target - 0.7) > 1e-6
        ):
            raise ValueError("Behavior cloning requires recordings with the 700 mL target.")
        observations = _matrix(
            arrays["observations"], 16, MAX_DEMONSTRATION_STEPS, "observations",
        )
        actions = _matrix(arrays["actions"], 6, MAX_DEMONSTRATION_STEPS, "actions")
        if len(observations) != len(actions) or np.any(np.abs(actions) > 1):
            raise ValueError("Demonstration actions must match observations and stay within [-1, 1].")
        if not np.allclose(observations[:, 14], 0.7, rtol=0, atol=1e-6):
            raise ValueError("Recorded observation targets must be 700 mL.")
        prepared.append((observations, actions))

    metrics: dict[str, Any] = {
        "demonstrations": len(prepared),
        "total_steps": sum(len(observations) for observations, _ in prepared),
        "validation_trajectories": 0,
        "validation_steps": 0,
        "validation_total_steps": 0,
        "heldout_action_mae": None,
    }
    if len(prepared) >= 2:
        states, actions = _fit_pairs(prepared[:-1], max_samples)
        observed, expected = prepared[-1]
        indices = np.linspace(
            0, len(observed) - 1, min(MAX_VALIDATION_STEPS, len(observed)), dtype=np.int64,
        )
        absolute_error = 0.0
        for index in indices:
            predicted = _nearest_action(states, actions, _features(observed[index]))
            absolute_error += float(np.abs(predicted - expected[index]).mean())
        metrics.update({
            "validation_trajectories": 1,
            "validation_steps": len(indices),
            "validation_total_steps": len(observed),
            "heldout_action_mae": absolute_error / len(indices),
        })
    states, actions = _fit_pairs(prepared, max_samples)
    metrics["training_samples"] = len(states)
    return {
        # A worker from before configurable spacing must reject a wider model,
        # including tabs left open while the server deploys a new classroom.
        "schema_version": (
            LEGACY_SCHEMA_VERSION
            if arm_base_distance == CoffeePouringEnv.DEFAULT_ARM_BASE_DISTANCE else SCHEMA_VERSION
        ),
        "algorithm": ALGORITHM,
        "arm_base_distance_m": arm_base_distance,
        "feature_indices": list(FEATURE_INDICES),
        "feature_scales": list(FEATURE_SCALES),
        "states": states.tolist(),
        "actions": actions.tolist(),
        "metrics": metrics,
    }


class NearestNeighborPolicy:
    """A bounded, NumPy-only inference model that also runs inside Pyodide."""

    def __init__(self, model: dict[str, Any]):
        if not isinstance(model, dict) or (
            type(model.get("schema_version")) is not int
            or model["schema_version"] not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)
            or model.get("algorithm") != ALGORITHM
            or model.get("feature_indices") != list(FEATURE_INDICES)
            or model.get("feature_scales") != list(FEATURE_SCALES)
        ):
            raise ValueError("Unsupported behavior-cloning model.")
        if model["schema_version"] == SCHEMA_VERSION and "arm_base_distance_m" not in model:
            raise ValueError("Behavior-cloning model schema 2 requires arm_base_distance_m.")
        self.arm_base_distance = _arm_base_distance(model)
        self.states = _matrix(model.get("states"), len(FEATURE_INDICES), MAX_MODEL_SAMPLES, "states")
        self.actions = _matrix(model.get("actions"), 6, MAX_MODEL_SAMPLES, "actions")
        if len(self.states) != len(self.actions) or np.any(np.abs(self.actions) > 1):
            raise ValueError("Model actions must match states and stay within [-1, 1].")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        if (
            observation.shape != (16,)
            or observation.dtype.kind not in "fiu"
            or not np.isfinite(observation).all()
            or np.any(np.abs(observation) > 10_000)
        ):
            raise ValueError("Invalid behavior-cloning observation.")
        return _nearest_action(self.states, self.actions, _features(observation.astype(np.float32)))
