"""Reproducible, clearly labelled examples for the classroom cloning demo.

The geometry-based expert is used only to record these examples offline. It
starts in the student's normal empty-cup layout and controls the same six
velocity inputs. It never assigns the live simulation's joints, liquid, or
observations. The learned policy sees only the resulting observation/action
pairs, not this controller or its knowledge of the environment.

Regenerate the packaged archives with::

    python -m kaist_rl_lab.apps.coffee_expert
"""

from copy import copy
from importlib.resources import files
from itertools import pairwise
from pathlib import Path

import numpy as np

from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT
from kaist_rl_lab.apps.coffee_classroom import classroom_layout
from kaist_rl_lab.apps.coffee_demonstrations import encode_demonstration, read_demonstration
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession
from kaist_rl_lab.envs import CoffeePouringEnv

EXAMPLE_COUNT = 15
# Different approach and pouring speeds give separate demonstrations of the
# same safe maneuver. These are generated examples, never student submissions.
EXAMPLE_SPEEDS = ((1.0, 0.070), (0.92, 0.066), (0.84, 0.074), (0.96, 0.068), (0.88, 0.072))
# These seeds cover the two-dimensional start distribution in an approximate
# 5-by-3 grid. They remain separate from held-out rollout test seeds.
EXAMPLE_SEEDS = (7938, 8095, 7605, 7415, 8026, 7207, 7268, 7880, 7410, 7249, 7838, 7184, 7878, 7700, 7860)
MAX_EXAMPLE_STEPS = round(45 / BROWSER_DT)


def _reference_action(
    env: CoffeePouringEnv,
    pot_angle: float,
    motor_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold the cup upright and the spout over its mouth while tilting."""
    geometry = env.geometry
    cup_center = np.array([-0.05, 0.27])
    cup_wrist = cup_center + np.asarray(geometry.cup_grip)
    cup_q1, cup_q2 = env._inverse_kinematics(
        np.asarray(geometry.cup_base),
        cup_wrist,
        geometry.cup_upper,
        geometry.cup_fore,
        elbow_sign=-1.0,
    )
    # The stream follows a ballistic arc. Aim to the right of the mouth, not
    # directly above it; this is the same geometry as the environment solvability test.
    spout = cup_center + np.asarray(geometry.cup_mouth) + np.array([0.10, 0.20])
    pot_center = spout - env._rotation(pot_angle) @ np.asarray(geometry.pot_spout)
    pot_wrist = pot_center + env._rotation(pot_angle) @ np.asarray(geometry.pot_grip)
    pot_q1, pot_q2 = env._inverse_kinematics(
        np.asarray(geometry.pot_base),
        pot_wrist,
        geometry.pot_upper,
        geometry.pot_fore,
        elbow_sign=1.0,
    )
    target = np.array(
        [
            cup_q1,
            cup_q2,
            -cup_q1 - cup_q2,
            pot_q1,
            pot_q2,
            pot_angle - pot_q1 - pot_q2,
        ]
    )
    error = np.arctan2(np.sin(target - env.joint_angles), np.cos(target - env.joint_angles))
    action = np.clip(error / (env.dt * env.max_joint_speeds), -motor_limit, motor_limit)
    return action.astype(np.float32), error


def _return_volume(env: CoffeePouringEnv, angle: float, tilt_rate: float) -> float:
    """Estimate residual pouring during the slow return, on a separate model.

    This calculation never changes the real environment. The expert needs to
    start returning before the cup reaches 700 mL because the stream continues
    while the pot is moving. It uses the original flow model without overrides.
    """
    if angle <= 0:
        return 0.0
    forecast = copy(env)
    forecast.joint_angles = env.joint_angles.copy()
    grid = np.linspace(angle, 0.0, 257)
    remaining = env.source_remaining
    released = 0.0
    for start, end in pairwise(grid):
        forecast.joint_angles[5] = (start + end) / 2 - sum(forecast.joint_angles[3:5])
        rate, _, _ = forecast._flow_state(remaining)
        amount = min(rate * abs(end - start) / tilt_rate, remaining)
        released += amount
        remaining -= amount
    return released


def generate_example(index: int = 0) -> bytes:
    """Record one successful example through normal ``InteractiveSession.advance``.

    The only reset options are the existing browser layout and 700 mL target;
    success and all observations come from the unmodified environment itself.
    """
    if type(index) is not int or not 0 <= index < EXAMPLE_COUNT:
        raise ValueError(f"Example index must be between 0 and {EXAMPLE_COUNT - 1}.")
    motor_limit, tilt_rate = EXAMPLE_SPEEDS[index % len(EXAMPLE_SPEEDS)]
    seed = EXAMPLE_SEEDS[index]
    session = InteractiveSession(
        seed,
        700,
        dt=BROWSER_DT,
        steps_per_update=1,
        reset_options=classroom_layout(seed),
    )
    phase = "approach"
    angle = 0.0
    try:
        for _ in range(MAX_EXAMPLE_STEPS):
            action, error = _reference_action(session.env, angle, motor_limit)
            if phase == "approach" and np.max(np.abs(error)) < 0.002:
                phase = "pour"
            if phase == "pour":
                residual = _return_volume(session.env, angle, tilt_rate)
                if session.env.target_fill - session.env.fill <= residual + 0.002 or angle >= 1.05:
                    phase = "return"
                else:
                    angle = min(1.05, angle + tilt_rate * BROWSER_DT)
                action, _ = _reference_action(session.env, angle, motor_limit)
            elif phase == "return":
                angle = max(0.0, angle - tilt_rate * BROWSER_DT)
                action, _ = _reference_action(session.env, angle, motor_limit)
            for joint, command in enumerate(action):
                session.set_motor(joint, float(command))
            session.advance()
            if not session.running:
                break
        if session.running or not session.info["is_success"]:
            raise RuntimeError(f"Example {index + 1} did not achieve the environment's goal.")
        if abs(session.env.fill - 0.700) > 0.005 or session.env.spill > 0.001:
            raise RuntimeError(f"Example {index + 1} missed the stricter demonstration target.")
        return encode_demonstration(session, f"Generated example {index + 1}")
    finally:
        session.close()


def generate_examples() -> list[bytes]:
    """Regenerate fifteen examples offline; this is not a web request handler."""
    return [generate_example(index) for index in range(EXAMPLE_COUNT)]


def load_examples() -> list[bytes]:
    """Load the packaged recordings without running an expert on the web server."""
    directory = files("kaist_rl_lab.apps").joinpath("coffee_examples")
    return [
        directory.joinpath(f"example_{index + 1}.npz").read_bytes()
        for index in range(EXAMPLE_COUNT)
    ]


def main() -> None:
    directory = Path(__file__).with_name("coffee_examples")
    directory.mkdir(exist_ok=True)
    for index in range(EXAMPLE_COUNT):
        archive = generate_example(index)
        path = directory / f"example_{index + 1}.npz"
        path.write_bytes(archive)
        arrays, metadata = read_demonstration(archive)
        print(
            f"{path.name}: {len(arrays['actions']) * BROWSER_DT:.2f}s, "
            f"{metadata['fill_l'] * 1000:.2f} mL, "
            f"{metadata['spill_l'] * 1000:.4f} mL spill, "
            f"success={metadata['success']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
