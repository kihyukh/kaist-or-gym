"""Two-arm coffee pouring with rigid planar links.

This is a deliberately lightweight teaching environment.  Arm motion uses
exact planar forward kinematics; liquid flow is a smooth geometric
approximation, not a fluid simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ArmGeometry:
    """Immutable geometry in world coordinates (approximately metres)."""

    cup_base: tuple[float, float] = (-0.58, 0.10)
    pot_base: tuple[float, float] = (0.58, 0.10)
    cup_upper: float = 0.42
    cup_fore: float = 0.36
    pot_upper: float = 0.46
    pot_fore: float = 0.40
    cup_grip: tuple[float, float] = (0.075, 0.0)
    pot_grip: tuple[float, float] = (0.10, 0.0)
    cup_mouth: tuple[float, float] = (0.0, 0.13)
    pot_spout: tuple[float, float] = (-0.15, 0.06)
    cup_width: float = 0.16
    cup_height: float = 0.25
    pot_width: float = 0.23
    pot_height: float = 0.27
    table_y: float = 0.062


class CoffeePouringEnv(gym.Env):
    """Continuous-control coffee pouring with two fixed-link planar arms.

    The action is a six-vector in this order::

        [cup shoulder, cup elbow, cup wrist,
         pot shoulder, pot elbow, pot wrist]

    Each component is a normalized angular velocity in ``[-1, 1]``.  The six
    joint angles are the only robot configuration state.  Elbow, wrist, cup,
    pot, mouth, and spout positions are always derived by forward kinematics,
    so no link can telescope.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 8,
    }
    DEFAULT_HORIZON = 330
    DEFAULT_DT = 0.125
    FULL_SCALE_QUARTER_TURN_SECONDS = 10.0
    FLOW_START_SPOUT_DROP = 0.028
    FLOW_FULL_SPOUT_DROP = 0.118

    JOINT_NAMES = (
        "cup_shoulder",
        "cup_elbow",
        "cup_wrist",
        "pot_shoulder",
        "pot_elbow",
        "pot_wrist",
    )
    OBSERVATION_NAMES = (
        "cup_shoulder",
        "cup_elbow",
        "cup_wrist",
        "pot_shoulder",
        "pot_elbow",
        "pot_wrist",
        "cup_angle_sin",
        "cup_angle_cos",
        "pot_angle_sin",
        "pot_angle_cos",
        "spout_minus_mouth_x",
        "spout_height_gap",
        "fill",
        "spill",
        "target_fill",
        "elapsed_fraction",
    )

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        horizon: int | None = DEFAULT_HORIZON,
        dt: float = DEFAULT_DT,
        width: int = 960,
        height: int = 560,
    ) -> None:
        super().__init__()
        valid_modes = [None] + list(self.metadata["render_modes"])
        if render_mode not in valid_modes:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}; expected None, 'rgb_array', or 'human'."
            )
        if horizon is not None:
            if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)):
                raise TypeError("horizon must be a positive integer or None")
            if horizon <= 0:
                raise ValueError("horizon must be positive or None")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")

        self.render_mode = render_mode
        self.horizon = None if horizon is None else int(horizon)
        self.dt = float(dt)
        self.width = int(width)
        self.height = int(height)
        self.geometry = ArmGeometry()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        quarter_turn_speed = (np.pi / 2.0) / self.FULL_SCALE_QUARTER_TURN_SECONDS
        self.max_joint_speeds = np.full(6, quarter_turn_speed, dtype=np.float64)
        self.joint_low = np.array([-0.55, -2.85, -3.20, 0.35, 0.10, -3.20], dtype=np.float64)
        self.joint_high = np.array([2.65, -0.08, 3.20, 2.85, 2.75, 3.20], dtype=np.float64)
        # Observation: six normalized joints, sine/cosine for both vessel
        # angles, raw cup-to-spout geometry, fill, spill, target, and time.
        # The injective joint normalization keeps the observation Markov.
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 10 + [-4.0, -4.0, 0.0, 0.0, 0.50, -1.0], dtype=np.float32),
            high=np.array([1.0] * 10 + [4.0, 4.0, 1.02, 0.60, 0.90, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.max_flow_rate = 0.164  # litres per simulated second
        self.max_leak_rate = 0.064  # litres per simulated second

        self.joint_angles = np.zeros(6, dtype=np.float64)
        self.fill = 0.0
        self.spill = 0.0
        self.target_fill = 0.70
        self.elapsed_steps = 0
        self.last_flow = 0.0
        self.last_flow_rate = 0.0
        self.last_captured = 0.0
        self.last_stream_end = np.zeros(2, dtype=np.float64)
        self.action_energy = 0.0
        self.last_action = np.zeros(6, dtype=np.float32)
        self._episode_done = False
        self._last_termination_reason = None
        self._pygame = None
        self._window = None
        self._clock = None

    @staticmethod
    def _rotation(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _inverse_kinematics(
        base: np.ndarray,
        wrist: np.ndarray,
        upper: float,
        fore: float,
        elbow_sign: float,
    ) -> tuple[float, float]:
        delta = wrist - base
        raw_distance = float(np.linalg.norm(delta))
        distance = float(np.clip(raw_distance, abs(upper - fore) + 1e-5, upper + fore - 1e-5))
        cosine = np.clip(
            (distance**2 - upper**2 - fore**2) / (2.0 * upper * fore),
            -1.0,
            1.0,
        )
        q2 = float(np.sign(elbow_sign) * np.arccos(cosine))
        q1 = float(
            np.arctan2(delta[1], delta[0])
            - np.arctan2(fore * np.sin(q2), upper + fore * np.cos(q2))
        )
        return q1, q2

    @staticmethod
    def _arm_points(
        base: np.ndarray, q1: float, q2: float, upper: float, fore: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        elbow = base + upper * np.array([np.cos(q1), np.sin(q1)])
        wrist = elbow + fore * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
        return base.copy(), elbow, wrist

    def _initial_joints(self, cup_center: np.ndarray, pot_center: np.ndarray) -> np.ndarray:
        cup_wrist = cup_center + np.asarray(self.geometry.cup_grip)
        pot_wrist = pot_center + np.asarray(self.geometry.pot_grip)
        cup_q1, cup_q2 = self._inverse_kinematics(
            np.asarray(self.geometry.cup_base),
            cup_wrist,
            self.geometry.cup_upper,
            self.geometry.cup_fore,
            elbow_sign=-1.0,
        )
        pot_q1, pot_q2 = self._inverse_kinematics(
            np.asarray(self.geometry.pot_base),
            pot_wrist,
            self.geometry.pot_upper,
            self.geometry.pot_fore,
            elbow_sign=1.0,
        )
        return np.array(
            [
                cup_q1,
                cup_q2,
                -cup_q1 - cup_q2,
                pot_q1,
                pot_q2,
                -pot_q1 - pot_q2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _validate_reachable_wrist(
        name: str,
        base: np.ndarray,
        wrist: np.ndarray,
        upper: float,
        fore: float,
    ) -> None:
        distance = float(np.linalg.norm(wrist - base))
        minimum = abs(upper - fore) + 1e-5
        maximum = upper + fore - 1e-5
        if not minimum <= distance <= maximum:
            raise ValueError(
                f"{name} is unreachable: wrist distance {distance:.3f} must be between "
                f"{minimum:.3f} and {maximum:.3f}"
            )

    @property
    def cup_angle(self) -> float:
        return self._wrap_angle(float(np.sum(self.joint_angles[:3])))

    @property
    def pot_angle(self) -> float:
        return self._wrap_angle(float(np.sum(self.joint_angles[3:])))

    def joint_positions(self) -> dict[str, np.ndarray]:
        """Return ``base, elbow, wrist`` positions for both rigid arms."""

        g = self.geometry
        cup = self._arm_points(
            np.asarray(g.cup_base, dtype=np.float64),
            self.joint_angles[0],
            self.joint_angles[1],
            g.cup_upper,
            g.cup_fore,
        )
        pot = self._arm_points(
            np.asarray(g.pot_base, dtype=np.float64),
            self.joint_angles[3],
            self.joint_angles[4],
            g.pot_upper,
            g.pot_fore,
        )
        return {"cup": np.stack(cup), "pot": np.stack(pot)}

    def tool_positions(self) -> dict[str, Any]:
        """Return vessel and pouring landmarks derived from the joint state."""

        joints = self.joint_positions()
        cup_rotation = self._rotation(self.cup_angle)
        pot_rotation = self._rotation(self.pot_angle)
        cup_center = joints["cup"][2] - cup_rotation @ np.asarray(self.geometry.cup_grip)
        pot_center = joints["pot"][2] - pot_rotation @ np.asarray(self.geometry.pot_grip)
        cup_mouth = cup_center + cup_rotation @ np.asarray(self.geometry.cup_mouth)
        pot_spout = pot_center + pot_rotation @ np.asarray(self.geometry.pot_spout)
        return {
            "cup_center": cup_center,
            "cup_mouth": cup_mouth,
            "pot_center": pot_center,
            "pot_spout": pot_spout,
            "cup_angle": self.cup_angle,
            "pot_angle": self.pot_angle,
        }

    def _pour_intensity(self) -> float:
        """Return a continuous, periodic pour amount in ``[0, 1]``.

        Flow starts when the rotated spout falls below the pot centre.  This
        geometric rule has no discontinuity at the ``-pi``/``pi`` wrap.
        """

        rotated_spout = self._rotation(self.pot_angle) @ np.asarray(self.geometry.pot_spout)
        spout_drop = -float(rotated_spout[1])
        return float(
            np.clip(
                (spout_drop - self.FLOW_START_SPOUT_DROP)
                / (self.FLOW_FULL_SPOUT_DROP - self.FLOW_START_SPOUT_DROP),
                0.0,
                1.0,
            )
        )

    def _is_success(self) -> bool:
        return bool(
            abs(self.fill - self.target_fill) <= 0.040
            and self.spill <= 0.020
            and self.last_flow_rate <= 0.008
            and abs(self.cup_angle) <= np.deg2rad(8.0)
            and abs(self.pot_angle) <= np.deg2rad(12.0)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "target_fill" in options:
            target = float(options["target_fill"])
        else:
            target = float(self.np_random.uniform(0.60, 0.79))
        if not np.isfinite(target) or not 0.50 <= target <= 0.90:
            raise ValueError("target_fill must be between 0.50 and 0.90 litres")

        if "joint_angles" in options:
            if "cup_center" in options or "pot_center" in options:
                raise ValueError("joint_angles cannot be combined with cup_center or pot_center")
            initial_joints = np.asarray(options["joint_angles"], dtype=np.float64)
            if initial_joints.shape != (6,):
                raise ValueError("joint_angles must have shape (6,)")
            if not np.all(np.isfinite(initial_joints)):
                raise ValueError("joint_angles must contain only finite values")
            if np.any(initial_joints < self.joint_low) or np.any(initial_joints > self.joint_high):
                raise ValueError("joint_angles must lie within the mechanical limits")
        else:
            if "cup_center" in options:
                cup_center = np.asarray(options["cup_center"], dtype=np.float64)
            else:
                cup_center = np.array(
                    [
                        -0.20 + self.np_random.uniform(-0.025, 0.025),
                        0.26 + self.np_random.uniform(-0.012, 0.012),
                    ]
                )
            if "pot_center" in options:
                pot_center = np.asarray(options["pot_center"], dtype=np.float64)
            else:
                pot_center = np.array(
                    [
                        0.12 + self.np_random.uniform(-0.030, 0.030),
                        0.75 + self.np_random.uniform(-0.020, 0.020),
                    ]
                )
            if cup_center.shape != (2,) or pot_center.shape != (2,):
                raise ValueError("cup_center and pot_center must be two-dimensional")
            if not np.all(np.isfinite(cup_center)) or not np.all(np.isfinite(pot_center)):
                raise ValueError("cup_center and pot_center must contain only finite values")

            cup_wrist = cup_center + np.asarray(self.geometry.cup_grip)
            pot_wrist = pot_center + np.asarray(self.geometry.pot_grip)
            self._validate_reachable_wrist(
                "cup_center",
                np.asarray(self.geometry.cup_base),
                cup_wrist,
                self.geometry.cup_upper,
                self.geometry.cup_fore,
            )
            self._validate_reachable_wrist(
                "pot_center",
                np.asarray(self.geometry.pot_base),
                pot_wrist,
                self.geometry.pot_upper,
                self.geometry.pot_fore,
            )
            initial_joints = self._initial_joints(cup_center, pot_center)
            if np.any(initial_joints < self.joint_low) or np.any(initial_joints > self.joint_high):
                raise ValueError("requested vessel centres violate the mechanical joint limits")

        fill = float(options.get("fill", 0.0))
        spill = float(options.get("spill", 0.0))
        if not np.isfinite(fill) or not 0.0 <= fill <= 1.02:
            raise ValueError("fill must be between 0.0 and 1.02 litres")
        if not np.isfinite(spill) or not 0.0 <= spill <= 0.60:
            raise ValueError("spill must be between 0.0 and 0.60 litres")

        self.joint_angles = initial_joints.copy()
        self.fill = fill
        self.spill = spill
        self.target_fill = target
        self.elapsed_steps = 0
        self.last_flow = 0.0
        self.last_flow_rate = 0.0
        self.last_captured = 0.0
        self.last_stream_end = np.asarray(self.tool_positions()["pot_spout"]).copy()
        self.action_energy = 0.0
        self.last_action = np.zeros(6, dtype=np.float32)
        self._episode_done = False
        self._last_termination_reason = None

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_human()
        return observation, info

    def _get_obs(self) -> np.ndarray:
        tools = self.tool_positions()
        spout = np.asarray(tools["pot_spout"])
        mouth = np.asarray(tools["cup_mouth"])
        normalized_joints = (
            2.0 * (self.joint_angles - self.joint_low) / (self.joint_high - self.joint_low) - 1.0
        )
        elapsed_fraction = (
            -1.0 if self.horizon is None else min(self.elapsed_steps / self.horizon, 1.0)
        )
        observation = np.concatenate(
            [
                normalized_joints,
                np.array(
                    [
                        np.sin(self.cup_angle),
                        np.cos(self.cup_angle),
                        np.sin(self.pot_angle),
                        np.cos(self.pot_angle),
                        spout[0] - mouth[0],
                        spout[1] - mouth[1],
                        self.fill,
                        self.spill,
                        self.target_fill,
                        elapsed_fraction,
                    ]
                ),
            ]
        )
        return observation.astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        tools = self.tool_positions()
        error = abs(self.fill - self.target_fill)
        success = self._is_success()
        return {
            "is_success": bool(success),
            "fill": float(self.fill),
            "target_fill": float(self.target_fill),
            "fill_error": float(error),
            "spill": float(self.spill),
            "flow": float(self.last_flow),
            "flow_rate": float(self.last_flow_rate),
            "captured": float(self.last_captured),
            "elapsed_steps": int(self.elapsed_steps),
            "elapsed_time": float(self.elapsed_steps * self.dt),
            "time_remaining": (
                None
                if self.horizon is None
                else float(max(0, self.horizon - self.elapsed_steps) * self.dt)
            ),
            "has_time_limit": self.horizon is not None,
            "joint_angles": self.joint_angles.astype(np.float32).copy(),
            "cup_mouth": np.asarray(tools["cup_mouth"], dtype=np.float32),
            "pot_spout": np.asarray(tools["pot_spout"], dtype=np.float32),
            "stream_end": self.last_stream_end.astype(np.float32).copy(),
            "termination_reason": self._last_termination_reason,
        }

    def step(self, action: np.ndarray):
        if self._episode_done:
            raise RuntimeError("step() called after episode ended; call reset() first")

        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (6,):
            raise ValueError(f"action must have shape (6,), received {action_array.shape}")
        action_array = np.nan_to_num(action_array, nan=0.0, posinf=1.0, neginf=-1.0)
        action_array = np.clip(action_array, -1.0, 1.0)

        previous_error = abs(self.fill - self.target_fill)
        previous_spill = self.spill
        # Zero-order hold: the selected motor velocities remain constant over
        # this decision interval.  This integrates the continuous-time joint
        # model exactly; front ends may interpolate the two decision states for
        # a smoother display without adding extra Gymnasium transitions.
        angular_velocity = self.max_joint_speeds * action_array
        self.joint_angles = np.clip(
            self.joint_angles + self.dt * angular_velocity,
            self.joint_low,
            self.joint_high,
        )
        self.last_action = action_array.astype(np.float32)
        self.action_energy += self.dt * float(np.mean(action_array**2))

        tools = self.tool_positions()
        mouth = np.asarray(tools["cup_mouth"])
        spout = np.asarray(tools["pot_spout"])
        pour_intensity = self._pour_intensity()
        flow_rate = self.max_flow_rate * pour_intensity
        flow = flow_rate * self.dt
        vertical_gap = spout[1] - mouth[1]
        stream_x = spout[0] - 0.012 * pour_intensity
        horizontal_error = abs(stream_x - mouth[0])

        capture_fraction = 0.0
        if (
            flow > 0.0
            and 0.015 < vertical_gap < 0.31
            and horizontal_error < 0.075
            and abs(self.cup_angle) < np.deg2rad(25.0)
        ):
            alignment = float(np.clip(1.18 - horizontal_error / 0.065, 0.0, 1.0))
            uprightness = float(max(0.0, np.cos(self.cup_angle)) ** 2)
            capture_fraction = alignment * uprightness

        captured = flow * capture_fraction
        spilled = flow - captured
        room = max(0.0, 1.02 - self.fill)
        if captured > room:
            spilled += captured - room
            captured = room

        cup_tilt_excess = max(0.0, abs(self.cup_angle) - np.deg2rad(20.0))
        cup_leak_rate = self.max_leak_rate * cup_tilt_excess / np.deg2rad(35.0)
        cup_leak = min(self.fill, cup_leak_rate * self.dt)
        self.fill += captured - cup_leak
        self.spill += spilled + cup_leak
        self.last_flow = flow
        self.last_flow_rate = flow_rate
        self.last_captured = captured
        if flow > 0.0:
            table_height = self.geometry.table_y
            if capture_fraction > 0.0:
                end_y = mouth[1]
            elif spout[1] > table_height:
                end_y = table_height
            else:
                end_y = spout[1]
            self.last_stream_end = np.array([stream_x, min(spout[1], end_y)])
        else:
            self.last_stream_end = spout.copy()
        self.elapsed_steps += 1

        fill_error = abs(self.fill - self.target_fill)
        success = self._is_success()
        irrecoverable_failure = self.spill >= 0.40 or self.fill >= 1.019
        terminated = bool(success or irrecoverable_failure)
        time_limit_reached = self.horizon is not None and self.elapsed_steps >= self.horizon
        truncated = bool(time_limit_reached and not terminated)

        if success:
            self._last_termination_reason = "success"
        elif irrecoverable_failure:
            self._last_termination_reason = "spill_or_overflow"
        elif truncated:
            self._last_termination_reason = "time_limit"

        progress = previous_error - fill_error
        spill_delta = self.spill - previous_spill
        reward_terms = {
            "fill_progress": 20.0 * progress,
            "spill": -40.0 * spill_delta,
            "control": -0.024 * self.dt * float(np.sum(action_array**2)),
            "cup_tilt": -0.032 * self.dt * abs(self.cup_angle),
            "time": -0.008 * self.dt,
            "terminal": 0.0,
        }
        if terminated or truncated:
            reward_terms["terminal"] = (
                (15.0 if success else 0.0) - 10.0 * fill_error - 14.0 * self.spill
            )
        reward = float(sum(reward_terms.values()))

        self._episode_done = terminated or truncated
        observation = self._get_obs()
        info = self._get_info()
        info["reward_terms"] = reward_terms
        if self.render_mode == "human":
            self._render_human()
        return observation, reward, terminated, truncated, info

    def render_snapshot(self) -> dict[str, Any]:
        """Return a side-effect-free, JSON-safe scene keyframe.

        This compact representation lets notebook and browser front ends draw
        intermediate visual frames between discrete decision epochs.  Joint
        angles remain the source of truth, so rigid-link forward kinematics can
        be recomputed for every visual frame without creating extra transitions.
        """

        g = self.geometry
        tools = self.tool_positions()
        return {
            "schema_version": 1,
            "geometry": {
                "world_bounds_m": {"x": [-1.0, 1.0], "y": [0.0, 1.2]},
                "table_y_m": float(g.table_y),
                "joint_names": list(self.JOINT_NAMES),
                "joint_limits_rad": {
                    "low": self.joint_low.tolist(),
                    "high": self.joint_high.tolist(),
                },
                "max_joint_speeds_rad_s": self.max_joint_speeds.tolist(),
                "arms": {
                    "cup": {
                        "base_m": list(g.cup_base),
                        "link_lengths_m": [float(g.cup_upper), float(g.cup_fore)],
                    },
                    "pot": {
                        "base_m": list(g.pot_base),
                        "link_lengths_m": [float(g.pot_upper), float(g.pot_fore)],
                    },
                },
                "tools": {
                    "cup": {
                        "grip_offset_m": list(g.cup_grip),
                        "landmark_offset_m": list(g.cup_mouth),
                        "size_m": [float(g.cup_width), float(g.cup_height)],
                    },
                    "pot": {
                        "grip_offset_m": list(g.pot_grip),
                        "landmark_offset_m": list(g.pot_spout),
                        "size_m": [float(g.pot_width), float(g.pot_height)],
                    },
                },
            },
            "state": {
                "joint_angles_rad": self.joint_angles.tolist(),
                "liquid": {
                    "fill_l": float(self.fill),
                    "spill_l": float(self.spill),
                    "target_fill_l": float(self.target_fill),
                    "last_flow_l": float(self.last_flow),
                    "last_flow_rate_l_s": float(self.last_flow_rate),
                    "last_captured_l": float(self.last_captured),
                    "stream_end_m": self.last_stream_end.tolist(),
                },
                "landmarks_m": {
                    "cup_center": np.asarray(tools["cup_center"]).tolist(),
                    "cup_mouth": np.asarray(tools["cup_mouth"]).tolist(),
                    "pot_center": np.asarray(tools["pot_center"]).tolist(),
                    "pot_spout": np.asarray(tools["pot_spout"]).tolist(),
                },
                "step": int(self.elapsed_steps),
                "elapsed_time_s": float(self.elapsed_steps * self.dt),
                "dt_s": float(self.dt),
                "horizon_steps": self.horizon,
                "termination_reason": self._last_termination_reason,
            },
        }

    def render(self):
        if self.render_mode is None:
            return None
        frame = self._render_frame()
        if self.render_mode == "human":
            self._render_human(frame)
            return None
        return frame

    def _render_frame(self) -> np.ndarray:
        from kaist_or_gym.envs.coffee_pouring_rendering import render_frame

        return render_frame(self, width=self.width, height=self.height)

    def _render_human(self, frame: np.ndarray | None = None) -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - optional desktop dependency
            raise gym.error.DependencyNotInstalled(
                "Human rendering needs pygame; install kaist-or-gym[human]."
            ) from exc

        if self._pygame is None:
            self._pygame = pygame
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("KAIST OR Gym — Coffee Pouring")
            self._clock = pygame.time.Clock()
        if frame is None:
            frame = self._render_frame()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(max(1, round(1.0 / self.dt)))

    def close(self) -> None:
        if self._pygame is not None:  # pragma: no cover - optional desktop dependency
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._window = None
        self._clock = None
