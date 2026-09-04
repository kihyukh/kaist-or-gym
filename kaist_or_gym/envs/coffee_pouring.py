"""Two-arm coffee pouring with rigid planar links.

This is a deliberately lightweight teaching environment.  Arm motion uses
exact planar forward kinematics.  Liquid flow uses a deterministic,
physics-informed approximation: tilt exposes the spout, finite liquid head
sets the flow rate, gravity bends the stream, and the rendered stream must
intersect the rendered cup opening to be captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
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
    # The mouth landmark is exactly the centre of the rendered top edge.
    cup_mouth: tuple[float, float] = (0.0, 0.125)
    # The spout landmark is exactly the tip of the rendered spout triangle.
    pot_spout: tuple[float, float] = (-0.1794, 0.0594)
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
    CUP_CAPACITY = 1.02
    POT_CAPACITY = 2.40
    INITIAL_POT_VOLUME = 1.20
    GRAVITY = 9.81
    DISCHARGE_COEFFICIENT = 0.62
    SPOUT_AREA = 1.1e-4
    MAX_FLOW_RATE = 0.130
    EXIT_SPEED_COEFFICIENT = 0.72
    MAX_STREAM_EXIT_SPEED = 1.20
    FLOW_WETTING_START = 0.002
    FLOW_WETTING_FULL = 0.040
    # Maximum coherent-jet radius.  At smaller flows, continuity of volume
    # makes the stream narrower rather than drawing every drip at full width.
    JET_RADIUS = 0.0055
    CUP_RIM_THICKNESS = 0.004
    LINK_COLLISION_RADIUS = 0.029
    BODY_COLLISION_MARGIN = 0.003
    CUP_HANDLE_COLLISION_RADIUS = 0.008
    POT_HANDLE_COLLISION_RADIUS = 0.012
    TABLE_CONTACT_MARGIN = 0.002
    LIQUID_SUBSTEP = 1.0 / 64.0
    STREAM_PATH_SAMPLES = 25

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
            high=np.array(
                [
                    *([1.0] * 10),
                    4.0,
                    4.0,
                    self.CUP_CAPACITY,
                    self.INITIAL_POT_VOLUME,
                    0.90,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.max_flow_rate = self.MAX_FLOW_RATE  # litres per simulated second

        self.joint_angles = np.zeros(6, dtype=np.float64)
        self.fill = 0.0
        self.spill = 0.0
        self.target_fill = 0.70
        self.elapsed_steps = 0
        self.last_flow = 0.0
        self.last_flow_rate = 0.0
        self.last_captured = 0.0
        self.last_pour_intensity = 0.0
        self.last_exit_speed = 0.0
        self.last_jet_radius = 0.0
        self.last_capture_fraction = 0.0
        self.last_stream_end = np.zeros(2, dtype=np.float64)
        self.last_stream_path = np.zeros((self.STREAM_PATH_SAMPLES, 2), dtype=np.float64)
        self.last_spill_path = np.zeros((self.STREAM_PATH_SAMPLES, 2), dtype=np.float64)
        self.last_direct_spill = 0.0
        self.last_direct_spill_rate = 0.0
        self.last_direct_spill_path = np.zeros((self.STREAM_PATH_SAMPLES, 2), dtype=np.float64)
        self.last_cup_runoff = 0.0
        self.last_cup_runoff_rate = 0.0
        self.last_cup_runoff_path = np.zeros((self.STREAM_PATH_SAMPLES, 2), dtype=np.float64)
        self.spill_impact_x = 0.0
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
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

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

    @staticmethod
    def _point_segment_distance(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        segment = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-15:
            return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
        fraction = float(
            np.clip(
                np.dot(np.asarray(point) - np.asarray(start), segment) / length_squared,
                0.0,
                1.0,
            )
        )
        nearest = np.asarray(start) + fraction * segment
        return float(np.linalg.norm(np.asarray(point) - nearest))

    @staticmethod
    def _segments_intersect(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
        *,
        tolerance: float = 1e-12,
    ) -> bool:
        def cross(first: np.ndarray, second: np.ndarray) -> float:
            return float(first[0] * second[1] - first[1] * second[0])

        first_start = np.asarray(first_start, dtype=np.float64)
        first_end = np.asarray(first_end, dtype=np.float64)
        second_start = np.asarray(second_start, dtype=np.float64)
        second_end = np.asarray(second_end, dtype=np.float64)
        first_direction = first_end - first_start
        second_direction = second_end - second_start
        denominator = cross(first_direction, second_direction)
        offset = second_start - first_start
        if abs(denominator) > tolerance:
            first_fraction = cross(offset, second_direction) / denominator
            second_fraction = cross(offset, first_direction) / denominator
            return bool(
                -tolerance <= first_fraction <= 1.0 + tolerance
                and -tolerance <= second_fraction <= 1.0 + tolerance
            )

        if abs(cross(offset, first_direction)) > tolerance:
            return False
        axis = int(np.argmax(np.abs(first_direction)))
        if abs(first_direction[axis]) <= tolerance:
            return bool(np.linalg.norm(first_start - second_start) <= tolerance)
        first_interval = sorted((float(first_start[axis]), float(first_end[axis])))
        second_interval = sorted((float(second_start[axis]), float(second_end[axis])))
        return (
            max(first_interval[0], second_interval[0])
            <= min(first_interval[1], second_interval[1]) + tolerance
        )

    @classmethod
    def _segment_distance(
        cls,
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
    ) -> float:
        if cls._segments_intersect(first_start, first_end, second_start, second_end):
            return 0.0
        return min(
            cls._point_segment_distance(first_start, second_start, second_end),
            cls._point_segment_distance(first_end, second_start, second_end),
            cls._point_segment_distance(second_start, first_start, first_end),
            cls._point_segment_distance(second_end, first_start, first_end),
        )

    @staticmethod
    def _point_in_convex_polygon(
        point: np.ndarray,
        polygon: np.ndarray,
        *,
        tolerance: float = 1e-12,
    ) -> bool:
        polygon = np.asarray(polygon, dtype=np.float64)
        edges = np.roll(polygon, -1, axis=0) - polygon
        offsets = np.asarray(point, dtype=np.float64) - polygon
        crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
        return bool(np.all(crosses >= -tolerance) or np.all(crosses <= tolerance))

    @classmethod
    def _capsule_intersects_polygon(
        cls,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        polygon: np.ndarray,
    ) -> bool:
        if cls._point_in_convex_polygon(start, polygon) or cls._point_in_convex_polygon(
            end, polygon
        ):
            return True
        return any(
            cls._segment_distance(start, end, edge_start, edge_end) < radius
            for edge_start, edge_end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
        )

    @staticmethod
    def _convex_polygons_overlap(
        first: np.ndarray,
        second: np.ndarray,
        *,
        margin: float,
    ) -> bool:
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        for polygon in (first, second):
            for edge in np.roll(polygon, -1, axis=0) - polygon:
                normal = np.array([-edge[1], edge[0]], dtype=np.float64)
                length = float(np.linalg.norm(normal))
                if length <= 1e-15:
                    continue
                normal /= length
                first_projection = first @ normal
                second_projection = second @ normal
                if float(np.max(first_projection)) + margin <= float(
                    np.min(second_projection)
                ) or float(np.max(second_projection)) + margin <= float(np.min(first_projection)):
                    return False
        return True

    @staticmethod
    def _bounding_boxes_overlap(
        first: np.ndarray,
        second: np.ndarray,
        *,
        margin: float,
    ) -> bool:
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        return bool(
            np.all(np.max(first, axis=0) + margin >= np.min(second, axis=0))
            and np.all(np.max(second, axis=0) + margin >= np.min(first, axis=0))
        )

    def _cross_robot_collision(self, angles: np.ndarray) -> bool:
        """Whether the two rendered robots or held vessels overlap."""

        angles = np.asarray(angles, dtype=np.float64)
        g = self.geometry
        cup_arm = np.stack(
            self._arm_points(np.asarray(g.cup_base), angles[0], angles[1], g.cup_upper, g.cup_fore)
        )
        pot_arm = np.stack(
            self._arm_points(np.asarray(g.pot_base), angles[3], angles[4], g.pot_upper, g.pot_fore)
        )
        cup_angle = self._wrap_angle(float(np.sum(angles[:3])))
        pot_angle = self._wrap_angle(float(np.sum(angles[3:])))
        cup_rotation = self._rotation(cup_angle)
        pot_rotation = self._rotation(pot_angle)
        cup_center = cup_arm[2] - cup_rotation @ np.asarray(g.cup_grip)
        pot_center = pot_arm[2] - pot_rotation @ np.asarray(g.pot_grip)
        cup_polygon = (
            cup_center
            + np.array(
                [
                    (-0.50 * g.cup_width, 0.50 * g.cup_height),
                    (0.50 * g.cup_width, 0.50 * g.cup_height),
                    (0.38 * g.cup_width, -0.50 * g.cup_height),
                    (-0.38 * g.cup_width, -0.50 * g.cup_height),
                ]
            )
            @ cup_rotation.T
        )
        pot_polygons = (
            pot_center
            + np.array(
                [
                    (-0.47 * g.pot_width, 0.48 * g.pot_height),
                    (0.47 * g.pot_width, 0.48 * g.pot_height),
                    (0.43 * g.pot_width, -0.48 * g.pot_height),
                    (-0.43 * g.pot_width, -0.48 * g.pot_height),
                ]
            )
            @ pot_rotation.T,
            pot_center
            + np.array(
                [
                    (-0.43 * g.pot_width, 0.34 * g.pot_height),
                    (-0.78 * g.pot_width, 0.22 * g.pot_height),
                    (-0.43 * g.pot_width, 0.06 * g.pot_height),
                ]
            )
            @ pot_rotation.T,
        )
        # Eight capsules approximate each smooth rendered half-ellipse within
        # two millimetres; the collision radii include that chord error.
        handle_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 9)
        cup_handle = (
            cup_center
            + np.column_stack(
                [
                    0.45 * g.cup_width + 0.34 * g.cup_width * np.cos(handle_angles),
                    0.25 * g.cup_height * np.sin(handle_angles),
                ]
            )
            @ cup_rotation.T
        )
        pot_handle = (
            pot_center
            + np.column_stack(
                [
                    0.43 * g.pot_width + 0.38 * g.pot_width * np.cos(handle_angles),
                    0.30 * g.pot_height * np.sin(handle_angles),
                ]
            )
            @ pot_rotation.T
        )

        for cup_start, cup_end in pairwise(cup_arm):
            for pot_start, pot_end in pairwise(pot_arm):
                if self._segment_distance(cup_start, cup_end, pot_start, pot_end) < (
                    2.0 * self.LINK_COLLISION_RADIUS
                ):
                    return True
            for pot_polygon in pot_polygons:
                if self._capsule_intersects_polygon(
                    cup_start,
                    cup_end,
                    self.LINK_COLLISION_RADIUS + self.BODY_COLLISION_MARGIN,
                    pot_polygon,
                ):
                    return True

        for pot_start, pot_end in pairwise(pot_arm):
            if self._capsule_intersects_polygon(
                pot_start,
                pot_end,
                self.LINK_COLLISION_RADIUS + self.BODY_COLLISION_MARGIN,
                cup_polygon,
            ):
                return True

        cup_handle_arm_margin = self.CUP_HANDLE_COLLISION_RADIUS + self.LINK_COLLISION_RADIUS
        if self._bounding_boxes_overlap(cup_handle, pot_arm, margin=cup_handle_arm_margin):
            for handle_start, handle_end in pairwise(cup_handle):
                for pot_start, pot_end in pairwise(pot_arm):
                    if (
                        self._segment_distance(handle_start, handle_end, pot_start, pot_end)
                        < cup_handle_arm_margin
                    ):
                        return True
        for pot_polygon in pot_polygons:
            margin = self.CUP_HANDLE_COLLISION_RADIUS + self.BODY_COLLISION_MARGIN
            if self._bounding_boxes_overlap(cup_handle, pot_polygon, margin=margin):
                for handle_start, handle_end in pairwise(cup_handle):
                    if self._capsule_intersects_polygon(
                        handle_start,
                        handle_end,
                        margin,
                        pot_polygon,
                    ):
                        return True

        pot_handle_arm_margin = self.POT_HANDLE_COLLISION_RADIUS + self.LINK_COLLISION_RADIUS
        if self._bounding_boxes_overlap(pot_handle, cup_arm, margin=pot_handle_arm_margin):
            for handle_start, handle_end in pairwise(pot_handle):
                for cup_start, cup_end in pairwise(cup_arm):
                    if (
                        self._segment_distance(handle_start, handle_end, cup_start, cup_end)
                        < pot_handle_arm_margin
                    ):
                        return True
        pot_handle_body_margin = self.POT_HANDLE_COLLISION_RADIUS + self.BODY_COLLISION_MARGIN
        if self._bounding_boxes_overlap(
            pot_handle,
            cup_polygon,
            margin=pot_handle_body_margin,
        ):
            for handle_start, handle_end in pairwise(pot_handle):
                if self._capsule_intersects_polygon(
                    handle_start,
                    handle_end,
                    pot_handle_body_margin,
                    cup_polygon,
                ):
                    return True

        handle_margin = self.CUP_HANDLE_COLLISION_RADIUS + self.POT_HANDLE_COLLISION_RADIUS
        if self._bounding_boxes_overlap(cup_handle, pot_handle, margin=handle_margin):
            for cup_start, cup_end in pairwise(cup_handle):
                for pot_start, pot_end in pairwise(pot_handle):
                    if (
                        self._segment_distance(cup_start, cup_end, pot_start, pot_end)
                        < handle_margin
                    ):
                        return True

        return any(
            self._convex_polygons_overlap(
                cup_polygon,
                pot_polygon,
                margin=self.BODY_COLLISION_MARGIN,
            )
            for pot_polygon in pot_polygons
        )

    def _constrain_cross_robot_motion(
        self,
        current: np.ndarray,
        proposed: np.ndarray,
    ) -> np.ndarray:
        """Project simultaneous six-joint motion to first robot contact."""

        if not self._cross_robot_collision(proposed):
            return proposed
        if self._cross_robot_collision(current):
            # A legacy/custom state may already overlap.  Let it move so a
            # separating command can escape instead of permanently trapping it.
            return proposed

        low = 0.0
        high = 1.0
        delta = proposed - current
        for _ in range(20):
            middle = 0.5 * (low + high)
            candidate = current + middle * delta
            if self._cross_robot_collision(candidate):
                high = middle
            else:
                low = middle
        return current + low * delta

    def _arm_table_clearance(self, arm_name: str, angles: np.ndarray) -> float:
        """Minimum signed clearance of one rendered arm/vessel above the table."""

        g = self.geometry
        angles = np.asarray(angles, dtype=np.float64)
        if arm_name == "cup":
            base = np.asarray(g.cup_base, dtype=np.float64)
            points = self._arm_points(base, angles[0], angles[1], g.cup_upper, g.cup_fore)
            vessel_angle = self._wrap_angle(float(np.sum(angles)))
            rotation = self._rotation(vessel_angle)
            center = points[2] - rotation @ np.asarray(g.cup_grip)
            vessel_local = np.array(
                [
                    (-0.50 * g.cup_width, 0.50 * g.cup_height),
                    (0.50 * g.cup_width, 0.50 * g.cup_height),
                    (0.38 * g.cup_width, -0.50 * g.cup_height),
                    (-0.38 * g.cup_width, -0.50 * g.cup_height),
                ],
                dtype=np.float64,
            )
            handle_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 24)
            handle_local = np.column_stack(
                [
                    0.45 * g.cup_width + 0.34 * g.cup_width * np.cos(handle_angles),
                    0.25 * g.cup_height * np.sin(handle_angles),
                ]
            )
            handle_radius = self.CUP_HANDLE_COLLISION_RADIUS
        elif arm_name == "pot":
            base = np.asarray(g.pot_base, dtype=np.float64)
            points = self._arm_points(base, angles[0], angles[1], g.pot_upper, g.pot_fore)
            vessel_angle = self._wrap_angle(float(np.sum(angles)))
            rotation = self._rotation(vessel_angle)
            center = points[2] - rotation @ np.asarray(g.pot_grip)
            vessel_local = np.array(
                [
                    (-0.47 * g.pot_width, 0.48 * g.pot_height),
                    (0.47 * g.pot_width, 0.48 * g.pot_height),
                    (0.43 * g.pot_width, -0.48 * g.pot_height),
                    (-0.43 * g.pot_width, -0.48 * g.pot_height),
                    (-0.43 * g.pot_width, 0.34 * g.pot_height),
                    (-0.78 * g.pot_width, 0.22 * g.pot_height),
                    (-0.43 * g.pot_width, 0.06 * g.pot_height),
                ],
                dtype=np.float64,
            )
            handle_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 24)
            handle_local = np.column_stack(
                [
                    0.43 * g.pot_width + 0.38 * g.pot_width * np.cos(handle_angles),
                    0.30 * g.pot_height * np.sin(handle_angles),
                ]
            )
            handle_radius = self.POT_HANDLE_COLLISION_RADIUS
        else:  # pragma: no cover - internal programming error
            raise ValueError(f"unknown arm name: {arm_name}")

        arm_clearance = min(point[1] for point in points) - self.LINK_COLLISION_RADIUS
        vessel_points = center + vessel_local @ rotation.T
        handle_points = center + handle_local @ rotation.T
        vessel_clearance = float(np.min(vessel_points[:, 1])) - self.TABLE_CONTACT_MARGIN
        handle_clearance = float(np.min(handle_points[:, 1])) - handle_radius
        return float(min(arm_clearance, vessel_clearance, handle_clearance) - g.table_y)

    def _constrain_arm_above_table(
        self,
        arm_name: str,
        current: np.ndarray,
        proposed: np.ndarray,
    ) -> np.ndarray:
        """Project one substep to first contact with the tabletop."""

        if self._arm_table_clearance(arm_name, proposed) >= 0.0:
            return proposed
        if self._arm_table_clearance(arm_name, current) < -1e-9:
            # Reset validation normally makes this unreachable.  Keeping an
            # escape path avoids trapping a state restored from legacy data.
            return proposed

        low = 0.0
        high = 1.0
        delta = proposed - current
        for _ in range(32):
            middle = 0.5 * (low + high)
            candidate = current + middle * delta
            if self._arm_table_clearance(arm_name, candidate) >= 0.0:
                low = middle
            else:
                high = middle
        return current + low * delta

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

    @property
    def source_remaining(self) -> float:
        """Coffee still in the pot, derived by conservation of volume."""

        return float(max(0.0, self.INITIAL_POT_VOLUME - self.fill - self.spill))

    @staticmethod
    def _symmetric_uniform_sum_quantile(
        fraction: float,
        first_half_width: float,
        second_half_width: float,
    ) -> float:
        """Quantile of two centred uniforms, used for a horizontal free surface."""

        fraction = float(np.clip(fraction, 0.0, 1.0))
        long = float(max(abs(first_half_width), abs(second_half_width)))
        short = float(min(abs(first_half_width), abs(second_half_width)))
        if long < 1e-12:
            return 0.0
        if short < 1e-12:
            return long * (2.0 * fraction - 1.0)
        shoulder_probability = short / (2.0 * long)
        if fraction < shoulder_probability:
            return float(-(long + short) + np.sqrt(8.0 * long * short * fraction))
        if fraction <= 1.0 - shoulder_probability:
            return float(2.0 * long * fraction - long)
        return float(long + short - np.sqrt(8.0 * long * short * (1.0 - fraction)))

    def _pot_surface_relative_y(self, remaining: float | None = None) -> float:
        """World-vertical liquid surface height relative to the pot centre."""

        if remaining is None:
            remaining = self.source_remaining
        fraction = float(np.clip(remaining / self.POT_CAPACITY, 0.0, 1.0))
        angle = self.pot_angle
        half_width = 0.43 * self.geometry.pot_width * abs(np.sin(angle))
        half_height = 0.45 * self.geometry.pot_height * abs(np.cos(angle))
        return self._symmetric_uniform_sum_quantile(fraction, half_width, half_height)

    def _pot_surface_world_y(
        self,
        tools: dict[str, Any] | None = None,
        remaining: float | None = None,
    ) -> float:
        if tools is None:
            tools = self.tool_positions()
        return float(np.asarray(tools["pot_center"])[1] + self._pot_surface_relative_y(remaining))

    def _flow_state(self, remaining: float | None = None) -> tuple[float, float, float]:
        """Return volumetric rate, exit speed, and hydraulic head.

        The rate follows a Torricelli-style ``sqrt(2*g*h)`` law.  A smooth
        wetting factor models the spout opening progressively as the free
        surface rises above it, avoiding an abrupt on/off threshold.
        """

        if remaining is None:
            remaining = self.source_remaining
        if remaining <= 1e-12:
            return 0.0, 0.0, 0.0
        rotated_spout = self._rotation(self.pot_angle) @ np.asarray(self.geometry.pot_spout)
        head = max(0.0, self._pot_surface_relative_y(remaining) - float(rotated_spout[1]))
        wetting = self._smoothstep(
            (head - self.FLOW_WETTING_START) / (self.FLOW_WETTING_FULL - self.FLOW_WETTING_START)
        )
        ideal_speed = float(np.sqrt(2.0 * self.GRAVITY * head))
        flow_rate = min(
            self.max_flow_rate,
            1000.0 * self.DISCHARGE_COEFFICIENT * self.SPOUT_AREA * wetting * ideal_speed,
        )
        exit_speed = min(
            self.MAX_STREAM_EXIT_SPEED,
            self.EXIT_SPEED_COEFFICIENT * ideal_speed,
        )
        return float(flow_rate), float(exit_speed), float(head)

    def _jet_radius(self, flow_rate: float, exit_speed: float) -> float:
        """Radius implied by volumetric continuity, capped by the spout."""

        if flow_rate <= 1e-12 or exit_speed <= 1e-12:
            return 0.0
        cubic_metres_per_second = flow_rate / 1000.0
        radius = np.sqrt(cubic_metres_per_second / (np.pi * exit_speed))
        return float(np.clip(radius, 0.0, self.JET_RADIUS))

    def _pour_intensity(self) -> float:
        """Current flow rate normalized to ``[0, 1]`` for diagnostics."""

        flow_rate, _, _ = self._flow_state()
        return 0.0 if self.max_flow_rate <= 0.0 else float(flow_rate / self.max_flow_rate)

    def _cup_polygon(self, tools: dict[str, Any] | None = None) -> np.ndarray:
        """Return the same cup trapezoid used by both renderers."""

        if tools is None:
            tools = self.tool_positions()
        g = self.geometry
        local = np.array(
            [
                (-0.50 * g.cup_width, 0.50 * g.cup_height),
                (0.50 * g.cup_width, 0.50 * g.cup_height),
                (0.38 * g.cup_width, -0.50 * g.cup_height),
                (-0.38 * g.cup_width, -0.50 * g.cup_height),
            ],
            dtype=np.float64,
        )
        return np.asarray(tools["cup_center"]) + local @ self._rotation(self.cup_angle).T

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _clip_polygon_below(points: np.ndarray, surface_y: float) -> np.ndarray:
        """Clip a polygon to the half-plane at or below a horizontal surface."""

        clipped: list[np.ndarray] = []
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        previous = np.asarray(points[-1], dtype=np.float64)
        previous_inside = previous[1] <= surface_y
        for raw_current in points:
            current = np.asarray(raw_current, dtype=np.float64)
            current_inside = current[1] <= surface_y
            if current_inside != previous_inside:
                fraction = float((surface_y - previous[1]) / (current[1] - previous[1]))
                clipped.append(previous + fraction * (current - previous))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        if not clipped:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(clipped, dtype=np.float64)

    def _stable_cup_capacity(self, tools: dict[str, Any] | None = None) -> float:
        """Maximum coffee retained below the lower rim at the current tilt."""

        polygon = self._cup_polygon(tools)
        lower_rim_y = float(min(polygon[0, 1], polygon[1, 1]))
        retained_area = self._polygon_area(self._clip_polygon_below(polygon, lower_rim_y))
        full_area = self._polygon_area(polygon)
        return float(self.CUP_CAPACITY * retained_area / full_area)

    def _cup_runoff_path(
        self,
        tools: dict[str, Any],
        *,
        preferred_x: float | None = None,
    ) -> np.ndarray:
        """Return a gravity-driven path from the physically lower cup rim.

        With a level cup, an incoming overflow leaves from the rim closest to
        its impact.  Once the cup tilts, gravity selects the lower rim.
        """

        polygon = self._cup_polygon(tools)
        rims = polygon[:2]
        height_difference = float(rims[0, 1] - rims[1, 1])
        if abs(height_difference) > 1e-9:
            side_index = 0 if height_difference < 0.0 else 1
        elif preferred_x is not None:
            side_index = int(np.argmin([abs(float(point[0]) - preferred_x) for point in rims]))
        else:
            side_index = 0

        return self._cup_exterior_runoff_path(
            tools,
            np.asarray(rims[side_index], dtype=np.float64),
            side_index=side_index,
        )

    def _gravity_fall_path(
        self,
        start: np.ndarray,
        *,
        horizontal_speed: float,
        samples: int | None = None,
    ) -> np.ndarray:
        """Sample a free-fall path from a rim or exterior collision point."""

        sample_count = self.STREAM_PATH_SAMPLES if samples is None else int(samples)
        start = np.asarray(start, dtype=np.float64)
        if start[1] <= self.geometry.table_y:
            impact = np.array([start[0], self.geometry.table_y], dtype=np.float64)
            return np.repeat(impact[None, :], sample_count, axis=0)
        fall_time = float(np.sqrt(2.0 * (start[1] - self.geometry.table_y) / self.GRAVITY))
        times = np.linspace(0.0, fall_time, sample_count)
        path = np.empty((sample_count, 2), dtype=np.float64)
        path[:, 0] = start[0] + horizontal_speed * times
        path[:, 1] = start[1] - 0.5 * self.GRAVITY * times**2
        path[-1, 1] = self.geometry.table_y
        return path

    @staticmethod
    def _path_enters_convex_polygon(
        path: np.ndarray,
        polygon: np.ndarray,
        *,
        tolerance: float = 1e-10,
    ) -> bool:
        """Whether any point or connecting segment enters a convex polygon."""

        polygon = np.asarray(polygon, dtype=np.float64)
        edges = np.roll(polygon, -1, axis=0) - polygon
        path = np.asarray(path, dtype=np.float64)
        for point in path:
            offsets = point - polygon
            crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
            if np.all(crosses > tolerance) or np.all(crosses < -tolerance):
                return True

        signed_double_area = float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        orientation = 1.0 if signed_double_area >= 0.0 else -1.0
        for start, end in pairwise(path):
            lower = 0.0
            upper = 1.0
            for vertex, edge in zip(polygon, edges, strict=True):
                start_offset = start - vertex
                end_offset = end - vertex
                start_side = orientation * (edge[0] * start_offset[1] - edge[1] * start_offset[0])
                end_side = orientation * (edge[0] * end_offset[1] - edge[1] * end_offset[0])
                slope = end_side - start_side
                if abs(slope) <= 1e-15:
                    if start_side <= tolerance:
                        upper = lower
                        break
                    continue
                crossing = (tolerance - start_side) / slope
                if slope > 0.0:
                    lower = max(lower, crossing)
                else:
                    upper = min(upper, crossing)
                if lower >= upper - 1e-12:
                    break
            if lower < upper - 1e-12 and upper > 0.0 and lower < 1.0:
                return True
        return False

    def _exterior_fall_path(
        self,
        start: np.ndarray,
        polygon: np.ndarray,
        *,
        preferred_direction: float,
        samples: int | None = None,
    ) -> np.ndarray:
        """Choose an outward free-fall arc that never crosses the cup body."""

        preferred_sign = -1.0 if preferred_direction < 0.0 else 1.0
        fallback: np.ndarray | None = None
        for speed in (0.0, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12):
            directions = (preferred_sign,) if speed == 0.0 else (preferred_sign, -preferred_sign)
            for direction in directions:
                candidate = self._gravity_fall_path(
                    start,
                    horizontal_speed=direction * speed,
                    samples=samples,
                )
                if fallback is None:
                    fallback = candidate
                if not self._path_enters_convex_polygon(candidate, polygon):
                    return candidate
        # A finite convex cup always admits an outward direction at a boundary
        # point; this fallback is defensive for pathological floating inputs.
        assert fallback is not None
        return fallback

    @staticmethod
    def _sample_polyline(
        points: np.ndarray,
        sample_count: int,
        *,
        endpoint: bool,
    ) -> np.ndarray:
        """Sample a polyline uniformly by distance without changing its shape."""

        points = np.asarray(points, dtype=np.float64)
        if sample_count <= 0:
            return np.empty((0, 2), dtype=np.float64)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-15])
        points = points[keep]
        if len(points) == 1:
            return np.repeat(points, sample_count, axis=0)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        final_reserved = 1 if endpoint else 0
        interval_samples = sample_count - final_reserved
        if interval_samples <= 0:
            return points[-1:].copy()

        # Every segment receives its own samples, so every corner is preserved
        # and the renderer never replaces two exterior wall segments with a
        # straight chord through the cup.
        counts = np.ones(len(lengths), dtype=int)
        remaining = interval_samples - len(lengths)
        if remaining < 0:
            # Routes have at most three segments and normal rendering reserves
            # twelve samples; retain the earliest corners defensively.
            selected = points[:interval_samples]
            return np.vstack([selected, points[-1:]]) if endpoint else selected
        if remaining:
            exact = remaining * lengths / float(np.sum(lengths))
            additions = np.floor(exact).astype(int)
            counts += additions
            leftover = remaining - int(np.sum(additions))
            if leftover:
                order = np.argsort(-(exact - additions))
                counts[order[:leftover]] += 1

        pieces = [
            np.linspace(first, second, count, endpoint=False)
            for first, second, count in zip(points[:-1], points[1:], counts, strict=True)
        ]
        sampled = np.vstack(pieces)
        return np.vstack([sampled, points[-1:]]) if endpoint else sampled

    def _cup_boundary_runoff_path(
        self,
        tools: dict[str, Any],
        start: np.ndarray,
        *,
        edge: tuple[int, int],
        preferred_direction: float,
    ) -> np.ndarray:
        """Follow the open cup's solid boundary downhill, then enter free fall."""

        polygon = self._cup_polygon(tools)
        start = np.asarray(start, dtype=np.float64)
        # The cup is open between vertices 0 and 1.  Its solid wall is the
        # chain 0 -> 3 -> 2 -> 1, including both sides and the bottom.
        chain = (0, 3, 2, 1)
        if set(edge) == {0, 1}:
            # For a downward-facing cup, this is coffee leaving through the
            # open mouth rather than liquid meeting another solid wall.
            return self._exterior_fall_path(
                start,
                polygon,
                preferred_direction=preferred_direction,
            )
        first_position = chain.index(edge[0])
        second_position = chain.index(edge[1])
        if abs(first_position - second_position) != 1:
            raise ValueError("runoff edge must be a solid cup-wall edge")

        candidates: list[np.ndarray] = []
        for endpoint_position, direction in (
            (first_position, -1 if first_position < second_position else 1),
            (second_position, 1 if first_position < second_position else -1),
        ):
            endpoint = polygon[chain[endpoint_position]]
            if endpoint[1] > start[1] + 1e-9:
                continue
            route = [start, endpoint]
            position = endpoint_position
            while 0 <= position + direction < len(chain):
                next_position = position + direction
                next_point = polygon[chain[next_position]]
                if next_point[1] > route[-1][1] + 1e-9:
                    break
                route.append(next_point)
                position = next_position
            candidates.append(np.asarray(route, dtype=np.float64))

        if candidates:
            # Prefer the route with the lowest release point.  Equal-height
            # alternatives use the requested exterior side as a stable tie-break.
            def route_key(route: np.ndarray) -> tuple[float, float]:
                release = route[-1]
                directional_preference = -preferred_direction * float(release[0])
                return float(release[1]), directional_preference

            route = min(candidates, key=route_key)
        else:
            route = start[None, :]

        release = route[-1]
        if len(route) == 1 or np.linalg.norm(route[-1] - route[0]) <= 1e-9:
            return self._exterior_fall_path(
                release,
                polygon,
                preferred_direction=preferred_direction,
            )

        wall_samples = min(12, max(4, self.STREAM_PATH_SAMPLES // 2))
        wall_path = self._sample_polyline(route, wall_samples, endpoint=False)
        fall_path = self._exterior_fall_path(
            release,
            polygon,
            preferred_direction=preferred_direction,
            samples=self.STREAM_PATH_SAMPLES - wall_samples,
        )
        combined = np.vstack([wall_path, fall_path])
        if self._path_enters_convex_polygon(combined, polygon):
            return self._exterior_fall_path(
                release,
                polygon,
                preferred_direction=preferred_direction,
            )
        return combined

    def _cup_exterior_runoff_path(
        self,
        tools: dict[str, Any],
        start: np.ndarray,
        *,
        side_index: int,
    ) -> np.ndarray:
        """Route runoff along a cup wall, then let it fall outside the cup."""

        start = np.asarray(start, dtype=np.float64)
        outward = -1.0 if side_index == 0 else 1.0
        edge = (0, 3) if side_index == 0 else (1, 2)
        return self._cup_boundary_runoff_path(
            tools,
            start,
            edge=edge,
            preferred_direction=outward,
        )

    def _ballistic_edge_collision(
        self,
        spout: np.ndarray,
        velocity: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
        maximum_time: float,
    ) -> tuple[float, np.ndarray] | None:
        """Return the first centreline collision with one finite cup edge."""

        edge = np.asarray(edge_end, dtype=np.float64) - np.asarray(edge_start, dtype=np.float64)
        length_squared = float(np.dot(edge, edge))
        if length_squared <= 1e-15:
            return None
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        offset = np.asarray(spout, dtype=np.float64) - np.asarray(edge_start, dtype=np.float64)
        roots = self._quadratic_roots(
            -0.5 * self.GRAVITY * normal[1],
            float(np.dot(velocity, normal)),
            float(np.dot(offset, normal)),
        )
        for candidate in sorted(roots):
            if not 1e-9 < candidate <= maximum_time + 1e-9:
                continue
            point = spout + velocity * candidate
            point[1] -= 0.5 * self.GRAVITY * candidate**2
            along_edge = float(np.dot(point - edge_start, edge) / length_squared)
            if -1e-9 <= along_edge <= 1.0 + 1e-9:
                return float(candidate), point
        return None

    def _add_spill(self, volume: float, impact_x: float) -> None:
        """Accumulate spilled volume and its volume-weighted table location."""

        volume = float(max(0.0, volume))
        if volume <= 1e-15:
            return
        previous = float(self.spill)
        updated = previous + volume
        self.spill_impact_x = float(
            (previous * self.spill_impact_x + volume * float(impact_x)) / updated
        )
        self.spill = updated

    def _cup_surface_world_y(
        self,
        tools: dict[str, Any] | None = None,
        volume: float | None = None,
    ) -> float:
        """Horizontal surface giving the requested volume in the rendered cup."""

        polygon = self._cup_polygon(tools)
        if volume is None:
            volume = self.fill
        target_fraction = float(np.clip(volume / self.CUP_CAPACITY, 0.0, 1.0))
        low = float(np.min(polygon[:, 1]))
        high = float(np.max(polygon[:, 1]))
        full_area = self._polygon_area(polygon)
        for _ in range(48):
            middle = 0.5 * (low + high)
            fraction = self._polygon_area(self._clip_polygon_below(polygon, middle)) / full_area
            if fraction < target_fraction:
                low = middle
            else:
                high = middle
        return float(0.5 * (low + high))

    @staticmethod
    def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
        """Return finite real roots of ``a*t**2 + b*t + c``."""

        epsilon = 1e-12
        if abs(a) < epsilon:
            if abs(b) < epsilon:
                return []
            return [float(-c / b)]
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return []
        root = float(np.sqrt(max(0.0, discriminant)))
        return [float((-b - root) / (2.0 * a)), float((-b + root) / (2.0 * a))]

    def _ballistic_stream(
        self,
        tools: dict[str, Any],
        flow_rate: float,
        exit_speed: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Trace the rendered coffee stream and test it against the cup mouth.

        Coffee leaves along the pot's spout direction and then follows a
        ballistic arc under gravity.  Capture is the overlap of the finite jet
        with the finite, rotated cup-opening segment.  Returning the sampled
        path from this same calculation keeps rendering and liquid dynamics
        geometrically identical.
        """

        spout = np.asarray(tools["pot_spout"], dtype=np.float64)
        table_y = float(self.geometry.table_y)
        if spout[1] <= table_y:
            path = np.repeat(spout[None, :], self.STREAM_PATH_SAMPLES, axis=0)
            return path, 0.0, path.copy()

        nozzle_direction = self._rotation(self.pot_angle) @ np.array([-1.0, 0.0])
        velocity = float(exit_speed) * nozzle_direction
        fall_height = float(spout[1] - table_y)
        table_time = float(
            (velocity[1] + np.sqrt(velocity[1] ** 2 + 2.0 * self.GRAVITY * fall_height))
            / self.GRAVITY
        )

        contact_time: float | None = None
        capture_fraction = 0.0
        rim_runoff_start: np.ndarray | None = None
        rim_runoff_side: int | None = None
        body_collision: np.ndarray | None = None
        body_collision_edge: tuple[int, int] | None = None
        mouth = np.asarray(tools["cup_mouth"], dtype=np.float64)
        cup_polygon = self._cup_polygon(tools)
        cup_rotation = self._rotation(self.cup_angle)
        tangent = cup_rotation @ np.array([1.0, 0.0])
        opening_normal = cup_rotation @ np.array([0.0, 1.0])
        opening_faces_up = opening_normal[1] > 0.0
        if opening_faces_up:
            offset = spout - mouth
            roots = self._quadratic_roots(
                -0.5 * self.GRAVITY * opening_normal[1],
                float(np.dot(velocity, opening_normal)),
                float(np.dot(offset, opening_normal)),
            )
            for candidate in sorted(roots):
                if not 1e-9 < candidate <= table_time + 1e-9:
                    continue
                point = spout + velocity * candidate
                point[1] -= 0.5 * self.GRAVITY * candidate**2
                along_opening = float(np.dot(point - mouth, tangent))
                impact_velocity = velocity + np.array([0.0, -self.GRAVITY * candidate])
                approaching = float(np.dot(impact_velocity, opening_normal)) < 0.0
                impact_radius = self._jet_radius(flow_rate, float(np.linalg.norm(impact_velocity)))
                inner_half_width = 0.5 * self.geometry.cup_width - self.CUP_RIM_THICKNESS
                outer_half_width = 0.5 * self.geometry.cup_width
                opening_overlap = max(
                    0.0,
                    min(along_opening + impact_radius, inner_half_width)
                    - max(along_opening - impact_radius, -inner_half_width),
                )
                candidate_capture = float(
                    np.clip(opening_overlap / (2.0 * impact_radius), 0.0, 1.0)
                )
                vessel_overlap = max(
                    0.0,
                    min(along_opening + impact_radius, outer_half_width)
                    - max(along_opening - impact_radius, -outer_half_width),
                )
                if approaching and (candidate_capture > 0.0 or vessel_overlap > 0.0):
                    contact_time = float(candidate)
                    capture_fraction = candidate_capture
                    if candidate_capture < 1.0 - 1e-12:
                        rim_runoff_side = 1 if along_opening >= 0.0 else 0
                        rim_runoff_start = cup_polygon[rim_runoff_side].copy()
                    break

        if contact_time is None:
            # The opening is empty space, but the solid cup edges must stop a
            # near miss instead of letting the jet pass through the vessel.
            solid_edges = [(1, 2), (2, 3), (3, 0)]
            if not opening_faces_up:
                solid_edges.append((0, 1))
            collisions: list[tuple[float, np.ndarray, int, int]] = []
            for start_index, end_index in solid_edges:
                collision = self._ballistic_edge_collision(
                    spout,
                    velocity,
                    cup_polygon[start_index],
                    cup_polygon[end_index],
                    table_time,
                )
                if collision is not None:
                    collisions.append((*collision, start_index, end_index))
            if collisions:
                contact_time, body_collision, start_index, end_index = min(
                    collisions, key=lambda item: item[0]
                )
                body_collision_edge = (start_index, end_index)

        end_time = table_time if contact_time is None else contact_time
        times = np.linspace(0.0, end_time, self.STREAM_PATH_SAMPLES)
        path = spout[None, :] + times[:, None] * velocity[None, :]
        path[:, 1] -= 0.5 * self.GRAVITY * times**2
        if contact_time is None:
            path[-1, 1] = table_y

        spill_path = np.repeat(path[-1][None, :], self.STREAM_PATH_SAMPLES, axis=0)
        if contact_time is not None and capture_fraction < 1.0 - 1e-12:
            runoff_start = rim_runoff_start if rim_runoff_start is not None else body_collision
            if runoff_start is not None:
                if rim_runoff_start is not None and rim_runoff_side is not None:
                    spill_path = self._cup_exterior_runoff_path(
                        tools, runoff_start, side_index=rim_runoff_side
                    )
                elif body_collision_edge is not None:
                    cup_center_x = float(np.asarray(tools["cup_center"])[0])
                    outward = -1.0 if runoff_start[0] < cup_center_x else 1.0
                    spill_path = self._cup_boundary_runoff_path(
                        tools,
                        runoff_start,
                        edge=body_collision_edge,
                        preferred_direction=outward,
                    )
        return path, capture_fraction, spill_path

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

        table_collisions = [
            arm_name
            for arm_name, arm_slice in (("cup", slice(0, 3)), ("pot", slice(3, 6)))
            if self._arm_table_clearance(arm_name, initial_joints[arm_slice]) < -1e-9
        ]
        if table_collisions:
            names = " and ".join(table_collisions)
            raise ValueError(f"initial {names} arm configuration intersects the table")
        if self._cross_robot_collision(initial_joints):
            raise ValueError("initial robot configuration contains an arm or vessel collision")

        fill = float(options.get("fill", 0.0))
        spill = float(options.get("spill", 0.0))
        if not np.isfinite(fill) or not 0.0 <= fill <= self.CUP_CAPACITY:
            raise ValueError(f"fill must be between 0.0 and {self.CUP_CAPACITY:.2f} litres")
        if not np.isfinite(spill) or not 0.0 <= spill <= self.INITIAL_POT_VOLUME:
            raise ValueError(f"spill must be between 0.0 and {self.INITIAL_POT_VOLUME:.2f} litres")
        if fill + spill > self.INITIAL_POT_VOLUME + 1e-12:
            raise ValueError("fill plus spill cannot exceed the pot's initial coffee volume")

        self.joint_angles = initial_joints.copy()
        self.fill = fill
        self.spill = spill
        self.target_fill = target
        self.elapsed_steps = 0
        self.last_flow = 0.0
        self.last_flow_rate = 0.0
        self.last_captured = 0.0
        self.last_pour_intensity = 0.0
        self.last_exit_speed = 0.0
        self.last_jet_radius = 0.0
        self.last_capture_fraction = 0.0
        self.last_stream_end = np.asarray(self.tool_positions()["pot_spout"]).copy()
        self.last_stream_path = np.repeat(
            self.last_stream_end[None, :], self.STREAM_PATH_SAMPLES, axis=0
        )
        self.last_spill_path = self.last_stream_path.copy()
        self.last_direct_spill = 0.0
        self.last_direct_spill_rate = 0.0
        self.last_direct_spill_path = self.last_stream_path.copy()
        runoff_start = self._cup_runoff_path(self.tool_positions())[0]
        self.last_cup_runoff = 0.0
        self.last_cup_runoff_rate = 0.0
        self.last_cup_runoff_path = np.repeat(
            runoff_start[None, :], self.STREAM_PATH_SAMPLES, axis=0
        )
        self.spill_impact_x = 0.0
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
            "capture_fraction": float(self.last_capture_fraction),
            "pour_intensity": float(self.last_pour_intensity),
            "stream_exit_speed": float(self.last_exit_speed),
            "jet_radius": float(self.last_jet_radius),
            "source_remaining": self.source_remaining,
            "stable_cup_capacity": self._stable_cup_capacity(tools),
            "cup_surface_y": self._cup_surface_world_y(tools),
            "pot_surface_y": self._pot_surface_world_y(tools),
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
            "stream_path": self.last_stream_path.astype(np.float32).copy(),
            "spill_path": self.last_spill_path.astype(np.float32).copy(),
            "direct_spill": float(self.last_direct_spill),
            "direct_spill_rate": float(self.last_direct_spill_rate),
            "direct_spill_path": self.last_direct_spill_path.astype(np.float32).copy(),
            "cup_runoff": float(self.last_cup_runoff),
            "cup_runoff_rate": float(self.last_cup_runoff_rate),
            "cup_runoff_path": self.last_cup_runoff_path.astype(np.float32).copy(),
            "spill_impact_x": float(self.spill_impact_x),
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
        # this decision interval.  Fixed physics substeps integrate both the
        # joints and liquid while the browser interpolates only the resulting
        # decision keyframes for a smooth display.
        angular_velocity = self.max_joint_speeds * action_array
        self.last_action = action_array.astype(np.float32)
        self.action_energy += self.dt * float(np.mean(action_array**2))

        total_flow = 0.0
        total_captured = 0.0
        total_direct_spill = 0.0
        total_cup_runoff = 0.0
        final_tools = self.tool_positions()
        final_spout = np.asarray(final_tools["pot_spout"])
        final_path = np.repeat(final_spout[None, :], self.STREAM_PATH_SAMPLES, axis=0)
        final_spill_path = final_path.copy()
        direct_spill_path = final_path.copy()
        final_flow_rate = 0.0
        final_exit_speed = 0.0
        final_jet_radius = 0.0
        final_capture_fraction = 0.0
        runoff_start = self._cup_runoff_path(final_tools)[0]
        runoff_path = np.repeat(runoff_start[None, :], self.STREAM_PATH_SAMPLES, axis=0)
        runoff_preferred_x: float | None = None
        remaining_time = self.dt
        while remaining_time > 1e-12:
            substep = min(self.LIQUID_SUBSTEP, remaining_time)
            proposed_angles = np.clip(
                self.joint_angles + substep * angular_velocity,
                self.joint_low,
                self.joint_high,
            )
            for arm_name, arm_slice in (("cup", slice(0, 3)), ("pot", slice(3, 6))):
                proposed_angles[arm_slice] = self._constrain_arm_above_table(
                    arm_name,
                    self.joint_angles[arm_slice],
                    proposed_angles[arm_slice],
                )
            proposed_angles = self._constrain_cross_robot_motion(
                self.joint_angles,
                proposed_angles,
            )
            self.joint_angles = proposed_angles
            final_tools = self.tool_positions()
            final_spout = np.asarray(final_tools["pot_spout"])

            # A tilted open cup can retain only the part of its contents below
            # the lower rim.  The excess leaves immediately in this quasi-static
            # model, using the same trapezoid that is rendered on screen.
            stable_capacity = self._stable_cup_capacity(final_tools)
            tilt_spill = max(0.0, self.fill - stable_capacity)
            self.fill -= tilt_spill
            if tilt_spill > 0.0:
                runoff_path = self._cup_runoff_path(final_tools)
                runoff_preferred_x = float(runoff_path[0, 0])
                self._add_spill(tilt_spill, runoff_path[-1, 0])
                total_cup_runoff += tilt_spill

            flow_rate, exit_speed, _ = self._flow_state()
            flow = min(flow_rate * substep, self.source_remaining)
            if flow > 0.0:
                stream_path, capture_fraction, spill_path = self._ballistic_stream(
                    final_tools, flow_rate, exit_speed
                )
            else:
                stream_path = np.repeat(final_spout[None, :], self.STREAM_PATH_SAMPLES, axis=0)
                spill_path = stream_path.copy()
                capture_fraction = 0.0

            capture_fraction = float(np.clip(capture_fraction, 0.0, 1.0))
            incoming = min(flow, flow * capture_fraction)
            direct_spill = max(0.0, flow - incoming)
            room = max(0.0, stable_capacity - self.fill)
            accepted = min(incoming, room)
            overflow = max(0.0, incoming - accepted)
            self.fill += accepted

            if direct_spill > 0.0:
                has_exterior_runoff = np.linalg.norm(spill_path[-1] - spill_path[0]) > 1e-9
                direct_path = spill_path if has_exterior_runoff else stream_path
                self._add_spill(direct_spill, direct_path[-1, 0])
                total_direct_spill += direct_spill
                # Preserve the most recent geometry that actually produced
                # direct spill.  The endpoint can already be a clean capture,
                # so relying only on endpoint geometry would make a newly
                # growing puddle appear without its causal stream.
                direct_spill_path = direct_path.copy()
            if overflow > 0.0:
                runoff_path = self._cup_runoff_path(
                    final_tools, preferred_x=float(stream_path[-1, 0])
                )
                runoff_preferred_x = float(runoff_path[0, 0])
                self._add_spill(overflow, runoff_path[-1, 0])
                total_cup_runoff += overflow

            total_flow += flow
            total_captured += accepted
            remaining_time -= substep

        # Rendering describes the instantaneous state at the decision
        # endpoint, while ``last_flow`` and ``last_captured`` describe the
        # integrated volume transferred during the whole interval.
        final_flow_rate, final_exit_speed, _ = self._flow_state()
        final_jet_radius = self._jet_radius(final_flow_rate, final_exit_speed)
        if final_flow_rate > 1e-12:
            final_path, final_capture_fraction, final_spill_path = self._ballistic_stream(
                final_tools, final_flow_rate, final_exit_speed
            )
            final_capture_fraction = float(np.clip(final_capture_fraction, 0.0, 1.0))
        else:
            final_path = np.repeat(final_spout[None, :], self.STREAM_PATH_SAMPLES, axis=0)
            final_spill_path = final_path.copy()
            final_capture_fraction = 0.0

        self.last_flow = total_flow
        self.last_flow_rate = final_flow_rate
        self.last_captured = total_captured
        self.last_pour_intensity = (
            0.0 if self.max_flow_rate <= 0.0 else final_flow_rate / self.max_flow_rate
        )
        self.last_exit_speed = final_exit_speed
        self.last_jet_radius = final_jet_radius
        self.last_capture_fraction = final_capture_fraction
        self.last_stream_path = final_path
        self.last_spill_path = final_spill_path
        self.last_stream_end = final_path[-1].copy()
        self.last_direct_spill = total_direct_spill
        self.last_direct_spill_rate = total_direct_spill / self.dt
        self.last_direct_spill_path = direct_spill_path
        self.last_cup_runoff = total_cup_runoff
        self.last_cup_runoff_rate = total_cup_runoff / self.dt
        if total_cup_runoff > 1e-15:
            self.last_cup_runoff_path = self._cup_runoff_path(
                final_tools, preferred_x=runoff_preferred_x
            )
        else:
            runoff_start = self._cup_runoff_path(final_tools)[0]
            self.last_cup_runoff_path = np.repeat(
                runoff_start[None, :], self.STREAM_PATH_SAMPLES, axis=0
            )
        self.elapsed_steps += 1

        fill_error = abs(self.fill - self.target_fill)
        success = self._is_success()
        irrecoverable_failure = self.spill >= 0.40 or self.fill >= self.CUP_CAPACITY - 0.001
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
            "schema_version": 4,
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
                    "cup_capacity_l": float(self.CUP_CAPACITY),
                    "stable_cup_capacity_l": self._stable_cup_capacity(tools),
                    "source_initial_l": float(self.INITIAL_POT_VOLUME),
                    "source_capacity_l": float(self.POT_CAPACITY),
                    "source_remaining_l": self.source_remaining,
                    "last_flow_l": float(self.last_flow),
                    "last_flow_rate_l_s": float(self.last_flow_rate),
                    "last_captured_l": float(self.last_captured),
                    "last_capture_fraction": float(self.last_capture_fraction),
                    "last_pour_intensity": float(self.last_pour_intensity),
                    "last_exit_speed_m_s": float(self.last_exit_speed),
                    "last_jet_radius_m": float(self.last_jet_radius),
                    "cup_surface_y_m": self._cup_surface_world_y(tools),
                    "target_surface_y_m": self._cup_surface_world_y(tools, self.target_fill),
                    "pot_surface_y_m": self._pot_surface_world_y(tools),
                    "stream_end_m": self.last_stream_end.tolist(),
                    "stream_path_m": self.last_stream_path.tolist(),
                    "spill_path_m": self.last_spill_path.tolist(),
                    "direct_spill_l": float(self.last_direct_spill),
                    "direct_spill_rate_l_s": float(self.last_direct_spill_rate),
                    "direct_spill_path_m": self.last_direct_spill_path.tolist(),
                    "cup_runoff_l": float(self.last_cup_runoff),
                    "cup_runoff_rate_l_s": float(self.last_cup_runoff_rate),
                    "cup_runoff_path_m": self.last_cup_runoff_path.tolist(),
                    "spill_impact_x_m": float(self.spill_impact_x),
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
