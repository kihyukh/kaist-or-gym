"""Bimanual towel straightening and folding with lightweight XPBD cloth physics.

The environment is intended for reinforcement-learning demonstrations rather
than textile engineering.  It nevertheless keeps one authoritative physical
state: fixed-link spatial robot kinematics, a triangular cloth mesh with
stretch/shear/bending compliance, gravity, friction, capsule contact, two-sided
self-contact, and compliant three-vertex pinch grasps.

Actions are normalized joint and gripper velocities.  A decision is held for
``dt`` seconds while the robot and cloth advance through smaller fixed physics
steps, so changing the visualization refresh rate never changes the task.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class LaundryGeometry:
    """Dimensions in metres for the table, towel, and mirrored robot arms."""

    table_half_width: float = 0.72
    table_half_depth: float = 0.58
    table_height: float = 0.0
    table_thickness: float = 0.075
    shoulder_x: float = 0.78
    shoulder_y: float = -0.02
    shoulder_z: float = 0.57
    upper_length: float = 0.38
    fore_length: float = 0.38
    hand_length: float = 0.105
    finger_length: float = 0.085
    arm_radius: float = 0.026
    hand_radius: float = 0.017
    finger_radius: float = 0.007
    towel_width: float = 0.70
    towel_depth: float = 0.48
    towel_thickness: float = 0.010
    gripper_min_opening: float = 0.012
    gripper_max_opening: float = 0.080


class LaundryFoldingEnv(gym.Env):
    """Continuous-control towel straightening and folding with two 3R arms.

    Each arm has three fixed-axis revolute joints: shoulder yaw, elbow pitch,
    and wrist pitch.  Two rigid fingers move symmetrically, adding one gripper
    aperture degree of freedom.  The eight-dimensional action is ordered as::

        [left shoulder, left elbow, left wrist, left gripper,
         right shoulder, right elbow, right wrist, right gripper]

    Positive gripper commands open the fingers and negative commands close
    them.  All components are normalized velocity commands in ``[-1, 1]``.

    The observation is a flat vector containing normalized robot state,
    position, velocity, and grasp membership for every cloth vertex, followed
    by task progress.  ``observation_layout`` exposes the corresponding slices.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 10,
    }
    PHYSICS_MODEL = "xpbd_cloth_v1"
    DEFAULT_DT = 0.10
    DEFAULT_HORIZON = 500
    MAX_PHYSICS_STEP = 1.0 / 200.0
    SOLVER_ITERATIONS = 5
    STRAIGHTNESS_THRESHOLD = 0.88
    FOLD_SUCCESS_THRESHOLD = 0.78
    GRAVITY = 9.81
    AIR_DAMPING = 0.55
    TABLE_FRICTION = 7.0
    STATIC_FRICTION_DISTANCE = 0.0012
    DYNAMIC_FRICTION_DISTANCE = 0.00045
    SELF_COLLISION_PASSES = 1
    GRASP_CAPTURE_RADIUS = 0.050
    GRASP_CAPTURE_OPENING = 0.030
    GRASP_RELEASE_OPENING = 0.052

    ACTION_NAMES = (
        "left_shoulder_yaw",
        "left_elbow_pitch",
        "left_wrist_pitch",
        "left_gripper",
        "right_shoulder_yaw",
        "right_elbow_pitch",
        "right_wrist_pitch",
        "right_gripper",
    )
    JOINT_NAMES = (
        "left_shoulder_yaw",
        "left_elbow_pitch",
        "left_wrist_pitch",
        "right_shoulder_yaw",
        "right_elbow_pitch",
        "right_wrist_pitch",
    )
    TASK_OBSERVATION_NAMES = (
        "straightness",
        "fold_score",
        "straightened_phase",
        "straight_streak",
        "bimanual_tension",
        "dropped_fraction",
        "mean_speed",
        "elapsed_fraction",
    )

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        horizon: int | None = DEFAULT_HORIZON,
        dt: float = DEFAULT_DT,
        mesh_rows: int = 9,
        mesh_cols: int = 13,
        width: int = 1000,
        height: int = 680,
        solver_iterations: int = SOLVER_ITERATIONS,
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
        if not isinstance(mesh_rows, (int, np.integer)) or int(mesh_rows) < 5:
            raise ValueError("mesh_rows must be an integer of at least 5")
        if not isinstance(mesh_cols, (int, np.integer)) or int(mesh_cols) < 5:
            raise ValueError("mesh_cols must be an integer of at least 5")
        if int(mesh_rows) % 2 == 0:
            raise ValueError("mesh_rows must be odd so the target crease is a material row")
        if not isinstance(solver_iterations, (int, np.integer)) or solver_iterations < 1:
            raise ValueError("solver_iterations must be a positive integer")
        if int(width) < 320 or int(height) < 240:
            raise ValueError("render width and height must be at least 320 by 240")

        self.render_mode = render_mode
        self.horizon = None if horizon is None else int(horizon)
        self.dt = float(dt)
        self.mesh_rows = int(mesh_rows)
        self.mesh_cols = int(mesh_cols)
        self.width = int(width)
        self.height = int(height)
        self.solver_iterations = int(solver_iterations)
        self.geometry = LaundryGeometry()
        self.vertex_count = self.mesh_rows * self.mesh_cols

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.max_joint_speeds = np.array([[0.42, 0.40, 0.52], [0.42, 0.40, 0.52]], dtype=np.float64)
        self.max_gripper_speed = 0.045
        self.joint_low = np.array([[-1.15, -1.55, -1.30], [-1.15, -1.55, -1.30]], dtype=np.float64)
        self.joint_high = np.array([[1.15, -0.30, 1.45], [1.15, -0.30, 1.45]], dtype=np.float64)
        self.default_joint_angles = np.array(
            [[0.0, -1.35, 0.40], [0.0, -1.35, 0.40]], dtype=np.float64
        )

        self.rest_positions = self._make_rest_positions()
        self.faces = self._make_faces()
        (
            self.constraint_i,
            self.constraint_j,
            self.constraint_rest,
            self.constraint_compliance,
            self.constraint_kind,
        ) = self._make_distance_constraints()
        self.constraint_batches = self._color_constraints()
        self._self_contact_exclusion = self._make_self_contact_exclusion()
        self._structural_mask = self.constraint_kind == 0

        robot_size = 2 * 4
        cloth_size = self.vertex_count * 8
        task_size = len(self.TASK_OBSERVATION_NAMES)
        self.observation_layout = {
            "robot": slice(0, robot_size),
            "cloth": slice(robot_size, robot_size + cloth_size),
            "task": slice(robot_size + cloth_size, robot_size + cloth_size + task_size),
        }
        self.OBSERVATION_NAMES = self._make_observation_names()
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(robot_size + cloth_size + task_size,),
            dtype=np.float32,
        )

        self.joint_angles = self.default_joint_angles.copy()
        self.gripper_openings = np.full(2, self.geometry.gripper_max_opening)
        self.cloth_positions = self.rest_positions.copy()
        self.cloth_velocities = np.zeros_like(self.cloth_positions)
        self.grasped_vertices: list[np.ndarray] = [
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        ]
        self.grasp_local_offsets: list[np.ndarray] = [
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        ]
        self.elapsed_steps = 0
        self.last_action = np.zeros(8, dtype=np.float32)
        self.straightened_once = False
        self.straight_streak = 0
        self._previous_metrics: dict[str, float] = {}
        self._episode_done = False
        self._last_termination_reason: str | None = None
        self.camera_azimuth = -52.0
        self.camera_elevation = 32.0
        self.camera_distance = 2.10
        self._pygame = None
        self._window = None
        self._clock = None

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------
    def _make_rest_positions(self) -> np.ndarray:
        g = self.geometry
        xs = np.linspace(-0.5 * g.towel_width, 0.5 * g.towel_width, self.mesh_cols)
        ys = np.linspace(-0.5 * g.towel_depth, 0.5 * g.towel_depth, self.mesh_rows)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, g.table_height + 0.5 * g.towel_thickness)
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    def _index(self, row: int, col: int) -> int:
        return row * self.mesh_cols + col

    def _make_faces(self) -> np.ndarray:
        faces: list[tuple[int, int, int]] = []
        for row in range(self.mesh_rows - 1):
            for col in range(self.mesh_cols - 1):
                a = self._index(row, col)
                b = self._index(row, col + 1)
                c = self._index(row + 1, col)
                d = self._index(row + 1, col + 1)
                if (row + col) % 2 == 0:
                    faces.extend(((a, b, d), (a, d, c)))
                else:
                    faces.extend(((a, b, c), (b, d, c)))
        return np.asarray(faces, dtype=np.int64)

    def _make_distance_constraints(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # kind: 0 structural, 1 shear, 2 bending surrogate.  The long-range
        # bending constraints are intentionally compliant; self-contact and
        # the visible crease remain free to form.
        edges: list[tuple[int, int, float, int]] = []
        structural_compliance = 2.0e-7
        shear_compliance = 8.0e-7
        bending_compliance = 8.0e-3
        for row in range(self.mesh_rows):
            for col in range(self.mesh_cols - 1):
                edges.append(
                    (self._index(row, col), self._index(row, col + 1), structural_compliance, 0)
                )
        for row in range(self.mesh_rows - 1):
            for col in range(self.mesh_cols):
                edges.append(
                    (self._index(row, col), self._index(row + 1, col), structural_compliance, 0)
                )
        for row in range(self.mesh_rows - 1):
            for col in range(self.mesh_cols - 1):
                edges.append(
                    (self._index(row, col), self._index(row + 1, col + 1), shear_compliance, 1)
                )
                edges.append(
                    (self._index(row, col + 1), self._index(row + 1, col), shear_compliance, 1)
                )
        for row in range(self.mesh_rows):
            for col in range(self.mesh_cols - 2):
                edges.append(
                    (self._index(row, col), self._index(row, col + 2), bending_compliance, 2)
                )
        for row in range(self.mesh_rows - 2):
            for col in range(self.mesh_cols):
                edges.append(
                    (self._index(row, col), self._index(row + 2, col), bending_compliance, 2)
                )

        first = np.asarray([edge[0] for edge in edges], dtype=np.int64)
        second = np.asarray([edge[1] for edge in edges], dtype=np.int64)
        rest = np.linalg.norm(self.rest_positions[first] - self.rest_positions[second], axis=1)
        compliance = np.asarray([edge[2] for edge in edges], dtype=np.float64)
        kind = np.asarray([edge[3] for edge in edges], dtype=np.int8)
        return first, second, rest, compliance, kind

    def _color_constraints(self) -> list[np.ndarray]:
        """Greedily create disjoint batches for vectorized Gauss-Seidel solves."""

        batch_members: list[list[int]] = []
        occupied: list[set[int]] = []
        for constraint_index, (first, second) in enumerate(
            zip(self.constraint_i, self.constraint_j, strict=True)
        ):
            for members, used in zip(batch_members, occupied, strict=True):
                if int(first) not in used and int(second) not in used:
                    members.append(constraint_index)
                    used.update((int(first), int(second)))
                    break
            else:
                batch_members.append([constraint_index])
                occupied.append({int(first), int(second)})
        return [np.asarray(batch, dtype=np.int64) for batch in batch_members]

    def _make_self_contact_exclusion(self) -> np.ndarray:
        rows = np.arange(self.vertex_count) // self.mesh_cols
        cols = np.arange(self.vertex_count) % self.mesh_cols
        face_rows = rows[self.faces]
        face_cols = cols[self.faces]
        exclusion = np.zeros((self.vertex_count, len(self.faces)), dtype=bool)
        for vertex in range(self.vertex_count):
            close_row = np.any(np.abs(face_rows - rows[vertex]) <= 1, axis=1)
            close_col = np.any(np.abs(face_cols - cols[vertex]) <= 1, axis=1)
            exclusion[vertex] = close_row & close_col
        return exclusion

    def _make_observation_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for arm in ("left", "right"):
            names.extend(
                [
                    f"{arm}_shoulder_yaw",
                    f"{arm}_elbow_pitch",
                    f"{arm}_wrist_pitch",
                    f"{arm}_gripper_opening",
                ]
            )
        for vertex in range(self.vertex_count):
            names.extend(
                [
                    f"cloth_{vertex}_x",
                    f"cloth_{vertex}_y",
                    f"cloth_{vertex}_z",
                    f"cloth_{vertex}_vx",
                    f"cloth_{vertex}_vy",
                    f"cloth_{vertex}_vz",
                    f"cloth_{vertex}_left_grasp",
                    f"cloth_{vertex}_right_grasp",
                ]
            )
        names.extend(self.TASK_OBSERVATION_NAMES)
        return tuple(names)

    # ------------------------------------------------------------------
    # Robot kinematics and collision
    # ------------------------------------------------------------------
    def arm_kinematics(
        self,
        arm_index: int,
        joint_angles: np.ndarray | None = None,
        opening: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return all rigid landmarks for one mirrored fixed-length 3R arm."""

        if arm_index not in (0, 1):
            raise ValueError("arm_index must be 0 (left) or 1 (right)")
        angles = self.joint_angles[arm_index] if joint_angles is None else np.asarray(joint_angles)
        aperture = self.gripper_openings[arm_index] if opening is None else float(opening)
        if angles.shape != (3,):
            raise ValueError("joint_angles must have shape (3,)")
        g = self.geometry
        side = 1.0 if arm_index == 0 else -1.0
        base = np.array([-side * g.shoulder_x, g.shoulder_y, g.shoulder_z], dtype=np.float64)
        base_heading = 0.0 if arm_index == 0 else np.pi
        heading = base_heading + float(angles[0])
        horizontal = np.array([np.cos(heading), np.sin(heading), 0.0])
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        elbow = base + g.upper_length * horizontal
        fore_pitch = float(angles[1])
        fore_direction = np.array(
            [
                np.cos(fore_pitch) * np.cos(heading),
                np.cos(fore_pitch) * np.sin(heading),
                np.sin(fore_pitch),
            ],
            dtype=np.float64,
        )
        wrist = elbow + g.fore_length * fore_direction
        hand_pitch = fore_pitch + float(angles[2])
        hand_direction = np.array(
            [
                np.cos(hand_pitch) * np.cos(heading),
                np.cos(hand_pitch) * np.sin(heading),
                np.sin(hand_pitch),
            ],
            dtype=np.float64,
        )
        normal = np.cross(hand_direction, lateral)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        palm = wrist + g.hand_length * hand_direction
        finger_bases = np.stack([palm + 0.5 * aperture * lateral, palm - 0.5 * aperture * lateral])
        finger_tips = finger_bases + g.finger_length * hand_direction
        pinch = np.mean(finger_tips, axis=0)
        frame = np.column_stack((hand_direction, lateral, normal))
        return {
            "base": base,
            "elbow": elbow,
            "wrist": wrist,
            "palm": palm,
            "finger_bases": finger_bases,
            "finger_tips": finger_tips,
            "pinch": pinch,
            "frame": frame,
            "opening": float(aperture),
        }

    def joint_positions(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for arm_index, name in enumerate(("left", "right")):
            kinematics = self.arm_kinematics(arm_index)
            result[name] = np.stack(
                [
                    kinematics["base"],
                    kinematics["elbow"],
                    kinematics["wrist"],
                    kinematics["palm"],
                ]
            )
        return result

    def _robot_segments(
        self,
        joint_angles: np.ndarray | None = None,
        openings: np.ndarray | None = None,
    ) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
        angles = self.joint_angles if joint_angles is None else np.asarray(joint_angles)
        apertures = self.gripper_openings if openings is None else np.asarray(openings)
        all_segments: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
        for arm_index in range(2):
            kin = self.arm_kinematics(arm_index, angles[arm_index], float(apertures[arm_index]))
            fingers_base = np.asarray(kin["finger_bases"])
            fingers_tip = np.asarray(kin["finger_tips"])
            all_segments.append(
                [
                    (np.asarray(kin["base"]), np.asarray(kin["elbow"]), self.geometry.arm_radius),
                    (np.asarray(kin["elbow"]), np.asarray(kin["wrist"]), self.geometry.arm_radius),
                    (np.asarray(kin["wrist"]), np.asarray(kin["palm"]), self.geometry.hand_radius),
                    (fingers_base[0], fingers_tip[0], self.geometry.finger_radius),
                    (fingers_base[1], fingers_tip[1], self.geometry.finger_radius),
                ]
            )
        return all_segments

    @staticmethod
    def _segment_distance(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
    ) -> float:
        """Shortest distance between two closed 3-D line segments."""

        u = first_end - first_start
        v = second_end - second_start
        w = first_start - second_start
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w))
        e = float(np.dot(v, w))
        denominator = a * c - b * b
        small = 1e-12
        if a <= small and c <= small:
            return float(np.linalg.norm(first_start - second_start))
        if a <= small:
            first_parameter = 0.0
            second_parameter = np.clip(e / c, 0.0, 1.0)
        elif c <= small:
            second_parameter = 0.0
            first_parameter = np.clip(-d / a, 0.0, 1.0)
        else:
            first_parameter = (
                0.0 if denominator <= small else np.clip((b * e - c * d) / denominator, 0.0, 1.0)
            )
            second_parameter = (b * first_parameter + e) / c
            if second_parameter < 0.0:
                second_parameter = 0.0
                first_parameter = np.clip(-d / a, 0.0, 1.0)
            elif second_parameter > 1.0:
                second_parameter = 1.0
                first_parameter = np.clip((b - d) / a, 0.0, 1.0)
        first_point = first_start + first_parameter * u
        second_point = second_start + second_parameter * v
        return float(np.linalg.norm(first_point - second_point))

    def _robot_pose_valid(self, angles: np.ndarray, openings: np.ndarray) -> bool:
        segments = self._robot_segments(angles, openings)
        g = self.geometry
        for arm_segments in segments:
            for start, end, radius in arm_segments:
                samples = np.stack((start, 0.5 * (start + end), end))
                over_table = (np.abs(samples[:, 0]) <= g.table_half_width) & (
                    np.abs(samples[:, 1]) <= g.table_half_depth
                )
                if np.any(over_table & (samples[:, 2] < g.table_height + radius - 1e-6)):
                    return False
        for first in segments[0]:
            for second in segments[1]:
                if self._segment_distance(first[0], first[1], second[0], second[1]) < (
                    first[2] + second[2] + 0.003
                ):
                    return False
        return True

    def _project_robot_target(
        self,
        start_angles: np.ndarray,
        target_angles: np.ndarray,
        start_openings: np.ndarray,
        target_openings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._robot_pose_valid(target_angles, target_openings):
            return target_angles, target_openings
        low, high = 0.0, 1.0
        for _ in range(22):
            middle = 0.5 * (low + high)
            candidate_angles = start_angles + middle * (target_angles - start_angles)
            candidate_openings = start_openings + middle * (target_openings - start_openings)
            if self._robot_pose_valid(candidate_angles, candidate_openings):
                low = middle
            else:
                high = middle
        return (
            start_angles + low * (target_angles - start_angles),
            start_openings + low * (target_openings - start_openings),
        )

    # ------------------------------------------------------------------
    # Cloth constraints, contacts, and grasps
    # ------------------------------------------------------------------
    def _grasp_mask(self, arm_index: int) -> np.ndarray:
        mask = np.zeros(self.vertex_count, dtype=bool)
        mask[self.grasped_vertices[arm_index]] = True
        return mask

    def _grasp_anchors(self, arm_index: int) -> np.ndarray:
        vertices = self.grasped_vertices[arm_index]
        if len(vertices) == 0:
            return np.empty((0, 3), dtype=np.float64)
        kin = self.arm_kinematics(arm_index)
        return (
            np.asarray(kin["pinch"])
            + self.grasp_local_offsets[arm_index] @ np.asarray(kin["frame"]).T
        )

    def _release_grasp(self, arm_index: int) -> None:
        self.grasped_vertices[arm_index] = np.empty(0, dtype=np.int64)
        self.grasp_local_offsets[arm_index] = np.empty((0, 3), dtype=np.float64)

    def _try_acquire_grasp(self, arm_index: int) -> None:
        if len(self.grasped_vertices[arm_index]) or (
            self.gripper_openings[arm_index] > self.GRASP_CAPTURE_OPENING
        ):
            return
        kin = self.arm_kinematics(arm_index)
        pinch = np.asarray(kin["pinch"])
        frame = np.asarray(kin["frame"])
        triangle_centres = np.mean(self.cloth_positions[self.faces], axis=1)
        local = (triangle_centres - pinch) @ frame
        # A short pinch prism between the two fingers.  Using a small triangular
        # patch avoids the unrealistic freely rotating single-particle grasp.
        aperture_margin = 0.5 * self.gripper_openings[arm_index] + self.geometry.towel_thickness
        eligible = (
            (np.abs(local[:, 0]) <= self.GRASP_CAPTURE_RADIUS)
            & (np.abs(local[:, 1]) <= aperture_margin)
            & (np.abs(local[:, 2]) <= 0.035)
        )
        if not np.any(eligible):
            return
        occupied = {int(index) for other in self.grasped_vertices for index in other}
        candidate_indices = np.flatnonzero(eligible)
        order = candidate_indices[np.argsort(np.linalg.norm(local[candidate_indices], axis=1))]
        chosen: np.ndarray | None = None
        for face_index in order:
            candidate = self.faces[int(face_index)]
            if not occupied.intersection(int(index) for index in candidate):
                chosen = candidate.copy()
                break
        if chosen is None:
            return
        offsets = (self.cloth_positions[chosen] - pinch) @ frame
        self.grasped_vertices[arm_index] = chosen.astype(np.int64)
        self.grasp_local_offsets[arm_index] = offsets

    def _update_grasps(self, gripper_commands: np.ndarray) -> None:
        for arm_index in range(2):
            if self.gripper_openings[arm_index] >= self.GRASP_RELEASE_OPENING:
                self._release_grasp(arm_index)
            elif gripper_commands[arm_index] < -0.05:
                self._try_acquire_grasp(arm_index)

    def _solve_distance_constraints(
        self,
        positions: np.ndarray,
        inverse_mass: np.ndarray,
        lambdas: np.ndarray,
        physics_dt: float,
        reverse: bool,
    ) -> None:
        batches: Iterable[np.ndarray] = (
            reversed(self.constraint_batches) if reverse else self.constraint_batches
        )
        for batch in batches:
            first = self.constraint_i[batch]
            second = self.constraint_j[batch]
            delta = positions[first] - positions[second]
            distance = np.linalg.norm(delta, axis=1)
            safe_distance = np.maximum(distance, 1e-12)
            direction = delta / safe_distance[:, None]
            constraint = distance - self.constraint_rest[batch]
            alpha = self.constraint_compliance[batch] / (physics_dt * physics_dt)
            denominator = inverse_mass[first] + inverse_mass[second] + alpha
            active = denominator > 1e-15
            delta_lambda = np.zeros_like(distance)
            delta_lambda[active] = (
                -constraint[active] - alpha[active] * lambdas[batch][active]
            ) / denominator[active]
            lambdas[batch] += delta_lambda
            correction = delta_lambda[:, None] * direction
            positions[first] += inverse_mass[first, None] * correction
            positions[second] -= inverse_mass[second, None] * correction

    def _project_table(self, positions: np.ndarray) -> np.ndarray:
        g = self.geometry
        inside = (np.abs(positions[:, 0]) <= g.table_half_width) & (
            np.abs(positions[:, 1]) <= g.table_half_depth
        )
        floor = g.table_height + 0.5 * g.towel_thickness
        contact = inside & (positions[:, 2] < floor)
        positions[contact, 2] = floor
        return contact

    def _apply_table_position_friction(
        self,
        positions: np.ndarray,
        previous: np.ndarray,
        pinned: np.ndarray,
    ) -> None:
        """Approximate Coulomb sticking/sliding at the cloth/table interface."""

        g = self.geometry
        floor = g.table_height + 0.5 * g.towel_thickness
        contact = (
            (np.abs(positions[:, 0]) <= g.table_half_width)
            & (np.abs(positions[:, 1]) <= g.table_half_depth)
            & (positions[:, 2] <= floor + 2e-6)
            & ~pinned
        )
        indices = np.flatnonzero(contact)
        if not len(indices):
            return
        displacement = positions[indices, :2] - previous[indices, :2]
        distance = np.linalg.norm(displacement, axis=1)
        sticking = distance <= self.STATIC_FRICTION_DISTANCE
        if np.any(sticking):
            positions[indices[sticking], :2] = previous[indices[sticking], :2]
        sliding = ~sticking & (distance > 1e-12)
        if np.any(sliding):
            correction = np.minimum(self.DYNAMIC_FRICTION_DISTANCE / distance[sliding], 1.0)
            positions[indices[sliding], :2] -= correction[:, None] * displacement[sliding]

    def _project_robot_capsules(self, positions: np.ndarray, pinned: np.ndarray) -> None:
        clearance = 0.55 * self.geometry.towel_thickness
        for arm_segments in self._robot_segments():
            for start, end, radius in arm_segments:
                segment = end - start
                length_squared = float(np.dot(segment, segment))
                if length_squared <= 1e-15:
                    closest = np.repeat(start[None, :], self.vertex_count, axis=0)
                else:
                    fraction = np.clip(
                        ((positions - start) @ segment) / length_squared,
                        0.0,
                        1.0,
                    )
                    closest = start + fraction[:, None] * segment
                offset = positions - closest
                distance = np.linalg.norm(offset, axis=1)
                minimum = radius + clearance
                colliding = (distance < minimum) & ~pinned
                if not np.any(colliding):
                    continue
                safe = np.maximum(distance[colliding], 1e-12)
                normal = offset[colliding] / safe[:, None]
                zero = distance[colliding] < 1e-12
                if np.any(zero):
                    normal[zero] = np.array([0.0, 0.0, 1.0])
                positions[colliding] = closest[colliding] + minimum * normal

    def _project_self_contact(
        self, positions: np.ndarray, previous: np.ndarray, pinned: np.ndarray
    ) -> None:
        """Two-sided vertex/triangle contact with topology-neighbour exclusion."""

        thickness = self.geometry.towel_thickness
        triangles = positions[self.faces]
        first = triangles[:, 0]
        edge_one = triangles[:, 1] - first
        edge_two = triangles[:, 2] - first
        normals = np.cross(edge_one, edge_two)
        normal_length = np.linalg.norm(normals, axis=1)
        valid_triangle = normal_length > 1e-10
        normals[valid_triangle] /= normal_length[valid_triangle, None]
        for vertex in range(self.vertex_count):
            if pinned[vertex]:
                continue
            valid = valid_triangle & ~self._self_contact_exclusion[vertex]
            if not np.any(valid):
                continue
            point = positions[vertex]
            signed = np.einsum("ij,ij->i", point - first, normals)
            projection = point - signed[:, None] * normals
            relative = projection - first
            dot00 = np.einsum("ij,ij->i", edge_one, edge_one)
            dot01 = np.einsum("ij,ij->i", edge_one, edge_two)
            dot11 = np.einsum("ij,ij->i", edge_two, edge_two)
            dot20 = np.einsum("ij,ij->i", relative, edge_one)
            dot21 = np.einsum("ij,ij->i", relative, edge_two)
            denominator = dot00 * dot11 - dot01 * dot01
            safe = np.abs(denominator) > 1e-14
            bary_one = np.zeros(len(self.faces))
            bary_two = np.zeros(len(self.faces))
            bary_one[safe] = (dot11[safe] * dot20[safe] - dot01[safe] * dot21[safe]) / denominator[
                safe
            ]
            bary_two[safe] = (dot00[safe] * dot21[safe] - dot01[safe] * dot20[safe]) / denominator[
                safe
            ]
            inside = (bary_one >= -1e-5) & (bary_two >= -1e-5) & (bary_one + bary_two <= 1.0 + 1e-5)
            previous_signed = np.einsum("ij,ij->i", previous[vertex] - first, normals)
            crossed = previous_signed * signed < 0.0
            contact = valid & safe & inside & ((np.abs(signed) < thickness) | crossed)
            if not np.any(contact):
                continue
            candidates = np.flatnonzero(contact)
            face_index = int(candidates[np.argmin(np.abs(signed[candidates]))])
            side = np.sign(previous_signed[face_index])
            if side == 0.0:
                side = np.sign(signed[face_index]) or 1.0
            correction = (side * thickness - signed[face_index]) * normals[face_index]
            positions[vertex] += correction

    def _pin_grasps(self, positions: np.ndarray) -> None:
        for arm_index in range(2):
            vertices = self.grasped_vertices[arm_index]
            if len(vertices):
                positions[vertices] = self._grasp_anchors(arm_index)

    def _cloth_substep(self, physics_dt: float) -> None:
        previous = self.cloth_positions.copy()
        damping = np.exp(-self.AIR_DAMPING * physics_dt)
        self.cloth_velocities *= damping
        self.cloth_velocities[:, 2] -= self.GRAVITY * physics_dt
        positions = previous + physics_dt * self.cloth_velocities

        pinned = self._grasp_mask(0) | self._grasp_mask(1)
        inverse_mass = np.ones(self.vertex_count, dtype=np.float64)
        inverse_mass[pinned] = 0.0
        lambdas = np.zeros(len(self.constraint_i), dtype=np.float64)
        contact = np.zeros(self.vertex_count, dtype=bool)
        for iteration in range(self.solver_iterations):
            self._solve_distance_constraints(
                positions,
                inverse_mass,
                lambdas,
                physics_dt,
                reverse=bool(iteration % 2),
            )
            contact |= self._project_table(positions)
            self._project_robot_capsules(positions, pinned)
            self._pin_grasps(positions)
        for _ in range(self.SELF_COLLISION_PASSES):
            self._project_self_contact(positions, previous, pinned)
            self._project_table(positions)
            self._pin_grasps(positions)
        self._apply_table_position_friction(positions, previous, pinned)
        self._project_table(positions)
        self._pin_grasps(positions)

        if not np.all(np.isfinite(positions)):
            raise FloatingPointError("cloth solver produced a non-finite position")
        velocities = (positions - previous) / physics_dt
        if np.any(contact):
            friction = np.exp(-self.TABLE_FRICTION * physics_dt)
            velocities[contact, :2] *= friction
            slow = contact & (np.linalg.norm(velocities[:, :2], axis=1) < 0.012)
            velocities[slow, :2] = 0.0
            velocities[contact, 2] = np.maximum(velocities[contact, 2], 0.0)
        self.cloth_positions = positions
        self.cloth_velocities = velocities

    # ------------------------------------------------------------------
    # Task metrics, Gym API, and observations
    # ------------------------------------------------------------------
    def cloth_metrics(self) -> dict[str, float]:
        structural_length = np.linalg.norm(
            self.cloth_positions[self.constraint_i[self._structural_mask]]
            - self.cloth_positions[self.constraint_j[self._structural_mask]],
            axis=1,
        )
        structural_rest = self.constraint_rest[self._structural_mask]
        strain_rms = float(np.sqrt(np.mean(((structural_length / structural_rest) - 1.0) ** 2)))

        centred = self.cloth_positions - np.mean(self.cloth_positions, axis=0)
        singular_values = np.linalg.svd(centred, full_matrices=False, compute_uv=False)
        plane_rms = float(singular_values[-1] / np.sqrt(self.vertex_count))
        planarity = float(np.exp(-((plane_rms / 0.026) ** 2)))

        grid = self.cloth_positions.reshape(self.mesh_rows, self.mesh_cols, 3)
        width_span = float(np.mean(np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1)))
        depth_span = float(np.mean(np.linalg.norm(grid[-1] - grid[0], axis=1)))
        coverage_error = 0.5 * (
            abs(width_span - self.geometry.towel_width) / self.geometry.towel_width
            + abs(depth_span - self.geometry.towel_depth) / self.geometry.towel_depth
        )
        coverage = float(np.exp(-((coverage_error / 0.13) ** 2)))
        strain_score = float(np.exp(-((strain_rms / 0.065) ** 2)))
        straightness = float(
            np.clip(0.42 * planarity + 0.38 * strain_score + 0.20 * coverage, 0.0, 1.0)
        )

        middle = self.mesh_rows // 2
        near_half = grid[:middle]
        far_half = grid[:middle:-1]
        pair_delta = far_half - near_half
        pair_xy = np.linalg.norm(pair_delta[:, :, :2], axis=2)
        alignment = float(np.exp(-((float(np.mean(pair_xy)) / 0.095) ** 2)))
        layer_error = float(
            np.mean(np.abs(np.abs(pair_delta[:, :, 2]) - 1.25 * self.geometry.towel_thickness))
        )
        layer_score = float(np.exp(-((layer_error / 0.028) ** 2)))
        width_score = float(np.exp(-((abs(width_span - self.geometry.towel_width) / 0.12) ** 2)))
        g = self.geometry
        supported = (
            (np.abs(self.cloth_positions[:, 0]) <= g.table_half_width)
            & (np.abs(self.cloth_positions[:, 1]) <= g.table_half_depth)
            & (self.cloth_positions[:, 2] <= g.table_height + 0.15)
        )
        support_score = float(np.mean(supported))
        fold_score = float(
            np.clip(
                alignment * (0.35 + 0.20 * layer_score + 0.25 * width_score + 0.20 * support_score),
                0.0,
                1.0,
            )
        )

        outside = (np.abs(self.cloth_positions[:, 0]) > g.table_half_width + 0.02) | (
            np.abs(self.cloth_positions[:, 1]) > g.table_half_depth + 0.02
        )
        dropped = outside & (self.cloth_positions[:, 2] < g.table_height - 0.06)
        dropped_fraction = float(np.mean(dropped))
        mean_speed = float(np.mean(np.linalg.norm(self.cloth_velocities, axis=1)))
        mean_height = float(np.mean(self.cloth_positions[:, 2] - g.table_height))
        if len(self.grasped_vertices[0]) and len(self.grasped_vertices[1]):
            left_material = np.mean(self.rest_positions[self.grasped_vertices[0]], axis=0)
            right_material = np.mean(self.rest_positions[self.grasped_vertices[1]], axis=0)
            material_span = float(np.linalg.norm(left_material - right_material))
            left_world = np.mean(self.cloth_positions[self.grasped_vertices[0]], axis=0)
            right_world = np.mean(self.cloth_positions[self.grasped_vertices[1]], axis=0)
            world_span = float(np.linalg.norm(left_world - right_world))
            span_ratio = world_span / max(material_span, 1e-6)
            # Two nearby pinches must not satisfy the straightening milestone;
            # the hands need to hold materially separated towel regions.
            separated_material = np.clip((material_span - 0.38) / 0.22, 0.0, 1.0)
            bimanual_tension = float(
                separated_material * np.clip((span_ratio - 0.72) / 0.20, 0.0, 1.0)
            )
        else:
            material_span = 0.0
            world_span = 0.0
            bimanual_tension = 0.0
        return {
            "straightness": straightness,
            "planarity": planarity,
            "coverage": coverage,
            "strain_rms": strain_rms,
            "plane_rms": plane_rms,
            "width_span": width_span,
            "depth_span": depth_span,
            "fold_alignment": alignment,
            "layer_score": layer_score,
            "fold_score": fold_score,
            "support_score": support_score,
            "dropped_fraction": dropped_fraction,
            "mean_speed": mean_speed,
            "mean_height": mean_height,
            "grasp_material_span": material_span,
            "grasp_world_span": world_span,
            "bimanual_tension": bimanual_tension,
        }

    def _get_obs(self) -> np.ndarray:
        joint_mid = 0.5 * (self.joint_low + self.joint_high)
        joint_half = 0.5 * (self.joint_high - self.joint_low)
        normalized_joints = (self.joint_angles - joint_mid) / joint_half
        g = self.geometry
        normalized_opening = (
            2.0
            * (
                (self.gripper_openings - g.gripper_min_opening)
                / (g.gripper_max_opening - g.gripper_min_opening)
            )
            - 1.0
        )
        robot = np.column_stack((normalized_joints, normalized_opening)).ravel()

        position_scale = np.array([1.15, 0.90, 0.80])
        velocity_scale = np.array([1.5, 1.5, 1.5])
        normalized_positions = np.clip(self.cloth_positions / position_scale, -1.0, 1.0)
        normalized_velocities = np.clip(self.cloth_velocities / velocity_scale, -1.0, 1.0)
        left_grasp = self._grasp_mask(0).astype(np.float64)
        right_grasp = self._grasp_mask(1).astype(np.float64)
        cloth = np.column_stack(
            (normalized_positions, normalized_velocities, left_grasp, right_grasp)
        ).ravel()

        metrics = self.cloth_metrics()
        if self.horizon is None:
            elapsed_fraction = float(np.tanh(self.elapsed_steps / 500.0))
        else:
            elapsed_fraction = 2.0 * min(self.elapsed_steps / self.horizon, 1.0) - 1.0
        task = np.array(
            [
                metrics["straightness"],
                metrics["fold_score"],
                1.0 if self.straightened_once else -1.0,
                np.clip(self.straight_streak / 5.0, 0.0, 1.0),
                metrics["bimanual_tension"],
                metrics["dropped_fraction"],
                np.clip(metrics["mean_speed"] / 1.5, 0.0, 1.0),
                elapsed_fraction,
            ],
            dtype=np.float64,
        )
        observation = np.concatenate((robot, cloth, task))
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _get_info(self, metrics: dict[str, float] | None = None) -> dict[str, Any]:
        current = self.cloth_metrics() if metrics is None else metrics
        elapsed_time = self.elapsed_steps * self.dt
        time_remaining = (
            None
            if self.horizon is None
            else max(0.0, (self.horizon - self.elapsed_steps) * self.dt)
        )
        is_success = bool(
            self.straightened_once
            and current["fold_score"] >= self.FOLD_SUCCESS_THRESHOLD
            and current["mean_height"] <= 0.13
            and current["mean_speed"] <= 0.20
            and current["dropped_fraction"] == 0.0
        )
        return {
            **current,
            "stage": "fold" if self.straightened_once else "straighten",
            "straightened_once": bool(self.straightened_once),
            "straight_streak": int(self.straight_streak),
            "is_success": is_success,
            "joint_angles": self.joint_angles.copy(),
            "gripper_openings": self.gripper_openings.copy(),
            "grasped_vertices": tuple(vertices.copy() for vertices in self.grasped_vertices),
            "elapsed_steps": int(self.elapsed_steps),
            "elapsed_time": float(elapsed_time),
            "time_remaining": time_remaining,
            "termination_reason": self._last_termination_reason,
            "physics_model": self.PHYSICS_MODEL,
        }

    def _randomize_cloth(self, wrinkle_amplitude: float) -> None:
        g = self.geometry
        # Start from an approximately inextensible accordion instead of adding
        # independent particle noise.  The projected depth shrinks as the
        # material rises, while neighbouring arc lengths stay close to rest.
        u = np.linspace(-1.0, 1.0, self.mesh_cols)
        v = np.linspace(-1.0, 1.0, self.mesh_rows)
        phase = float(self.np_random.uniform(-0.35, 0.35))
        wave = np.sin(np.pi * (v + 1.0) + phase)
        envelope = np.maximum(0.0, 1.0 - v * v)
        row_height = wrinkle_amplitude * (0.20 + 0.80 * wave * wave) * envelope
        row_height -= np.min(row_height)
        rest_dy = g.towel_depth / (self.mesh_rows - 1)
        delta_z = np.diff(row_height)
        delta_y = np.sqrt(np.maximum(rest_dy * rest_dy - delta_z * delta_z, 0.20 * rest_dy**2))
        y = np.concatenate(([0.0], np.cumsum(delta_y)))
        y -= 0.5 * (y[0] + y[-1])
        x = 0.5 * g.towel_width * u
        xx, yy = np.meshgrid(x, y)
        zz = np.repeat(row_height[:, None], self.mesh_cols, axis=1)
        transverse = 0.10 * wrinkle_amplitude * np.sin(2.0 * np.pi * u)[None, :] * envelope[:, None]
        zz += transverse
        zz -= np.min(zz)
        positions = np.stack((xx, yy, zz + 0.5 * g.towel_thickness), axis=2).reshape(-1, 3)
        yaw = float(self.np_random.uniform(-0.22, 0.22))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        positions[:, :2] = positions[:, :2] @ rotation.T
        positions[:, :2] += self.np_random.uniform([-0.055, -0.045], [0.055, 0.045])
        self.cloth_positions = positions
        self.cloth_velocities = np.zeros_like(positions)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        self.joint_angles = self.default_joint_angles.copy()
        if "joint_angles" in options:
            candidate = np.asarray(options["joint_angles"], dtype=np.float64)
            if candidate.shape == (6,):
                candidate = candidate.reshape(2, 3)
            if candidate.shape != (2, 3) or not np.all(np.isfinite(candidate)):
                raise ValueError("joint_angles must be a finite array with shape (2, 3) or (6,)")
            if np.any(candidate < self.joint_low) or np.any(candidate > self.joint_high):
                raise ValueError("joint_angles must satisfy the mechanical limits")
            self.joint_angles = candidate.copy()

        self.gripper_openings = np.full(2, self.geometry.gripper_max_opening)
        if "gripper_openings" in options:
            openings = np.asarray(options["gripper_openings"], dtype=np.float64)
            if openings.shape != (2,) or not np.all(np.isfinite(openings)):
                raise ValueError("gripper_openings must be a finite array with shape (2,)")
            if np.any(openings < self.geometry.gripper_min_opening) or np.any(
                openings > self.geometry.gripper_max_opening
            ):
                raise ValueError("gripper_openings are outside the mechanical limits")
            self.gripper_openings = openings.copy()
        if not self._robot_pose_valid(self.joint_angles, self.gripper_openings):
            raise ValueError("robot configuration collides with the table or other arm")

        wrinkle_amplitude = float(
            options.get("wrinkle_amplitude", self.np_random.uniform(0.055, 0.095))
        )
        if not np.isfinite(wrinkle_amplitude) or not 0.0 <= wrinkle_amplitude <= 0.12:
            raise ValueError("wrinkle_amplitude must be between 0 and 0.12 metres")
        if "cloth_positions" in options:
            positions = np.asarray(options["cloth_positions"], dtype=np.float64)
            if positions.shape != (self.vertex_count, 3) or not np.all(np.isfinite(positions)):
                raise ValueError(
                    f"cloth_positions must be finite with shape ({self.vertex_count}, 3)"
                )
            self.cloth_positions = positions.copy()
            velocities = np.asarray(
                options.get("cloth_velocities", np.zeros_like(positions)), dtype=np.float64
            )
            if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
                raise ValueError("cloth_velocities must match cloth_positions and be finite")
            self.cloth_velocities = velocities.copy()
        else:
            self._randomize_cloth(wrinkle_amplitude)

        self.grasped_vertices = [np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)]
        self.grasp_local_offsets = [
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        ]
        settle_steps = int(options.get("settle_steps", 8))
        if settle_steps < 0 or settle_steps > 200:
            raise ValueError("settle_steps must be between 0 and 200")
        settle_dt = min(self.MAX_PHYSICS_STEP, self.dt / max(1, settle_steps))
        for _ in range(settle_steps):
            self._cloth_substep(settle_dt)
        self.cloth_velocities.fill(0.0)

        self.elapsed_steps = 0
        self.last_action = np.zeros(8, dtype=np.float32)
        self.straightened_once = False
        self.straight_streak = 0
        self._episode_done = False
        self._last_termination_reason = None
        self._previous_metrics = self.cloth_metrics()
        observation = self._get_obs()
        return observation, self._get_info(self._previous_metrics)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("step() called after the episode ended; call reset()")
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (8,):
            raise ValueError("action must have shape (8,)")
        command = np.nan_to_num(command, nan=0.0, posinf=1.0, neginf=-1.0)
        command = np.clip(command, -1.0, 1.0)
        grouped = command.reshape(2, 4)

        start_angles = self.joint_angles.copy()
        start_openings = self.gripper_openings.copy()
        target_angles = np.clip(
            start_angles + self.dt * self.max_joint_speeds * grouped[:, :3],
            self.joint_low,
            self.joint_high,
        )
        target_openings = np.clip(
            start_openings + self.dt * self.max_gripper_speed * grouped[:, 3],
            self.geometry.gripper_min_opening,
            self.geometry.gripper_max_opening,
        )
        target_angles, target_openings = self._project_robot_target(
            start_angles, target_angles, start_openings, target_openings
        )

        substeps = max(1, int(np.ceil(self.dt / self.MAX_PHYSICS_STEP)))
        physics_dt = self.dt / substeps
        for substep in range(substeps):
            fraction = (substep + 1) / substeps
            self.joint_angles = start_angles + fraction * (target_angles - start_angles)
            self.gripper_openings = start_openings + fraction * (target_openings - start_openings)
            self._update_grasps(grouped[:, 3])
            self._cloth_substep(physics_dt)
        self.joint_angles = target_angles
        self.gripper_openings = target_openings
        self.last_action = command.astype(np.float32)
        self.elapsed_steps += 1

        previous = self._previous_metrics
        metrics = self.cloth_metrics()
        reached_straightness = (
            metrics["straightness"] >= self.STRAIGHTNESS_THRESHOLD
            and metrics["strain_rms"] <= 0.09
            and metrics["bimanual_tension"] >= 0.65
            and metrics["dropped_fraction"] == 0.0
        )
        if reached_straightness:
            self.straight_streak += 1
        else:
            self.straight_streak = 0
        just_straightened = bool(self.straight_streak >= 5 and not self.straightened_once)
        if self.straight_streak >= 5:
            self.straightened_once = True

        effort = float(np.mean(command * command))
        if self.straightened_once:
            reward = 7.0 * (metrics["fold_score"] - previous["fold_score"])
            reward += 0.6 * (metrics["straightness"] - previous["straightness"])
        else:
            reward = 5.0 * (metrics["straightness"] - previous["straightness"])
            reward += 0.8 * (metrics["coverage"] - previous["coverage"])
            reward += 0.8 * (metrics["bimanual_tension"] - previous["bimanual_tension"])
        reward -= 0.004 * effort
        reward -= 4.0 * max(0.0, metrics["dropped_fraction"] - previous["dropped_fraction"])
        if just_straightened:
            reward += 1.5

        info = self._get_info(metrics)
        terminated = False
        truncated = False
        if info["is_success"]:
            terminated = True
            reward += 10.0
            self._last_termination_reason = "success"
        elif metrics["dropped_fraction"] >= 0.20 or np.min(self.cloth_positions[:, 2]) < -0.45:
            terminated = True
            reward -= 8.0
            self._last_termination_reason = "towel_dropped"
        elif self.horizon is not None and self.elapsed_steps >= self.horizon:
            truncated = True
            self._last_termination_reason = "time_limit"
        self._episode_done = bool(terminated or truncated)
        self._previous_metrics = metrics
        info = self._get_info(metrics)
        observation = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Rendering contract
    # ------------------------------------------------------------------
    def set_camera(
        self,
        *,
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        preset: str | None = None,
    ) -> None:
        presets = {
            "perspective": (-52.0, 32.0, 2.10),
            "top": (-90.0, 88.0, 2.20),
            "front": (-90.0, 14.0, 2.25),
            "side": (0.0, 18.0, 2.25),
        }
        if preset is not None:
            if preset not in presets:
                raise ValueError(f"unknown camera preset {preset!r}")
            azimuth, elevation, distance = presets[preset]
        if azimuth is not None:
            if not np.isfinite(azimuth):
                raise ValueError("camera azimuth must be finite")
            self.camera_azimuth = float(azimuth)
        if elevation is not None:
            if not np.isfinite(elevation) or not 5.0 <= elevation <= 89.0:
                raise ValueError("camera elevation must be between 5 and 89 degrees")
            self.camera_elevation = float(elevation)
        if distance is not None:
            if not np.isfinite(distance) or not 1.2 <= distance <= 4.0:
                raise ValueError("camera distance must be between 1.2 and 4.0 metres")
            self.camera_distance = float(distance)

    def render_snapshot(self) -> dict[str, Any]:
        arms: dict[str, Any] = {}
        for arm_index, name in enumerate(("left", "right")):
            kin = self.arm_kinematics(arm_index)
            arms[name] = {
                "joints_m": np.stack(
                    [kin["base"], kin["elbow"], kin["wrist"], kin["palm"]]
                ).tolist(),
                "finger_bases_m": np.asarray(kin["finger_bases"]).tolist(),
                "finger_tips_m": np.asarray(kin["finger_tips"]).tolist(),
                "pinch_m": np.asarray(kin["pinch"]).tolist(),
                "opening_m": float(self.gripper_openings[arm_index]),
                "grasped_vertices": self.grasped_vertices[arm_index].tolist(),
            }
        metrics = self.cloth_metrics()
        return {
            "schema_version": 1,
            "physics_model": self.PHYSICS_MODEL,
            "geometry": {
                "table": {
                    "half_width_m": self.geometry.table_half_width,
                    "half_depth_m": self.geometry.table_half_depth,
                    "height_m": self.geometry.table_height,
                    "thickness_m": self.geometry.table_thickness,
                },
                "towel": {
                    "width_m": self.geometry.towel_width,
                    "depth_m": self.geometry.towel_depth,
                    "thickness_m": self.geometry.towel_thickness,
                    "mesh_rows": self.mesh_rows,
                    "mesh_cols": self.mesh_cols,
                    "faces": self.faces.tolist(),
                    "crease_row": self.mesh_rows // 2,
                },
                "robot": {
                    "upper_length_m": self.geometry.upper_length,
                    "fore_length_m": self.geometry.fore_length,
                    "hand_length_m": self.geometry.hand_length,
                    "finger_length_m": self.geometry.finger_length,
                },
            },
            "state": {
                "step": int(self.elapsed_steps),
                "elapsed_time_s": float(self.elapsed_steps * self.dt),
                "horizon": self.horizon,
                "joint_angles_rad": self.joint_angles.tolist(),
                "gripper_openings_m": self.gripper_openings.tolist(),
                "cloth_vertices_m": self.cloth_positions.tolist(),
                "arms": arms,
                "stage": "fold" if self.straightened_once else "straighten",
                "metrics": {key: float(value) for key, value in metrics.items()},
                "termination_reason": self._last_termination_reason,
            },
            "camera": {
                "azimuth_deg": float(self.camera_azimuth),
                "elevation_deg": float(self.camera_elevation),
                "distance_m": float(self.camera_distance),
                "target_m": [0.0, 0.0, 0.18],
            },
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        from .laundry_folding_rendering import render_frame

        frame = render_frame(self, width=self.width, height=self.height)
        if self.render_mode == "rgb_array":
            return frame
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - optional desktop path
            raise RuntimeError(
                "Human rendering needs pygame. Install with: pip install 'kaist-rl-lab[human]'"
            ) from exc
        if self._pygame is None:  # pragma: no cover - optional desktop path
            pygame.init()
            pygame.display.init()
            self._pygame = pygame
            self._window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("KAIST RL Lab — Laundry Folding")
            self._clock = pygame.time.Clock()
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return None
        surface = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surface, (0, 0))
        self._pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self) -> None:
        if self._pygame is not None:  # pragma: no cover - optional desktop path
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._window = None
        self._clock = None
