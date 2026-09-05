"""Deterministic software renderer for the laundry-folding environment.

The simulator remains the only owner of scene state.  This module consumes a
JSON-safe :meth:`LaundryFoldingEnv.render_snapshot` and projects that state to
an RGB image with Pillow.  A small depth-sorted painter is sufficient for the
coarse educational cloth mesh and, unlike an OpenGL renderer, works unchanged
in headless test runners and notebook runtimes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

Color = tuple[int, int, int]
Point2D = tuple[float, float]


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=max(8, int(size)))
    except OSError:  # pragma: no cover - depends on the host font set
        return ImageFont.load_default()


def _mix(first: Color, second: Color, amount: float) -> Color:
    weight = float(np.clip(amount, 0.0, 1.0))
    return tuple(
        round((1.0 - weight) * left + weight * right)
        for left, right in zip(first, second, strict=True)
    )


def _scale_color(color: Color, scale: float) -> Color:
    return tuple(int(np.clip(round(channel * scale), 0, 255)) for channel in color)


def _unit(vector: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / length


def _points(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape (n, 3)")
    return result


@dataclass(frozen=True)
class _PerspectiveCamera:
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal_length: float
    centre_x: float
    centre_y: float

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        relative = values - self.position
        depth = relative @ self.forward
        safe_depth = np.maximum(depth, 0.04)
        projected = np.column_stack(
            (
                self.centre_x + self.focal_length * (relative @ self.right) / safe_depth,
                self.centre_y - self.focal_length * (relative @ self.up) / safe_depth,
            )
        )
        return projected, depth

    def radius_pixels(self, radius_metres: float, depth: float) -> int:
        safe_depth = max(float(depth), 0.10)
        return max(1, round(self.focal_length * radius_metres / safe_depth))


def _camera_from_snapshot(camera: dict[str, Any], width: int, height: int) -> _PerspectiveCamera:
    azimuth = np.deg2rad(float(camera["azimuth_deg"]))
    elevation = np.deg2rad(float(camera["elevation_deg"]))
    distance = float(camera["distance_m"])
    target = np.asarray(camera.get("target_m", [0.0, 0.0, 0.18]), dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("camera target must contain three finite coordinates")
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("camera distance must be finite and positive")

    offset = distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )
    position = target + offset
    forward = _unit(target - position, (0.0, 1.0, 0.0))
    right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])), (1.0, 0.0, 0.0))
    up = _unit(np.cross(right, forward), (0.0, 0.0, 1.0))
    vertical_fov = np.deg2rad(46.0)
    # Preserve the full workcell in unusually narrow render targets.  A
    # vertical-FOV-only camera would crop both shoulders in portrait frames.
    effective_height = min(float(height), float(width) / 1.55)
    focal_length = 0.5 * effective_height / np.tan(0.5 * vertical_fov)
    return _PerspectiveCamera(
        position=position,
        right=right,
        up=up,
        forward=forward,
        focal_length=float(focal_length),
        centre_x=0.50 * width,
        centre_y=0.52 * height,
    )


def _screen_points(points: np.ndarray, camera: _PerspectiveCamera) -> list[Point2D]:
    projected, _ = camera.project(points)
    return [(float(x), float(y)) for x, y in projected]


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Point2D,
    end: Point2D,
    *,
    fill: Color,
    width: int,
    dash: float = 7.0,
    gap: float = 5.0,
) -> None:
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    delta = second - first
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return
    direction = delta / length
    cursor = 0.0
    while cursor < length:
        finish = min(length, cursor + dash)
        a = first + cursor * direction
        b = first + finish * direction
        draw.line((tuple(a), tuple(b)), fill=fill, width=width)
        cursor += dash + gap


def _rounded_progress(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    value: float,
    threshold: float,
    color: Color,
) -> None:
    left, top, right, bottom = bounds
    radius = max(2, (bottom - top) // 2)
    draw.rounded_rectangle(bounds, radius=radius, fill=(224, 230, 232))
    amount = float(np.clip(value, 0.0, 1.0))
    filled_right = left + round(amount * (right - left))
    if filled_right > left:
        draw.rounded_rectangle(
            (left, top, max(left + 2 * radius, filled_right), bottom),
            radius=radius,
            fill=color,
        )
        if filled_right < left + 2 * radius:
            draw.rectangle((filled_right, top, left + 2 * radius, bottom), fill=(224, 230, 232))
    marker = left + round(float(np.clip(threshold, 0.0, 1.0)) * (right - left))
    draw.line((marker, top - 2, marker, bottom + 2), fill=(54, 68, 78), width=1)


def _draw_capsule(
    draw: ImageDraw.ImageDraw,
    start: Point2D,
    end: Point2D,
    radius: int,
    color: Color,
    outline: Color,
) -> None:
    core_width = max(2, 2 * radius)
    border_width = core_width + max(2, radius // 2)
    draw.line((start, end), fill=outline, width=border_width)
    draw.line((start, end), fill=color, width=core_width)
    for x, y in (start, end):
        draw.ellipse(
            (
                x - 0.5 * border_width,
                y - 0.5 * border_width,
                x + 0.5 * border_width,
                y + 0.5 * border_width,
            ),
            fill=outline,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )


def _draw_scene(
    snapshot: dict[str, Any],
    env: Any,
    *,
    width: int,
    height: int,
) -> Image.Image:
    sky_top: Color = (239, 243, 245)
    sky_bottom: Color = (250, 251, 251)
    image = Image.new("RGB", (width, height), sky_bottom)
    draw = ImageDraw.Draw(image)
    for row in range(height):
        fraction = row / max(1, height - 1)
        draw.line((0, row, width, row), fill=_mix(sky_top, sky_bottom, fraction))

    geometry = snapshot["geometry"]
    table = geometry["table"]
    towel_geometry = geometry["towel"]
    state = snapshot["state"]
    camera = _camera_from_snapshot(snapshot["camera"], width, height)

    table_z = float(table["height_m"])
    table_bottom = table_z - float(table["thickness_m"])
    half_width = float(table["half_width_m"])
    half_depth = float(table["half_depth_m"])
    top = np.array(
        [
            [-half_width, -half_depth, table_z],
            [half_width, -half_depth, table_z],
            [half_width, half_depth, table_z],
            [-half_width, half_depth, table_z],
        ],
        dtype=np.float64,
    )
    bottom = top.copy()
    bottom[:, 2] = table_bottom
    table_sides = [
        np.stack((top[index], top[(index + 1) % 4], bottom[(index + 1) % 4], bottom[index]))
        for index in range(4)
    ]
    for side_index, side in enumerate(table_sides):
        polygon = _screen_points(side, camera)
        fill = (190, 196, 196) if side_index % 2 == 0 else (181, 188, 189)
        draw.polygon(polygon, fill=fill, outline=(147, 157, 160))
    draw.polygon(
        _screen_points(top, camera),
        fill=(225, 225, 219),
        outline=(133, 146, 150),
        width=2,
    )

    # A measured depth grid makes height and perspective readable without
    # turning the scene into a decorative illustration.
    grid_color: Color = (201, 205, 202)
    for x_value in np.linspace(-half_width, half_width, 9):
        line = np.array(
            [[x_value, -half_depth, table_z + 0.001], [x_value, half_depth, table_z + 0.001]]
        )
        draw.line(_screen_points(line, camera), fill=grid_color, width=1)
    for y_value in np.linspace(-half_depth, half_depth, 7):
        line = np.array(
            [[-half_width, y_value, table_z + 0.001], [half_width, y_value, table_z + 0.001]]
        )
        draw.line(_screen_points(line, camera), fill=grid_color, width=1)

    towel_width = float(towel_geometry["width_m"])
    towel_depth = float(towel_geometry["depth_m"])
    target_z = table_z + 0.003
    folded_target = np.array(
        [
            [-0.5 * towel_width, -0.5 * towel_depth, target_z],
            [0.5 * towel_width, -0.5 * towel_depth, target_z],
            [0.5 * towel_width, 0.0, target_z],
            [-0.5 * towel_width, 0.0, target_z],
        ]
    )
    draw.polygon(
        _screen_points(folded_target, camera),
        fill=(211, 224, 222),
        outline=(93, 128, 130),
    )
    crease_world = np.array(
        [[-0.5 * towel_width, 0.0, target_z + 0.001], [0.5 * towel_width, 0.0, target_z + 0.001]]
    )
    crease_screen = _screen_points(crease_world, camera)
    _draw_dashed_line(
        draw,
        crease_screen[0],
        crease_screen[1],
        fill=(171, 128, 48),
        width=max(1, round(height / 420)),
    )

    vertices = _points(state["cloth_vertices_m"], name="cloth vertices")
    faces = np.asarray(towel_geometry["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("towel faces must have shape (n, 3)")
    if len(faces) and (int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices)):
        raise ValueError("towel faces contain an out-of-range vertex index")
    projected_vertices, vertex_depth = camera.project(vertices)

    # Soft projected shadows are based on the exact simulated mesh and rigid
    # links.  The small light-direction offset exposes vertices that lift from
    # the table, which is especially useful in a folding task.
    shadow_vertices = vertices.copy()
    elevation = np.maximum(0.0, vertices[:, 2] - table_z)
    shadow_vertices[:, 0] -= 0.17 * elevation
    shadow_vertices[:, 1] -= 0.10 * elevation
    shadow_vertices[:, 2] = table_z + 0.004
    projected_shadow, _ = camera.project(shadow_vertices)
    shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    for face in faces:
        shadow_draw.polygon(
            [tuple(projected_shadow[index]) for index in face],
            fill=(52, 65, 69, 25),
        )

    arm_colors: dict[str, Color] = {
        "left": (40, 117, 116),
        "right": (43, 79, 108),
    }
    arm_outline: Color = (52, 68, 78)
    arm_radius = float(getattr(env.geometry, "arm_radius", 0.026))
    hand_radius = float(getattr(env.geometry, "hand_radius", 0.017))
    finger_radius = float(getattr(env.geometry, "finger_radius", 0.007))
    arms = state["arms"]
    for arm_name in ("left", "right"):
        arm = arms[arm_name]
        joints = _points(arm["joints_m"], name=f"{arm_name} arm joints")
        finger_bases = _points(arm["finger_bases_m"], name=f"{arm_name} finger bases")
        finger_tips = _points(arm["finger_tips_m"], name=f"{arm_name} finger tips")
        shadow_points = np.vstack((joints, finger_bases, finger_tips))
        height_above = np.maximum(0.0, shadow_points[:, 2] - table_z)
        shadow_points[:, 0] -= 0.17 * height_above
        shadow_points[:, 1] -= 0.10 * height_above
        shadow_points[:, 2] = table_z + 0.005
        shadow_screen, shadow_depth = camera.project(shadow_points)
        shadow_pairs = (
            (0, 1, arm_radius),
            (1, 2, arm_radius),
            (2, 3, hand_radius),
            (4, 6, finger_radius),
            (5, 7, finger_radius),
        )
        for first, second, radius in shadow_pairs:
            mean_depth = 0.5 * (shadow_depth[first] + shadow_depth[second])
            line_width = 2 * camera.radius_pixels(radius * 1.25, mean_depth)
            shadow_draw.line(
                (tuple(shadow_screen[first]), tuple(shadow_screen[second])),
                fill=(52, 65, 69, 38),
                width=max(2, line_width),
            )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, height / 260.0)))
    image = Image.alpha_composite(image.convert("RGBA"), shadow_layer).convert("RGB")
    draw = ImageDraw.Draw(image)

    primitives: list[dict[str, Any]] = []
    light = _unit(np.array([-0.35, -0.45, 0.82]), (0.0, 0.0, 1.0))
    rows = int(towel_geometry["mesh_rows"])
    cols = int(towel_geometry["mesh_cols"])
    crease_row = int(towel_geometry["crease_row"])
    if rows * cols != len(vertices):
        raise ValueError("towel mesh dimensions do not match the vertex array")
    half_colors = ((84, 148, 158), (112, 168, 175))
    for face in faces:
        triangle = vertices[face]
        edge_one = triangle[1] - triangle[0]
        edge_two = triangle[2] - triangle[0]
        raw_normal = np.cross(edge_one, edge_two)
        normal = _unit(raw_normal, (0.0, 0.0, 1.0))
        centroid = np.mean(triangle, axis=0)
        to_camera = _unit(camera.position - centroid, (0.0, 0.0, 1.0))
        facing_normal = normal if float(np.dot(normal, to_camera)) >= 0.0 else -normal
        diffuse = max(0.0, float(np.dot(facing_normal, light)))
        slope_light = 0.08 * abs(float(facing_normal[2]))
        brightness = 0.58 + 0.34 * diffuse + slope_light
        material_row = float(np.mean(face // cols))
        half_index = 0 if material_row < crease_row else 1
        base = half_colors[half_index]
        if float(np.dot(normal, to_camera)) < 0.0:
            base = _scale_color(base, 0.80)
        fill = _scale_color(base, brightness)
        outline = _mix(fill, (42, 73, 79), 0.18)
        primitives.append(
            {
                "kind": "triangle",
                "depth": float(np.mean(vertex_depth[face])),
                "points": [tuple(projected_vertices[index]) for index in face],
                "fill": fill,
                "outline": outline,
            }
        )

    # Arm segments are inserted into the same depth queue as cloth triangles.
    # Segment widths are perspective-scaled; their endpoints always come from
    # rigid forward kinematics, so visual links cannot telescope.
    for arm_name in ("left", "right"):
        arm = arms[arm_name]
        joints = _points(arm["joints_m"], name=f"{arm_name} arm joints")
        bases = _points(arm["finger_bases_m"], name=f"{arm_name} finger bases")
        tips = _points(arm["finger_tips_m"], name=f"{arm_name} finger tips")
        joint_screen, joint_depth = camera.project(joints)
        base_screen, base_depth = camera.project(bases)
        tip_screen, tip_depth = camera.project(tips)
        color = arm_colors[arm_name]
        segment_specs = (
            (joint_screen[0], joint_screen[1], joint_depth[0], joint_depth[1], arm_radius),
            (joint_screen[1], joint_screen[2], joint_depth[1], joint_depth[2], arm_radius),
            (joint_screen[2], joint_screen[3], joint_depth[2], joint_depth[3], hand_radius),
            (base_screen[0], tip_screen[0], base_depth[0], tip_depth[0], finger_radius),
            (base_screen[1], tip_screen[1], base_depth[1], tip_depth[1], finger_radius),
        )
        for start, end, start_depth, end_depth, radius in segment_specs:
            depth = 0.5 * float(start_depth + end_depth)
            primitives.append(
                {
                    "kind": "capsule",
                    "depth": depth,
                    "start": tuple(start),
                    "end": tuple(end),
                    "radius": camera.radius_pixels(radius, depth),
                    "fill": color,
                    "outline": arm_outline,
                }
            )
        for joint_index, (point, depth) in enumerate(zip(joint_screen, joint_depth, strict=True)):
            radius_m = arm_radius * (1.32 if joint_index < 3 else 1.10)
            primitives.append(
                {
                    "kind": "joint",
                    "depth": float(depth) - 1e-5,
                    "point": tuple(point),
                    "radius": camera.radius_pixels(radius_m, float(depth)),
                    "fill": (245, 248, 248),
                    "outline": color,
                }
            )
        for point, depth in zip(tip_screen, tip_depth, strict=True):
            primitives.append(
                {
                    "kind": "joint",
                    "depth": float(depth) - 2e-5,
                    "point": tuple(point),
                    "radius": max(2, camera.radius_pixels(finger_radius * 1.45, float(depth))),
                    "fill": _mix(color, (245, 248, 248), 0.20),
                    "outline": arm_outline,
                }
            )

    # Material-coordinate cues remain attached to the deforming vertices.
    grid_vertices = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    material_lines: list[tuple[np.ndarray, Color, int]] = []
    boundary_color: Color = (41, 86, 92)
    crease_color: Color = (190, 137, 45)
    for indices in (
        grid_vertices[0, :],
        grid_vertices[-1, :],
        grid_vertices[:, 0],
        grid_vertices[:, -1],
    ):
        material_lines.append((indices, boundary_color, max(2, round(height / 300))))
    material_lines.append((grid_vertices[crease_row, :], crease_color, max(2, round(height / 260))))
    for indices, color, line_width in material_lines:
        for first, second in pairwise(indices):
            primitives.append(
                {
                    "kind": "line",
                    "depth": 0.5 * float(vertex_depth[first] + vertex_depth[second]) - 3e-5,
                    "start": tuple(projected_vertices[first]),
                    "end": tuple(projected_vertices[second]),
                    "fill": color,
                    "width": line_width,
                }
            )

    corner_indices = (
        int(grid_vertices[0, 0]),
        int(grid_vertices[0, -1]),
        int(grid_vertices[-1, -1]),
        int(grid_vertices[-1, 0]),
    )
    corner_colors: tuple[Color, ...] = (
        (39, 83, 112),
        (190, 137, 45),
        (166, 79, 63),
        (45, 124, 119),
    )
    for label, index, color in zip("ABCD", corner_indices, corner_colors, strict=True):
        primitives.append(
            {
                "kind": "corner",
                "depth": float(vertex_depth[index]) - 6e-5,
                "point": tuple(projected_vertices[index]),
                "radius": max(4, round(height / 105)),
                "fill": color,
                "label": label,
            }
        )

    for arm_name in ("left", "right"):
        color = arm_colors[arm_name]
        for index in arms[arm_name].get("grasped_vertices", []):
            vertex = int(index)
            if 0 <= vertex < len(vertices):
                primitives.append(
                    {
                        "kind": "grasp_vertex",
                        "depth": float(vertex_depth[vertex]) - 8e-5,
                        "point": tuple(projected_vertices[vertex]),
                        "radius": max(4, round(height / 92)),
                        "fill": color,
                    }
                )

    primitives.sort(key=lambda item: item["depth"], reverse=True)
    corner_font = _font(max(8, round(height / 62)), bold=True)
    for primitive in primitives:
        kind = primitive["kind"]
        if kind == "triangle":
            draw.polygon(
                primitive["points"],
                fill=primitive["fill"],
                outline=primitive["outline"],
                width=1,
            )
        elif kind == "capsule":
            _draw_capsule(
                draw,
                primitive["start"],
                primitive["end"],
                primitive["radius"],
                primitive["fill"],
                primitive["outline"],
            )
        elif kind == "joint":
            x, y = primitive["point"]
            radius = primitive["radius"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=primitive["fill"],
                outline=primitive["outline"],
                width=max(1, radius // 3),
            )
        elif kind == "line":
            draw.line(
                (primitive["start"], primitive["end"]),
                fill=primitive["fill"],
                width=primitive["width"],
            )
        elif kind == "corner":
            x, y = primitive["point"]
            radius = primitive["radius"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=primitive["fill"],
                outline=(249, 250, 250),
                width=max(1, radius // 3),
            )
            if radius >= 6:
                draw.text(
                    (x, y - 0.5),
                    primitive["label"],
                    font=corner_font,
                    fill=(255, 255, 255),
                    anchor="mm",
                )
        elif kind == "grasp_vertex":
            x, y = primitive["point"]
            radius = primitive["radius"]
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(244, 190, 65),
                width=max(2, radius // 3),
            )

    # Grasp rings sit at the physical pinch point and are shown only when a
    # cloth patch is actually latched by the environment.
    for arm_name in ("left", "right"):
        arm = arms[arm_name]
        if not arm.get("grasped_vertices"):
            continue
        pinch = np.asarray(arm["pinch_m"], dtype=np.float64)[None, :]
        pinch_screen, pinch_depth = camera.project(pinch)
        x, y = pinch_screen[0]
        radius = max(7, camera.radius_pixels(0.038, float(pinch_depth[0])))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(244, 190, 65),
            width=max(2, round(height / 250)),
        )

    return image


def _draw_metric_panel(
    image: Image.Image,
    snapshot: dict[str, Any],
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    draw = ImageDraw.Draw(image)
    state = snapshot["state"]
    metrics = state["metrics"]
    stage = str(state["stage"])
    straightness = float(np.clip(metrics["straightness"], 0.0, 1.0))
    fold_score = float(np.clip(metrics["fold_score"], 0.0, 1.0))
    straight_threshold = 0.88
    fold_threshold = 0.78

    width = right - left
    height = bottom - top
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(7, round(height * 0.12)),
        fill=(255, 255, 255),
        outline=(214, 222, 224),
        width=1,
    )
    stage_width = max(76, round(width * 0.205))
    metric_gap = max(8, round(width * 0.018))
    metric_width = (width - stage_width - 3 * metric_gap) // 2
    label_font = _font(max(9, min(13, round(height * 0.20))), bold=True)
    small_font = _font(max(8, min(11, round(height * 0.16))))
    value_font = _font(max(11, min(18, round(height * 0.27))), bold=True)

    stage_number = "1 / 2" if stage == "straighten" else "2 / 2"
    stage_label = "STRAIGHTEN" if stage == "straighten" else "FOLD"
    stage_color: Color = (43, 122, 120) if stage == "straighten" else (180, 130, 38)
    stage_x = left + metric_gap
    draw.text((stage_x, top + round(height * 0.18)), "STAGE", font=small_font, fill=(93, 108, 116))
    draw.text(
        (stage_x, top + round(height * 0.47)),
        stage_number,
        font=value_font,
        fill=(35, 52, 64),
        anchor="lm",
    )
    draw.text(
        (stage_x, bottom - round(height * 0.17)),
        stage_label,
        font=small_font,
        fill=stage_color,
        anchor="ls",
    )

    entries = (
        ("STRAIGHTNESS", straightness, straight_threshold, (43, 122, 120)),
        ("FOLD SCORE", fold_score, fold_threshold, (44, 85, 116)),
    )
    for metric_index, (label, value, threshold, color) in enumerate(entries):
        x = left + stage_width + metric_gap + metric_index * (metric_width + metric_gap)
        draw.text((x, top + round(height * 0.18)), label, font=label_font, fill=(66, 84, 94))
        draw.text(
            (x + metric_width, top + round(height * 0.18)),
            f"{100.0 * value:.0f}%",
            font=value_font,
            fill=(29, 44, 57),
            anchor="ra",
        )
        bar_top = top + round(height * 0.59)
        bar_bottom = min(bottom - round(height * 0.17), bar_top + max(6, round(height * 0.13)))
        _rounded_progress(
            draw,
            (x, bar_top, x + metric_width, bar_bottom),
            value,
            threshold,
            color,
        )
        if width >= 650:
            draw.text(
                (x + metric_width, bottom - round(height * 0.12)),
                f"target {100.0 * threshold:.0f}%",
                font=small_font,
                fill=(107, 120, 126),
                anchor="rs",
            )


def render_frame(env: Any, *, width: int, height: int) -> np.ndarray:
    """Render an ``H x W x 3`` RGB frame without changing environment state."""

    width = int(width)
    height = int(height)
    if width < 320 or height < 240:
        raise ValueError("render width and height must be at least 320 by 240")

    snapshot = env.render_snapshot()
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError("unsupported laundry-folding render snapshot")

    background: Color = (244, 247, 248)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    margin = max(7, round(min(width, height) * 0.018))
    header_height = max(40, min(68, round(height * 0.10)))
    footer_height = max(48, min(84, round(height * 0.13)))
    footer_gap = max(5, round(height * 0.012))
    scene_top = header_height
    footer_top = height - margin - footer_height
    scene_bottom = footer_top - footer_gap
    scene_left = margin
    scene_right = width - margin
    scene_width = max(1, scene_right - scene_left)
    scene_height = max(1, scene_bottom - scene_top)

    scene = _draw_scene(snapshot, env, width=scene_width, height=scene_height)
    corner_radius = max(7, round(min(width, height) * 0.018))
    scene_mask = Image.new("L", scene.size, 0)
    ImageDraw.Draw(scene_mask).rounded_rectangle(
        (0, 0, scene_width - 1, scene_height - 1),
        radius=corner_radius,
        fill=255,
    )
    image.paste(scene, (scene_left, scene_top), scene_mask)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (scene_left, scene_top, scene_right - 1, scene_bottom - 1),
        radius=corner_radius,
        outline=(207, 216, 219),
        width=1,
    )

    state = snapshot["state"]
    camera = snapshot["camera"]
    title_font = _font(max(15, min(24, round(height * 0.034))), bold=True)
    lab_font = _font(max(8, min(12, round(height * 0.017))), bold=True)
    detail_font = _font(max(8, min(12, round(height * 0.017))))
    lab_x = margin + 2
    draw.text(
        (lab_x, max(3, round(header_height * 0.16))),
        "KAIST RL LAB",
        font=lab_font,
        fill=(43, 122, 120),
    )
    title_y = max(16, round(header_height * 0.48))
    draw.text(
        (lab_x, title_y), "Bimanual towel folding", font=title_font, fill=(29, 43, 58), anchor="lm"
    )

    stage = str(state["stage"])
    stage_text = "STRAIGHTEN" if stage == "straighten" else "FOLD"
    stage_fill: Color = (224, 239, 237) if stage == "straighten" else (244, 232, 207)
    stage_text_color: Color = (35, 105, 103) if stage == "straighten" else (145, 98, 28)
    if width >= 540:
        title_width = draw.textlength("Bimanual towel folding", font=title_font)
        pill_left = min(width - margin - 154, int(lab_x + title_width + 18))
        pill_top = max(8, title_y - 12)
        pill_right = pill_left + max(74, round(width * 0.095))
        pill_bottom = pill_top + max(20, round(header_height * 0.36))
        draw.rounded_rectangle(
            (pill_left, pill_top, pill_right, pill_bottom),
            radius=(pill_bottom - pill_top) // 2,
            fill=stage_fill,
        )
        draw.text(
            ((pill_left + pill_right) / 2, (pill_top + pill_bottom) / 2),
            stage_text,
            font=lab_font,
            fill=stage_text_color,
            anchor="mm",
        )

    step = int(state["step"])
    horizon = state.get("horizon")
    step_text = f"STEP {step}" if horizon is None else f"STEP {step} / {int(horizon)}"
    time_text = f"{float(state['elapsed_time_s']):.1f} s"
    right_x = width - margin - 2
    draw.text(
        (right_x, max(5, round(header_height * 0.25))),
        step_text,
        font=lab_font,
        fill=(50, 73, 87),
        anchor="ra",
    )
    draw.text(
        (right_x, max(18, round(header_height * 0.62))),
        time_text,
        font=detail_font,
        fill=(96, 112, 120),
        anchor="ra",
    )

    # Compact camera and grasp readouts live inside the scene rather than
    # competing with the progress panel.
    if width >= 500 and scene_height >= 120:
        camera_label = (
            f"az {float(camera['azimuth_deg']):.0f}\N{DEGREE SIGN}  \u00b7  "
            f"el {float(camera['elevation_deg']):.0f}\N{DEGREE SIGN}"
        )
        label_padding = 7
        text_width = int(draw.textlength(camera_label, font=detail_font))
        label_right = scene_right - 8
        label_bottom = scene_bottom - 7
        draw.rounded_rectangle(
            (
                label_right - text_width - 2 * label_padding,
                label_bottom - 22,
                label_right,
                label_bottom,
            ),
            radius=6,
            fill=(248, 250, 250),
            outline=(210, 219, 221),
        )
        draw.text(
            (label_right - label_padding, label_bottom - 11),
            camera_label,
            font=detail_font,
            fill=(76, 94, 103),
            anchor="rm",
        )

        grasp_labels = []
        for short_name, arm_name in (("L", "left"), ("R", "right")):
            grasping = bool(state["arms"][arm_name].get("grasped_vertices"))
            grasp_labels.append(f"{short_name} {'GRASP' if grasping else 'OPEN'}")
        grasp_text = "   ".join(grasp_labels)
        grasp_left = scene_left + 8
        grasp_top = scene_top + 7
        grasp_width = int(draw.textlength(grasp_text, font=lab_font)) + 2 * label_padding
        draw.rounded_rectangle(
            (grasp_left, grasp_top, grasp_left + grasp_width, grasp_top + 22),
            radius=6,
            fill=(248, 250, 250),
            outline=(210, 219, 221),
        )
        draw.text(
            (grasp_left + label_padding, grasp_top + 11),
            grasp_text,
            font=lab_font,
            fill=(58, 79, 91),
            anchor="lm",
        )

    _draw_metric_panel(
        image,
        snapshot,
        left=margin,
        top=footer_top,
        right=width - margin,
        bottom=height - margin,
    )
    result = np.asarray(image, dtype=np.uint8)
    if result.shape != (height, width, 3):  # defensive guard for Pillow mode changes
        raise RuntimeError("laundry renderer produced an invalid RGB frame")
    return result.copy()


__all__ = ["render_frame"]
