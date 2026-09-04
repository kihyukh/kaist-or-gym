"""Raster renderer for :mod:`kaist_or_gym.envs.coffee_pouring`.

The renderer deliberately receives the environment itself so the Gymnasium
environment remains the sole owner of scene state and geometry.  Interactive
front ends can consume ``env.render_snapshot()`` for smooth interpolation;
``env.render()`` remains the canonical raster output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Color = tuple[int, int, int]


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:  # pragma: no cover - depends on local font installation
        return ImageFont.load_default()


def _rotated_points(
    center: np.ndarray, angle: float, points: Iterable[Sequence[float]]
) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray([center + rotation @ np.asarray(point) for point in points])


def _horizontal_span(points: np.ndarray, world_y: float) -> tuple[float, float]:
    """Exact horizontal intersection span of a polygon at ``world_y``."""

    intersections: list[float] = []
    for first, second in zip(points, np.roll(points, -1, axis=0), strict=True):
        delta_y = float(second[1] - first[1])
        if abs(delta_y) < 1e-12:
            if abs(float(first[1]) - world_y) < 1e-12:
                intersections.extend([float(first[0]), float(second[0])])
            continue
        fraction = (world_y - float(first[1])) / delta_y
        if -1e-12 <= fraction <= 1.0 + 1e-12:
            intersections.append(float(first[0] + fraction * (second[0] - first[0])))
    if len(intersections) < 2:
        centre = float(np.mean(points[:, 0]))
        return centre, centre
    return min(intersections), max(intersections)


def render_frame(env, *, width: int, height: int) -> np.ndarray:
    """Render the current environment state as an ``H x W x 3`` RGB array."""

    image = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(image)
    world_x = (-1.0, 1.0)
    header = 76
    footer = 50
    scale = min((width - 80) / (world_x[1] - world_x[0]), (height - header - footer) / 1.2)
    center_x = width / 2.0
    bottom = height - footer

    def xy(point: Sequence[float]) -> tuple[int, int]:
        return (
            round(center_x + float(point[0]) * scale),
            round(bottom - float(point[1]) * scale),
        )

    def polygon(points: np.ndarray, **kwargs) -> None:
        draw.polygon([xy(point) for point in points], **kwargs)

    navy: Color = (34, 74, 103)
    teal: Color = (43, 122, 120)
    dark: Color = (29, 43, 58)
    muted: Color = (82, 101, 111)
    joint_fill: Color = (250, 252, 252)
    coffee: Color = (112, 68, 43)
    amber: Color = (189, 139, 41)
    red: Color = (180, 82, 59)

    # Header and plotting field.
    draw.rectangle((0, 0, width, header), fill=(255, 255, 255))
    draw.text((32, 18), "KAIST OR GYM", font=_font(14, bold=True), fill=teal)
    draw.text((32, 37), "Two-arm coffee pouring", font=_font(25, bold=True), fill=dark)
    draw.text(
        (width - 32, 26),
        "fixed links · six revolute joints",
        font=_font(15, bold=True),
        fill=navy,
        anchor="ra",
    )

    # Light world grid and table.
    for x_value in np.linspace(-0.8, 0.8, 5):
        x_pixel, _ = xy((x_value, 0.0))
        draw.line((x_pixel, header, x_pixel, bottom), fill=(223, 229, 231), width=1)
    for y_value in np.linspace(0.2, 1.0, 5):
        _, y_pixel = xy((0.0, y_value))
        draw.line((center_x - scale, y_pixel, center_x + scale, y_pixel), fill=(223, 229, 231))
    table_y = xy((0.0, env.geometry.table_y))[1]
    draw.rectangle((0, table_y, width, bottom), fill=(218, 213, 202))
    draw.line((0, table_y, width, table_y), fill=(142, 151, 151), width=2)

    joints = env.joint_positions()

    def draw_arm(points: np.ndarray, color: Color, label: str) -> None:
        pixels = [xy(point) for point in points]
        draw.line(pixels, fill=(181, 193, 197), width=20, joint="curve")
        draw.line(pixels, fill=color, width=12, joint="curve")
        for point in pixels:
            radius = 10
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=joint_fill,
                outline=color,
                width=4,
            )
        draw.text(
            (pixels[0][0], pixels[0][1] + 17),
            label,
            font=_font(12, bold=True),
            fill=muted,
            anchor="ma",
        )

    draw_arm(joints["cup"], teal, "cup arm")
    draw_arm(joints["pot"], navy, "pot arm")

    tools = env.tool_positions()
    cup_center = np.asarray(tools["cup_center"])
    pot_center = np.asarray(tools["pot_center"])
    cup_angle = float(tools["cup_angle"])
    pot_angle = float(tools["pot_angle"])
    g = env.geometry

    # Cup body: an outlined trapezoid with world-horizontal coffee fill.
    cup_outer = _rotated_points(
        cup_center,
        cup_angle,
        [
            (-0.50 * g.cup_width, 0.50 * g.cup_height),
            (0.50 * g.cup_width, 0.50 * g.cup_height),
            (0.38 * g.cup_width, -0.50 * g.cup_height),
            (-0.38 * g.cup_width, -0.50 * g.cup_height),
        ],
    )
    polygon(cup_outer, fill=(250, 252, 252), outline=muted)
    cup_mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(cup_mask)
    mask_draw.polygon([xy(point) for point in cup_outer], fill=255)
    surface_world_y = env._cup_surface_world_y(tools)
    surface_pixel_y = int(np.clip(xy((0.0, surface_world_y))[1], 0, height))
    liquid_layer = Image.new("RGB", (width, height), coffee)
    liquid_mask = Image.new("L", (width, height), 0)
    liquid_draw = ImageDraw.Draw(liquid_mask)
    liquid_draw.rectangle((0, surface_pixel_y, width, height), fill=255)
    liquid_mask = Image.fromarray(
        np.minimum(np.asarray(cup_mask), np.asarray(liquid_mask)).astype(np.uint8)
    )
    image.paste(liquid_layer, mask=liquid_mask)
    draw = ImageDraw.Draw(image)
    draw.line([xy(point) for point in np.vstack([cup_outer, cup_outer[0]])], fill=muted, width=3)

    # Target fill line is intentionally world-horizontal, matching the liquid.
    target_y = env._cup_surface_world_y(tools, env.target_fill)
    target_left, target_right = _horizontal_span(cup_outer, target_y)
    target_y_px = xy((0.0, target_y))[1]
    draw.line(
        (xy((target_left, 0.0))[0], target_y_px, xy((target_right, 0.0))[0], target_y_px),
        fill=navy,
        width=2,
    )
    # Handles are sampled in vessel coordinates and rotate with the vessels.
    cup_handle_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 24)
    cup_handle_local = np.column_stack(
        [
            0.45 * g.cup_width + 0.34 * g.cup_width * np.cos(cup_handle_angles),
            0.25 * g.cup_height * np.sin(cup_handle_angles),
        ]
    )
    cup_handle = _rotated_points(cup_center, cup_angle, cup_handle_local)
    draw.line([xy(point) for point in cup_handle], fill=muted, width=4, joint="curve")

    # Pot body, spout, and handle rotate rigidly about its wrist attachment.
    pot_outer = _rotated_points(
        pot_center,
        pot_angle,
        [
            (-0.47 * g.pot_width, 0.48 * g.pot_height),
            (0.47 * g.pot_width, 0.48 * g.pot_height),
            (0.43 * g.pot_width, -0.48 * g.pot_height),
            (-0.43 * g.pot_width, -0.48 * g.pot_height),
        ],
    )
    polygon(pot_outer, fill=(121, 132, 135), outline=(57, 79, 88))
    spout_shape = _rotated_points(
        pot_center,
        pot_angle,
        [
            (-0.43 * g.pot_width, 0.34 * g.pot_height),
            (-0.78 * g.pot_width, 0.22 * g.pot_height),
            (-0.43 * g.pot_width, 0.06 * g.pot_height),
        ],
    )
    polygon(spout_shape, fill=(86, 102, 106), outline=(57, 79, 88))
    coffee_window = _rotated_points(
        pot_center,
        pot_angle,
        [
            (-0.43 * g.pot_width, 0.45 * g.pot_height),
            (0.43 * g.pot_width, 0.45 * g.pot_height),
            (0.43 * g.pot_width, -0.45 * g.pot_height),
            (-0.43 * g.pot_width, -0.45 * g.pot_height),
        ],
    )
    pot_mask = Image.new("L", (width, height), 0)
    pot_mask_draw = ImageDraw.Draw(pot_mask)
    pot_mask_draw.polygon([xy(point) for point in coffee_window], fill=255)
    pot_liquid_mask = Image.new("L", (width, height), 0)
    pot_liquid_draw = ImageDraw.Draw(pot_liquid_mask)
    pot_surface_pixel_y = xy((0.0, env._pot_surface_world_y(tools)))[1]
    pot_liquid_draw.rectangle((0, pot_surface_pixel_y, width, height), fill=255)
    pot_clipped_mask = Image.fromarray(
        np.minimum(np.asarray(pot_mask), np.asarray(pot_liquid_mask)).astype(np.uint8)
    )
    coffee_layer = Image.new("RGB", (width, height), coffee)
    image.paste(coffee_layer, mask=pot_clipped_mask)
    draw.line(
        [xy(point) for point in np.vstack([coffee_window, coffee_window[0]])],
        fill=(57, 79, 88),
        width=2,
    )
    pot_handle_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 24)
    pot_handle_local = np.column_stack(
        [
            0.43 * g.pot_width + 0.38 * g.pot_width * np.cos(pot_handle_angles),
            0.30 * g.pot_height * np.sin(pot_handle_angles),
        ]
    )
    pot_handle = _rotated_points(pot_center, pot_angle, pot_handle_local)
    draw.line(
        [xy(point) for point in pot_handle],
        fill=(57, 79, 88),
        width=7,
        joint="curve",
    )

    # The environment supplies the exact ballistic path used for capture.
    if env.last_flow_rate > 1e-5:
        stream_width = max(1, round(2.0 * env.last_jet_radius * scale))
        draw.line(
            [xy(point) for point in env.last_stream_path],
            fill=coffee,
            width=stream_width,
            joint="curve",
        )
    # This is an interval event rather than endpoint state.  Keeping its
    # actual substep path explains a puddle that began just before the endpoint
    # geometry became a clean capture.
    if env.last_direct_spill > 1e-8:
        direct_radius = env._jet_radius(env.last_direct_spill_rate, 0.35)
        direct_width = max(1, round(2.0 * direct_radius * scale))
        draw.line(
            [xy(point) for point in env.last_direct_spill_path],
            fill=coffee,
            width=direct_width,
            joint="curve",
        )
    if env.last_cup_runoff > 1e-8:
        runoff_radius = env._jet_radius(env.last_cup_runoff_rate, 0.35)
        runoff_width = max(1, round(2.0 * runoff_radius * scale))
        draw.line(
            [xy(point) for point in env.last_cup_runoff_path],
            fill=coffee,
            width=runoff_width,
            joint="curve",
        )
    if env.spill > 1e-8:
        puddle_center = xy((env.spill_impact_x, env.geometry.table_y))
        puddle_world_radius = float(np.clip(0.13 * np.sqrt(env.spill / 0.10), 0.005, 0.26))
        puddle_width = max(2, round(puddle_world_radius * scale))
        puddle_height = max(2, round(0.16 * puddle_width))
        draw.ellipse(
            (
                puddle_center[0] - puddle_width,
                puddle_center[1] - puddle_height,
                puddle_center[0] + puddle_width,
                puddle_center[1] + puddle_height,
            ),
            fill=(148, 100, 74),
        )

    # Time and outcome readout.
    draw.rounded_rectangle((25, height - 42, width - 25, height - 10), 8, fill=(255, 255, 255))
    draw.text(
        (40, height - 34),
        f"cup {env.fill * 1000:>4.0f}/{env.target_fill * 1000:>4.0f} mL · "
        f"pot {env.source_remaining * 1000:>4.0f} mL",
        font=_font(14, bold=True),
        fill=dark,
    )
    draw.text(
        (width // 2, height - 34),
        f"spill {env.spill * 1000:>4.0f} mL · pour {env.last_flow_rate * 1000:>3.0f} mL/s",
        font=_font(14, bold=True),
        fill=red if env.spill > 0.02 else muted,
        anchor="ma",
    )
    if env.horizon is None:
        time_text = f"step {env.elapsed_steps} · elapsed {env.elapsed_steps * env.dt:>4.1f} s"
    else:
        remaining = max(0, env.horizon - env.elapsed_steps) * env.dt
        time_text = f"step {env.elapsed_steps}/{env.horizon} · {remaining:>4.1f} s left"
    draw.text(
        (width - 40, height - 34),
        time_text,
        font=_font(14, bold=True),
        fill=amber,
        anchor="ra",
    )

    return np.asarray(image, dtype=np.uint8).copy()
