"""Configurable spacing preserves old recordings and exact replay geometry."""

import json
from dataclasses import asdict
from io import BytesIO
from itertools import product

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_classroom import (
    CUP_POSITION_JITTER,
    INITIAL_LAYOUT,
    POT_POSITION_JITTER,
    classroom_layout,
    fixed_policy_layout,
)
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession
from kaist_rl_lab.envs.coffee_pouring import ArmGeometry, CoffeePouringEnv


def _archive_with_metadata(data, transform):
    arrays, metadata = read_demonstration(data)
    transform(metadata)
    output = BytesIO()
    np.savez_compressed(output, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return output.getvalue()


def _record(distance):
    session = InteractiveSession(
        123, 700, dt=1 / 32, steps_per_update=1,
        reset_options=fixed_policy_layout(), arm_base_distance=distance,
    )
    snapshots = [session.env.render_snapshot()]
    try:
        for action in ([.5, -.2, -.3, -.2, .1, .1], [-.1, .2, -.1, .4, -.2, -.2]):
            for index, direction in enumerate(action):
                session.set_motor(index, direction)
            for _ in range(4):
                session.advance()
                snapshots.append(session.env.render_snapshot())
        return session.save_demonstration("spacing-test").read_bytes(), snapshots
    finally:
        session.close()


def test_default_remains_original_geometry_and_only_bases_move():
    legacy = CoffeePouringEnv()
    widened = CoffeePouringEnv(arm_base_distance=1.28)
    try:
        assert legacy.arm_base_distance == 1.16
        assert legacy.geometry == ArmGeometry()
        expected = asdict(legacy.geometry)
        expected.update(cup_base=(-.64, .10), pot_base=(.64, .10))
        assert asdict(widened.geometry) == expected
        for env in (legacy, widened):
            env.reset(seed=0, options={**fixed_policy_layout(), "target_fill": .7})
            arms = env.render_snapshot()["geometry"]["arms"]
            assert arms["cup"]["base_m"] == list(env.geometry.cup_base)
            assert arms["pot"]["base_m"] == list(env.geometry.pot_base)
    finally:
        legacy.close()
        widened.close()


@pytest.mark.parametrize("distance", [True, False, np.bool_(True), None, "1.28", [], {}])
def test_spacing_rejects_non_numbers(distance):
    with pytest.raises(TypeError, match="arm_base_distance"):
        CoffeePouringEnv(arm_base_distance=distance)


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -float("inf"), .7999, 1.4001])
def test_spacing_rejects_nonfinite_and_unsupported_distances(distance):
    with pytest.raises(ValueError, match="arm_base_distance"):
        CoffeePouringEnv(arm_base_distance=distance)


def test_session_restart_preserves_spacing_and_new_archive_records_it():
    session = InteractiveSession(
        0, 700, dt=1 / 32, reset_options=fixed_policy_layout(), arm_base_distance=1.28,
    )
    try:
        original = session.observation.copy()
        session.advance()
        session.restart(0, 700, speed=1, horizon=None)
        assert session.arm_base_distance == session.env.arm_base_distance == 1.28
        np.testing.assert_array_equal(session.observation, original)
        session.advance()
        _, metadata = read_demonstration(session.save_demonstration().read_bytes())
        assert metadata["arm_base_distance_m"] == 1.28
        assert metadata["schema_version"] == 2
    finally:
        session.close()


@pytest.mark.parametrize("distance", [None, True, "1.28", float("nan"), .79, 1.41])
def test_archive_rejects_invalid_explicit_spacing(distance):
    data, _ = _record(1.28)
    corrupted = _archive_with_metadata(data, lambda metadata: metadata.update(
        arm_base_distance_m=distance,
    ))
    with pytest.raises((ValueError, TypeError), match="arm_base_distance"):
        read_demonstration(corrupted)


def test_new_archive_schema_requires_explicit_spacing():
    data, _ = _record(1.28)
    missing_spacing = _archive_with_metadata(
        data, lambda metadata: metadata.pop("arm_base_distance_m"),
    )
    with pytest.raises(ValueError, match="schema 2 requires arm_base_distance_m"):
        read_demonstration(missing_spacing)


@pytest.mark.parametrize("distance,legacy", [(1.16, True), (1.16, False), (1.28, False)])
def test_recorded_geometry_and_all_render_frames_replay_exactly(distance, legacy):
    pytest.importorskip("fastapi")
    from kaist_rl_lab.apps.coffee_web import _replay

    data, snapshots = _record(distance)
    if legacy:
        data = _archive_with_metadata(data, lambda metadata: metadata.pop("arm_base_distance_m"))
    arrays, metadata = read_demonstration(data)
    assert ("arm_base_distance_m" not in metadata) == legacy
    assert metadata["schema_version"] == (1 if distance == 1.16 else 2)
    replay = _replay(data)
    assert len(replay["frames"]) == len(snapshots)
    for frame, expected in zip(replay["frames"], snapshots, strict=True):
        actual = frame["snapshot"].copy()
        actual.pop("playback")
        assert actual == expected
    assert replay["total_reward"] == np.sum(arrays["rewards"], dtype=np.float64)


@pytest.mark.parametrize("distance", [1.22, 1.28, 1.32])
def test_wider_classroom_layout_corners_and_seeded_starts_remain_valid(distance):
    layouts = [fixed_policy_layout(), *(classroom_layout(seed) for seed in range(32))]
    for offsets in product((-1, 1), repeat=4):
        layouts.append({
            "cup_center": [INITIAL_LAYOUT["cup_center"][axis]
                           + offsets[axis] * CUP_POSITION_JITTER[axis] for axis in range(2)],
            "pot_center": [INITIAL_LAYOUT["pot_center"][axis]
                           + offsets[axis + 2] * POT_POSITION_JITTER[axis] for axis in range(2)],
        })
    env = CoffeePouringEnv(arm_base_distance=distance, include_render_info=False)
    try:
        for layout in layouts:
            observation, _ = env.reset(seed=0, options={**layout, "target_fill": .7})
            assert np.isfinite(observation).all()
            assert not env._cross_robot_collision(env.joint_angles)
            for index, vessel in enumerate(("cup", "pot")):
                np.testing.assert_allclose(
                    env.tool_positions()[vessel + "_center"], layout[vessel + "_center"], atol=1e-12,
                )
                angles = env.joint_angles[index * 3:index * 3 + 3]
                assert abs(sum(angles)) < 1e-12
                assert env._arm_table_clearance(vessel, angles) >= 0
    finally:
        env.close()
