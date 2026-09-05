import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_demonstrations import (
    TrajectoryCollector,
    load_behavior_cloning_data,
    read_demonstration,
)
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession, build_app


@pytest.fixture
def recorded_session():
    session = InteractiveSession(seed=7001, target_ml=700)
    session.set_motor(0, 1)
    session.advance_batch()
    yield session
    session.close()


def test_batch_preserves_every_transition_and_stops_at_episode_end():
    batched = InteractiveSession(seed=7001, target_ml=700)
    reference = InteractiveSession(seed=7001, target_ml=700)
    batched.motors[:] = reference.motors[:] = [1, 0, 0, 0, -1, 0]
    assert batched.advance_batch() == 4
    for _ in range(4):
        reference.advance()
    np.testing.assert_array_equal(batched.observation, reference.observation)
    assert len(batched.trajectory) == len(reference.trajectory) == 4
    for actual, expected in zip(batched.trajectory, reference.trajectory):
        for key in ("observation", "action", "next_observation"):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual["reward"] == expected["reward"]
    finite = InteractiveSession(seed=7001, target_ml=700, horizon=2)
    assert finite.advance_batch() == 2
    assert len(finite.trajectory) == 2
    assert finite.trajectory[-1]["truncated"]
    for session in (batched, reference, finite):
        session.close()


def test_saved_attempt_is_frozen_retryable_and_ready_for_behavior_cloning(recorded_session):
    session = recorded_session
    path = session.save_demonstration("student-7")
    original = path.read_bytes()
    arrays, metadata = read_demonstration(original)
    assert arrays["observations"].shape == (4, 16)
    assert arrays["actions"].shape == (4, 6)
    assert arrays["truncated"].tolist() == [False, False, False, True]
    assert metadata["dt"] == 0.125
    assert metadata["physics_substep"] == 1 / 64
    assert metadata["participant"] == "student-7"
    assert not session.running
    assert not session.advance()
    assert session.save_demonstration("another-code").read_bytes() == original


def test_empty_attempt_does_not_finish():
    session = InteractiveSession(seed=7001, target_ml=700, start_paused=True)
    with pytest.raises(ValueError, match="No trajectory"):
        session.save_demonstration()
    assert session.running and session.paused
    session.close()


def test_collector_authentication_duplicates_and_dataset_loading(recorded_session, tmp_path):
    collector = TrajectoryCollector(tmp_path, "lecture-code-123")
    data = recorded_session.save_demonstration("student-7").read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    with pytest.raises(ValueError, match="Incorrect"):
        collector.receive("wrong", encoded)
    assert not list(tmp_path.iterdir())
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(
            pool.map(lambda _: collector.receive("lecture-code-123", encoded), range(8))
        )
    assert sum(not r["duplicate"] for r in receipts) == 1
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert collector.summary()[0]["participant"] == "student-7"
    dataset = load_behavior_cloning_data(tmp_path)
    assert dataset["observations"].shape == (4, 16)
    np.testing.assert_array_equal(dataset["actions"], [[1, 0, 0, 0, 0, 0]] * 4)
    assert len(set(dataset["episode_ids"])) == 1


@pytest.mark.parametrize("corruption", ["nan", "action", "continuity", "timing", "flags", "object"])
def test_collector_rejects_invalid_training_data(recorded_session, tmp_path, corruption):
    arrays, metadata = read_demonstration(recorded_session.save_demonstration().read_bytes())
    if corruption == "nan":
        arrays["observations"][0, 0] = np.nan
    elif corruption == "action":
        arrays["actions"][0, 0] = 2
    elif corruption == "continuity":
        arrays["observations"][1, 0] += 1
    elif corruption == "timing":
        metadata["dt"] = 0.5
    elif corruption == "flags":
        arrays["truncated"][:] = False
    else:
        arrays["actions"] = np.asarray([[object()] * 6] * 4, dtype=object)
    stream = BytesIO()
    np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata)))
    collector = TrajectoryCollector(tmp_path, "lecture-code-123")
    with pytest.raises(ValueError):
        collector.receive("lecture-code-123", base64.b64encode(stream.getvalue()).decode("ascii"))
    assert not list(tmp_path.iterdir())


def test_upload_failure_keeps_download_and_can_be_retried(monkeypatch):
    gr = pytest.importorskip("gradio")
    from kaist_rl_lab.apps import coffee_demonstrations

    attempts = []

    def upload(url, code, data):
        attempts.append(data)
        if len(attempts) == 1:
            raise OSError("offline")
        arrays, metadata = read_demonstration(data)
        return {"episode_id": metadata["episode_id"], "transitions": len(arrays["actions"])}

    monkeypatch.setattr(coffee_demonstrations, "submit_demonstration", upload)
    app = build_app(collector_url="https://example.gradio.live", lecture_code="lecture-code-123")
    handlers = {h.fn.__name__: h.fn for h in app.fns.values() if h.fn}
    session = InteractiveSession(seed=7001, target_ml=700, dt=1/32)
    session.advance_batch()
    encoded = base64.b64encode(session.save_demonstration("student-2").read_bytes()).decode()
    event = gr.EventData(None, {"archive": encoded})
    try:
        failed = json.loads(handlers["upload"](event))
        assert "not confirmed" in failed["submission"]
        assert not failed["confirmed"]
        retried = json.loads(handlers["upload"](event))
        assert "Submitted 4 transitions" in retried["submission"]
        assert retried["confirmed"]
        assert attempts[0] == attempts[1]
    finally:
        session.close()
        app.close()
