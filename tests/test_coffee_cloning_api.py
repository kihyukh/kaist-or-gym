"""Instructor-only cloning source boundaries, limits and nonblocking execution."""

import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from kaist_rl_lab.apps import coffee_cloning_api
from kaist_rl_lab.apps.coffee_browser_runtime import BrowserRuntime
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_web import create_app

ORIGIN = "https://coffee.example"
PASSWORD = "test-instructor-password-only"
ENDPOINT = "/api/instructor/cloning/train"


@pytest.fixture
def client(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("Student")
    (static / "instructor.html").write_text("Instructor")
    app = create_app(data_dir=tmp_path / "private", static_dir=static, build_assets=False,
                     password=PASSWORD, session_secret="a" * 32, public_base_url=ORIGIN)
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as value:
        yield value


@pytest.fixture
def archive():
    runtime = BrowserRuntime()
    try:
        runtime.session.set_motor(0, 1)
        runtime.session.toggle_pause()
        for _ in range(3):
            runtime.session.advance()
        original = runtime.session.save_demonstration("private-student-code").read_bytes()
    finally:
        runtime.session.close()

    def create(*, success=False, dt=1 / 32, target=0.7):
        arrays, metadata = read_demonstration(original)
        # Boundary tests vary the metadata accepted by the existing collector;
        # physics correctness and successful examples have their own rollout tests.
        metadata.update(episode_id=str(uuid4()), success=success, dt=dt, target_fill_l=target)
        arrays["observations"][:, 14] = target
        arrays["next_observations"][:, 14] = target
        output = BytesIO()
        np.savez_compressed(output, **arrays, metadata=np.asarray(json.dumps(metadata)))
        return output.getvalue()

    return create


def sign_in(client):
    assert client.post("/api/instructor/login", json={"password": PASSWORD}).status_code == 200


def make_class(client, name="Fall 2026"):
    response = client.post("/api/instructor/sessions", json={"name": name})
    assert response.status_code == 200
    session = response.json()
    return session["id"], parse_qs(urlsplit(session["join_url"]).query)["class"][0]


def upload(client, token, data):
    response = client.post("/api/submissions", params={"class": token}, content=data,
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200, response.text
    return response.json()


def mock_training(monkeypatch):
    received = []

    def fit(demonstrations):
        received.append(demonstrations)
        return {"kind": "test-policy", "metrics": {"samples": sum(
            len(arrays["actions"]) for arrays, _ in demonstrations)}}

    monkeypatch.setattr(coffee_cloning_api, "_fit_policy", fit)
    return received


def test_authentication_origin_and_examples_do_not_create_student_data(client, archive, monkeypatch):
    body = {"source": "examples"}
    assert client.post(ENDPOINT, json=body).status_code == 401
    sign_in(client)
    assert client.post(ENDPOINT, json=body, headers={"Origin": "https://other.example"}).status_code == 403
    received = mock_training(monkeypatch)
    monkeypatch.setattr(coffee_cloning_api, "_example_archives", lambda: [archive(success=True)])
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    assert result["source"] == "examples" and result["session_id"] is None
    assert result["selection"]["used_trajectories"] == 1
    assert len(received) == 1
    assert set(received[0][0][0]) == {"observations", "actions"}
    assert received[0][0][1] == {"dt": 1 / 32, "target_fill_l": 0.7}
    assert "private-student-code" not in response.text
    assert client.get("/api/instructor/sessions").json() == []
    with client.app.state.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0


def test_selected_class_and_success_filter_are_respected(client, archive, monkeypatch):
    sign_in(client)
    first_id, first_token = make_class(client, "Selected")
    _, other_token = make_class(client, "Other")
    upload(client, first_token, archive(success=True))
    upload(client, first_token, archive(success=False))
    upload(client, other_token, archive(success=True))
    received = mock_training(monkeypatch)
    first = client.post(ENDPOINT, json={"source": "students", "session_id": first_id})
    assert first.status_code == 200, first.text
    selection = first.json()["selection"]
    assert selection["available_trajectories"] == 2
    assert selection["eligible_trajectories"] == 1
    assert selection["used_trajectories"] == 1
    assert selection["skipped_unsuccessful"] == 1
    all_attempts = client.post(ENDPOINT, json={
        "source": "students", "session_id": first_id, "successful_only": False,
    })
    assert all_attempts.status_code == 200, all_attempts.text
    assert all_attempts.json()["selection"]["used_trajectories"] == 2
    assert [len(value) for value in received] == [1, 2]


@pytest.mark.parametrize("body", [
    {}, {"source": []}, {"source": "students"}, {"source": "students", "session_id": 42},
    {"source": "examples", "successful_only": "true"},
    {"source": "examples", "session_id": "a-student-class"},
])
def test_invalid_requests_fail_clearly(client, body):
    sign_in(client)
    assert client.post(ENDPOINT, json=body).status_code == 400


def test_empty_or_missing_class_and_no_successful_trajectories(client, archive):
    sign_in(client)
    assert client.post(ENDPOINT, json={"source": "students", "session_id": "missing"}).status_code == 404
    identifier, token = make_class(client)
    body = {"source": "students", "session_id": identifier}
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 409
    assert "no submitted trajectories" in response.json()["detail"]
    upload(client, token, archive(success=False))
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 409
    assert "no successful submitted trajectories" in response.json()["detail"]


def test_incompatible_and_invalid_archives_are_counted(client, archive, monkeypatch):
    sign_in(client)
    identifier, token = make_class(client)
    upload(client, token, archive(success=True, dt=0.125))
    upload(client, token, archive(success=True, target=0.5))
    broken = upload(client, token, archive(success=True))
    client.app.state.store.archive_path(broken["episode_id"]).write_bytes(b"broken")
    received = mock_training(monkeypatch)
    body = {"source": "students", "session_id": identifier}
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 409 and "32 Hz" in response.json()["detail"]
    assert received == []
    upload(client, token, archive(success=True))
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 200, response.text
    selection = response.json()["selection"]
    assert selection["used_trajectories"] == 1
    assert selection["skipped_invalid"] == 1
    assert selection["skipped_incompatible"] == 2


def test_selection_is_bounded_latest_first_and_keeps_float32_pairs(client, archive, monkeypatch):
    sign_in(client)
    identifier, token = make_class(client)
    records = [upload(client, token, archive(success=True)) for _ in range(4)]
    # Explicit timestamps make the selection deterministic without timing assumptions.
    with client.app.state.store.connect() as db:
        for index, record in enumerate(records):
            db.execute("UPDATE submissions SET received_at=? WHERE episode_id=?",
                       (f"2026-09-19T00:00:0{index}", record["episode_id"]))
    client.app.state.store.archive_path(records[0]["episode_id"]).write_bytes(b"old broken archive")
    monkeypatch.setattr(coffee_cloning_api, "MAX_TRAINING_TRAJECTORIES", 2)
    received = mock_training(monkeypatch)
    response = client.post(ENDPOINT, json={"source": "students", "session_id": identifier})
    assert response.status_code == 200, response.text
    selection = response.json()["selection"]
    assert selection["used_trajectories"] == 2
    assert selection["inspected_trajectories"] == 2
    assert selection["capped_trajectories"] == 2
    assert selection["skipped_invalid"] == 0
    assert selection["available_transitions"] == 6
    assert all(array.dtype == np.float32 for pairs, _ in received[0] for array in pairs.values())


def test_training_lock_does_not_block_health_and_is_released(client, archive, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(coffee_cloning_api, "_example_archives", lambda: [archive(success=True)])
    entered, release = Event(), Event()

    def blocking_fit(_):
        entered.set()
        assert release.wait(timeout=5)
        return {"kind": "test-policy"}

    monkeypatch.setattr(coffee_cloning_api, "_fit_policy", blocking_fit)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, ENDPOINT, json={"source": "examples"})
        try:
            assert entered.wait(timeout=5)
            assert client.get("/health").status_code == 200
            response = client.post(ENDPOINT, json={"source": "examples"})
            assert response.status_code == 429
            assert "Another policy" in response.json()["detail"]
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    assert client.post(ENDPOINT, json={"source": "examples"}).status_code == 200


def test_three_training_requests_per_minute(client, archive, monkeypatch):
    sign_in(client)
    monkeypatch.setattr(coffee_cloning_api, "_example_archives", lambda: [archive(success=True)])
    mock_training(monkeypatch)
    for _ in range(3):
        assert client.post(ENDPOINT, json={"source": "examples"}).status_code == 200
    limited = client.post(ENDPOINT, json={"source": "examples"})
    assert limited.status_code == 429 and limited.headers["retry-after"] == "60"


def test_prepared_examples_train_through_real_endpoint(client):
    sign_in(client)
    response = client.post(ENDPOINT, json={"source": "examples"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["selection"]["used_trajectories"] == 5
    assert result["selection"]["available_transitions"] > 4000
    model = result["model"]
    assert model["algorithm"] == "nearest_neighbor"
    assert model["metrics"]["demonstrations"] == 5
    assert model["metrics"]["validation_trajectories"] == 1
    assert len(model["states"]) == len(model["actions"]) > 4000
