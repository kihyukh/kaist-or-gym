"""Classroom boundaries, persistence, retry receipts and actual-physics replay."""

import json
from io import BytesIO
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from kaist_rl_lab.apps.coffee_browser_runtime import BrowserRuntime
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession
from kaist_rl_lab.apps.coffee_web import COOKIE_NAME, create_app

PASSWORD = "test-instructor-password-only"
SECRET = "test-session-signing-secret-not-for-deployment"


@pytest.fixture
def setup(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("Coffee student demo")
    (static / "instructor.html").write_text("Instructor login")
    kwargs = {"data_dir": tmp_path / "private", "static_dir": static, "build_assets": False,
              "password": PASSWORD, "session_secret": SECRET,
              "public_base_url": "https://coffee.example"}
    app = create_app(**kwargs)
    with TestClient(app, base_url="https://coffee.example",
                    headers={"Origin": "https://coffee.example"}) as client:
        yield client, kwargs


def login(client):
    response = client.post("/api/instructor/login", json={"password": PASSWORD})
    assert response.status_code == 200
    return response


def new_class(client, participant_required=True):
    login(client)
    response = client.post("/api/instructor/sessions", json={
        "name": "Fall 2026", "participant_required": participant_required,
    })
    assert response.status_code == 200
    classroom = response.json()
    token = parse_qs(urlsplit(classroom["join_url"]).query)["class"][0]
    return classroom, token


def recording(participant="student-7", steps=3):
    runtime = BrowserRuntime()
    try:
        session = runtime.session
        session.set_motor(0, 1)
        session.toggle_pause()
        for _ in range(steps):
            session.advance()
        return session.save_demonstration(participant).read_bytes()
    finally:
        runtime.session.close()


def upload(client, token, data):
    return client.post("/api/submissions", params={"class": token}, content=data,
                       headers={"Content-Type": "application/octet-stream"})


def test_public_play_private_data_and_secure_login(setup):
    client, _ = setup
    assert client.get("/").text == "Coffee student demo"
    assert client.get("/instructor").text == "Instructor login"
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/instructor/sessions").status_code == 401
    assert client.post("/api/instructor/login", json={"password": "wrong"}).status_code == 401
    denied = client.post("/api/instructor/login", json={"password": PASSWORD},
                         headers={"Origin": "https://unrelated.example"})
    assert denied.status_code == 403
    saved_headers = client.headers.copy()
    del client.headers["Origin"]
    assert client.post("/api/instructor/login", json={"password": PASSWORD}).status_code == 403
    client.headers = saved_headers
    response = login(client)
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=strict" in cookie
    assert PASSWORD not in cookie
    assert client.get("/api/instructor/sessions").json() == []
    saved_cookie = client.cookies.get(COOKIE_NAME)
    client.post("/api/instructor/logout")
    client.cookies.set(COOKIE_NAME, saved_cookie, path="/api/instructor")
    assert client.get("/api/instructor/sessions").status_code == 401


def test_classroom_submission_retry_close_and_persistence(setup):
    client, kwargs = setup
    classroom, token = new_class(client)
    public = client.get("/api/session", params={"class": token}).json()
    assert public == {"id": classroom["id"], "name": "Fall 2026", "open": True,
                      "participant_required": True}
    assert "join_token" not in public
    assert client.get("/api/session", params={"class": "wrong"}).status_code == 404
    assert upload(client, token, recording("")).status_code == 409
    data = recording()
    result = upload(client, token, data)
    assert result.status_code == 200
    receipt = result.json()
    assert receipt["status"] == "saved" and not receipt["duplicate"]
    repeated = upload(client, token, data).json()
    assert repeated == {**receipt, "duplicate": True}
    endpoint = f"/api/instructor/submissions/{receipt['episode_id']}"
    rows = client.get("/api/instructor/submissions", params={"session": classroom["id"]}).json()
    assert len(rows) == 1 and rows[0]["participant"] == "student-7"
    assert rows[0]["steps"] == 3 and rows[0]["duration_seconds"] == 3 / 32
    assert client.get(endpoint + "/download").content == data
    assert client.get("/archives/coffee_" + receipt["episode_id"] + ".npz").status_code == 404
    assert client.get("/classroom.sqlite3").status_code == 404
    svg = client.get(f"/api/instructor/sessions/{classroom['id']}/qr.svg")
    assert svg.status_code == 200 and "<svg" in svg.text
    assert not client.post(f"/api/instructor/sessions/{classroom['id']}/close").json()["open"]
    assert upload(client, token, recording()).status_code == 409
    assert upload(client, token, data).json() == {**receipt, "duplicate": True}
    client.post("/api/instructor/logout")
    assert client.get(endpoint + "/download").status_code == 401
    assert client.get(endpoint + "/replay").status_code == 401
    with TestClient(create_app(**kwargs), base_url="https://coffee.example",
                    headers={"Origin": "https://coffee.example"}) as restarted:
        login(restarted)
        assert len(restarted.get("/api/instructor/sessions").json()) == 1
        assert restarted.get(endpoint + "/download").content == data


def test_replay_matches_recorded_frames_and_rejects_conflicting_archive(setup):
    client, _ = setup
    _, token = new_class(client)
    data = recording(steps=6)
    receipt = upload(client, token, data).json()
    arrays, metadata = read_demonstration(data)
    replay = client.get(f"/api/instructor/submissions/{receipt['episode_id']}/replay")
    assert replay.status_code == 200, replay.text
    frames = replay.json()["frames"]
    assert len(frames) == 7 and frames[0]["time"] == 0
    assert frames[-1]["time"] == 6 / 32
    assert frames[-1]["snapshot"]["state"]["step"] == 6
    assert frames[-1]["snapshot"]["playback"]["running"] is False
    metadata["participant"] = "changed participant"
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays, metadata=np.asarray(json.dumps(metadata)))
    assert upload(client, token, buffer.getvalue()).status_code == 409


def test_rejects_invalid_uploads_and_mismatched_summaries(setup, monkeypatch):
    client, _ = setup
    _, token = new_class(client, participant_required=False)
    assert upload(client, token, b"broken npz").status_code == 400
    assert client.post("/api/submissions", params={"class": token}, content=b"bad").status_code == 415
    arrays, metadata = read_demonstration(recording())
    metadata["fill_l"] = 0.75
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays, metadata=np.asarray(json.dumps(metadata)))
    assert upload(client, token, buffer.getvalue()).status_code == 400
    monkeypatch.setattr("kaist_rl_lab.apps.coffee_web.MAX_ARCHIVE_BYTES", 8)
    assert upload(client, token, b"012345678").status_code == 413


def test_missing_secrets_and_unsafe_public_origin_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="COFFEE_INSTRUCTOR_PASSWORD"):
        create_app(password="", session_secret="")
    with pytest.raises(ValueError, match="HTTPS"):
        create_app(password=PASSWORD, session_secret=SECRET, public_base_url="http://coffee.example")


def test_timed_out_submission_is_replayable_and_excluded_by_successful_only_cloning(setup):
    client, _ = setup
    classroom, token = new_class(client)
    runtime = BrowserRuntime(seed=33)
    try:
        runtime.session.paused = False
        for _ in range(1921):
            runtime.session.advance()
        data = runtime.session.save_demonstration("deadline-student").read_bytes()
    finally:
        runtime.session.close()
    response = upload(client, token, data)
    assert response.status_code == 200, response.text
    episode = response.json()["episode_id"]
    rows = client.get("/api/instructor/submissions", params={"session": classroom["id"]}).json()
    assert len(rows) == 1
    assert rows[0]["termination_reason"] == "time_limit"
    assert rows[0]["duration_seconds"] == 60
    assert rows[0]["steps"] == 1920
    assert rows[0]["success"] is False
    replay = client.get(f"/api/instructor/submissions/{episode}/replay")
    assert replay.status_code == 200, replay.text
    result = replay.json()
    assert result["metadata"]["termination_reason"] == "time_limit"
    assert result["frames"][-1]["time"] == 60
    assert result["frames"][-1]["snapshot"]["state"]["termination_reason"] == "time_limit"
    filtered = client.post("/api/instructor/cloning/train", json={
        "source": "students", "session_id": classroom["id"], "successful_only": True,
    })
    assert filtered.status_code == 409
    assert "no successful" in filtered.json()["detail"]
    arrays, metadata = read_demonstration(data)
    metadata["success"] = True
    invalid = BytesIO()
    np.savez_compressed(invalid, **arrays, metadata=np.asarray(json.dumps(metadata)))
    assert upload(client, token, invalid.getvalue()).status_code == 400
    metadata.pop("termination_reason")
    invalid = BytesIO()
    np.savez_compressed(invalid, **arrays, metadata=np.asarray(json.dumps(metadata)))
    assert upload(client, token, invalid.getvalue()).status_code == 400


def test_new_overlong_upload_is_rejected_but_legacy_archive_can_still_replay(setup):
    from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M, classroom_layout
    from kaist_rl_lab.apps.coffee_web import _replay

    client, _ = setup
    _, token = new_class(client)
    session = InteractiveSession(33, 700, dt=1 / 32, horizon=None,
                                 arm_base_distance=ARM_BASE_DISTANCE_M,
                                 reset_options=classroom_layout(33))
    try:
        for _ in range(1921):
            session.advance()
        data = session.save_demonstration("legacy-long-attempt").read_bytes()
    finally:
        session.close()
    arrays, metadata = read_demonstration(data)
    # An old recording predates the horizon/outcome metadata entirely.
    metadata.pop("horizon_steps")
    metadata.pop("termination_reason")
    legacy = BytesIO()
    np.savez_compressed(legacy, **arrays, metadata=np.asarray(json.dumps(metadata)))
    response = upload(client, token, legacy.getvalue())
    assert response.status_code == 409
    assert "60 simulated seconds" in response.json()["detail"]
    replay = _replay(legacy.getvalue())
    assert replay["frames"][-1]["time"] == 1921 / 32


@pytest.mark.parametrize("custom_origin", [None, "https://coffee.example"])
def test_render_url_drives_class_links_and_instructor_origin(setup, monkeypatch, custom_origin):
    _, kwargs = setup
    render_origin = "https://coffee-classroom.onrender.com"
    monkeypatch.setenv("RENDER_EXTERNAL_URL", render_origin)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    if custom_origin:
        monkeypatch.setenv("PUBLIC_BASE_URL", custom_origin)
    origin = custom_origin or render_origin
    with TestClient(create_app(**{**kwargs, "public_base_url": None}), base_url=origin,
                    headers={"Origin": origin}) as client:
        classroom, _ = new_class(client)
        assert classroom["join_url"].startswith(origin + "/?class=")
        assert client.post("/api/instructor/sessions", json={"name": "Wrong origin"},
                           headers={"Origin": "https://unrelated.example"}).status_code == 403
