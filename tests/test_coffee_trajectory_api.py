"""Reward summaries and exact replays for both private trajectory sources."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from kaist_rl_lab.apps import coffee_web
from kaist_rl_lab.apps.coffee_browser_runtime import BrowserRuntime
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import EXAMPLE_COUNT, load_examples

PASSWORD = "trajectory-test-password-only"
SECRET = "trajectory-test-signing-secret-not-for-deployment"
ORIGIN = "https://coffee.example"


def app_options(tmp_path):
    static = tmp_path / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text("Coffee")
    (static / "instructor.html").write_text("Instructor")
    return {"data_dir": tmp_path / "private", "static_dir": static, "build_assets": False,
            "password": PASSWORD, "session_secret": SECRET, "public_base_url": ORIGIN}


@pytest.fixture
def setup(tmp_path):
    options = app_options(tmp_path)
    app = coffee_web.create_app(**options)
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        yield client, app.state.store, options


def login(client):
    assert client.post("/api/instructor/login", json={"password": PASSWORD}).status_code == 200


def new_class(client, name="Trajectory class"):
    response = client.post("/api/instructor/sessions", json={"name": name})
    assert response.status_code == 200
    classroom = response.json()
    return classroom["id"], parse_qs(urlsplit(classroom["join_url"]).query)["class"][0]


def recording(rewards=None):
    runtime = BrowserRuntime()
    try:
        runtime.session.set_motor(0, 1)
        runtime.session.toggle_pause()
        for _ in range(6):
            runtime.session.advance()
        data = runtime.session.save_demonstration("student-7").read_bytes()
    finally:
        runtime.session.close()
    if rewards is None:
        return data
    arrays, metadata = read_demonstration(data)
    arrays["rewards"] = np.asarray(rewards, dtype=np.float64)
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return buffer.getvalue()


def upload(client, token, data):
    return client.post("/api/submissions", params={"class": token}, content=data,
                       headers={"Content-Type": "application/octet-stream"})


def test_student_rewards_use_recorded_sequence_and_survive_restart(setup):
    client, store, options = setup
    login(client)
    identifier, token = new_class(client)
    rewards = [-2, 1, 3, -0.25, 0.5, -4]
    data = recording(rewards)
    receipt = upload(client, token, data)
    assert receipt.status_code == 200
    episode_id = receipt.json()["episode_id"]
    endpoint = f"/api/instructor/submissions/{episode_id}"
    rows = client.get("/api/instructor/submissions", params={"session": identifier}).json()
    assert len(rows) == 1
    assert rows[0]["total_reward"] == np.sum(rewards, dtype=np.float64)
    assert "reward_checked" not in rows[0]
    with store.connect() as db:
        saved = db.execute("SELECT * FROM submissions WHERE episode_id=?", (episode_id,)).fetchone()
    assert saved["total_reward"] == -1.75 and saved["reward_checked"] == 1
    replay = client.get(endpoint + "/replay")
    assert replay.status_code == 200, replay.text
    assert replay.headers["cache-control"] == "no-store"
    result = replay.json()
    assert result["total_reward"] == -1.75
    assert [frame["cumulative_reward"] for frame in result["frames"]] == [
        0, -2, -1, 2, 1.75, 2.25, -1.75,
    ]
    assert result["frames"][-1]["snapshot"]["state"]["step"] == 6
    assert client.get(endpoint + "/download").content == data
    assert upload(client, token, data).json()["duplicate"]
    with TestClient(coffee_web.create_app(**options), base_url=ORIGIN,
                    headers={"Origin": ORIGIN}) as restarted:
        login(restarted)
        saved = restarted.get("/api/instructor/submissions", params={"session": identifier}).json()
        assert saved[0]["total_reward"] == -1.75


def test_nonfinite_reward_totals_are_rejected_before_storage(setup):
    client, store, _ = setup
    login(client)
    identifier, token = new_class(client)
    huge = np.finfo(np.float64).max
    # Every entry is finite; the float64 aggregate cannot be represented.
    response = upload(client, token, recording([huge, huge, 0, 0, 0, 0]))
    assert response.status_code == 400
    assert client.get("/api/instructor/submissions", params={"session": identifier}).json() == []
    assert list(store.archives.iterdir()) == []
    # A finite pairwise sum is insufficient if a displayed prefix overflows.
    with pytest.raises(ValueError, match="finite totals"):
        coffee_web._recorded_rewards({"rewards": np.array([huge, huge, -huge, -huge])})


def make_legacy_store(directory, archives):
    directory.mkdir()
    (directory / "archives").mkdir()
    with sqlite3.connect(directory / "classroom.sqlite3") as db:
        db.executescript("""
            CREATE TABLE classes (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, join_token TEXT UNIQUE NOT NULL,
                open INTEGER NOT NULL, participant_required INTEGER NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE submissions (
                episode_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES classes(id),
                participant TEXT NOT NULL, received_at TEXT NOT NULL, steps INTEGER NOT NULL,
                success INTEGER NOT NULL, fill_ml REAL NOT NULL, spill_ml REAL NOT NULL,
                duration_seconds REAL NOT NULL, sha256 TEXT NOT NULL, receipt TEXT NOT NULL
            );
            INSERT INTO classes VALUES ('legacy-class', 'Legacy', 'legacy-token', 1, 1, '2026');
        """)
        ids = []
        for index, (data, contents) in enumerate(archives):
            _, metadata = read_demonstration(data)
            episode_id = metadata["episode_id"]
            ids.append(episode_id)
            db.execute("INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (episode_id, "legacy-class", "legacy-student", str(index), 6, 0, 0, 0,
                        6 / 32, "legacy-digest", "legacy-receipt"))
            if contents is not None:
                (directory / "archives" / f"coffee_{episode_id}.npz").write_bytes(contents)
        return ids


def test_legacy_migration_backfills_bounded_batches_and_failed_archives_only_once(tmp_path, monkeypatch):
    options = app_options(tmp_path)
    valid = recording([-2, 1, 3, -0.25, 0.5, -4])
    corrupt = recording()
    missing = recording()
    ids = make_legacy_store(options["data_dir"], [(missing, None), (corrupt, b"broken"), (valid, valid)])
    monkeypatch.setattr(coffee_web, "LEGACY_REWARD_BATCH_SIZE", 2)
    app = coffee_web.create_app(**options)
    # Initialization can safely repeat before and after the lazy migration.
    coffee_web.ClassroomStore(options["data_dir"])
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        login(client)
        rows = client.get("/api/instructor/submissions", params={"session": "legacy-class"}).json()
        by_id = {row["episode_id"]: row for row in rows}
        assert by_id[ids[2]]["total_reward"] == -1.75
        assert by_id[ids[1]]["total_reward"] is None
        assert by_id[ids[0]]["total_reward"] is None
        with app.state.store.connect() as db:
            assert db.execute("SELECT SUM(reward_checked) FROM submissions").fetchone()[0] == 2
        client.get("/api/instructor/submissions", params={"session": "legacy-class"})
        with app.state.store.connect() as db:
            assert db.execute("SELECT SUM(reward_checked) FROM submissions").fetchone()[0] == 3
        # Restoring failed files does not trigger repeated request-time reads.
        app.state.store.archive_path(ids[0]).write_bytes(missing)
        app.state.store.archive_path(ids[1]).write_bytes(corrupt)
        monkeypatch.setattr(coffee_web, "_validated_archive", lambda _: pytest.fail("Archive reread"))
        again = client.get("/api/instructor/submissions", params={"session": "legacy-class"}).json()
        assert {row["episode_id"]: row["total_reward"] for row in again} == {
            ids[0]: None, ids[1]: None, ids[2]: -1.75,
        }
    coffee_web.ClassroomStore(options["data_dir"])


def test_generated_examples_list_download_and_replay_without_student_records(setup, monkeypatch):
    client, store, _ = setup
    login(client)
    identifier, _ = new_class(client)
    packaged = load_examples()
    monkeypatch.setattr("kaist_rl_lab.apps.coffee_expert.generate_example",
                        lambda _: pytest.fail("Online example generation"))
    response = client.get("/api/instructor/examples")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    rows = response.json()
    assert len(rows) == EXAMPLE_COUNT
    for index, (row, data) in enumerate(zip(rows, packaged, strict=True), start=1):
        arrays, metadata = read_demonstration(data)
        assert row["example_id"] == f"example-{index}"
        assert row["label"] == f"Generated example {index}"
        assert row["episode_id"] == metadata["episode_id"]
        assert row["steps"] == len(arrays["actions"])
        assert row["total_reward"] == np.sum(arrays["rewards"], dtype=np.float64)
        assert row["success"] is True
        assert 695 <= row["fill_ml"] <= 705
        assert row["spill_ml"] < 1
        download = client.get(f"/api/instructor/examples/example-{index}/download")
        assert download.status_code == 200 and download.content == data
        assert download.headers["cache-control"] == "no-store"
    replay = client.get("/api/instructor/examples/example-1/replay")
    assert replay.status_code == 200, replay.text
    result = replay.json()
    assert result["total_reward"] == rows[0]["total_reward"]
    assert result["frames"][0]["cumulative_reward"] == 0
    assert result["frames"][-1]["cumulative_reward"] == pytest.approx(rows[0]["total_reward"])
    assert result["frames"][-1]["snapshot"]["state"]["step"] == rows[0]["steps"]
    assert result["metadata"]["success"] is True
    assert client.get("/api/instructor/submissions", params={"session": identifier}).json() == []
    assert list(store.archives.iterdir()) == []
    # Generated IDs never resolve through the student namespace, and vice versa.
    assert client.get("/api/instructor/submissions/example-1/replay").status_code == 404
    assert client.get(f"/api/instructor/examples/{rows[0]['episode_id']}/replay").status_code == 404
    for invalid in ("example-0", f"example-{EXAMPLE_COUNT + 1}", "example-01", "example_1.npz", "classroom.sqlite3"):
        assert client.get(f"/api/instructor/examples/{invalid}/replay").status_code == 404
        assert client.get(f"/api/instructor/examples/{invalid}/download").status_code == 404


def test_all_trajectory_data_requires_authentication_before_reads(setup, monkeypatch):
    client, store, _ = setup
    monkeypatch.setattr("kaist_rl_lab.apps.coffee_expert.load_examples",
                        lambda: pytest.fail("Unauthenticated example read"))
    monkeypatch.setattr(store, "backfill_rewards", lambda _: pytest.fail("Unauthenticated backfill"))
    for endpoint in (
        "/api/instructor/examples", "/api/instructor/examples/example-1/replay",
        "/api/instructor/examples/example-1/replay-stream",
        "/api/instructor/examples/example-1/download",
        "/api/instructor/submissions?session=unknown",
        "/api/instructor/submissions/unknown/replay", "/api/instructor/submissions/unknown/download",
        "/api/instructor/submissions/unknown/replay-stream",
    ):
        response = client.get(endpoint)
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"


def stream_result(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["kind"] == "start" and events[-1]["kind"] == "complete"
    return events, [events[0]["frame"]] + [
        frame for event in events[1:] if event["kind"] == "frames" for frame in event["frames"]
    ]


def test_streamed_replay_matches_complete_physics_and_reuses_verified_cache(setup, monkeypatch):
    client, _, _ = setup
    login(client)
    _, token = new_class(client)
    episode_id = upload(client, token, recording([-2, 1, 3, -0.25, 0.5, -4])).json()["episode_id"]
    endpoint = f"/api/instructor/submissions/{episode_id}"
    complete = client.get(endpoint + "/replay").json()
    first = client.get(endpoint + "/replay-stream")
    events, frames = stream_result(first)
    assert frames == complete["frames"]
    assert events[0]["total_reward"] == complete["total_reward"]
    assert events[0]["frame_count"] == len(frames)
    assert events[0]["duration_seconds"] == frames[-1]["time"]
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["x-accel-buffering"] == "no"
    monkeypatch.setattr(coffee_web, "_replay_events", lambda _: pytest.fail("Cached replay simulated again"))
    assert client.get(endpoint + "/replay-stream").content == first.content
    client.post("/api/instructor/logout")
    assert client.get(endpoint + "/replay-stream").status_code == 401


def test_stream_initial_frame_needs_no_steps_and_close_releases_environment(setup, monkeypatch):
    from types import SimpleNamespace

    import anyio
    from starlette.requests import ClientDisconnect

    client, _, _ = setup
    login(client)
    _, token = new_class(client)
    data = recording()
    episode_id = upload(client, token, data).json()["episode_id"]
    steps, closed = [], []
    original_step = coffee_web.CoffeePouringEnv.step
    original_close = coffee_web.CoffeePouringEnv.close

    def step(env, action):
        steps.append(1)
        return original_step(env, action)

    def close(env):
        closed.append(1)
        return original_close(env)

    monkeypatch.setattr(coffee_web.CoffeePouringEnv, "step", step)
    monkeypatch.setattr(coffee_web.CoffeePouringEnv, "close", close)
    route = next(route for route in client.app.routes
                 if getattr(route, "path", "") == "/api/instructor/submissions/{episode_id}/replay-stream")
    request = SimpleNamespace(cookies={coffee_web.COOKIE_NAME: client.cookies.get(coffee_web.COOKIE_NAME)})

    async def cancel_after_first_frame():
        response = route.endpoint(episode_id, request)

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                event = json.loads(message["body"])
                assert event["kind"] == "start" and event["frame"]["time"] == 0
                assert steps == []
                raise OSError("The browser closed the response")

        # Exercise the response lifecycle, not just iterator.aclose(): ASGI
        # disconnects can interrupt iteration while it is suspended at a yield.
        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    anyio.run(cancel_after_first_frame)
    assert closed == [1]
    # The aborted response neither holds the shared lock nor populates the cache.
    stream_result(client.get(f"/api/instructor/submissions/{episode_id}/replay-stream"))
    assert len(steps) == 6


def test_generated_stream_uses_same_frames_and_cache_is_bounded(setup, monkeypatch):
    client, _, _ = setup
    login(client)
    # Keep this API/cache test fast; actual packaged physics is covered above.
    first, second = recording(), recording()
    monkeypatch.setattr("kaist_rl_lab.apps.coffee_expert.load_examples", lambda: [first, second])
    monkeypatch.setattr(coffee_web, "REPLAY_CACHE_ITEMS", 1)
    calls = []
    original = coffee_web._replay_events

    def replay(data):
        calls.append(data)
        yield from original(data)

    monkeypatch.setattr(coffee_web, "_replay_events", replay)
    for identifier in ("example-1", "example-2", "example-2", "example-1"):
        _, frames = stream_result(client.get(f"/api/instructor/examples/{identifier}/replay-stream"))
        assert frames[-1]["snapshot"]["state"]["step"] == 6
    assert calls == [first, second, first]


def test_stream_reports_invalid_physics_without_caching_partial_results(setup, monkeypatch):
    client, store, _ = setup
    login(client)
    _, token = new_class(client)
    data = recording()
    episode_id = upload(client, token, data).json()["episode_id"]
    arrays, metadata = read_demonstration(data)
    arrays["next_observations"][-1, 0] += .01
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays, metadata=np.asarray(json.dumps(metadata)))
    store.archive_path(episode_id).write_bytes(buffer.getvalue())
    endpoint = f"/api/instructor/submissions/{episode_id}/replay-stream"
    for _ in range(2):
        events = [json.loads(line) for line in client.get(endpoint).text.splitlines()]
        assert events[0]["kind"] == "start"
        assert events[-1]["kind"] == "error"
        assert "physics" in events[-1]["detail"]
        assert not any(event["kind"] == "complete" for event in events)
    store.archive_path(episode_id).write_bytes(data)
    stream_result(client.get(endpoint))


def test_replay_lock_is_shared_across_sources_and_released_after_errors(setup, monkeypatch):
    client, _, _ = setup
    login(client)
    _, token = new_class(client)
    episode_id = upload(client, token, recording()).json()["episode_id"]
    entered, release = Event(), Event()
    original = coffee_web._replay

    def wait_replay(_):
        entered.set()
        assert release.wait(timeout=10)
        return {"frames": [], "total_reward": 0}

    monkeypatch.setattr(coffee_web, "_replay", wait_replay)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.get, "/api/instructor/examples/example-1/replay")
        try:
            assert entered.wait(timeout=5)
            assert client.get(f"/api/instructor/submissions/{episode_id}/replay").status_code == 429
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200

    def invalid_replay(_):
        raise ValueError("Physics mismatch")

    monkeypatch.setattr(coffee_web, "_replay", invalid_replay)
    assert client.get("/api/instructor/examples/example-1/replay").status_code == 409
    monkeypatch.setattr(coffee_web, "_replay", original)
    assert client.get(f"/api/instructor/submissions/{episode_id}/replay").status_code == 200


def test_missing_student_archive_has_clear_errors(setup):
    client, store, _ = setup
    login(client)
    _, token = new_class(client)
    episode_id = upload(client, token, recording()).json()["episode_id"]
    store.archive_path(episode_id).unlink()
    assert client.get(f"/api/instructor/submissions/{episode_id}/download").status_code == 404
    assert client.get(f"/api/instructor/submissions/{episode_id}/replay").status_code == 409
