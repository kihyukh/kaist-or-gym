"""Standalone coffee classroom: public play/upload and private instructor tools.

Physics stays in each browser. SQLite and archives live in COFFEE_DATA_DIR, which
must be a persistent volume when deployed. Run exactly one application worker.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from uuid import UUID, uuid4
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

from kaist_rl_lab.apps.coffee_demonstrations import (
    MAX_ARCHIVE_BYTES,
    read_demonstration,
    validate_collection_duration,
)
from kaist_rl_lab.envs import CoffeePouringEnv

COOKIE_NAME = "coffee_instructor"
SESSION_SECONDS = 12 * 60 * 60
MAX_CLASS_SUBMISSIONS = 5000
LEGACY_REWARD_BATCH_SIZE = 20
REPLAY_CACHE_BYTES = 24 * 1024 * 1024
REPLAY_CACHE_ITEMS = 4
ARCHIVE_ERRORS = (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError,
                  OverflowError, EOFError, BadZipFile, RuntimeError, ZlibError)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recorded_rewards(arrays):
    """Sum saved step rewards without recomputing physics or terminal bonuses."""
    with np.errstate(over="ignore", invalid="ignore"):
        rewards = np.asarray(arrays["rewards"], dtype=np.float64)
        total = float(np.sum(rewards, dtype=np.float64))
        cumulative = np.concatenate(([0.0], np.cumsum(rewards, dtype=np.float64)))
    if not math.isfinite(total) or not np.isfinite(cumulative).all():
        raise ValueError("Recorded rewards must have finite totals.")
    return total, cumulative


def _validated_archive(data: bytes):
    """Keep archive validation shared, then validate fields used by the website."""
    arrays, metadata = read_demonstration(data)
    if not isinstance(metadata.get("participant"), str):
        raise TypeError("Participant code must be text.")
    if type(metadata.get("seed")) is not int or not 0 <= metadata["seed"] < 2**63:
        raise ValueError("Invalid episode seed.")
    if type(metadata.get("success")) is not bool:
        raise ValueError("Invalid success flag.")
    angles = metadata.get("initial_joint_angles_rad")
    if not isinstance(angles, list) or len(angles) != 6:
        raise ValueError("Invalid initial joint angles.")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in angles):
        raise ValueError("Invalid initial joint angles.")
    for field in ("fill_l", "spill_l", "target_fill_l"):
        value = metadata.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
            raise ValueError("Invalid liquid amount.")
    if not 0 < metadata["target_fill_l"] <= CoffeePouringEnv.CUP_CAPACITY:
        raise ValueError("Invalid target amount.")
    final = arrays["next_observations"][-1]
    for field, column in (("fill_l", 12), ("spill_l", 13), ("target_fill_l", 14)):
        if not np.isclose(metadata[field], final[column], rtol=1e-5, atol=1e-7):
            raise ValueError("Liquid summary does not match the recording.")
    _recorded_rewards(arrays)
    return arrays, metadata


class ClassroomStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.archives = directory / "archives"
        self.archives.mkdir(parents=True, exist_ok=True)
        self.database = directory / "classroom.sqlite3"
        self.write_lock = Lock()
        self.reward_backfill_lock = Lock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS classes (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, join_token TEXT UNIQUE NOT NULL,
                    open INTEGER NOT NULL, participant_required INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS submissions (
                    episode_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES classes(id),
                    participant TEXT NOT NULL, received_at TEXT NOT NULL, steps INTEGER NOT NULL,
                    success INTEGER NOT NULL, fill_ml REAL NOT NULL, spill_ml REAL NOT NULL,
                    duration_seconds REAL NOT NULL, sha256 TEXT NOT NULL, receipt TEXT NOT NULL,
                    total_reward REAL, reward_checked INTEGER NOT NULL DEFAULT 0,
                    termination_reason TEXT,
                    reward_model TEXT NOT NULL DEFAULT 'legacy'
                );
                CREATE INDEX IF NOT EXISTS submissions_session ON submissions(session_id);
                CREATE TABLE IF NOT EXISTS instructor_sessions (
                    token_hash TEXT PRIMARY KEY, expires INTEGER NOT NULL
                );
            """)
            # Existing installations retain their rows and archives. Taking the
            # write transaction before inspecting columns makes restarts and
            # concurrent application initialization idempotent.
            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(submissions)")}
            if "total_reward" not in columns:
                db.execute("ALTER TABLE submissions ADD COLUMN total_reward REAL")
            if "reward_checked" not in columns:
                db.execute("ALTER TABLE submissions ADD COLUMN reward_checked INTEGER NOT NULL DEFAULT 0")
            if "termination_reason" not in columns:
                db.execute("ALTER TABLE submissions ADD COLUMN termination_reason TEXT")
            if "reward_model" not in columns:
                db.execute("ALTER TABLE submissions ADD COLUMN reward_model TEXT NOT NULL DEFAULT 'legacy'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def archive_path(self, episode_id: str) -> Path:
        return self.archives / f"coffee_{UUID(episode_id)}.npz"

    def backfill_rewards(self, session_id: str):
        """Read a bounded batch once; unavailable legacy rewards remain unknown."""
        if not self.reward_backfill_lock.acquire(blocking=False):
            return
        try:
            with self.connect() as db:
                rows = db.execute(
                    "SELECT episode_id FROM submissions WHERE session_id=? AND reward_checked=0 "
                    "ORDER BY received_at DESC, episode_id DESC LIMIT ?",
                    (session_id, LEGACY_REWARD_BATCH_SIZE),
                ).fetchall()
            updates = []
            for row in rows:
                total = None
                try:
                    with self.archive_path(row["episode_id"]).open("rb") as archive:
                        arrays, metadata = _validated_archive(archive.read(MAX_ARCHIVE_BYTES + 1))
                    if str(UUID(metadata["episode_id"])) != row["episode_id"]:
                        raise ValueError("The archive belongs to another episode.")
                    total, _ = _recorded_rewards(arrays)
                except ARCHIVE_ERRORS:
                    pass
                updates.append((total, row["episode_id"]))
            with self.connect() as db:
                db.executemany(
                    "UPDATE submissions SET total_reward=?, reward_checked=1 "
                    "WHERE episode_id=? AND reward_checked=0", updates,
                )
        finally:
            self.reward_backfill_lock.release()

    def receive(self, token: str, data: bytes, arrays, metadata):
        episode_id = str(UUID(metadata["episode_id"]))
        digest = hashlib.sha256(data).hexdigest()
        total_reward, _ = _recorded_rewards(arrays)
        with self.write_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            classroom = db.execute("SELECT * FROM classes WHERE join_token=?", (token,)).fetchone()
            if not classroom:
                raise ValueError("Classroom link not found.")
            existing = db.execute(
                "SELECT * FROM submissions WHERE episode_id=?", (episode_id,)
            ).fetchone()
            if existing:
                if existing["sha256"] != digest or existing["session_id"] != classroom["id"]:
                    raise ValueError("A different recording already uses this episode ID.")
                return {"status": "saved", "episode_id": episode_id,
                        "receipt": existing["receipt"], "duplicate": True}
            validate_collection_duration(arrays, metadata)
            if not classroom["open"]:
                raise ValueError("Submissions for this classroom are closed.")
            if classroom["participant_required"] and not metadata["participant"].strip():
                raise ValueError("Enter a participant code before submitting.")
            count = db.execute(
                "SELECT COUNT(*) FROM submissions WHERE session_id=?", (classroom["id"],)
            ).fetchone()[0]
            if count >= MAX_CLASS_SUBMISSIONS:
                raise ValueError("This classroom has reached its submission limit.")
            received, receipt = _now(), secrets.token_urlsafe(12)
            path = self.archive_path(episode_id)
            created = False
            try:
                # An orphan left by a process interruption is adopted only if identical.
                try:
                    with path.open("xb") as output:
                        created = True
                        output.write(data)
                        output.flush()
                        os.fsync(output.fileno())
                except FileExistsError:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise ValueError("A different recording already uses this episode ID.")
                db.execute(
                    "INSERT INTO submissions (episode_id, session_id, participant, received_at, "
                    "steps, success, fill_ml, spill_ml, duration_seconds, sha256, receipt, "
                    "total_reward, reward_checked, termination_reason, reward_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (episode_id, classroom["id"], metadata["participant"].strip(), received,
                     len(arrays["actions"]), int(metadata["success"]),
                     float(arrays["next_observations"][-1, 12]) * 1000,
                     float(arrays["next_observations"][-1, 13]) * 1000,
                     len(arrays["actions"]) * metadata["dt"], digest, receipt, total_reward,
                     metadata.get("termination_reason"), metadata.get("reward_model", "legacy")),
                )
                db.commit()
            except Exception:
                if created:
                    path.unlink(missing_ok=True)
                raise
            return {"status": "saved", "episode_id": episode_id,
                    "receipt": receipt, "duplicate": False}


def _replay_events(data: bytes):
    """Yield verified frames immediately, keeping the original renderer/physics."""
    arrays, metadata = _validated_archive(data)
    total_reward, cumulative_rewards = _recorded_rewards(arrays)
    env = CoffeePouringEnv(
        dt=metadata["dt"], horizon=metadata.get("horizon_steps"),
        arm_base_distance=metadata.get(
            "arm_base_distance_m", CoffeePouringEnv.DEFAULT_ARM_BASE_DISTANCE,
        ),
    )
    count = len(arrays["actions"])
    selected = set(np.linspace(0, count, min(count + 1, 400), dtype=int).tolist())
    def frame(step, motors):
        snapshot = env.render_snapshot()
        snapshot["playback"] = {
            "generation": 1, "revision": step, "input_sequence": step, "kind": "replay",
            "speed": 1.0, "paused": True, "running": False,
            "motors": list(map(float, motors)), "decision_interval_wall_ms": metadata["dt"] * 1000,
        }
        return {"time": step * metadata["dt"], "snapshot": snapshot,
                "cumulative_reward": float(cumulative_rewards[step])}

    try:
        observation, _ = env.reset(seed=metadata["seed"], options={
            "joint_angles": metadata["initial_joint_angles_rad"],
            "target_fill": metadata["target_fill_l"],
        })
        if not np.array_equal(observation, arrays["observations"][0]):
            raise ValueError("The recording cannot be replayed by this environment version.")
        # The first response needs only archive validation and reset; no
        # transition has to be simulated before the instructor sees the scene.
        yield {"kind": "start", "metadata": metadata, "total_reward": total_reward,
               "duration_seconds": count * metadata["dt"], "frame_count": len(selected),
               "frame": frame(0, [0] * 6)}
        for index, action in enumerate(arrays["actions"], start=1):
            observation, _, terminated, _, _ = env.step(action)
            if not np.array_equal(observation, arrays["next_observations"][index - 1]):
                raise ValueError("The recording does not match replayed physics.")
            if terminated and index < count:
                raise ValueError("The recording continues after its environment ended.")
            if index in selected:
                yield {"kind": "frames", "frames": [frame(index, action)]}
        yield {"kind": "complete"}
    finally:
        env.close()


def _replay(data: bytes) -> dict:
    """Compatibility response for clients that need the complete frame array."""
    result = None
    for event in _replay_events(data):
        if event["kind"] == "start":
            result = {"metadata": event["metadata"], "total_reward": event["total_reward"],
                      "frames": [event["frame"]]}
        elif event["kind"] == "frames":
            result["frames"].extend(event["frames"])
    return result


def create_app(*, data_dir=None, public_base_url=None, password=None, session_secret=None,
               static_dir=None, build_assets=True, allow_missing_origin=False):
    """Application factory; importing the module never starts a server."""
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.concurrency import run_in_threadpool

    # FastAPI resolves these annotations in module globals, even for nested routes.
    globals()["Request"] = Request
    password = password if password is not None else os.environ.get("COFFEE_INSTRUCTOR_PASSWORD", "")
    session_secret = (session_secret if session_secret is not None
                      else os.environ.get("COFFEE_SESSION_SECRET", ""))
    if len(password) < 12 or len(session_secret) < 32:
        raise ValueError("Set COFFEE_INSTRUCTOR_PASSWORD (12+ characters) and "
                         "COFFEE_SESSION_SECRET (32+ characters) before starting.")
    base = (public_base_url or os.environ.get("PUBLIC_BASE_URL")
            or os.environ.get("RENDER_EXTERNAL_URL") or "http://localhost:8000").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("PUBLIC_BASE_URL must be the website origin, without a path or credentials.")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for a public website; HTTP is supported only on localhost.")
    directory = Path(data_dir or os.environ.get("COFFEE_DATA_DIR", "./data/coffee-web")).resolve()
    store = ClassroomStore(directory)
    assets = Path(static_dir or directory / "site").resolve()
    if assets == directory or assets in directory.parents:
        raise ValueError("Static assets must not expose the data directory.")
    if build_assets:
        from kaist_rl_lab.apps.coffee_web_assets import build_static_site
        build_static_site(assets)
    if not assets.is_dir():
        raise ValueError("The static website directory does not exist.")
    app = FastAPI(title="Coffee classroom", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    replay_lock = Lock()
    replay_cache = OrderedDict()
    replay_cache_lock = Lock()
    rate_lock = Lock()
    rates = defaultdict(deque)

    def rate_limit(request, group, limit):
        key = (group, request.client.host if request.client else "unknown")
        now = time.monotonic()
        with rate_lock:
            if len(rates) > 8192:
                for old in list(rates):
                    if not rates[old] or rates[old][-1] < now - 60:
                        del rates[old]
            events = rates[key]
            while events and events[0] < now - 60:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(429, "Please wait a minute before trying again.", headers={"Retry-After": "60"})
            events.append(now)

    def instructor(request):
        token = request.cookies.get(COOKIE_NAME, "")
        try:
            value, signature = token.rsplit(".", 1)
            expected = hmac.new(session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError()
        except ValueError:
            raise HTTPException(401, "Sign in as the instructor.") from None
        with store.connect() as db:
            row = db.execute("SELECT expires FROM instructor_sessions WHERE token_hash=?",
                             (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not row or row["expires"] <= time.time():
            raise HTTPException(401, "Sign in as the instructor.")

    def classroom(row):
        return {"id": row["id"], "name": row["name"], "open": bool(row["open"]),
                "participant_required": bool(row["participant_required"]),
                "created_at": row["created_at"], "join_url": f"{base}/?class={row['join_token']}"}

    async def small_json(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 4096:
                raise HTTPException(413, "Request is too large.")
        try:
            result = json.loads(data)
        except (ValueError, UnicodeError):
            raise HTTPException(400, "Invalid request.") from None
        if not isinstance(result, dict):
            raise HTTPException(400, "Invalid request.")
        return result

    @app.middleware("http")
    async def security_headers(request, call_next):
        if request.url.path.startswith("/api/instructor/") and request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin != base and not (origin is None and allow_missing_origin):
                return JSONResponse({"detail": "Use the instructor page on this website."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/") or request.url.path == "/instructor":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        with store.connect() as db:
            db.execute("SELECT 1").fetchone()
        return {"status": "ok"}

    @app.get("/api/session")
    def session_info(class_token: str = Query(alias="class", min_length=1, max_length=128)):
        with store.connect() as db:
            row = db.execute("SELECT * FROM classes WHERE join_token=?", (class_token,)).fetchone()
        if not row:
            raise HTTPException(404, "Classroom link not found. Ask your instructor for the QR code.")
        return {key: value for key, value in classroom(row).items()
                if key in {"id", "name", "open", "participant_required"}}

    @app.get("/api/session/default")
    def default_session():
        # The main student website joins the sole open class. Never choose an
        # arbitrary recipient when several classes are collecting recordings.
        with store.connect() as db:
            rows = db.execute("SELECT * FROM classes WHERE open=1 LIMIT 2").fetchall()
        if len(rows) != 1:
            return {"session": None,
                    "reason": "no_open_class" if not rows else "multiple_open_classes"}
        row = rows[0]
        session = {key: value for key, value in classroom(row).items()
                   if key in {"id", "name", "open", "participant_required"}}
        session["join_token"] = row["join_token"]
        return {"session": session}

    @app.post("/api/submissions")
    async def submit(request: Request, class_token: str = Query(alias="class", min_length=1, max_length=128)):
        rate_limit(request, "upload", 240)
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
            raise HTTPException(415, "Submit a trajectory archive.")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_ARCHIVE_BYTES:
                raise HTTPException(413, "Trajectory archive is too large.")
        try:
            arrays, metadata = await run_in_threadpool(_validated_archive, bytes(data))
        except ARCHIVE_ERRORS:
            raise HTTPException(400, "Invalid trajectory archive.") from None
        try:
            return await run_in_threadpool(store.receive, class_token, bytes(data), arrays, metadata)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/instructor/login")
    async def login(request: Request):
        rate_limit(request, "login", 10)
        body = await small_json(request)
        supplied = body.get("password")
        if not isinstance(supplied, str) or not hmac.compare_digest(
                hashlib.sha256(supplied.encode()).digest(), hashlib.sha256(password.encode()).digest()):
            raise HTTPException(401, "Incorrect instructor password.")
        value = secrets.token_urlsafe(32)
        token = value + "." + hmac.new(session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()
        with store.connect() as db:
            db.execute("DELETE FROM instructor_sessions WHERE expires<=?", (int(time.time()),))
            db.execute("INSERT INTO instructor_sessions VALUES (?, ?)",
                       (hashlib.sha256(token.encode()).hexdigest(), int(time.time()) + SESSION_SECONDS))
        response = JSONResponse({"authenticated": True})
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                            secure=parsed.scheme == "https", samesite="strict", path="/api/instructor")
        return response

    @app.post("/api/instructor/logout")
    def logout(request: Request):
        token = request.cookies.get(COOKIE_NAME, "")
        with store.connect() as db:
            db.execute("DELETE FROM instructor_sessions WHERE token_hash=?",
                       (hashlib.sha256(token.encode()).hexdigest(),))
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE_NAME, path="/api/instructor")
        return response

    @app.get("/api/instructor/sessions")
    def list_sessions(request: Request):
        instructor(request)
        with store.connect() as db:
            return [classroom(row) for row in db.execute("SELECT * FROM classes ORDER BY created_at DESC")]

    @app.post("/api/instructor/sessions")
    async def create_session(request: Request):
        instructor(request)
        body = await small_json(request)
        name, required = body.get("name", ""), body.get("participant_required", True)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120 or type(required) is not bool:
            raise HTTPException(400, "Enter a classroom name of 1–120 characters.")
        identifier = str(uuid4())
        with store.connect() as db:
            db.execute("INSERT INTO classes VALUES (?, ?, ?, 1, ?, ?)",
                       (identifier, name.strip(), secrets.token_urlsafe(24), int(required), _now()))
            row = db.execute("SELECT * FROM classes WHERE id=?", (identifier,)).fetchone()
        return classroom(row)

    @app.post("/api/instructor/sessions/{identifier}/close")
    def close_session(identifier: str, request: Request):
        instructor(request)
        with store.connect() as db:
            db.execute("UPDATE classes SET open=0 WHERE id=?", (identifier,))
            row = db.execute("SELECT * FROM classes WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise HTTPException(404, "Classroom not found.")
        return classroom(row)

    @app.get("/api/instructor/sessions/{identifier}/qr.svg")
    def session_qr(identifier: str, request: Request):
        instructor(request)
        import qrcode
        import qrcode.image.svg
        with store.connect() as db:
            row = db.execute("SELECT * FROM classes WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise HTTPException(404, "Classroom not found.")
        output = BytesIO()
        qrcode.make(classroom(row)["join_url"], image_factory=qrcode.image.svg.SvgPathImage).save(output)
        return Response(output.getvalue(), media_type="image/svg+xml")

    @app.get("/api/instructor/submissions")
    def submissions(request: Request, session: str = Query(min_length=1, max_length=128)):
        instructor(request)
        with store.connect() as db:
            if not db.execute("SELECT id FROM classes WHERE id=?", (session,)).fetchone():
                raise HTTPException(404, "Classroom not found.")
        store.backfill_rewards(session)
        with store.connect() as db:
            rows = db.execute("""SELECT episode_id, participant, received_at, steps, success,
                                 fill_ml, spill_ml, duration_seconds, total_reward, termination_reason, reward_model FROM submissions
                                 WHERE session_id=? ORDER BY received_at DESC""", (session,))
            return [{**dict(row), "success": bool(row["success"])} for row in rows]

    def submitted_path(episode_id):
        with store.connect() as db:
            row = db.execute("SELECT episode_id FROM submissions WHERE episode_id=?", (episode_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Recording not found.")
        return store.archive_path(row["episode_id"])

    @app.get("/api/instructor/submissions/{episode_id}/download")
    def download(episode_id: str, request: Request):
        instructor(request)
        path = submitted_path(episode_id)
        if not path.is_file():
            raise HTTPException(404, "Recording archive is unavailable.")
        return FileResponse(path, media_type="application/octet-stream", filename=path.name)

    def prepared_replay(read_archive):
        if not replay_lock.acquire(blocking=False):
            raise HTTPException(429, "Another replay is being prepared. Try again shortly.")
        try:
            return _replay(read_archive())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except ARCHIVE_ERRORS:
            raise HTTPException(409, "This recording archive is unavailable or invalid.") from None
        finally:
            replay_lock.release()

    def streamed_replay(read_archive):
        """Stream bounded physics work; disconnects close the environment/lock."""
        import anyio

        class ReplayResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # Starlette may cancel response iteration while it is
                    # suspended at a yield. Explicitly close it rather than
                    # leaving lock cleanup to async-generator garbage collection.
                    with anyio.CancelScope(shield=True):
                        await self.body_iterator.aclose()

        def encode(event):
            return (json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode()

        def advance(iterator):
            return next(iterator, None)

        async def events():
            iterator = None
            acquired = False
            try:
                data = await run_in_threadpool(read_archive)
                digest = hashlib.sha256(data).hexdigest()
                with replay_cache_lock:
                    cached = replay_cache.get(digest)
                    if cached is not None:
                        replay_cache.move_to_end(digest)
                if cached is not None:
                    yield cached
                    return
                # Switching recordings aborts the previous response. Give its
                # bounded in-flight physics chunk time to finish and release
                # the lock instead of reporting a spurious busy error.
                acquired = await run_in_threadpool(replay_lock.acquire, True, 0.5)
                if not acquired:
                    yield encode({"kind": "error", "detail":
                                  "Another replay is being prepared. Try again shortly."})
                    return
                iterator = _replay_events(data)
                chunks = []
                size = 0
                completed = False
                while (event := await run_in_threadpool(advance, iterator)) is not None:
                    chunk = encode(event)
                    size += len(chunk)
                    if size <= REPLAY_CACHE_BYTES:
                        chunks.append(chunk)
                    else:
                        chunks.clear()
                    completed = event["kind"] == "complete"
                    yield chunk
                # Only complete, physics-verified streams are cached. Bound
                # memory independently of archive count and student input size.
                if completed and size <= REPLAY_CACHE_BYTES:
                    with replay_cache_lock:
                        replay_cache[digest] = b"".join(chunks)
                        while (len(replay_cache) > REPLAY_CACHE_ITEMS
                               or sum(map(len, replay_cache.values())) > REPLAY_CACHE_BYTES):
                            replay_cache.popitem(last=False)
            except ARCHIVE_ERRORS as exc:
                detail = str(exc) if isinstance(exc, ValueError) else "This recording archive is unavailable or invalid."
                yield encode({"kind": "error", "detail": detail})
            finally:
                # A cancelled response can arrive between worker-thread chunks.
                # Shield cleanup so a second request can start immediately.
                with anyio.CancelScope(shield=True):
                    if iterator is not None:
                        await run_in_threadpool(iterator.close)
                    if acquired:
                        replay_lock.release()

        return ReplayResponse(events(), media_type="application/x-ndjson",
                              headers={"X-Accel-Buffering": "no"})

    @app.get("/api/instructor/submissions/{episode_id}/replay")
    def replay(episode_id: str, request: Request):
        instructor(request)
        path = submitted_path(episode_id)

        def read_archive():
            with path.open("rb") as archive:
                return archive.read(MAX_ARCHIVE_BYTES + 1)

        return prepared_replay(read_archive)

    @app.get("/api/instructor/submissions/{episode_id}/replay-stream")
    def replay_stream(episode_id: str, request: Request):
        instructor(request)
        path = submitted_path(episode_id)

        def read_archive():
            with path.open("rb") as archive:
                return archive.read(MAX_ARCHIVE_BYTES + 1)

        return streamed_replay(read_archive)

    def example_archive(example_id):
        from kaist_rl_lab.apps.coffee_expert import EXAMPLE_COUNT, load_examples

        # Look up opaque IDs before loading anything. They are never filenames
        # supplied by the caller, and this only reads the bundled BC recordings.
        indexes = {f"example-{index + 1}": index for index in range(EXAMPLE_COUNT)}
        if example_id not in indexes:
            raise HTTPException(404, "Generated example not found.")
        return load_examples()[indexes[example_id]]

    @app.get("/api/instructor/examples")
    def examples(request: Request):
        instructor(request)
        from kaist_rl_lab.apps.coffee_expert import load_examples

        rows = []
        for index, data in enumerate(load_examples(), start=1):
            arrays, metadata = _validated_archive(data)
            total_reward, _ = _recorded_rewards(arrays)
            rows.append({
                "example_id": f"example-{index}", "label": f"Practice demonstration {index}",
                "episode_id": metadata["episode_id"], "steps": len(arrays["actions"]),
                "success": metadata["success"],
                "fill_ml": float(arrays["next_observations"][-1, 12]) * 1000,
                "spill_ml": float(arrays["next_observations"][-1, 13]) * 1000,
                "duration_seconds": len(arrays["actions"]) * metadata["dt"],
                "total_reward": total_reward,
                "reward_model": metadata.get("reward_model", "legacy"),
            })
        return rows

    @app.get("/api/instructor/examples/{example_id}/download")
    def download_example(example_id: str, request: Request):
        instructor(request)
        data = example_archive(example_id)
        return Response(data, media_type="application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="coffee_{example_id}.npz"',
        })

    @app.get("/api/instructor/examples/{example_id}/replay")
    def replay_example(example_id: str, request: Request):
        instructor(request)
        return prepared_replay(lambda: example_archive(example_id))

    @app.get("/api/instructor/examples/{example_id}/replay-stream")
    def replay_example_stream(example_id: str, request: Request):
        instructor(request)
        data = example_archive(example_id)
        return streamed_replay(lambda: data)

    @app.get("/instructor")
    def instructor_page():
        return FileResponse(assets / "instructor.html")

    from kaist_rl_lab.apps.coffee_cloning_api import register_cloning_routes
    register_cloning_routes(app, store, instructor, small_json, rate_limit)

    # These pages contain no credentials or submitted data. Every instructor data
    # endpoint performs authentication independently; hiding a URL is not access control.
    app.mount("/", StaticFiles(directory=assets, html=True), name="website")
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
