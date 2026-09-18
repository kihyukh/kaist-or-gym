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
from collections import defaultdict, deque
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

from kaist_rl_lab.apps.coffee_demonstrations import MAX_ARCHIVE_BYTES, read_demonstration
from kaist_rl_lab.envs import CoffeePouringEnv

COOKIE_NAME = "coffee_instructor"
SESSION_SECONDS = 12 * 60 * 60
MAX_CLASS_SUBMISSIONS = 5000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return arrays, metadata


class ClassroomStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.archives = directory / "archives"
        self.archives.mkdir(parents=True, exist_ok=True)
        self.database = directory / "classroom.sqlite3"
        self.write_lock = Lock()
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
                    duration_seconds REAL NOT NULL, sha256 TEXT NOT NULL, receipt TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS submissions_session ON submissions(session_id);
                CREATE TABLE IF NOT EXISTS instructor_sessions (
                    token_hash TEXT PRIMARY KEY, expires INTEGER NOT NULL
                );
            """)

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

    def receive(self, token: str, data: bytes, arrays, metadata):
        episode_id = str(UUID(metadata["episode_id"]))
        digest = hashlib.sha256(data).hexdigest()
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
                    "INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (episode_id, classroom["id"], metadata["participant"].strip(), received,
                     len(arrays["actions"]), int(metadata["success"]),
                     float(arrays["next_observations"][-1, 12]) * 1000,
                     float(arrays["next_observations"][-1, 13]) * 1000,
                     len(arrays["actions"]) * metadata["dt"], digest, receipt),
                )
                db.commit()
            except Exception:
                if created:
                    path.unlink(missing_ok=True)
                raise
            return {"status": "saved", "episode_id": episode_id,
                    "receipt": receipt, "duplicate": False}


def _replay(data: bytes) -> dict:
    """Sample actual physics frames; never interpolate recorded observations."""
    arrays, metadata = _validated_archive(data)
    env = CoffeePouringEnv(dt=metadata["dt"], horizon=None)
    count = len(arrays["actions"])
    selected = set(np.linspace(0, count, min(count + 1, 400), dtype=int).tolist())
    frames = []

    def append_frame(step, motors):
        snapshot = env.render_snapshot()
        snapshot["playback"] = {
            "generation": 1, "revision": step, "input_sequence": step, "kind": "replay",
            "speed": 1.0, "paused": True, "running": False,
            "motors": list(map(float, motors)), "decision_interval_wall_ms": metadata["dt"] * 1000,
        }
        frames.append({"time": step * metadata["dt"], "snapshot": snapshot})

    try:
        observation, _ = env.reset(seed=metadata["seed"], options={
            "joint_angles": metadata["initial_joint_angles_rad"],
            "target_fill": metadata["target_fill_l"],
        })
        if not np.array_equal(observation, arrays["observations"][0]):
            raise ValueError("The recording cannot be replayed by this environment version.")
        append_frame(0, [0] * 6)
        for index, action in enumerate(arrays["actions"], start=1):
            observation, _, terminated, _, _ = env.step(action)
            if not np.array_equal(observation, arrays["next_observations"][index - 1]):
                raise ValueError("The recording does not match replayed physics.")
            if terminated and index < count:
                raise ValueError("The recording continues after its environment ended.")
            if index in selected:
                append_frame(index, action)
        return {"metadata": metadata, "frames": frames}
    finally:
        env.close()


def create_app(*, data_dir=None, public_base_url=None, password=None, session_secret=None,
               static_dir=None, build_assets=True, allow_missing_origin=False):
    """Application factory; importing the module never starts a server."""
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, JSONResponse, Response
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
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError,
                OverflowError, EOFError, BadZipFile, RuntimeError, ZlibError):
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
            rows = db.execute("""SELECT episode_id, participant, received_at, steps, success,
                                 fill_ml, spill_ml, duration_seconds FROM submissions
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
        return FileResponse(path, media_type="application/octet-stream", filename=path.name)

    @app.get("/api/instructor/submissions/{episode_id}/replay")
    def replay(episode_id: str, request: Request):
        instructor(request)
        path = submitted_path(episode_id)
        if not replay_lock.acquire(blocking=False):
            raise HTTPException(429, "Another replay is being prepared. Try again shortly.")
        try:
            return _replay(path.read_bytes())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        finally:
            replay_lock.release()

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
