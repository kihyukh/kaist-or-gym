"""Private, bounded behavior-cloning jobs for the classroom instructor.

Training reads the selected classroom's archives without changing them. The
example recordings are a separate source and never enter the student dataset.
"""

from __future__ import annotations

from threading import Lock
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

MAX_TRAINING_TRAJECTORIES = 20
MAX_INSPECTED_ARCHIVES = 60
MAX_TRAINING_SAMPLES = 50_000


def _example_archives():
    from kaist_rl_lab.apps.coffee_expert import load_examples

    return load_examples()


def _fit_policy(demonstrations):
    from kaist_rl_lab.apps.coffee_cloning import train_behavior_cloning

    return train_behavior_cloning(demonstrations, max_samples=MAX_TRAINING_SAMPLES)


def _prepare_and_train(store, *, source, session_id, successful_only):
    from fastapi import HTTPException

    from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M
    from kaist_rl_lab.apps.coffee_demonstrations import MAX_ARCHIVE_BYTES
    from kaist_rl_lab.apps.coffee_web import _validated_archive
    from kaist_rl_lab.envs.coffee_pouring import CoffeePouringEnv

    selection = {
        "available_trajectories": 0,
        "eligible_trajectories": 0,
        "inspected_trajectories": 0,
        "used_trajectories": 0,
        "skipped_unsuccessful": 0,
        "skipped_invalid": 0,
        "skipped_incompatible": 0,
        "capped_trajectories": 0,
        "available_transitions": 0,
        "max_trajectories": MAX_TRAINING_TRAJECTORIES,
        "max_training_samples": MAX_TRAINING_SAMPLES,
        "successful_only": successful_only,
    }
    if source == "students":
        with store.connect() as db:
            if not db.execute("SELECT id FROM classes WHERE id=?", (session_id,)).fetchone():
                raise HTTPException(404, "Classroom not found.")
            total = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(success), 0) FROM submissions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            selection["available_trajectories"] = total[0]
            selection["skipped_unsuccessful"] = total[0] - total[1] if successful_only else 0
            selection["eligible_trajectories"] = total[0] - selection["skipped_unsuccessful"]
            rows = db.execute(
                "SELECT episode_id FROM submissions WHERE session_id=? AND (?=0 OR success=1) "
                "ORDER BY received_at DESC, episode_id DESC LIMIT ?",
                (session_id, int(successful_only), MAX_INSPECTED_ARCHIVES),
            ).fetchall()

        # Read one bounded archive at a time. Keeping only float32 state/action
        # pairs bounds retained data to about 53 MB even for 20 maximal archives.
        def archives():
            for row in rows:
                try:
                    with store.archive_path(row["episode_id"]).open("rb") as file:
                        yield file.read(MAX_ARCHIVE_BYTES + 1)
                except OSError:
                    yield b""

        candidates = archives()
    else:
        examples = _example_archives()
        selection["available_trajectories"] = len(examples)
        selection["eligible_trajectories"] = len(examples)
        candidates = iter(examples[:MAX_INSPECTED_ARCHIVES])

    demonstrations = []
    invalid_errors = (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError,
                      OverflowError, EOFError, BadZipFile, RuntimeError, ZlibError)
    for data in candidates:
        selection["inspected_trajectories"] += 1
        try:
            arrays, metadata = _validated_archive(data)
        except invalid_errors:
            selection["skipped_invalid"] += 1
            continue
        distance = metadata.get("arm_base_distance_m", CoffeePouringEnv.DEFAULT_ARM_BASE_DISTANCE)
        if metadata["dt"] != 1 / 32 or not np.isclose(
                metadata["target_fill_l"], 0.7, rtol=0, atol=1e-7) or distance != ARM_BASE_DISTANCE_M:
            selection["skipped_incompatible"] += 1
            del arrays, metadata
            continue
        if successful_only and not metadata["success"]:
            # Normally accounted for by SQL, but also honor the actual archive
            # if a locally restored database and archive disagree.
            selection["skipped_unsuccessful"] += 1
            selection["eligible_trajectories"] -= 1
            del arrays, metadata
            continue
        pairs = {key: arrays[key].astype(np.float32, copy=True)
                 for key in ("observations", "actions")}
        selection["available_transitions"] += len(pairs["actions"])
        demonstrations.append((pairs, {"dt": metadata["dt"],
                                       "target_fill_l": metadata["target_fill_l"],
                                       "arm_base_distance_m": distance}))
        # Drop the unused next states and potentially wider source arrays before
        # decoding another archive or allocating the learner's working arrays.
        del arrays, metadata, data
        if len(demonstrations) >= MAX_TRAINING_TRAJECTORIES:
            break
    selection["used_trajectories"] = len(demonstrations)
    selection["capped_trajectories"] = max(
        0, selection["available_trajectories"] - selection["skipped_unsuccessful"]
        - selection["skipped_invalid"] - selection["skipped_incompatible"]
        - selection["used_trajectories"],
    )
    if not demonstrations:
        if source == "students" and not selection["available_trajectories"]:
            detail = "This class has no submitted trajectories yet. Try the prepared examples."
        elif source == "students" and successful_only and not selection["eligible_trajectories"]:
            detail = ("This class has no successful submitted trajectories yet. Try the prepared "
                      "examples, or turn off the successful-only filter to study all attempts.")
        else:
            detail = ("No compatible trajectories were found. Use recordings made at 32 Hz "
                      f"with the 700 mL target and the current {ARM_BASE_DISTANCE_M:g} m arm spacing, "
                      "or try the prepared examples.")
        raise HTTPException(409, detail)
    try:
        model = _fit_policy(demonstrations)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"model": model, "source": source,
            "source_label": ("Selected class submissions" if source == "students"
                             else "Generated practice demonstrations"),
            "session_id": session_id if source == "students" else None,
            "selection": selection}


def register_cloning_routes(app, store, instructor, small_json, rate_limit):
    """Install the route before the catch-all static mount in the web factory."""
    from fastapi import HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    globals()["Request"] = Request
    training_lock = Lock()

    def training_response(**options):
        from fastapi.responses import JSONResponse

        # Large fitted models are also encoded off the event loop, so student
        # uploads and the service health check can proceed during serialization.
        return JSONResponse(_prepare_and_train(store, **options))

    @app.post("/api/instructor/cloning/train")
    async def train(request: Request):
        instructor(request)
        body = await small_json(request)
        source = body.get("source")
        session_id = body.get("session_id")
        successful_only = body.get("successful_only", True)
        if source not in ("students", "examples"):
            raise HTTPException(400, "Choose student submissions or the prepared examples.")
        if type(successful_only) is not bool:
            raise HTTPException(400, "The successful-only option must be true or false.")
        if source == "students" and (not isinstance(session_id, str)
                                     or not 1 <= len(session_id) <= 128):
            raise HTTPException(400, "Select a class before training from student submissions.")
        if source == "examples" and session_id is not None:
            raise HTTPException(400, "Prepared examples are separate from student classes.")
        if not training_lock.acquire(blocking=False):
            raise HTTPException(429, "Another policy is being trained. Try again shortly.",
                                headers={"Retry-After": "5"})
        try:
            rate_limit(request, "cloning", 3)
            return await run_in_threadpool(
                training_response, source=source, session_id=session_id,
                successful_only=successful_only,
            )
        finally:
            training_lock.release()
