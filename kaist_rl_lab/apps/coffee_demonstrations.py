"""Portable coffee trajectories and an upload-only classroom collector.

Only the instructor mounts Drive. Student runtimes send validated NumPy archives
through the collector API and never receive filesystem paths or Drive credentials.
"""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID
from zipfile import ZipFile

import numpy as np

from kaist_rl_lab.envs import CoffeePouringEnv
from kaist_rl_lab.version import __version__

MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_TRANSITIONS = 30_000
ARRAY_NAMES = {"observations", "actions", "rewards", "next_observations", "terminated", "truncated"}


def encode_demonstration(session: Any, participant: str = "") -> bytes:
    """Serialize a finished attempt without pickle or browser-only animation frames."""
    rows = session.trajectory
    if not rows:
        raise ValueError("No trajectory yet. Resume time and make an attempt first.")
    if len(rows) > MAX_TRANSITIONS:
        raise ValueError("This attempt is too long to submit. Save shorter classroom attempts.")
    participant = str(participant).strip()
    if len(participant) > 64:
        raise ValueError("Use a participant code of at most 64 characters.")
    metadata = {
        "schema_version": 1,
        "environment": "kaist-or/CoffeePouringEnv-v0",
        "package_version": __version__,
        "physics_model": "torricelli_ballistic_v3",
        "episode_id": session.episode_id,
        "participant": participant,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": session.seed,
        "dt": session.env.dt,
        "physics_substep": session.env.LIQUID_SUBSTEP,
        "steps_per_update": session.steps_per_update,
        "target_fill_l": session.env.target_fill,
        "joint_names": list(CoffeePouringEnv.JOINT_NAMES),
        "observation_names": list(CoffeePouringEnv.OBSERVATION_NAMES),
        "manual_finish": session.manual_finish,
        "success": bool(session.info["is_success"]),
        "fill_l": float(session.info["fill"]),
        "spill_l": float(session.info["spill"]),
    }
    data = BytesIO()
    np.savez_compressed(
        data,
        observations=np.stack([r["observation"] for r in rows]).astype(np.float32),
        actions=np.stack([r["action"] for r in rows]).astype(np.float32),
        rewards=np.asarray([r["reward"] for r in rows], dtype=np.float32),
        next_observations=np.stack([r["next_observation"] for r in rows]).astype(np.float32),
        terminated=np.asarray([r["terminated"] for r in rows], dtype=np.bool_),
        truncated=np.asarray([r["truncated"] for r in rows], dtype=np.bool_),
        metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
    )
    return data.getvalue()


def read_demonstration(data: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate an untrusted upload before it can enter the teaching dataset."""
    if not data or len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("Trajectory archive is empty or too large.")
    with ZipFile(BytesIO(data)) as archive:
        if {f.filename for f in archive.infolist()} != {
            f"{k}.npy" for k in ARRAY_NAMES | {"metadata"}
        }:
            raise ValueError("Unexpected trajectory archive contents.")
        if (
            len(archive.infolist()) != 7
            or sum(f.file_size for f in archive.infolist()) > MAX_EXPANDED_BYTES
        ):
            raise ValueError("Expanded trajectory archive is too large.")
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        raw_metadata = archive["metadata"]
        if (
            raw_metadata.shape != ()
            or raw_metadata.dtype.kind != "U"
            or raw_metadata.nbytes > 65536
        ):
            raise ValueError("Invalid trajectory metadata.")
        metadata = json.loads(str(raw_metadata))
        arrays = {key: archive[key] for key in ARRAY_NAMES}
    if (
        metadata.get("schema_version") != 1
        or metadata.get("environment") != "kaist-or/CoffeePouringEnv-v0"
    ):
        raise ValueError("Unsupported trajectory format or environment.")
    UUID(metadata["episode_id"])
    if len(str(metadata.get("participant", ""))) > 64:
        raise ValueError("Participant code is too long.")
    if (
        metadata.get("dt") != CoffeePouringEnv.DEFAULT_DT
        or metadata.get("physics_substep") != CoffeePouringEnv.LIQUID_SUBSTEP
        or metadata.get("physics_model") != "torricelli_ballistic_v3"
        or metadata.get("joint_names") != list(CoffeePouringEnv.JOINT_NAMES)
        or metadata.get("observation_names") != list(CoffeePouringEnv.OBSERVATION_NAMES)
    ):
        raise ValueError("Trajectory timing or observation/action definitions do not match.")
    count = len(arrays["observations"])
    if not 1 <= count <= MAX_TRANSITIONS:
        raise ValueError("Invalid number of trajectory transitions.")
    widths = {"observations": 16, "next_observations": 16, "actions": 6}
    for key, value in arrays.items():
        expected = (count, widths[key]) if key in widths else (count,)
        if (
            value.shape != expected
            or value.dtype.kind not in "fbiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid {key} array.")
    if np.any(np.abs(arrays["actions"]) > 1):
        raise ValueError("Motor commands must be between -1 and 1.")
    for key in ("terminated", "truncated"):
        if arrays[key].dtype != np.bool_:
            raise ValueError("Episode flags must be boolean arrays.")
    ended = arrays["terminated"] | arrays["truncated"]
    if ended[:-1].any() or not ended[-1]:
        raise ValueError("A submitted attempt must end at its last transition.")
    if not np.array_equal(arrays["observations"][1:], arrays["next_observations"][:-1]):
        raise ValueError("Trajectory observations are not consecutive.")
    return arrays, metadata


class TrajectoryCollector:
    """Write unique, validated attempts into an instructor-owned Drive directory."""

    def __init__(self, directory: str | Path, lecture_code: str):
        if len(lecture_code) < 12:
            raise ValueError("Use a lecture code with at least 12 characters.")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lecture_code = lecture_code
        self._lock = RLock()

    def receive(self, lecture_code: str, encoded_archive: str) -> dict[str, Any]:
        if not isinstance(lecture_code, str) or not hmac.compare_digest(
            lecture_code, self._lecture_code
        ):
            raise ValueError("Incorrect lecture code.")
        if (
            not isinstance(encoded_archive, str)
            or len(encoded_archive) > 4 * MAX_ARCHIVE_BYTES // 3 + 4
        ):
            raise ValueError("Trajectory upload is too large.")
        data = base64.b64decode(encoded_archive, validate=True)
        arrays, metadata = read_demonstration(data)
        episode_id = str(UUID(metadata["episode_id"]))
        path = self.directory / f"coffee_{episode_id}.npz"
        with self._lock:
            duplicate = path.exists()
            if duplicate:
                if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
                    raise ValueError("A different recording already exists for this episode.")
            else:
                created = False
                try:
                    with path.open("xb") as file:
                        created = True
                        file.write(data)
                        file.flush()
                except Exception:
                    if created:
                        path.unlink(missing_ok=True)
                    raise
        receipt = {
            "episode_id": episode_id,
            "transitions": len(arrays["actions"]),
            "duplicate": duplicate,
        }
        print(f"Received {episode_id}: {receipt['transitions']} transitions", flush=True)
        return receipt

    def summary(self) -> list[dict[str, Any]]:
        """Instructor-side summary; deliberately not exposed by the public API."""
        result = []
        with self._lock:
            for path in sorted(self.directory.glob("coffee_*.npz")):
                arrays, metadata = read_demonstration(path.read_bytes())
                result.append(
                    {
                        "participant": metadata["participant"],
                        "episode_id": metadata["episode_id"],
                        "transitions": len(arrays["actions"]),
                        "success": metadata["success"],
                        "fill_mL": round(metadata["fill_l"] * 1000),
                        "spill_mL": round(metadata["spill_l"] * 1000),
                    }
                )
        return result


def build_collector(collector: TrajectoryCollector):
    """Expose an upload-only endpoint; Drive files never become Gradio outputs."""
    import gradio as gr

    def receive(lecture_code: str, encoded_archive: str):
        try:
            return collector.receive(lecture_code, encoded_archive)
        except Exception as exc:
            raise gr.Error("Submission rejected. Check the lecture code and recording.") from exc

    with gr.Blocks(title="Coffee trajectory collector") as app:
        gr.Markdown("# Classroom trajectory collector\nSubmit your attempt from the coffee demo.")
        code = gr.Textbox(type="password", visible=False)
        payload = gr.Textbox(visible=False)
        receipt = gr.JSON(visible=False)
        send = gr.Button("Submit", visible=False)
        send.click(
            receive,
            inputs=[code, payload],
            outputs=receipt,
            api_name="submit_trajectory",
            queue=False,
            show_progress="hidden",
        )
    return app


def submit_demonstration(url: str, lecture_code: str, data: bytes) -> dict[str, Any]:
    from gradio_client import Client

    client = Client(url, verbose=False)
    try:
        job = client.submit(
            lecture_code, base64.b64encode(data).decode("ascii"), api_name="/submit_trajectory"
        )
        receipt = job.result(timeout=60)
    finally:
        client.close()
    expected_id = read_demonstration(data)[1]["episode_id"]
    if not isinstance(receipt, dict) or receipt.get("episode_id") != expected_id:
        raise ValueError("The collector did not confirm receipt of this recording.")
    return receipt


def load_behavior_cloning_data(directory: str | Path) -> dict[str, np.ndarray]:
    """Load validated attempts, keeping episode IDs available for train/test splits."""
    observations, actions, episodes = [], [], []
    for path in sorted(Path(directory).glob("coffee_*.npz")):
        arrays, metadata = read_demonstration(path.read_bytes())
        observations.append(arrays["observations"])
        actions.append(arrays["actions"])
        episodes.append(np.full(len(arrays["actions"]), metadata["episode_id"]))
    if not observations:
        raise ValueError("No submitted trajectories found in this directory.")
    return {
        "observations": np.concatenate(observations),
        "actions": np.concatenate(actions),
        "episode_ids": np.concatenate(episodes),
    }
