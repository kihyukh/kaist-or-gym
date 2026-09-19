"""Recompute coffee fine-tuning comparisons using real physics, never saved answers.

Run from a repository checkout; the prior PPO implementation is loaded from the
exact local Git revision below rather than copied into this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kaist_rl_lab.apps import coffee_finetuning as current
from kaist_rl_lab.apps.coffee_cloning import train_behavior_cloning
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import load_examples
from kaist_rl_lab.apps.coffee_finetuning_reward import fine_tuning_reward
from kaist_rl_lab.apps.coffee_finetuning_runtime import FineTuningRuntime
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession

PRIOR_REVISION = "29b4adb6da78d5aab348b6e7e7d64c6706356de7"
PRIOR_PATH = "kaist_rl_lab/apps/coffee_finetuning.py"
CONFIGURATIONS = ("prior-original", "prior-new-objective", "tuned-ppo", "policy-search")
TUNED_PARAMETERS = {
    "SPEED_BOUND": 0.5, "LATENT_STD": 0.5, "MAX_UPDATE_KL": 0.08,
    "MAX_MEAN_CHANGE": 0.6, "MAX_LATENT_MEAN": 1.25, "DECISION_STEPS": 16,
    "ACTOR_LEARNING_RATE": 0.15, "PPO_EPOCHS": 12,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


class MeasuredSession(InteractiveSession):
    """Observe both rewards without changing the environment or its recordings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_measurements()

    def _reset_measurements(self):
        self.original_discounted_return = 0.0
        self.precision_discounted_return = 0.0
        self.measurement_discount = 1.0

    def restart(self, *args, **kwargs):
        super().restart(*args, **kwargs)
        self._reset_measurements()

    def advance(self):
        advanced = super().advance()
        if advanced:
            self.original_discounted_return += (
                self.measurement_discount * self.trajectory[-1]["reward"]
            )
            self.precision_discounted_return += (
                self.measurement_discount * fine_tuning_reward(self.info, self.env.dt)
            )
            self.measurement_discount *= 0.99**self.env.dt
        return advanced


def load_prior(source: str, use_new_objective: bool):
    """Freeze the algorithm; the optional single replacement changes only reward."""
    if use_new_objective:
        original = 'reward = self.session.trajectory[-1]["reward"]'
        if source.count(original) != 1:
            raise RuntimeError("The frozen PPO reward call no longer matches the reference.")
        source = source.replace(original, "reward = fine_tuning_reward(self.session.info, BROWSER_DT)")
    module = types.ModuleType("frozen_coffee_ppo_benchmark")
    module.fine_tuning_reward = fine_tuning_reward
    # Execute only this checkout's explicitly pinned historical implementation.
    exec(compile(source, f"git:{PRIOR_REVISION}:{PRIOR_PATH}", "exec"), module.__dict__)  # noqa: S102
    return module


def watch_best(model, policy, result):
    """Exercise the same live Watch best runtime used by the instructor page."""
    runtime = FineTuningRuntime()
    try:
        runtime.dispatch(json.dumps({"kind": "ft-load", "model": model}))
        runtime.best_policy = deepcopy(policy)
        runtime.result = {"evaluation_seed": result["evaluation_seed"], "baseline": result["baseline"]}
        runtime.dispatch('{"kind":"ft-run","policy":"best","speed":0}')
        while runtime.rollout_active:
            state = json.loads(runtime.dispatch('{"kind":"tick","max_steps":32}'))
        rollout = state["finetuning"]["rollout"]
        actual = {
            "precision_discounted_return": rollout["reward"], "raw_return": rollout["raw_return"],
            "seconds": rollout["elapsed_seconds"], "fill_ml": runtime.session.env.fill * 1000,
            "spill_ml": runtime.session.env.spill * 1000,
            "success": bool(runtime.session.info["is_success"]), "outcome": rollout["outcome"],
        }
        for field, value in actual.items():
            if value != result["best"][field]:
                raise AssertionError(("Watch best mismatch", field, value, result["best"][field]))
        return actual
    finally:
        runtime.close()


def run_case(name, seed, iterations, model, prior_source, search_cap):
    started = time.perf_counter()
    is_prior = name.startswith("prior-")
    learner = load_prior(prior_source, name == "prior-new-objective") if is_prior else current
    if not is_prior:
        for key, value in TUNED_PARAMETERS.items():
            setattr(learner, key, value)
        learner.SEARCH_SPEED_MAX = search_cap
    # Each worker executes runs sequentially. Restoring this hook avoids leaking
    # benchmark instrumentation into another run or the Watch best runtime.
    original_session = learner.InteractiveSession
    learner.InteractiveSession = MeasuredSession

    class MeasuredTrainer(learner.FineTuningTrainer):
        def _metrics(self):
            metrics = super()._metrics()
            metrics.update(
                original_discounted_return=self.session.original_discounted_return,
                precision_discounted_return=self.session.precision_discounted_return,
            )
            selected = (metrics["original_discounted_return"] if name == "prior-original"
                        else metrics["precision_discounted_return"])
            if metrics["return"] != selected:
                raise AssertionError("The independent reward measurement disagrees with training.")
            return metrics

    kwargs = {} if is_prior else {"strategy": "policy_search" if name == "policy-search" else "ppo"}
    trainer = None
    try:
        trainer = MeasuredTrainer(model, seed=seed, episodes=iterations, **kwargs)
        while not trainer.done:
            trainer.step_chunk(max_steps=32)
        result = trainer.result()
        if len(result["history"]) != iterations or any(
            row["evaluation"] is None for row in result["history"]
        ):
            raise AssertionError("An exploration or independent evaluation is missing.")
        replay = watch_best(model, trainer.best_policy, result)
        parameters = {
            key: getattr(learner, key) for key in (
                "SPEED_BOUND", "LATENT_STD", "MAX_UPDATE_KL", "MAX_MEAN_CHANGE",
                "MAX_LATENT_MEAN", "DECISION_STEPS", "GAE_LAMBDA", "CRITIC_RETENTION",
            )
        }
        parameters.update(
            ACTOR_LEARNING_RATE=0.025 if is_prior else learner.ACTOR_LEARNING_RATE,
            PPO_EPOCHS=4 if is_prior else learner.PPO_EPOCHS,
            search_speed_range=[0.7, search_cap],
            search_radius_range=[current.SEARCH_RADIUS_MIN, current.SEARCH_RADIUS_MAX],
            search_radius_floor=current.SEARCH_RADIUS_FLOOR,
            search_parameterization="approach_pour_and_return_gains",
        )
        return {
            "configuration": name, "seed": seed, "iterations": iterations,
            "training_objective": "original" if name == "prior-original" else "precision",
            "parameters": parameters, "baseline": result["baseline"], "best": result["best"],
            "final": result["history"][-1]["evaluation"],
            "successful_candidates": sum(row["training"]["success"] for row in result["history"]),
            "successful_evaluations": sum(row["evaluation"]["success"] for row in result["history"]),
            "precise_candidates": sum(row["training"]["success"] and abs(row["training"]["fill_ml"] - 700) <= 5 for row in result["history"]),
            "precise_evaluations": sum(row["evaluation"]["success"] and abs(row["evaluation"]["fill_ml"] - 700) <= 5 for row in result["history"]),
            "first_precise_evaluation": next((row["episode"] for row in result["history"] if row["evaluation"]["success"] and abs(row["evaluation"]["fill_ml"] - 700) <= 5), None),
            "watch_best_exact": replay, "wall_seconds": time.perf_counter() - started,
            "result": result,
        }
    finally:
        if trainer is not None:
            trainer.close()
        learner.InteractiveSession = original_session


def write_results(output: Path, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    lines = [
        "# Coffee fine-tuning benchmark", "",
        ("Every result below was recomputed from real environment steps. Each training iteration "
        "contains an exploration/candidate rollout and a fresh evaluation without noise. "
        "Watch best was also replayed through the instructor runtime and matched exactly."), "",
        (f"Completed runs: {len(report['runs'])}/{report['expected_runs']}. "
        f"Iterations per run: {report['iterations']}. "
        f"Search speed ceiling: {report['search_cap']:.2f}×."), "",
        ("| Configuration | Seed | Best / final seconds | Best fill (mL) | Best precision reward | "
        "Best original reward | First ±5 mL evaluation | ±5 mL exploration / evaluation |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        best, final = run["best"], run["final"]
        lines.append(
            f"| {run['configuration']} | {run['seed']} | {best['seconds']:.3f} / "
            f"{final['seconds']:.3f} | {best['fill_ml']:.3f} | {best['precision_discounted_return']:.6f} "
            f"| {best['original_discounted_return']:.6f} | {run['first_precise_evaluation'] or '—'} | {run['precise_candidates']} / "
            f"{run['precise_evaluations']} |"
        )
    if report["runs"]:
        baseline = report["runs"][0]["baseline"]
        lines.extend([
            "", (f"Shared BC baseline: {baseline['seconds']:.5f} s, {baseline['fill_ml']:.5f} mL, "
            f"precision reward {baseline['precision_discounted_return']:.6f}, original reward "
            f"{baseline['original_discounted_return']:.6f}."),
        ])
    lines.extend([
        "", ("The specified search/training seeds use the same fixed classroom starting pose. "
        "They measure optimization-seed variability, not generalization to unseen poses. "
        "The environment's success criterion remains unchanged, including its ±40 mL volume tolerance. "
        "The RL objective separately emphasizes ±5 mL, with a smooth reward peak at exactly 700 mL. "
        "The table reports precision explicitly rather than conflating it with environment success."), "",
        ("The controlled algorithm comparison is prior-new-objective versus tuned-ppo versus "
        "policy-search. Prior-original additionally shows the original algorithm and original objective. "
        "Best checkpoints are selected using each configuration's own training objective; both "
        "objectives are measured independently for every rollout."), "",
        (f"Reference PPO revision: `{PRIOR_REVISION}`. Complete histories, parameter values, "
        "runtime versions, source hashes, and archive hashes are in the adjacent JSON file."), "",
    ])
    output.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--search-cap", type=float, default=1.4)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100 or not 1 <= args.jobs <= 8 or not 1 <= args.search_cap <= 1.6:
        parser.error("Use 1–100 iterations, 1–8 jobs, and a speed ceiling between 1 and 1.6.")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Seeds must be unsigned 32-bit integers.")
    prior_source = git("show", f"{PRIOR_REVISION}:{PRIOR_PATH}")
    archives = load_examples()
    model = train_behavior_cloning([read_demonstration(data) for data in archives])
    source_paths = [
        PRIOR_PATH, "kaist_rl_lab/apps/coffee_finetuning_reward.py",
        "kaist_rl_lab/apps/coffee_finetuning_runtime.py", "kaist_rl_lab/apps/coffee_classroom.py",
        "kaist_rl_lab/apps/coffee_cloning.py", "kaist_rl_lab/apps/coffee_pouring_app.py",
        "kaist_rl_lab/envs/coffee_pouring.py", "benchmarks/coffee_finetuning/benchmark.py",
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
        "expected_runs": len(args.configurations) * len(args.seeds), "iterations": args.iterations,
        "configurations": args.configurations, "seeds": args.seeds, "search_cap": args.search_cap,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": version("numpy"), "gymnasium": version("gymnasium"),
                        "jobs": args.jobs, "dt": 1 / 32, "arm_base_distance_m": model["arm_base_distance_m"]},
        "git_revision": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "reference_ppo_revision": PRIOR_REVISION, "reference_ppo_sha256": digest(prior_source.encode()),
        "source_sha256": {path: digest((ROOT / path).read_bytes()) for path in source_paths},
        "archives_sha256": [digest(data) for data in archives], "model_metrics": model["metrics"],
        "runs": [],
    }
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = [
            pool.submit(run_case, name, seed, args.iterations, model, prior_source, args.search_cap)
            for name in args.configurations for seed in args.seeds
        ]
        for completed in as_completed(pending):
            run = completed.result()
            report["runs"].append(run)
            report["runs"].sort(key=lambda row: (args.configurations.index(row["configuration"]), row["seed"]))
            write_results(args.output, report)
            print(json.dumps({"configuration": run["configuration"], "seed": run["seed"],
                              "best_seconds": run["best"]["seconds"],
                              "best_time_reward": run["best"]["precision_discounted_return"]}), flush=True)
    report["complete"] = True
    report["wall_seconds"] = time.perf_counter() - started
    write_results(args.output, report)


if __name__ == "__main__":
    main()
