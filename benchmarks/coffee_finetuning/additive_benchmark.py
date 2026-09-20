"""Recompute the current additive-score search and actual manual-finish checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kaist_rl_lab.apps.coffee_classroom import (
    ARM_BASE_DISTANCE_M,
    POLICY_START_SEED,
    fixed_policy_layout,
)
from kaist_rl_lab.apps.coffee_cloning import train_behavior_cloning
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import load_examples
from kaist_rl_lab.apps.coffee_finetuning import FineTuningTrainer
from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession
from kaist_rl_lab.envs.coffee_reward import REWARD_MODEL


def state_metrics(session, *, completed=None):
    """Independently evaluate the advertised formula, without using reward terms."""
    info = session.info
    fill = float(session.env.fill * 1000)
    spill = float(session.env.spill * 1000)
    seconds = float(session.env.elapsed_steps * session.env.dt)
    pot = math.degrees(float(session.env.pot_angle))
    flow = float(info["flow_rate"] * 1000)
    success = bool(info["is_success"]) if completed is None else completed
    precise = abs(fill - 700) <= 5 + 1e-9
    bonus = 100.0 if success and precise and flow <= 1 + 1e-9 else 0.0
    score = 700 - abs(fill - 700) + bonus - 10 * seconds - spill - 2 * abs(pot) - 5 * flow
    return {
        "fill_ml": fill, "spill_ml": spill, "seconds": seconds,
        "pot_degrees": pot, "flow_ml_s": flow, "success": success,
        "bonus": bonus, "formula_score": score, "within_5ml": precise,
        "termination_reason": info.get("termination_reason"),
    }


class MeasuredTrainer(FineTuningTrainer):
    def _metrics(self):
        result = super()._metrics()
        measured = state_metrics(self.session)
        result.update(measured)
        if not math.isclose(result["return"], measured["formula_score"], rel_tol=0, abs_tol=1e-7):
            raise AssertionError(("Literal additive formula mismatch", result))
        return result


def new_session():
    return InteractiveSession(
        POLICY_START_SEED, 700, dt=1 / 32, steps_per_update=1, horizon=60 * 32,
        reset_options=fixed_policy_layout(), arm_base_distance=ARM_BASE_DISTANCE_M,
        include_render_info=False,
    )


def advance_policy(session, policy):
    for index, value in enumerate(policy.predict(session.observation)):
        session.set_motor(index, float(value))
    session.advance()


def replay(policy):
    session = new_session()
    near_flowing, near_unfinished = [], []
    try:
        while session.running:
            advance_policy(session, policy)
            if session.running and abs(float(session.env.fill) - 0.7) <= 0.005:
                stop = state_metrics(session, completed=False)
                near_unfinished.append(stop)
                if stop["flow_ml_s"] > 1:
                    near_flowing.append(stop)
        return {
            "final": state_metrics(session), "reward_sum": float(session.cumulative_reward),
            "near_target_flowing_count": len(near_flowing),
            "first_near_target_flowing": near_flowing[0] if near_flowing else None,
            "best_stop_while_flowing": max(near_flowing, key=lambda row: row["formula_score"])
            if near_flowing else None,
            "best_stop_while_unfinished": max(near_unfinished, key=lambda row: row["formula_score"])
            if near_unfinished else None,
        }
    finally:
        session.close()


def check_manual_finish(policy, target, completed_score):
    """Recompute the chosen stop state, then invoke the actual manual-save finish."""
    session = new_session()
    try:
        target_step = round(target["seconds"] * 32)
        while session.env.elapsed_steps < target_step:
            advance_policy(session, policy)
        before = state_metrics(session, completed=False)
        if before != target:
            raise AssertionError("Manual-stop replay did not reach the recorded state.")
        session.finish()
        total = float(session.cumulative_reward)
        if not math.isclose(total, target["formula_score"], rel_tol=0, abs_tol=1e-7):
            raise AssertionError("Manual finish does not match the additive formula.")
        if not math.isclose(sum(row["reward"] for row in session.trajectory), total, abs_tol=1e-9):
            raise AssertionError("The saved transition rewards do not match the displayed total.")
        session.finish()
        if session.cumulative_reward != total:
            raise AssertionError("Repeated finish charged a second terminal cost.")
        return {
            "saved_score": total, "completed_score": completed_score,
            "margin_for_finishing": completed_score - total, "state": before,
            "final_transition_truncated": bool(session.trajectory[-1]["truncated"]),
        }
    finally:
        session.close()


def run_case(seed, model):
    started = time.perf_counter()
    trainer = MeasuredTrainer(model, seed=seed, episodes=10, strategy="policy_search")
    try:
        while not trainer.done:
            trainer.step_chunk(32)
        result = trainer.result()
        best_policy = deepcopy(trainer.best_policy)
        checked = replay(best_policy)
        for field in ("fill_ml", "spill_ml", "seconds", "pot_degrees", "flow_ml_s", "formula_score", "success"):
            if checked["final"][field] != result["best"][field]:
                raise AssertionError(("Best-policy replay mismatch", field))
        if not math.isclose(checked["reward_sum"], result["best"]["return"], rel_tol=0, abs_tol=1e-7):
            raise AssertionError("Best-policy replay reward does not match training.")
        stop_checks = {}
        for name in ("best_stop_while_flowing", "best_stop_while_unfinished"):
            if checked[name] is not None:
                stop_checks[name] = check_manual_finish(best_policy, checked[name], result["best"]["return"])
        return {
            "seed": seed, "result": result, "replay": checked,
            "manual_stop_checks": stop_checks,
            "first_precise_evaluation": next(
                (row["episode"] for row in result["history"]
                 if row["evaluation"]["success"] and row["evaluation"]["within_5ml"]), None,
            ),
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        trainer.close()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Additive coffee-score benchmark", "",
        f"Reward model: `{report['reward_model']}`. Completed runs: {len(report['runs'])}/{len(report['seeds'])}.",
        "", ("A fresh clone is fit once; each run uses ten real candidate trials and ten fresh "
        "evaluations with unchanged two-phase search. All completed rollouts are checked against "
        "the literal additive formula. The selected policy is independently replayed through physics."),
        "", "| Seed | BC points | Best points | Best fill (mL) | Seconds | Pot tilt (degrees) | Flow (mL/s) | First ±5 mL evaluation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        result, best = run["result"], run["result"]["best"]
        lines.append(
            f"| {run['seed']} | {result['baseline']['return']:.6f} | {best['return']:.6f} | "
            f"{best['fill_ml']:.6f} | {best['seconds']:.5f} | {best['pot_degrees']:.6f} | "
            f"{best['flow_ml_s']:.6f} | {run['first_precise_evaluation'] or '—'} |"
        )
    lines += [
        "", ("The best near-target early-stop states are recomputed, then finished using the same "
        "`InteractiveSession.finish()` method invoked when a student saves. Their saved reward sums "
        "are checked against the formula, including physical flow and pot tilt; finishing twice "
        "must not charge twice. The table reports measurements, not a universal no-exploit guarantee."),
        "", "| Seed | Best stop while flowing | Best stop before upright | Completed score |",
        "|---|---:|---:|---:|",
    ]
    for run in report["runs"]:
        checks = run["manual_stop_checks"]
        values = [f"{checks[name]['saved_score']:.6f}" if name in checks else "Not observed"
                  for name in ("best_stop_while_flowing", "best_stop_while_unfinished")]
        lines.append(f"| {run['seed']} | {values[0]} | {values[1]} | {run['result']['best']['return']:.6f} |")
    lines += [
        "", ("Full candidate/evaluation histories, final costs, replay checks, source hashes, archive "
        "hashes, and runtime versions are in the adjacent JSON. These seeds share the canonical "
        "starting pose; this checks optimization-seed variability, not unseen starting poses. "
        "The broader environment success tolerance remains ±40 mL; the 100-point bonus requires "
        "±5 mL, successful completion, and final flow at most 1 mL/s."),
        "", "Reproduce with `python benchmarks/coffee_finetuning/additive_benchmark.py --jobs 3`.",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("additive_results.json"))
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Use 1–8 workers and unsigned 32-bit seeds.")
    archives = load_examples()
    model = train_behavior_cloning([read_demonstration(data) for data in archives])
    paths = [
        "kaist_rl_lab/envs/coffee_reward.py", "kaist_rl_lab/envs/coffee_pouring.py",
        "kaist_rl_lab/apps/coffee_finetuning.py", "kaist_rl_lab/apps/coffee_finetuning_reward.py",
        "kaist_rl_lab/apps/coffee_pouring_app.py", "kaist_rl_lab/apps/coffee_cloning.py",
        "kaist_rl_lab/apps/coffee_classroom.py", "benchmarks/coffee_finetuning/additive_benchmark.py",
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "reward_model": REWARD_MODEL,
        "complete": False, "seeds": args.seeds, "iterations": 10,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": version("numpy"), "gymnasium": version("gymnasium"),
                        "dt": 1 / 32, "arm_base_distance_m": ARM_BASE_DISTANCE_M, "jobs": args.jobs},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
        "archives_sha256": [hashlib.sha256(data).hexdigest() for data in archives],
        "model_metrics": model["metrics"], "runs": [],
    }
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = [pool.submit(run_case, seed, model) for seed in args.seeds]
        for completed in as_completed(pending):
            run = completed.result()
            report["runs"].append(run)
            report["runs"].sort(key=lambda item: item["seed"])
            write_report(args.output, report)
            print(json.dumps({"seed": run["seed"], "best": run["result"]["best"]}), flush=True)
    report["complete"] = True
    report["wall_seconds"] = time.perf_counter() - started
    write_report(args.output, report)


if __name__ == "__main__":
    main()
