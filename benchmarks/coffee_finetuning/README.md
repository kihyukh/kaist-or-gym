# Coffee policy-learning benchmarks

## Current additive score

The environment, student recordings, policy playback, and fine-tuning now share
`additive_v1` points. From an empty cup, the undiscounted total is:

```text
700 − |final fill in mL − 700|
+ 100 if precisely and safely finished
− 10 × elapsed simulated seconds
− spilled mL
− 2 × |final pot tilt in degrees|
− 5 × final flow in mL/second
```

The 100-point bonus requires successful completion, final error at most 5 mL, and
flow at most 1 mL/s. The existing environment completion conditions additionally
require the pot/cup within 12°/8° of upright and spill at most 20 mL; its broader
fill tolerance remains ±40 mL. Manual saving uses the physical final tilt and
flow, even though the animation stops. Final costs apply once.

There is no extra discount: a second of simulated time costs exactly 10 points,
and pausing costs no simulated time. Fill receives one point per mL toward 700
and loses one point per mL beyond 700. Exact 700 is best at equal duration, spill,
tilt, and flow; the explicit time cost allows tradeoffs between speed and
accuracy. The old Gaussian bonus and its unconditional precision-priority claims
are historical, not part of this score.

## Reproduce the current bounded check

From the repository root with normal project dependencies installed:

```sh
python benchmarks/coffee_finetuning/additive_benchmark.py --jobs 3
```

This fits a fresh clone from the fifteen packaged demonstrations and runs ten
real candidate trials plus ten fresh evaluations for each seed 2026–2028. It
checks every completed return against an independent implementation of the
literal formula, replays each learned best policy through real physics, and
checks actual manual-finish behavior at near-target states. It never uses stored
answers to choose a policy.

[additive_results.md](additive_results.md) is the concise report;
[additive_results.json](additive_results.json) contains complete histories,
state measurements, manual-finish checks, and source/data hashes. The baseline
is **672.054437 mL in 36.0625 seconds**, scoring **287.443978 points**.

| Seed | First evaluation within ±5 mL | Best fill | Time | Points |
|---|---:|---:|---:|---:|
| 2026 | 4 | 702.087527 mL | 31.68750 s | 457.174745 |
| 2027 | 3 | 703.774911 mL | 31.34375 s | 458.889869 |
| 2028 | 3 | 703.437866 mL | 31.53125 s | 457.393665 |

All three selected policies have zero final flow and satisfy the precision
bonus conditions. Their final pot tilt is approximately 11.93–11.95°, within the
unchanged environment completion tolerance; its remaining tilt still costs
points. Every candidate, including failures, stays in the full history.

On these rollouts, the highest-scoring near-target stop while coffee still flows
scores **382.805–384.786**, below the completed scores. Even the best unfinished
near-target stop after flow nearly ceases scores only **386.793–388.547** because
the pot remains tilted and the completion bonus is not earned. These selected
states are recomputed and passed through the actual manual-finish method; saved
reward sums match, and finishing twice does not charge twice. These observations
validate the tested rollouts, not an absolute guarantee for every possible
policy or starting state.

To rerun one seed without overwriting the checked-in measurements:

```sh
python benchmarks/coffee_finetuning/additive_benchmark.py \
  --seeds 2026 --output /tmp/coffee-additive-check.json
```

## What policy search learns

The unchanged search has two speed gains: approach/pouring and returning upright.
Both start at 1.0 and remain between 0.7 and 1.4. Every physics step queries the
clone on the real current observation. The return gain applies when the sum of
the clone's three pot motor commands is below −0.0001; otherwise the approach/pour
gain applies. Motor directions are preserved and existing control limits apply.

Paired candidates change one gain by ± a seeded radius around a shared center.
Only measured reward decides acceptance; each candidate is followed by a fresh
retained-policy evaluation. A useful coordinate keeps its turn. Otherwise its
radius halves and search switches coordinates. Initial radii are 0.12–0.16 with
a 0.001 floor. No volume-error direction, expert controller, or known successful
gain is supplied to this optimizer. This is policy search; PPO remains a
separate actor-critic choice.

These three optimization seeds share one canonical starting pose. This measures
learning-seed variability, not generalization to unseen poses or guarantees for
future student datasets. No new multi-strategy comparison is claimed under the
additive score.

## Historical Gaussian-objective comparison

The files [results.md](results.md), [results.json](results.json),
[validation.md](validation.md), and [validation.json](validation.json) preserve
an earlier experiment with a 1000-point Gaussian precision bonus and 0.99 discount
per second. Their score values, selected policies, and PPO comparisons are
**historical** and must not be presented as current additive-score results.

The older `benchmark.py` belongs to that experiment. Its measurement assumptions
are not suitable for the current additive environment. Reproduce it from the
historical checkout, for example a separate worktree at `b17a734`, using:

```sh
python benchmarks/coffee_finetuning/benchmark.py --jobs 3
```

That historical script compares frozen prior PPO from
`29b4adb6da78d5aab348b6e7e7d64c6706356de7`, prior PPO with the then-new objective,
tuned PPO, and two-phase policy search. It also includes separate validation
seeds 7, 99, and 2029 at the same pose. Those runs tested the previous reward; they have
not been relabeled or reused as new measurements.

## Motor exploration with two generated demonstrations

`residual_benchmark.py` fits the unchanged clone using generated examples 1 and 2,
then runs ten paired motor-correction trials and ten separate policy evaluations.
`residual_results.json` stores all candidates, including rejected trials, and
checks each selected policy with an independent physics replay.

| Seed | Original score | Best score | Final fill | Time |
|---|---:|---:|---:|---:|
| 2026 | 299.78 | 416.79 | 696.02 mL | 35.53 s |
| 2027 | 299.78 | 432.49 | 695.72 mL | 33.94 s |
| 2028 | 299.78 | 427.38 | 696.09 mL | 34.47 s |

These are generated-data tests, not results on the class's two student recordings.
All use the canonical policy start. Sparse or poorly coordinated demonstrations
can still produce a policy that needs more exploration or additional examples.
The method keeps the original clone as a candidate and does not promise success.

Reproduce from the repository root:
`python benchmarks/coffee_finetuning/residual_benchmark.py`.
