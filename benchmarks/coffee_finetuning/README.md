# Reproduce the coffee fine-tuning comparison

From the repository root, run:

```sh
python benchmarks/coffee_finetuning/benchmark.py --jobs 3
```

This fits a fresh clone from the packaged 15 cautious demonstrations, then runs
ten actual exploration trials and ten fresh evaluations for each strategy and
seed. It also replays every best checkpoint through the instructor's **Watch
best** runtime and checks that reward, duration, fill, spill, and outcome match
exactly. It never loads saved answers or substitutes a prerecorded policy.

## Accuracy-first objective and comparison

The RL objective preserves the existing physics and completion conditions, but
places a narrow reward peak at exactly 700 mL. A safely completed pour receives
an additional `1000 * exp(-0.5 * (error_mL / 5)**2)` precision bonus, on top of
the 100-point completion bonus, time cost, spill/control penalties, and terminal
error cost. Discounting is 0.99 per second. The potential-based fill term gives
dense feedback without changing the ranking of completed trajectories. Thus a
fast 666 mL pour is no longer the preferred learning outcome. The environment's
legacy ±40 mL success flag and the RL target of ±5 mL are reported separately.

Four configurations run by default:

- `prior-original`: frozen PPO source at Git revision
  `29b4adb6da78d5aab348b6e7e7d64c6706356de7`, with its original environment reward.
- `prior-new-objective`: that same PPO source, changing only the reward call to
  the current precision objective.
- `tuned-ppo`: current actor-critic implementation, with the explicitly recorded
  wider exploration and faster update configuration.
- `policy-search`: reward-selected, paired coordinate search over two bounded
  speed gains: approach/pouring and returning upright.

The latter three provide the controlled comparison on the same objective.
Original and precision rewards are independently measured for every rollout in
all configurations. Best and final policies are reported separately. The prior
configuration retains its original single speed actor; it is not given the
search policy's two separate phase parameters.

## What policy search learns

Both gains start at 1.0 and remain between 0.7 and 1.4. Every physics step queries
the frozen clone on the actual current observation. The return gain applies
when the sum of the clone's three pot motor commands is below −0.0001; otherwise
the approach/pouring gain applies. The six controls retain their directions and
are clipped to their existing limits. No observation, target, environment state,
or volume measurement is changed.

A candidate pair changes one gain by ± a seeded radius around the same pair
center. Each candidate is accepted only if its measured reward improves, then
the retained policy is evaluated afresh. Search continues along a coordinate
while either candidate improves reward; otherwise it halves that coordinate's
radius and switches coordinates. Initial radii are uniformly drawn from
0.12–0.16, with a 0.001 floor after adaptation. Neither the direction of the fill
error nor a known successful gain is supplied to the optimizer. This method is
policy search, not actor-critic; the PPO choice remains available for comparison.

## Recorded results and validation

[results.md](results.md) summarizes the complete controlled comparison;
[results.json](results.json) contains every candidate, fresh evaluation, runtime
version, source and archive hash, and parameter. The cautious BC baseline is
**672.05444 mL in 36.0625 s**, outside the ±5 mL band. The first ten policy-search
trials produce:

| Optimization seed | Best fill | Best duration | First evaluation within ±5 mL |
|---|---:|---:|---:|
| 2026 | 699.87285 mL | 32.50000 s | 4 |
| 2027 | 699.38528 mL | 32.71875 s | 3 |
| 2028 | 702.07466 mL | 32.31250 s | 3 |

A separate cohort, not used to choose the search schedule, is recorded in
[validation.json](validation.json) and [validation.md](validation.md):

| Validation seed | Best fill | Best duration | First evaluation within ±5 mL |
|---|---:|---:|---:|
| 7 | 701.42932 mL | 32.18750 s | 3 |
| 99 | 699.08433 mL | 32.31250 s | 4 |
| 2029 | 699.94746 mL | 32.25000 s | 3 |

All six retained policies are within ±5 mL and faster than the clone. These are
measured results for this demonstration dataset and fixed classroom pose,
not a guarantee for other student datasets, optimization seeds, or starting
poses. Failed and imprecise exploratory candidates remain visible in the full
histories. The website recomputes its own policy when training starts.

Recompute the validation cohort with:

```sh
python benchmarks/coffee_finetuning/benchmark.py \
  --configurations policy-search --seeds 7 99 2029 --jobs 3 \
  --output /tmp/coffee-validation.json
```

For a smaller check or a separate report:

```sh
python benchmarks/coffee_finetuning/benchmark.py \
  --configurations policy-search --seeds 2026 --iterations 2 \
  --output /tmp/coffee-search-check.json
```

The checkout must contain the reference revision: the script reads it locally
with `git show`, without fetching code. NumPy, Gymnasium, and the project's
normal dependencies must be installed. Native Python measures learning behavior;
browser wall-clock speed differs. The optional `--search-cap` changes both gain
ceilings for calibration; the shipped policy uses 1.4.
