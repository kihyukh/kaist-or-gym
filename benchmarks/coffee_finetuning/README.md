# Reproduce the coffee fine-tuning comparison

From the repository root, run:

```sh
python benchmarks/coffee_finetuning/benchmark.py --jobs 3
```

This fits a fresh clone from the packaged 15 demonstrations, then runs ten real
exploration episodes and ten fresh evaluations for each strategy and seed. It
also replays every best checkpoint through the instructor's **Watch best**
runtime and checks that the reward, duration, final liquid amounts, and outcome
match exactly. It does not load saved benchmark answers, skip failed candidates,
or inject actions into the live website.

The default four configurations are:

- `prior-original`: original PPO source at Git revision
  `29b4adb6da78d5aab348b6e7e7d64c6706356de7`, with its original environment reward.
- `prior-new-objective`: the same frozen PPO source, changing only the reward
  call to the current time objective.
- `tuned-ppo`: current actor-critic implementation with the explicitly recorded
  wider exploration and faster update configuration.
- `policy-search`: bounded mirrored speed search, with a 1.4× maximum speed.

The controlled comparison uses the latter three configurations, which train on
the same reward. Both discounted objectives are independently measured on every
rollout in all configurations. Best and final policies are reported separately.

The checkout must contain the reference Git revision. The script reads it with
`git show`; it does not fetch code from the network. NumPy, Gymnasium, and the
project's normal runtime dependencies must be installed. Native Python results
measure learning behavior; browser wall-clock speed will differ.

Results are written to `results.json` and `results.md`, including all candidate
and evaluation histories, runtime versions, source and dataset hashes, and exact
parameters. To write a separate report or run a smaller check:

```sh
python benchmarks/coffee_finetuning/benchmark.py \
  --configurations policy-search --seeds 2026 --iterations 2 \
  --output /tmp/coffee-search-check.json
```

Every invocation recomputes the requested runs. Training seeds 2026–2028 share
the same fixed classroom pose: these checks assess variation in optimization,
not generalization to unseen starts. The target, physics, and environment success
criterion are unchanged. Faster completion can end farther from exactly 700 mL;
the report includes final fill and spill rather than hiding that tradeoff.

## Independent validation seeds

[validation.json](validation.json) records a separate production-trainer run on
seeds **7, 99, and 2029**, which were not used to choose the search parameters.
All three reached the 1.4× ceiling within **8, 7, and 7 iterations**, respectively.
Their retained policy finished in **27.75 s** with **666.37249 mL** and time reward
**62.279261**, versus the clone's **36.0625 s** and **51.005073**. All **30** fresh
current-policy evaluations succeeded; **27 of 30** exploratory candidates
succeeded, with unsuccessful candidates rejected. The archives were unchanged.

Recompute this validation cohort with:

```sh
python benchmarks/coffee_finetuning/benchmark.py \
  --configurations policy-search --seeds 7 99 2029 --jobs 3 \
  --output /tmp/coffee-validation.json
```

## Why the search speed ceiling is 1.4×

Calibration on seeds 2026–2028 compared larger ceilings using the same physics
and success test. A 1.45× ceiling reached **27.09375 s / 662.03432 mL** in all
three runs. A 1.5× ceiling produced **26.59375–27 s / 661.28958–662.98436 mL**.
The unrestricted 1.6× calibration reached **26.21875–27 s**, but one final policy
finished only **0.04281 mL above** the existing 660 mL lower success boundary.

The selected 1.4× ceiling keeps **6.37249 mL** of margin above that boundary,
while still reducing completion time by **23.05%**. This is a speed improvement;
it does not improve volume accuracy. The ceiling itself is an explicit policy
constraint, not a changed environment success rule. The CLI's `--search-cap`
option can reproduce these calibration comparisons.
