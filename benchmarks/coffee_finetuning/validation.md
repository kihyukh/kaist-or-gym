# Coffee fine-tuning benchmark

Every result below was recomputed from real environment steps. Each training iteration contains an exploration/candidate rollout and a fresh evaluation without noise. Watch best was also replayed through the instructor runtime and matched exactly.

Completed runs: 3/3. Iterations per run: 10. Search speed ceiling: 1.40×.

| Configuration | Seed | Best / final seconds | Best fill (mL) | Best precision reward | Best original reward | First ±5 mL evaluation | ±5 mL exploration / evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|
| policy-search | 7 | 32.188 / 32.188 | 701.429 | 753.093421 | 21.818657 | 3 | 3 / 8 |
| policy-search | 99 | 32.312 / 32.312 | 699.084 | 769.004042 | 21.801855 | 4 | 3 / 7 |
| policy-search | 2029 | 32.250 / 32.250 | 699.947 | 781.591481 | 21.836105 | 3 | 3 / 8 |

Shared BC baseline: 36.06250 s, 672.05444 mL, precision reward 51.005188, original reward 20.448061.

The specified search/training seeds use the same fixed classroom starting pose. They measure optimization-seed variability, not generalization to unseen poses. The environment's success criterion remains unchanged, including its ±40 mL volume tolerance. The RL objective separately emphasizes ±5 mL, with a smooth reward peak at exactly 700 mL. The table reports precision explicitly rather than conflating it with environment success.

The controlled algorithm comparison is prior-new-objective versus tuned-ppo versus policy-search. Prior-original additionally shows the original algorithm and original objective. Best checkpoints are selected using each configuration's own training objective; both objectives are measured independently for every rollout.

Reference PPO revision: `29b4adb6da78d5aab348b6e7e7d64c6706356de7`. Complete histories, parameter values, runtime versions, source hashes, and archive hashes are in the adjacent JSON file.
