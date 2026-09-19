# Coffee fine-tuning benchmark

Every result below was recomputed from real environment steps. Each training iteration contains an exploration/candidate rollout and a fresh evaluation without noise. Watch best was also replayed through the instructor runtime and matched exactly.

Completed runs: 12/12. Iterations per run: 10. Search speed ceiling: 1.40×.

| Configuration | Seed | Best / final seconds | Best fill (mL) | Best time reward | Best original reward | Successful exploration / evaluation |
|---|---:|---:|---:|---:|---:|---:|
| policy-search | 2026 | 27.750 / 27.750 | 666.372 | 62.279261 | 21.593300 | 10 / 10 |
| policy-search | 2027 | 27.750 / 27.750 | 666.372 | 62.279261 | 21.593300 | 10 / 10 |
| policy-search | 2028 | 27.750 / 27.750 | 666.372 | 62.279261 | 21.593300 | 9 / 10 |
| prior-original | 2026 | 35.812 / 36.000 | 686.027 | 52.305855 | 20.796980 | 10 / 10 |
| prior-original | 2027 | 35.500 / 35.500 | 682.768 | 52.507982 | 20.771355 | 10 / 10 |
| prior-original | 2028 | 36.062 / 35.906 | 686.584 | 52.005898 | 20.778795 | 10 / 10 |
| prior-new-objective | 2026 | 35.531 / 35.562 | 683.203 | 52.495596 | 20.771616 | 10 / 10 |
| prior-new-objective | 2027 | 35.500 / 35.531 | 682.805 | 52.510459 | 20.772099 | 10 / 10 |
| prior-new-objective | 2028 | 35.312 / 35.312 | 677.365 | 52.392137 | 20.658210 | 10 / 10 |
| tuned-ppo | 2026 | 30.562 / 30.531 | 671.132 | 58.586427 | 21.283687 | 10 / 10 |
| tuned-ppo | 2027 | 34.938 / 36.656 | 682.790 | 53.281222 | 20.879692 | 8 / 10 |
| tuned-ppo | 2028 | 34.562 / 34.562 | 682.699 | 53.793216 | 20.953151 | 9 / 10 |

Shared BC baseline: 36.06250 s, 672.05444 mL, time reward 51.005073, original reward 20.448061.

The specified search/training seeds use the same fixed classroom starting pose. They measure optimization-seed variability, not generalization to unseen poses. The environment's success criterion remains unchanged, including its ±40 mL volume tolerance. Faster completion does not necessarily mean a more accurate final volume.

The controlled algorithm comparison is prior-new-objective versus tuned-ppo versus policy-search. Prior-original additionally shows the original algorithm and original objective. Best checkpoints are selected using each configuration's own training objective; both objectives are measured independently for every rollout.

Reference PPO revision: `29b4adb6da78d5aab348b6e7e7d64c6706356de7`. Complete histories, parameter values, runtime versions, source hashes, and archive hashes are in the adjacent JSON file.
