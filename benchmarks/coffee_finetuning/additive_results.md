# Additive coffee-score benchmark

Reward model: `additive_v1`. Completed runs: 3/3.

A fresh clone is fit once; each run uses ten real candidate trials and ten fresh evaluations with unchanged two-phase search. All completed rollouts are checked against the literal additive formula. The selected policy is independently replayed through physics.

| Seed | BC points | Best points | Best fill (mL) | Seconds | Pot tilt (degrees) | Flow (mL/s) | First ±5 mL evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 287.443978 | 457.174745 | 702.087527 | 31.68750 | 11.931248 | 0.000000 | 4 |
| 2027 | 287.443978 | 458.889869 | 703.774911 | 31.34375 | 11.948700 | 0.000000 | 3 |
| 2028 | 287.443978 | 457.393665 | 703.437866 | 31.53125 | 11.927812 | 0.000000 | 3 |

The best near-target early-stop states are recomputed, then finished using the same `InteractiveSession.finish()` method invoked when a student saves. Their saved reward sums are checked against the formula, including physical flow and pot tilt; finishing twice must not charge twice. The table reports measurements, not a universal no-exploit guarantee.

| Seed | Best stop while flowing | Best stop before upright | Completed score |
|---|---:|---:|---:|
| 2026 | 382.804604 | 386.793032 | 457.174745 |
| 2027 | 384.786329 | 388.546657 | 458.889869 |
| 2028 | 383.385632 | 387.063809 | 457.393665 |

Full candidate/evaluation histories, final costs, replay checks, source hashes, archive hashes, and runtime versions are in the adjacent JSON. These seeds share the canonical starting pose; this checks optimization-seed variability, not unseen starting poses. The broader environment success tolerance remains ±40 mL; the 100-point bonus requires ±5 mL, successful completion, and final flow at most 1 mL/s.

Reproduce with `python benchmarks/coffee_finetuning/additive_benchmark.py --jobs 3`.
