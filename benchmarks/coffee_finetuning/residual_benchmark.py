"""Check motor exploration using two generated recordings, never student data."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
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
from kaist_rl_lab.envs.coffee_pouring import CoffeePouringEnv


def run(seed):
    model = train_behavior_cloning([read_demonstration(data) for data in load_examples()[:2]])
    trainer = FineTuningTrainer(model, seed=seed, episodes=10, strategy='residual_search')
    try:
        while not trainer.done:
            trainer.step_chunk()
        result = trainer.result()
        env = CoffeePouringEnv(dt=1 / 32, horizon=1920, arm_base_distance=ARM_BASE_DISTANCE_M,
                              include_render_info=False)
        obs, _ = env.reset(seed=POLICY_START_SEED, options={**fixed_policy_layout(), 'target_fill': .7})
        score = 0.
        while True:
            obs, reward, terminal, truncated, _info = env.step(trainer.best_policy.predict(obs))
            score += reward
            if terminal or truncated:
                break
        assert abs(score - result['best']['return']) < 1e-7
        assert result['best']['return'] > result['baseline']['return']
        assert all(row['evaluation'] is not None for row in result['history'])
        assert any(not row['update']['accepted'] for row in result['history'])
        env.close()
        return result
    finally:
        trainer.close()


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        runs = list(pool.map(run, [2026, 2027, 2028]))
    output = Path(__file__).with_name('residual_results.json')
    output.write_text(json.dumps({'dataset': 'generated examples 1 and 2; no student recordings',
                                  'strategy': 'residual_search', 'runs': runs}, indent=2) + '\n')
    for result in runs:
        print(json.dumps({key: result[key] for key in ['seed', 'baseline', 'best']}))
