"""Fine-tuning uses the environment's additive points without extra discount."""

from kaist_rl_lab.envs.coffee_reward import DISCOUNT_PER_SECOND

__all__ = ["DISCOUNT_PER_SECOND", "fine_tuning_reward"]


def fine_tuning_reward(info: dict, dt: float) -> float:
    """Return the same transition reward seen by students and cloned policies.

    The environment already charges 10 points per simulated second, using its
    physical dt. Keep this call signature shared by training and live playback.
    """
    return float(sum(info["reward_terms"].values()))
