"""Small random-rollout example for the fixed-link coffee-pouring environment."""

import gymnasium as gym
from PIL import Image

import kaist_or_gym  # noqa: F401 - registers kaist-or environments

env = gym.make("kaist-or/CoffeePouringEnv-v0", render_mode="rgb_array")
observation, info = env.reset(seed=7001)

while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break

Image.fromarray(env.render()).save("coffee_pouring_last_frame.png")
print(
    "fill={:.0f} mL, error={:.0f} mL, spill={:.0f} mL, success={}".format(
        info["fill"] * 1000,
        info["fill_error"] * 1000,
        info["spill"] * 1000,
        info["is_success"],
    )
)
env.close()
