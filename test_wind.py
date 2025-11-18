import gymnasium as gym
import kaist_or_gym
from kaist_or_gym.envs.windy_gridworld import WindyGridworld


env = WindyGridworld(render_mode="human")
obs, info = env.reset()
cumulative_reward = 0.0

for step in range(1000):
    # Sample a random action (uniform over 4 directions)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

    env.render()

    if terminated or truncated:
        print(f"Reached goal at step {step}. Final reward={reward:.2f} | Cumulative={cumulative_reward:.2f}")
        break