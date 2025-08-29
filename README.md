## Usage Example

Below is a minimal example of how to use the `TrafficControlEnv` environment for a fixed number of time steps:

```python
import gymnasium as gym
import kaist_or_gym

# Create the environment
env = gym.make("kaist-or/TrafficControlEnv-v0", render_mode="human")

observation, info = env.reset()

for _ in range(100):  # Run for 100 time steps
    action = env.action_space.sample()  # Replace with your policy
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()
```

This example demonstrates how to create the environment, take random actions, render the intersection, and run for a fixed number of steps.