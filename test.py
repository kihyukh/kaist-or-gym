import gymnasium as gym
import kaist_or_gym

env = gym.make("kaist-or/TrafficControlEnv-v0", render_mode="human")
env.reset()

# Run a few simulation steps
num_steps = 6000
action = 1
cumulative_time = 0.0
time_intervals = {1: 80, 2: 40}
dt = 0.1
for i in range(num_steps):
    if cumulative_time > time_intervals[action]:
        cumulative_time = 0.0
        action = 1 if action == 2 else 2
    else:
        cumulative_time += 1.0 * dt

    observation, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0: # Render every 10 steps for efficiency
        env.render()

print(env.unwrapped.get_avg_waiting_time())