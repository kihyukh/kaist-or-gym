import gymnasium as gym
import kaist_or_gym

env = gym.make("kaist-or/TrafficControlEnv-v0", render_mode="human")
env.reset()

# Run a few simulation steps
num_steps = 6000 # Increased the number of steps to better observe the effect of rendering frequency
action = 1
for i in range(num_steps):
    # Take action 1 to turn on GR, then action 2 to trigger yellow light transition
    if i % 600 == 0:
        if action == 1:
            action = 2
        else:
            action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0: # Render every 10 steps for efficiency
        env.render()
