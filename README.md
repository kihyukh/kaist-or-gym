# Traffic Control Environment

## Description

The `TrafficControlEnv` is a custom OpenAI Gymnasium-compatible environment for simulating traffic signal control at a four-way intersection. The environment models cars arriving from four directions (North, East, South, West), their movement through the intersection, and the effect of traffic signals on their waiting and travel times. This environment is designed for research and educational purposes in operations research, reinforcement learning, and traffic management.

## Actions

The environment uses a discrete action space with three possible actions at each time step:

- **Action 0:**
  *No change* — The traffic signal remains in its current state. No transition is triggered.

- **Action 1:**
  *Switch to North/South Green* —
  - If the current signal is red for all directions (`RR`) or already green for North/South (`GR`), this action sets or keeps the signal as green for North/South and red for East/West (`GR`).
  - If the current signal is green for East/West (`RG`), this action initiates a yellow light phase for East/West (`RY`), after which the signal will switch to green for North/South (`GR`).

- **Action 2:**
  *Switch to East/West Green* —
  - If the current signal is red for all directions (`RR`) or already green for East/West (`RG`), this action sets or keeps the signal as green for East/West and red for North/South (`RG`).
  - If the current signal is green for North/South (`GR`), this action initiates a yellow light phase for North/South (`YR`), after which the signal will switch to green for East/West (`RG`).

**Yellow Light Logic:**
When a transition between green signals is requested (e.g., from North/South green to East/West green), the environment enforces a yellow light phase (`YR` or `RY`) for safety. During the yellow phase, new actions are ignored until the yellow duration elapses, after which the signal switches to the target green state.

## Installation

You can install the package directly from PyPI using pip:

```sh
pip install kaist-or-gym
```

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

---

# Windy Gridworld Environment

## Description

`WindyGridworld` is a Gymnasium-compatible grid navigation environment inspired by Sutton & Barto’s windy gridworld. The agent moves on a 2D grid from a start state to a goal state while being affected by a column-dependent upward wind. Wind is stochastic: with probability specified per column, the agent is pushed up by one cell before its chosen action is applied.

Key properties (defaults in `windy_gridworld.py`):
- Grid size: 7 rows × 10 columns
- Start: `(3, 0)`; Goal: `(3, 7)`
- Wind probabilities by column (example): `[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.0]`
- Reward: +1 when the current state is the goal; −1 otherwise

## Spaces

- Action space: `Discrete(4)`
  - `0`: Up, `1`: Down, `2`: Left, `3`: Right
- Observation space: `Discrete(R*C)` where each state is the flattened index of `(row, col)`

## Dynamics

At each step:
1) With probability `wind[c]` (for the current column `c`), the agent is pushed one cell up (clipped to the top boundary).
2) The chosen action is then applied (with boundary clipping).
3) Episode terminates when the agent reaches the goal state.

In addition to the standard Gymnasium API, the environment provides:
- `transition_probability(state, action, next_state) -> float`: one-step `P(s'|s,a)`
- `possible_next_states(state, action) -> List[(next_state, prob)]`: enumerates the small set of feasible successors (wind/no-wind)
- `reward(state, action) -> float`: immediate reward based on the current state (overridable)

## Rendering

When `render_mode="human"`, the grid is plotted with:
- Blue translucent column shading indicating wind intensity (alpha ∝ wind probability)
- Upward arrows showing wind direction
- Start (S), Goal (G), and Agent (A) markers

## Usage Example

```python
from kaist_or_gym.envs.windy_gridworld import WindyGridworld

env = WindyGridworld(render_mode="human")
obs, info = env.reset(seed=0)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
```