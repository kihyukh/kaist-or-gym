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
pip install kaist-rl-lab
```

## Usage Example

Below is a minimal example of how to use the `TrafficControlEnv` environment for a fixed number of time steps:

```python
import gymnasium as gym
import kaist_rl_lab

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
from kaist_rl_lab.envs.windy_gridworld import WindyGridworld

env = WindyGridworld(render_mode="human")
obs, info = env.reset(seed=0)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
```

---

# Coffee Pouring Environment

## Description

`CoffeePouringEnv` is a continuous-control teaching environment for behavior cloning and
reward-based fine-tuning. Two planar robot arms hold a cup and a coffee pot. The goal is to reach
a requested fill amount while limiting spill, unnecessary motion, and cup tilt.

This is not a full fluid simulator or a real-robot controller. Arm geometry uses exact planar
forward kinematics, while the liquid uses a fast, deterministic physical approximation: a finite
pot reservoir with a horizontal free surface, tilt- and head-dependent Torricelli flow, a ballistic
stream under gravity, and geometric intersection with the rotated cup opening. Stream thickness
follows the current flow, tilted cups retain only the liquid below their lower rim, and runoff lands
on the table at the rendered impact location. Rim and wall impacts become exterior runoff, so a
missed stream never passes through the drawn cup.

## Rigid robot model

Each arm has two fixed-length links and one revolute wrist. The simulator stores only six joint
angles. Every visible elbow, wrist, cup, pot, mouth, and spout position is derived from those angles,
so an arm segment cannot extend or contract. Each physics substep also projects the links and
vessels to first contact with the tabletop or the other robot. Simultaneous six-joint contact
projection prevents the arms, cup, pot, and their handles from moving through the other system.

The six-dimensional continuous action is ordered as:

1. cup shoulder angular velocity
2. cup elbow angular velocity
3. cup wrist angular velocity
4. pot shoulder angular velocity
5. pot elbow angular velocity
6. pot wrist angular velocity

Every action component is normalized to `[-1, 1]`. The 16-dimensional observation includes the six
joint angles, sine/cosine encodings of both vessel angles, cup-to-spout geometry, fill, spill,
target amount, and elapsed time. At full command, every joint rotates at 9 degrees per simulated
second, so a 90-degree turn takes ten simulated seconds when a mechanical stop is not reached.
Each action is held as a constant angular velocity for the 0.125-second decision interval. Fixed
1/64-second physics substeps integrate joint motion and liquid flow together, while the agent still
observes and acts only at discrete Gymnasium decision epochs.

## Gymnasium usage

```python
import gymnasium as gym
import kaist_rl_lab

env = gym.make("kaist-or/CoffeePouringEnv-v0", render_mode="rgb_array")
observation, info = env.reset(seed=7001)

for _ in range(330):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frame = env.render()  # H x W x 3 uint8 image generated by the environment
    if terminated or truncated:
        break

env.close()
```

The coffee environment uses a 330-step horizon by default so training returns and evaluation
episodes are comparable. Pass `horizon=None` for a continuing attempt with no time-limit
truncation; success and irrecoverable spill still terminate the episode normally.

`info` reports fill, target, fill error, spill, remaining coffee, flow rate, stream and cup-runoff
paths, stream thickness and speed, capture fraction, spill impact location, success, joint angles,
tool positions, and the termination reason.
`reset(options=...)` accepts `target_fill`, `joint_angles`, `cup_center`, `pot_center`, `fill`, and
`spill` for controlled experiments.

`env.unwrapped.render_snapshot()` returns the same scene as a versioned, JSON-safe keyframe. Schema
version 4 includes the physical stream width, cup-runoff path, direct-spill event path, and
spill-impact location. It is
intended for interactive front ends that need smooth visual interpolation without advancing the
environment or changing the recorded trajectory.

## Interactive app and Google Colab

The interactive controller is a separate applet, but it does not duplicate the simulation. It
converts button clicks into a six-element action and calls `env.step()` once per decision epoch.
The environment supplies self-contained render keyframes; the browser interpolates their joint
angles at display refresh rate and recomputes rigid forward kinematics for every visual frame.
Animation frames never call `step()` and never create extra demonstration rows. The canonical
`env.render()` RGB renderer remains available for tests, videos, and non-interactive use.

Install the optional interface and launch it locally:

```sh
pip install -e ".[interactive]"
python -m kaist_rl_lab.apps.coffee_pouring_app
```

In the app, clockwise/counterclockwise commands latch. This allows several joints to rotate at the
same time while a timer advances the environment. The timeline above the scene shows the current
step and simulated time. **Pause / resume time** freezes the environment without clearing those
commands, while **Time speed** changes only the wall-clock playback rate—not the Gymnasium dynamics.

Human practice is unlimited by default. Turn on **Cap episode on reset** to use a chosen maximum
number of steps. **Finish** stops the timer and marks the final recorded transition as manually
truncated. The app exports the resulting human demonstration, including `dt`, joint-speed metadata,
and the effective horizon, as an `.npz` file suitable for behavior-cloning experiments; an exported
horizon of `-1` denotes unlimited practice.

The current exports identify the dynamics as `torricelli_ballistic_v3`. Re-record demonstrations
made with earlier coffee-flow rules before comparing behavior cloning and reward fine-tuning.

The environment remains a fast teaching approximation rather than a rigid-body/fluid simulator:
table, arm-to-arm, vessel, and handle contacts are enforced, while droplet breakup, splashing, and
surface tension are not modeled.

[Open the interactive notebook in Google Colab](https://colab.research.google.com/github/kihyukh/kaist-or-gym/blob/main/examples/coffee_pouring_colab.ipynb).

Colab displays a temporary Gradio share link because the notebook runtime cannot expose its local
server directly. Anyone with that temporary link can reach the app while the runtime is active, so
do not use private data in the classroom demo.
