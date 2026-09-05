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

The joint controls sit in a fixed dock below the scene, with the cup arm on the left and the pot
arm on the right. They stay outside the canvas so the robots and vessels remain visible throughout
their motion. Each joint has counterclockwise, hold, and clockwise commands; directions latch so
several joints can rotate together. On narrower screens, the two control groups stack vertically.

Each new demo starts **paused** so students can plan and select joint commands before moving.
The toolbar above the scene shows the current step and simulated time alongside three controls:
**Pause time** (which changes to **Resume time** while paused), **Reset + start**, and
**Stop all motors**. Pausing preserves joint commands; stopping all motors clears them without
changing whether time is paused. Reset starts a fresh episode with all joints held. The demo uses
a 700 mL target, normal playback speed, and unlimited practice time. Browser updates are batched
every 0.5 seconds to accommodate Colab tunnel latency. Each batch still records four separate
0.125-second Gym transitions; the 1/64-second physics integration and training dynamics are unchanged.
The browser interpolates between keyframes and adapts to modest network jitter.

The environment remains a fast teaching approximation rather than a rigid-body/fluid simulator:
table, arm-to-arm, vessel, and handle contacts are enforced, while droplet breakup, splashing, and
surface tension are not modeled.

[Open the interactive notebook in Google Colab](https://colab.research.google.com/github/kihyukh/kaist-or-gym/blob/main/examples/coffee_pouring_colab.ipynb).

For class, distribute `examples/coffee_pouring_colab.ipynb`. Its first cell installs
`kaist-rl-lab[interactive]==0.1.19` and Gradio 6.26.0 from PyPI; the second code cell launches
each student's own demo inside Colab and prints a link for a larger view. Students can use
**Runtime → Run all** with a standard Python 3 runtime; no GPU or Drive mount is needed.
The launch cell stays running while the demo is in use. If Colab requests a restart after
installation, restart the session and run both cells again.

Colab displays a temporary Gradio share link because the notebook runtime cannot expose its local
server directly. Anyone with that temporary link can reach the app while the runtime is active, so
do not use private data in the classroom demo.

### Collect classroom demonstrations in your personal Google Drive

Run `examples/coffee_trajectory_collector_colab.ipynb` in your own Google account. Its setup cell
connects your Drive and creates `My Drive / KAIST Coffee Trajectories` (or your chosen folder), then
prints a collector URL and a lecture code. Give those two values to students to enter in their
`coffee_pouring_colab.ipynb` launch cell. Students run independent demos in their own personal
accounts; they do not need access to your Drive folder or your Google credentials.

Students open **Save your demonstration**, optionally enter a participant code, and press
**Submit trajectory** before resetting. Submission ends the attempt and shows a receipt only after
the collector has saved it. A downloadable `.npz` backup remains available on upload failure;
retrying the same recording does not create duplicate files. With no collector configured,
**Save trajectory** provides a download only. New episodes are isolated by UUID.

The instructor notebook includes a live submission monitor and a behavior-cloning data loader.
The collector exposes only an upload endpoint; directory listings and saved Drive files are not
Gradio outputs. Keep the instructor runtime connected during class. Disconnecting it ends collection
but leaves the saved recordings in Drive.

Each `.npz` contains `observations` (N × 16), `actions` (N × 6), `rewards`, `next_observations`,
`terminated`, `truncated`, and JSON `metadata`. Load with `allow_pickle=False`. Metadata records
the episode/participant codes, environment and package versions, timing, ordered observation/joint
names, target, fill, spill, and success. Early submissions mark the final transition truncated.
The collector validates array shapes, finite values, action bounds, timing, and consecutive
observations before accepting an attempt.

```python
from kaist_rl_lab.apps.coffee_demonstrations import load_behavior_cloning_data

data = load_behavior_cloning_data("/content/drive/MyDrive/KAIST Coffee Trajectories")
observations, actions = data["observations"], data["actions"]
episode_ids = data["episode_ids"]  # Split whole episodes into training/validation sets.
```

---

# Laundry Folding Environment

## Task

`LaundryFoldingEnv` is a bimanual continuous-control environment in which two spatial robot arms
must first straighten a randomly posed, wrinkled towel and then fold its far half over its near
half. The task is deliberately staged: folding reward is unlocked only after both grippers hold
materially separated regions and keep the towel straight under bimanual tension for several
decisions. Letting the towel settle, pinching two nearby points, or crumpling it into a small area
does not satisfy the straightening milestone.

The default towel contains 117 simulated vertices and 192 triangles. Its state is much larger than
the coffee-pouring state: the default 952-dimensional observation contains robot configuration plus
the position, velocity, and left/right grasp membership of every cloth vertex. Material coordinates
remain fixed, allowing the environment to compare corresponding points across the intended fold.

## Robot and actions

Each arm is a mirrored fixed-length spatial 3R chain with a shoulder-yaw joint, elbow-pitch joint,
and wrist-pitch joint. Two rigid fingers move symmetrically at each wrist. The eight normalized
velocity commands are:

1. left shoulder, elbow, wrist, and gripper
2. right shoulder, elbow, wrist, and gripper

Positive gripper commands open the fingers; negative commands close them. A pinch is created only
when a cloth triangle is physically inside the closing finger prism. The environment attaches the
small triangular patch rather than magnetically teleporting one remote vertex. Opening the fingers
releases it.

## Cloth physics

The lightweight NumPy solver follows the position-based dynamics family described in the
[original PBD paper](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf) and the
[XPBD formulation](https://mmacklin.com/xpbd.pdf). It uses a checkerboard triangular mesh,
compliant structural, shear, and bending regularization, gravity, air damping, table friction,
capsule contacts with the robot, and two-sided vertex/triangle self-contact. Each 0.10-second Gym
decision is integrated using fixed steps no larger than 1/200 second. Robot motion is interpolated
through those physics steps, while the agent still chooses only one action per decision epoch.

This is a coarse educational shell model, not a fiber-level textile simulator. It models the
large-scale behavior needed to demonstrate straightening, grasping, lifting, self-contact, and
folding; it does not model individual yarns, detailed multilayer friction, air flow, or edge-edge
contact. Increase `mesh_rows` and `mesh_cols` for offline experiments when more resolution is worth
the additional computation; `mesh_rows` must remain odd so the target crease is a material row.

## Gymnasium usage

```python
import gymnasium as gym
import kaist_rl_lab

env = gym.make("kaist-or/LaundryFoldingEnv-v0", render_mode="rgb_array")
observation, info = env.reset(seed=17)

for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    if terminated or truncated:
        break

env.close()
```

`info` reports straightness, planarity, projected-coverage proxy, material strain, bimanual tension,
fold-pair alignment, layer separation, support, off-table fraction, cloth speed, grasped patches,
stage, success, and termination reason. `render_snapshot()` exposes the authoritative triangle mesh,
arm and finger landmarks, grasps, camera, and metrics as a versioned JSON-safe scene.

## Interactive app and Google Colab

Install the optional interface and launch locally:

```sh
pip install -e ".[interactive]"
python -m kaist_rl_lab.apps.laundry_folding_app
```

Joint and gripper commands latch, so one person can control several motors simultaneously with a
mouse. The app includes perspective, top, front, and side cameras; a prominent simulated clock;
pause/resume; wall-clock speed control; optional episode limits; and demonstration export for
behavior cloning. The 3-D image is always generated from the environment's actual cloth mesh and
robot geometry. The app advances physics only once per timer decision and never runs a second cloth
simulation in the browser.

[Open the laundry-folding notebook in Google Colab](https://colab.research.google.com/github/kihyukh/kaist-or-gym/blob/main/examples/laundry_folding_colab.ipynb).

The temporary Gradio share URL produced in Colab is reachable by anyone who has the link while the
runtime is active. Do not use private demonstration data in a publicly shared classroom session.
