import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class WindyGridworld(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 10,
    }

    def __init__(self, *, render_mode=None, grid_size=(7, 10), start_state=(3, 0), goal_state=(3, 7), wind=None):
        super().__init__()

        self.render_mode = render_mode

        self.grid_size = grid_size  # (rows, columns)
        self.start_state = start_state  # (row, col)
        self.goal_state = goal_state  # (row, col)
        self.current_state = self.start_state

        # Wind probabilities for each column. If rng.random() < wind_prob, then wind effect is 1.
        # Default probabilities similar to original example
        self.wind = wind if wind is not None else [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.0]

        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        # Observation is a single discrete index of the grid cell
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])

        # Rendering state (lazy init like TrafficControlEnv)
        self._plt = None
        self._display = None
        self.fig, self.ax = None, None
        self.display_handle = None

    def _state_to_int(self, state):
        """Converts a (row, col) state tuple to an integer index."""
        return state[0] * self.grid_size[1] + state[1]

    def _int_to_state(self, index):
        """Converts an integer index to a (row, col) state tuple."""
        return (index // self.grid_size[1], index % self.grid_size[1])

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the start state."""
        super().reset(seed=seed)
        # self.np_random is provided by gymnasium after super().reset

        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = None, None
        self.display_handle = None

        self.current_state = self.start_state
        observation = self._state_to_int(self.current_state)
        info = {}
        return observation, info

    def step(self, action):
        """Takes an action and returns (obs, reward, terminated, truncated, info)."""
        assert self.action_space.contains(action), "Invalid action"

        row, col = self.current_state

        # Apply stochastic wind effect (use gym's RNG for reproducibility)
        wind_effect = 1 if (self.wind[col] > 0 and self.np_random.random() < self.wind[col]) else 0
        new_row_after_wind = row - wind_effect

        # Apply action effect
        if action == 0:  # Up
            new_row = new_row_after_wind - 1
            new_col = col
        elif action == 1:  # Down
            new_row = new_row_after_wind + 1
            new_col = col
        elif action == 2:  # Left
            new_row = new_row_after_wind
            new_col = col - 1
        elif action == 3:  # Right
            new_row = new_row_after_wind
            new_col = col + 1
        else:
            # Should be unreachable due to assert
            raise ValueError("Invalid action. Must be 0 (Up), 1 (Down), 2 (Left), or 3 (Right).")

        # Boundary checks
        new_row = max(0, min(new_row, self.grid_size[0] - 1))
        new_col = max(0, min(new_col, self.grid_size[1] - 1))

        self.current_state = (new_row, new_col)

        # Reward and termination condition
        terminated = self.current_state == self.goal_state
        # Use reward function (expected immediate reward given state and action)
        reward = self.reward(self._state_to_int((row, col)), action)

        truncated = False
        observation = self._state_to_int(self.current_state)
        info = {}
        return observation, reward, terminated, truncated, info

    def transition_probability(self, state, action, next_state) -> float:
        """Return P(s' | s, a) for one step of dynamics.

        Args:
            state: int (Discrete index) or tuple (row, col)
            action: int in {0:Up, 1:Down, 2:Left, 3:Right}
            next_state: int (Discrete index) or tuple (row, col)

        Notes:
            - From the goal state, the process is absorbing: P(goal|goal,a)=1.
            - Stochasticity arises only from wind in the current column.
        """
        assert self.action_space.contains(action), "Invalid action"

        rows, cols = self.grid_size

        def to_rc(s):
            if isinstance(s, int):
                return (s // cols, s % cols)
            return s

        s_r, s_c = to_rc(state)
        ns_r, ns_c = to_rc(next_state)

        # Absorbing goal state behavior
        if (s_r, s_c) == self.goal_state:
            return 1.0 if (ns_r, ns_c) == self.goal_state else 0.0

        # Wind probability for the current column
        wind_p = 0.0
        if 0 <= s_c < len(self.wind):
            wind_p = float(self.wind[s_c])
        wind_p = max(0.0, min(1.0, wind_p))

        # Action deltas
        if action == 0:
            dr, dc = -1, 0
        elif action == 1:
            dr, dc = 1, 0
        elif action == 2:
            dr, dc = 0, -1
        else:  # action == 3
            dr, dc = 0, 1

        # Next state if wind occurs (agent pushed up by 1 first)
        wr = max(0, s_r - 1)
        wc = s_c
        nr_w = min(rows - 1, max(0, wr + dr))
        nc_w = min(cols - 1, max(0, wc + dc))

        # Next state if no wind
        wr = s_r
        wc = s_c
        nr_nw = min(rows - 1, max(0, wr + dr))
        nc_nw = min(cols - 1, max(0, wc + dc))

        prob = 0.0
        if (ns_r, ns_c) == (nr_w, nc_w):
            prob += wind_p
        if (ns_r, ns_c) == (nr_nw, nc_nw):
            prob += (1.0 - wind_p)
        return prob

    def possible_next_states(self, state, action):
        """Return a list of (next_state, probability) for (state, action).

        At most two outcomes exist in this environment: wind and no-wind. If
        the current state is the absorbing goal, returns [(goal, 1.0)].

        Args:
            state: int (Discrete index) or tuple (row, col)
            action: int in {0:Up, 1:Down, 2:Left, 3:Right}
        """
        assert self.action_space.contains(action), "Invalid action"

        rows, cols = self.grid_size

        def to_rc(s):
            if isinstance(s, int):
                return (s // cols, s % cols)
            return s

        sr, sc = to_rc(state)

        # Absorbing goal
        if (sr, sc) == self.goal_state:
            goal_idx = self._state_to_int(self.goal_state)
            return [(goal_idx, 1.0)]

        # Wind probability for current column
        wind_p = 0.0
        if 0 <= sc < len(self.wind):
            wind_p = float(self.wind[sc])
        wind_p = max(0.0, min(1.0, wind_p))

        # Action deltas
        if action == 0:
            dr, dc = -1, 0
        elif action == 1:
            dr, dc = 1, 0
        elif action == 2:
            dr, dc = 0, -1
        else:
            dr, dc = 0, 1

        # With wind: pushed up then apply action
        wr = max(0, sr - 1)
        wc = sc
        nr_w = min(rows - 1, max(0, wr + dr))
        nc_w = min(cols - 1, max(0, wc + dc))
        sp_w = self._state_to_int((nr_w, nc_w))

        # No wind: just apply action
        wr = sr
        wc = sc
        nr_nw = min(rows - 1, max(0, wr + dr))
        nc_nw = min(cols - 1, max(0, wc + dc))
        sp_nw = self._state_to_int((nr_nw, nc_nw))

        if sp_w == sp_nw:
            return [(sp_w, 1.0)]
        else:
            return [(sp_w, wind_p), (sp_nw, 1.0 - wind_p)]

    def reward(self, state, action) -> float:
        """Immediate reward R(s,a) that depends on the current state.

        Args:
            state: int (Discrete index) or tuple (row, col)
            action: int in {0:Up, 1:Down, 2:Left, 3:Right} (ignored by default)

        Reward scheme:
            - +1.0 if the CURRENT state s is the goal state
            - -1.0 otherwise
        """
        rows, cols = self.grid_size

        def to_rc(s):
            if isinstance(s, int):
                return (s // cols, s % cols)
            return s

        sr, sc = to_rc(state)
        return 1.0 if (sr, sc) == self.goal_state else 0.0

    def render_policy(self, Q, ax=None):
        """Renders the greedy policy derived from a Q-matrix.

        Args:
            Q (np.ndarray): A (S x A) matrix of state-action values.
            ax (matplotlib.axes.Axes, optional): An axis to plot on. If None,
                a new figure and axis are created.
        """
        if self._plt is None:
            import matplotlib.pyplot as plt
            self._plt = plt

        if ax is None:
            fig, ax = self._plt.subplots(figsize=(self.grid_size[1], self.grid_size[0]))
            ax.set_title("Policy Visualization")
        
        # Clear axis for redraw
        ax.clear()

        # Derive policy from Q
        policy = np.argmax(Q, axis=1)
        
        # Action to arrow mapping (dx, dy) for quiver
        action_arrows = {
            0: (0, -0.4),  # Up
            1: (0, 0.4),   # Down
            2: (-0.4, 0),  # Left
            3: (0.4, 0),   # Right
        }

        # Draw wind shades and arrows first (background)
        n_rows, n_cols = self.grid_size
        for c in range(n_cols):
            prob = self.wind[c] if c < len(self.wind) else 0.0
            if prob > 0:
                shade = self._plt.Rectangle((c - 0.5, -0.5), 1, n_rows, facecolor='blue', alpha=prob / 2, edgecolor=None, zorder=0)
                ax.add_patch(shade)
                ax.annotate('', xy=(c, -0.4), xytext=(c, n_rows - 0.6),
                            arrowprops=dict(arrowstyle='-|>', color='blue', alpha=prob / 2, lw=2), zorder=1)

        # Draw grid lines
        for r in range(self.grid_size[0]):
            ax.axhline(r - 0.5, color='black', linewidth=1)
        for c in range(self.grid_size[1]):
            ax.axvline(c - 0.5, color='black', linewidth=1)

        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()

        # Plot goal and start states
        goal_row, goal_col = self.goal_state
        ax.add_patch(self._plt.Rectangle((goal_col - 0.5, goal_row - 0.5), 1, 1, facecolor='green', alpha=0.6))
        ax.text(goal_col, goal_row, 'G', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

        start_row, start_col = self.start_state
        ax.add_patch(self._plt.Rectangle((start_col - 0.5, start_row - 0.5), 1, 1, facecolor='orange', alpha=0.3))
        ax.text(start_col, start_row, 'S', ha='center', va='center', color='black', fontsize=14, fontweight='bold')

        # Plot policy arrows
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) == self.goal_state:
                    continue
                s = self._state_to_int((r, c))
                a = policy[s]
                dx, dy = action_arrows[a]
                ax.arrow(c, r, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k', length_includes_head=True)

        ax.set_xticks([])
        ax.set_yticks([])
        
        if ax is None: # Only call show if we created the plot
            self._plt.show()

    def render(self):
        """Renders the current grid state using matplotlib without flickering (human mode)."""
        if self.render_mode != "human":
            return

        # Lazy import/setup like TrafficControlEnv
        if self._plt is None or self._display is None:
            import matplotlib.pyplot as plt  # noqa: F401
            try:
                from IPython.display import display  # noqa: F401
            except ImportError:
                display = None  # type: ignore
            self._plt = plt
            self._display = display  # type: ignore

        if self.fig is None or self.ax is None:
            self.fig, self.ax = self._plt.subplots(figsize=(self.grid_size[1], self.grid_size[0]))
            self.display_handle = (self._display(self.fig, display_id=True) if self._display is not None else None)
        else:
            self.ax.clear()

        # Draw wind intensity as transparent blue shade per column with upward arrow
        n_rows, n_cols = self.grid_size
        for c in range(n_cols):
            prob = self.wind[c] if c < len(self.wind) else 0.0
            if prob <= 0:
                continue
            # Column-wide translucent blue shade
            shade = self._plt.Rectangle((c - 0.5, -0.5), 1, n_rows, facecolor='blue', alpha=prob / 2, edgecolor=None, linewidth=0, zorder=0)
            self.ax.add_patch(shade)
            # Upward arrow (head near top, tail near bottom)
            self.ax.annotate(
                '',
                xy=(c, -0.4),              # arrow head near top inside axis
                xytext=(c, n_rows - 0.6),  # tail near bottom
                arrowprops=dict(arrowstyle='-|>', color='blue', alpha=prob / 2, lw=2),
                zorder=1,
            )

        # Draw grid lines
        for r in range(self.grid_size[0]):
            self.ax.axhline(r - 0.5, color='black', linewidth=1)
        for c in range(self.grid_size[1]):
            self.ax.axvline(c - 0.5, color='black', linewidth=1)

        self.ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.invert_yaxis()

        # Plot goal state
        goal_row, goal_col = self.goal_state
        self.ax.add_patch(self._plt.Rectangle((goal_col - 0.5, goal_row - 0.5), 1, 1, facecolor='green', alpha=0.6))
        self.ax.text(goal_col, goal_row, 'G', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

        # Plot start state
        start_row, start_col = self.start_state
        self.ax.add_patch(self._plt.Rectangle((start_col - 0.5, start_row - 0.5), 1, 1, facecolor='orange', alpha=0.3))
        self.ax.text(start_col, start_row, 'S', ha='center', va='center', color='black', fontsize=14, fontweight='bold')

        # Plot current agent state
        agent_row, agent_col = self.current_state
        self.ax.add_patch(self._plt.Rectangle((agent_col - 0.5, agent_row - 0.5), 1, 1, facecolor='red', alpha=0.8))
        self.ax.text(agent_col, agent_row, 'A', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

        # (Wind indicators are rendered as shaded columns and arrows above.)

        self.ax.set_title("Windy Gridworld (Stochastic Wind, Up by 1 with p)", fontsize=16)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Update the displayed plot
        if self.display_handle is not None:
            self.display_handle.update(self.fig)
        else:
            self._plt.pause(0.1)