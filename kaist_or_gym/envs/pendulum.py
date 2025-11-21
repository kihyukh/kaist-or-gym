import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import PendulumEnv
import numpy as np


class DiscretePendulumEnv(gym.Env):
    """
    A discretized version of the gymnasium Pendulum-v1 environment.

    The state space is discretized by binning the angle (theta) and the
    angular velocity (theta_dot). The action space (torque) is also
    discretized into a fixed number of torque values.

    This wrapper allows model-based RL algorithms that require discrete
    states and actions to be applied to the classic pendulum problem.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, n_theta_bins=16, n_thetadot_bins=16, n_torque_bins=9, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        # The continuous environment that this class wraps
        # For Jupyter-friendly rendering in 'human' mode, get rgb frames from the base env
        underlying_mode = 'rgb_array' if render_mode == 'human' else render_mode
        self.continuous_env = PendulumEnv(render_mode=underlying_mode)

        # Discretization parameters
        self.n_theta_bins = n_theta_bins
        self.n_thetadot_bins = n_thetadot_bins
        self.n_torque_bins = n_torque_bins

        # Define the discrete observation space
        self.observation_space = spaces.Discrete(n_theta_bins * n_thetadot_bins)

        # Define the discrete action space
        self.action_space = spaces.Discrete(n_torque_bins)
        # Create a mapping from discrete actions to continuous torque values
        self.torques = np.linspace(
            self.continuous_env.action_space.low[0],
            self.continuous_env.action_space.high[0],
            n_torque_bins
        )

        # Create bins for state discretization
        # Theta is in [-pi, pi]
        self.theta_bins = np.linspace(-np.pi, np.pi, n_theta_bins + 1)[1:-1]
        # Theta_dot is in [-max_speed, max_speed]
        self.thetadot_bins = np.linspace(-self.continuous_env.max_speed, self.continuous_env.max_speed, n_thetadot_bins + 1)[1:-1]

        # Lazy-rendering state for notebook-friendly rendering
        self._plt = None
        self._display = None
        self.fig = None
        self.ax = None
        self.img = None
        self.display_handle = None

    def _discretize_state(self, continuous_state):
        """Converts a continuous state to a discrete integer state."""
        cos_theta, sin_theta, theta_dot = continuous_state
        theta = np.arctan2(sin_theta, cos_theta)

        theta_bin = np.digitize(theta, bins=self.theta_bins)
        thetadot_bin = np.digitize(theta_dot, bins=self.thetadot_bins)

        return theta_bin * self.n_thetadot_bins + thetadot_bin

    def reset(self, *, seed=None, options=None):
        """Resets the environment and returns the initial discrete state."""
        super().reset(seed=seed)
        continuous_state, info = self.continuous_env.reset(seed=seed, options=options)
        discrete_state = self._discretize_state(continuous_state)
        return discrete_state, info

    def step(self, action):
        """
        Takes a discrete action, converts it to a continuous torque,
        and steps the continuous environment.
        """
        assert self.action_space.contains(action), "Invalid action"
        
        # Map discrete action to continuous torque
        torque = self.torques[action]
        
        # Step the continuous environment
        continuous_state, reward, terminated, truncated, info = self.continuous_env.step([torque])
        
        # Discretize the resulting state
        discrete_state = self._discretize_state(continuous_state)
        
        return discrete_state, reward, terminated, truncated, info

    def render(self):
        """Render suitable for Jupyter/Colab when render_mode='human'.

        - If render_mode is 'rgb_array', returns the frame from the underlying env.
        - If render_mode is 'human', uses matplotlib + IPython.display to update inline.
        """
        # If user wants raw frames, just return them
        if self.render_mode == 'rgb_array':
            return self.continuous_env.render()

        if self.render_mode != 'human':
            return None

        # Lazy imports/setup to avoid hard dependency when not rendering
        if self._plt is None or self._display is None:
            import matplotlib.pyplot as plt  # type: ignore
            try:
                from IPython.display import display  # type: ignore
            except ImportError:
                display = None  # type: ignore
            self._plt = plt
            self._display = display

        frame = self.continuous_env.render()  # ndarray HxWx3

        if self.fig is None or self.ax is None:
            self.fig, self.ax = self._plt.subplots(figsize=(5, 5))
            self.img = self.ax.imshow(frame)
            self.ax.axis('off')
            if self._display is not None:
                self.display_handle = self._display(self.fig, display_id=True)
            else:
                self._plt.show(block=False)
        else:
            if self.img is None:
                self.img = self.ax.imshow(frame)
                self.ax.axis('off')
            else:
                self.img.set_data(frame)
            if self.display_handle is not None:
                self.display_handle.update(self.fig)
            else:
                # Fallback to a tiny pause to refresh the canvas
                self._plt.pause(0.001)

    def close(self):
        """Closes the environment."""
        self.continuous_env.close()
        if self.fig is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.close(self.fig)
            except Exception:
                pass
        self.fig = None
        self.ax = None
        self.img = None
        self.display_handle = None