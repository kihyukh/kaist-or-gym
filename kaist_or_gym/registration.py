def _register():
    try:
        import gymnasium as gym
    except Exception:
        import gym

    # Map of env IDs to entry points
    env_specs = {
        "kaist-or/TrafficControlEnv-v0": "kaist_or_gym.envs:TrafficControlEnv",
    }

    for env_id, entry_point in env_specs.items():
        if env_id in getattr(gym.registry, "env_specs", {}) or \
           (hasattr(gym, "registry") and env_id in gym.registry):
            continue  # already registered
        gym.register(
            id=env_id,
            entry_point=entry_point,
            max_episode_steps=None,
        )