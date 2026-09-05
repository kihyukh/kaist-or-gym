"""Run a reproducible random rollout in the laundry-folding environment.

The random policy is intentionally poor: it is a small baseline that makes the
continuous action contract and the difficulty of deformable-object control
visible.  Rendering is optional so the default rollout stays lightweight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image

import kaist_rl_lab  # noqa: F401 - registers the kaist-or environments


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7001, help="episode and action seed")
    parser.add_argument("--steps", type=int, default=60, help="maximum decision steps")
    parser.add_argument(
        "--frame-every",
        type=int,
        default=3,
        help="capture one rendered frame every N decision steps",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="save the last frame as an image, or all captured frames as a .gif",
    )
    parser.add_argument("--show", action="store_true", help="display the final RGB frame")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.frame_every < 1:
        parser.error("--frame-every must be positive")
    return args


def main() -> None:
    args = _arguments()
    capture_frames = args.save is not None or args.show
    render_mode = "rgb_array" if capture_frames else None
    env = gym.make(
        "kaist-or/LaundryFoldingEnv-v0",
        render_mode=render_mode,
        horizon=args.steps,
    )
    env.action_space.seed(args.seed)
    frames: list[Image.Image] = []
    total_reward = 0.0

    try:
        observation, info = env.reset(seed=args.seed)
        if capture_frames:
            frames.append(Image.fromarray(env.render()))

        for step in range(1, args.steps + 1):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if capture_frames and (step % args.frame_every == 0 or terminated or truncated):
                frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break

        print(f"action shape: {env.action_space.shape}")
        print(f"observation shape: {observation.shape}")
        print(
            "steps={}, return={:.3f}, stage={}, straightness={:.3f}, "
            "fold_score={:.3f}, success={}".format(
                info["elapsed_steps"],
                total_reward,
                info["stage"],
                info["straightness"],
                info["fold_score"],
                info["is_success"],
            )
        )

        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if args.save.suffix.lower() == ".gif":
                frames[0].save(
                    args.save,
                    save_all=True,
                    append_images=frames[1:],
                    duration=100 * args.frame_every,
                    loop=0,
                )
            else:
                frames[-1].save(args.save)
            print(f"saved {args.save}")

        if args.show:
            plt.figure(figsize=(10, 6.8))
            plt.imshow(frames[-1])
            plt.axis("off")
            plt.tight_layout()
            plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
