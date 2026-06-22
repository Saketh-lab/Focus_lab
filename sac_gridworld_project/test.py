from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

MPL_CACHE_DIR = Path(".matplotlib-cache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

from src.gridworld_env import ContinuousGridWorldEnv, GridWorldConfig


MODEL_PATHS = [
    Path("checkpoints/best_model/best_model.zip"),
    Path("checkpoints/final_model.zip"),
]


@dataclass
class EpisodeResult:
    seed: int
    total_reward: float
    success: bool
    final_info: dict
    frames: list[np.ndarray]


def find_model_path() -> Path:
    for path in MODEL_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("No trained model found. Run `python train.py` first.")


def run_episode(model: SAC, seed: int) -> EpisodeResult:
    env = ContinuousGridWorldEnv(
        config=GridWorldConfig(size=7, max_steps=60, randomize_obstacles=False),
        render_mode="rgb_array",
        seed=seed,
    )
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(np.asarray(action))
        total_reward += reward
        frames.append(env.render())

    env.close()
    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        success=bool(info["is_success"]),
        final_info=info,
        frames=frames,
    )


def pick_video_episode(results: list[EpisodeResult]) -> EpisodeResult:
    successful = [result for result in results if result.success]
    if successful:
        return max(successful, key=lambda result: result.total_reward)
    return max(results, key=lambda result: result.total_reward)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model = SAC.load(find_model_path())
    results = [run_episode(model, args.seed + index) for index in range(args.episodes)]
    video_result = pick_video_episode(results)

    video_path = output_dir / "episode.mp4"
    imageio.mimsave(video_path, video_result.frames, fps=ContinuousGridWorldEnv.metadata["render_fps"])

    success_count = sum(result.success for result in results)
    rewards = [result.total_reward for result in results]
    print(f"Saved {video_path}")
    print(f"Evaluated episodes: {args.episodes}")
    print(f"Success rate: {success_count / args.episodes:.2f} ({success_count}/{args.episodes})")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Best reward: {max(rewards):.3f}")
    print(f"Video seed: {video_result.seed}")
    print(f"Video total reward: {video_result.total_reward:.3f}")
    print(f"Video final info: {video_result.final_info}")


if __name__ == "__main__":
    main()
