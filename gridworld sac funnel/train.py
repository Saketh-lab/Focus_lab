from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

MPL_CACHE_DIR = Path(".matplotlib-cache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.gridworld_env import ContinuousGridWorldFunnelEnv, GridWorldConfig


TOTAL_TIMESTEPS = 200_000
SEED = 7


def make_env(seed: int) -> gym.Env:
    env = ContinuousGridWorldFunnelEnv(
        config=GridWorldConfig(size=7, max_steps=60, randomize_obstacles=False),
        seed=seed,
    )
    return Monitor(env, info_keywords=("is_success",))


def main() -> None:
    checkpoint_dir = Path("checkpoints")
    log_dir = Path("logs")
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    tensorboard_log = str(log_dir) if find_spec("tensorboard") else None
    if tensorboard_log is None:
        print("TensorBoard is not installed, so TensorBoard logging is disabled.")

    check_env(ContinuousGridWorldFunnelEnv(seed=SEED), warn=True)

    train_env = DummyVecEnv([lambda: make_env(SEED)])
    eval_env = DummyVecEnv([lambda: make_env(SEED + 1)])

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=128,
        gamma=0.98,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1_000,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=SEED,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoint_dir),
        name_prefix="sac_gridworld_funnel",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )
    model.save(checkpoint_dir / "final_model")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
