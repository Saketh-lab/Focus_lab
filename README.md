# SAC GridWorld with Gymnasium

This project trains a Stable-Baselines3 Soft Actor-Critic agent on a custom Gymnasium GridWorld.

SAC normally expects a continuous action space, while GridWorld is usually discrete. To make SAC a good fit, this environment accepts a continuous 2D action vector in `[-1, 1]` and converts it into one grid move: up, down, left, or right.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

The best model is saved to:

```text
checkpoints/best_model/best_model.zip
```

Training logs are saved to:

```text
logs/
```

## Test and render video

```bash
python test.py
```

This writes:

```text
outputs/episode.mp4
```

## Project files

- `src/gridworld_env.py` - custom Gymnasium environment
- `train.py` - Stable-Baselines3 SAC training script
- `test.py` - runs a trained agent and records an episode
- `requirements.txt` - Python dependencies
