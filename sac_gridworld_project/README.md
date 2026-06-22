# SAC GridWorld

This project trains a Soft Actor-Critic agent from Stable-Baselines3 on a custom Gymnasium GridWorld.

SAC is designed for continuous action spaces, so this GridWorld uses a continuous 2D action:

```text
[dx, dy] in [-1, 1]
```

The environment converts that action into one grid move: left, right, up, or down.

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

The final model is saved to:

```text
checkpoints/final_model.zip
```

The best evaluated model is saved to:

```text
checkpoints/best_model/best_model.zip
```

## Test

```bash
python test.py --episodes 25
```

This evaluates the trained model and saves a video to:

```text
outputs/episode.mp4
```

## Files

- `src/gridworld_env.py` - custom Gymnasium environment
- `train.py` - trains SAC with Stable-Baselines3
- `test.py` - evaluates a trained policy and records a video
- `requirements.txt` - dependencies

## Notes

The default environment uses a fixed obstacle layout. That makes the first version easier to train and easier to explain. You can enable randomized obstacles by passing `randomize_obstacles=True` in `GridWorldConfig`.
