# SAC GridWorld with Funnel Reward

This project trains a Stable-Baselines3 Soft Actor-Critic agent on a custom Gymnasium GridWorld.

It is based on the earlier SAC GridWorld project, but the reward has been changed to use a funnel-shaped time-dependent reward inspired by the paper's funnel reward idea:

```text
r'(s, a, t) = rho(s) + gamma(t) - rho_max
```

For this GridWorld:

- `rho(s)` is robustness, represented as closeness to the goal.
- `rho_max` is the best possible robustness at the goal.
- `gamma(t)` is a decreasing funnel function.
- As time passes, the funnel shrinks, so the agent needs to get closer to the goal earlier.

This implementation ignores the STL-specific parts of the paper and only adapts the funnel reward shaping idea.

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

The script evaluates multiple episodes and saves the best successful episode to:

```text
outputs/episode.mp4
```

## Files

- `src/gridworld_env.py` - Gymnasium GridWorld with funnel reward
- `train.py` - trains Stable-Baselines3 SAC
- `test.py` - evaluates the trained policy and records a video
- `requirements.txt` - Python dependencies
