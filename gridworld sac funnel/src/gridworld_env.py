from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DEFAULT_OBSTACLES = frozenset(
    {
        (1, 1),
        (1, 2),
        (3, 2),
        (4, 4),
        (5, 3),
    }
)


@dataclass(frozen=True)
class GridWorldConfig:
    size: int = 7
    max_steps: int = 60
    obstacle_count: int = 5
    randomize_obstacles: bool = False
    wall_penalty: float = -0.05
    obstacle_penalty: float = -0.08
    goal_reward: float = 5.0
    funnel_gamma_0: float = 1.0
    funnel_gamma_inf: float = 0.05
    funnel_decay: float = 0.055


class ContinuousGridWorldFunnelEnv(gym.Env):
    """GridWorld with continuous actions and a funnel-shaped reward.

    The funnel reward follows the paper's reward-shaping idea:

        reward = rho(s) + gamma(t) - rho_max

    Here, rho(s) is closeness to the goal, rho_max is 1.0, and gamma(t)
    decreases over the episode. Early in the episode, the agent can be farther
    from the goal and still receive less-negative reward. Later, the shrinking
    funnel pushes it to be near the goal.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: GridWorldConfig | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GridWorldConfig()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        if self.config.size < 4:
            raise ValueError("Grid size must be at least 4.")
        if self.config.obstacle_count >= self.config.size * self.config.size - 2:
            raise ValueError("Too many obstacles for the configured grid size.")
        if self.config.funnel_gamma_0 < self.config.funnel_gamma_inf:
            raise ValueError("funnel_gamma_0 must be greater than or equal to funnel_gamma_inf.")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_size = 5 + self.config.size * self.config.size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32)

        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.goal_pos = np.array([self.config.size - 1, self.config.size - 1], dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([self.config.size - 1, self.config.size - 1], dtype=np.int32)
        self.obstacles = self._make_obstacles()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1

        proposed_pos = self.agent_pos + self._action_to_delta(action)
        blocked = False

        if not self._in_bounds(proposed_pos):
            blocked = True
        elif tuple(proposed_pos) in self.obstacles:
            blocked = True
        else:
            self.agent_pos = proposed_pos.astype(np.int32)

        reward = self._funnel_reward()
        if blocked and not self._in_bounds(proposed_pos):
            reward += self.config.wall_penalty
        elif blocked:
            reward += self.config.obstacle_penalty

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.steps >= self.config.max_steps

        if terminated:
            reward += self.config.goal_reward

        info = self._get_info()
        info["blocked"] = blocked
        info["is_success"] = terminated
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | str:
        if self.render_mode == "ansi":
            return self._render_ansi()
        return self._render_rgb_array()

    def _funnel_reward(self) -> float:
        rho = self._goal_robustness()
        gamma_t = self._funnel_gamma(self.steps)
        rho_max = 1.0
        return rho + gamma_t - rho_max

    def _goal_robustness(self) -> float:
        max_distance = 2 * (self.config.size - 1)
        distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        return 1.0 - (distance / max_distance)

    def _funnel_gamma(self, t: int) -> float:
        gamma_0 = self.config.funnel_gamma_0
        gamma_inf = self.config.funnel_gamma_inf
        decay = self.config.funnel_decay
        return (gamma_0 - gamma_inf) * float(np.exp(-decay * t)) + gamma_inf

    def _make_obstacles(self) -> set[tuple[int, int]]:
        if self.config.randomize_obstacles:
            return self._sample_obstacles()

        obstacles = set(DEFAULT_OBSTACLES)
        if self.config.size != 7:
            return set()
        if not self._has_path(obstacles):
            raise ValueError("Default obstacle layout does not contain a valid path.")
        return obstacles

    def _sample_obstacles(self) -> set[tuple[int, int]]:
        blocked = {(0, 0), (self.config.size - 1, self.config.size - 1)}
        cells = [
            (x, y)
            for x in range(self.config.size)
            for y in range(self.config.size)
            if (x, y) not in blocked
        ]

        for _ in range(1_000):
            indices = self._rng.choice(len(cells), size=self.config.obstacle_count, replace=False)
            obstacles = {cells[int(index)] for index in indices}
            if self._has_path(obstacles):
                return obstacles
        return set()

    def _action_to_delta(self, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if abs(clipped[0]) >= abs(clipped[1]):
            return np.array([1 if clipped[0] >= 0 else -1, 0], dtype=np.int32)
        return np.array([0, 1 if clipped[1] >= 0 else -1], dtype=np.int32)

    def _get_obs(self) -> np.ndarray:
        scale = self.config.size - 1
        obstacle_map = np.zeros((self.config.size, self.config.size), dtype=np.float32)
        for x, y in self.obstacles:
            obstacle_map[y, x] = 1.0

        time_fraction = min(self.steps / self.config.max_steps, 1.0)
        features = np.array(
            [
                self.agent_pos[0] / scale,
                self.agent_pos[1] / scale,
                self.goal_pos[0] / scale,
                self.goal_pos[1] / scale,
                time_fraction,
            ],
            dtype=np.float32,
        )
        return np.concatenate([features, obstacle_map.ravel()])

    def _get_info(self) -> dict:
        return {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "distance": self._manhattan_distance(self.agent_pos, self.goal_pos),
            "steps": self.steps,
            "robustness": self._goal_robustness(),
            "funnel_gamma": self._funnel_gamma(self.steps),
            "is_success": np.array_equal(self.agent_pos, self.goal_pos),
        }

    def _in_bounds(self, position: np.ndarray) -> bool:
        return bool(np.all(position >= 0) and np.all(position < self.config.size))

    @staticmethod
    def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.abs(a - b).sum())

    def _has_path(self, obstacles: set[tuple[int, int]]) -> bool:
        start = (0, 0)
        goal = (self.config.size - 1, self.config.size - 1)
        frontier = [start]
        visited = {start}

        while frontier:
            x, y = frontier.pop()
            if (x, y) == goal:
                return True

            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                neighbor = (nx, ny)
                if (
                    0 <= nx < self.config.size
                    and 0 <= ny < self.config.size
                    and neighbor not in obstacles
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    frontier.append(neighbor)

        return False

    def _render_ansi(self) -> str:
        rows: list[str] = []
        for y in reversed(range(self.config.size)):
            row = []
            for x in range(self.config.size):
                cell = (x, y)
                if np.array_equal(self.agent_pos, [x, y]):
                    row.append("A")
                elif np.array_equal(self.goal_pos, [x, y]):
                    row.append("G")
                elif cell in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def _render_rgb_array(self) -> np.ndarray:
        cell_size = 64
        border = 2
        image_size = self.config.size * cell_size
        image = np.full((image_size, image_size, 3), 245, dtype=np.uint8)

        colors = {
            "grid": np.array([210, 215, 220], dtype=np.uint8),
            "obstacle": np.array([55, 65, 81], dtype=np.uint8),
            "goal": np.array([34, 197, 94], dtype=np.uint8),
            "agent": np.array([59, 130, 246], dtype=np.uint8),
        }

        for x in range(self.config.size):
            for y in range(self.config.size):
                px = x * cell_size
                py = (self.config.size - 1 - y) * cell_size
                image[py : py + border, px : px + cell_size] = colors["grid"]
                image[py : py + cell_size, px : px + border] = colors["grid"]

                if (x, y) in self.obstacles:
                    image[py + 8 : py + cell_size - 8, px + 8 : px + cell_size - 8] = colors["obstacle"]

        gx, gy = self.goal_pos
        self._paint_square(image, gx, gy, colors["goal"], padding=12)

        ax, ay = self.agent_pos
        self._paint_square(image, ax, ay, colors["agent"], padding=18)
        return image

    def _paint_square(self, image: np.ndarray, x: int, y: int, color: np.ndarray, padding: int) -> None:
        cell_size = image.shape[0] // self.config.size
        px = x * cell_size
        py = (self.config.size - 1 - y) * cell_size
        image[py + padding : py + cell_size - padding, px + padding : px + cell_size - padding] = color
