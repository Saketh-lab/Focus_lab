from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class GridWorldConfig:
    size: int = 7
    max_steps: int = 60
    obstacle_count: int = 5
    step_penalty: float = -0.01
    wall_penalty: float = -0.05
    obstacle_penalty: float = -0.08
    goal_reward: float = 5.0
    distance_reward_scale: float = 0.10


class ContinuousGridWorldEnv(gym.Env):
    """A small GridWorld with continuous actions so it can be trained with SAC.

    Action format:
        A 2D continuous vector `[dx, dy]` in `[-1, 1]`.

    The larger absolute component decides the movement axis. The sign decides
    direction. For example, `[0.8, -0.2]` moves right and `[0.1, -0.9]` moves up.
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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_size = 4 + self.config.size * self.config.size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

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
        self.obstacles = self._sample_obstacles()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1

        previous_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        proposed_pos = self.agent_pos + self._action_to_delta(action)

        reward = self.config.step_penalty
        blocked = False

        if not self._in_bounds(proposed_pos):
            blocked = True
            reward += self.config.wall_penalty
        elif tuple(proposed_pos) in self.obstacles:
            blocked = True
            reward += self.config.obstacle_penalty
        else:
            self.agent_pos = proposed_pos.astype(np.int32)

        current_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        reward += self.config.distance_reward_scale * (previous_distance - current_distance)

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

    def _sample_obstacles(self) -> set[tuple[int, int]]:
        blocked = {(0, 0), (self.config.size - 1, self.config.size - 1)}
        all_cells = [
            (x, y)
            for x in range(self.config.size)
            for y in range(self.config.size)
            if (x, y) not in blocked
        ]
        for _ in range(1_000):
            indices = self._rng.choice(len(all_cells), size=self.config.obstacle_count, replace=False)
            obstacles = {all_cells[int(index)] for index in indices}
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
        positions = np.array(
            [
                self.agent_pos[0] / scale,
                self.agent_pos[1] / scale,
                self.goal_pos[0] / scale,
                self.goal_pos[1] / scale,
            ],
            dtype=np.float32,
        )
        return np.concatenate([positions, obstacle_map.ravel()])

    def _get_info(self) -> dict:
        return {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "distance": self._manhattan_distance(self.agent_pos, self.goal_pos),
            "steps": self.steps,
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

                cell = (x, y)
                if cell in self.obstacles:
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
