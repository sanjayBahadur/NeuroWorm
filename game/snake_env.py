"""
Headless Snake environment for use during neural optimization and testing.
No pygame dependency — pure numpy/Python for speed.
"""

import numpy as np
from typing import Optional


class SnakeEnv:
    """
    Grid-based Snake environment.

    Observation returned by step() and reset():
        (head_pos, food_pos, body_list, direction)
    where head_pos and food_pos are (x, y) tuples and
    body_list is a list of (x, y) tuples ordered head→tail.
    """

    DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
    DX = {"RIGHT": 1, "LEFT": -1, "UP": 0,  "DOWN": 0}
    DY = {"RIGHT": 0, "LEFT": 0,  "UP": -1, "DOWN": 1}
    OPPOSITES = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def __init__(self, grid_size: int = 20, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.score = 0
        self.steps = 0
        self.direction = "RIGHT"
        self.body: list[tuple[int, int]] = []
        self.food: tuple[int, int] = (0, 0)
        self.done = False
        self.death_cause: Optional[str] = None
        self.reset()

    def reset(self) -> tuple:
        """Reset to initial state. Returns first observation."""
        g = self.grid_size
        cx, cy = g // 2, g // 2
        self.body        = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction   = "RIGHT"
        self.score       = 0
        self.steps       = 0
        self.done        = False
        self.death_cause = None
        self.food        = self._place_food()
        return self._obs()

    def step(self, action: str) -> tuple[tuple, bool]:
        """
        Apply action and advance one step.

        Args:
            action: Direction string. Opposite of current direction is ignored.
        Returns:
            (observation, done)
        """
        if self.done:
            return self._obs(), True

        # Prevent 180° reversal
        if action == self.OPPOSITES[self.direction]:
            action = self.direction
        self.direction = action

        hx, hy = self.body[0]
        nx = hx + self.DX[action]
        ny = hy + self.DY[action]
        new_head = (nx, ny)
        self.steps += 1

        # Collision check
        if (
            nx < 0 or ny < 0
            or nx >= self.grid_size
            or ny >= self.grid_size
        ):
            self.done = True
            self.death_cause = "wall"
            return self._obs(), True
        elif new_head in self.body[:-1]:
            self.done = True
            self.death_cause = "self"
            return self._obs(), True

        # Move snake
        self.body.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
        else:
            self.body.pop()

        return self._obs(), False

    def _obs(self) -> tuple:
        return (self.body[0], self.food, list(self.body), self.direction)

    def _place_food(self) -> tuple[int, int]:
        """Place food at a random empty cell."""
        body_set = set(self.body)
        while True:
            pos = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
            if pos not in body_set:
                return pos
