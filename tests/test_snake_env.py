"""Unit tests for the headless SnakeEnv."""

import pytest
from game.snake_env import SnakeEnv


def test_reset_returns_valid_state():
    env = SnakeEnv(grid_size=10, seed=0)
    head, food, body, direction = env.reset()
    assert 0 <= head[0] < 10
    assert 0 <= head[1] < 10
    assert head == body[0]
    assert direction == "RIGHT"


def test_step_moves_head():
    env = SnakeEnv(grid_size=10, seed=0)
    (hx, hy), _, _, _ = env.reset()
    (nx, ny), _, _, _ = env.step("RIGHT")[0]
    assert nx == hx + 1
    assert ny == hy


def test_opposite_direction_ignored():
    env = SnakeEnv(grid_size=10, seed=0)
    env.reset()
    # Snake starts going RIGHT — trying LEFT should be ignored
    obs, done = env.step("LEFT")
    head, _, _, direction = obs
    assert direction == "RIGHT"


def test_wall_collision_ends_game():
    env = SnakeEnv(grid_size=5, seed=0)
    env.reset()
    # Force snake into top-left corner and try to go out of bounds
    env.body = [(0, 0), (1, 0), (2, 0)]
    env.direction = "LEFT"
    _, done = env.step("LEFT")
    assert done


def test_food_eaten_increases_score():
    env = SnakeEnv(grid_size=10, seed=0)
    env.reset()
    initial_score = env.score
    # Place food directly in front
    env.food = (env.body[0][0] + 1, env.body[0][1])
    env.step("RIGHT")
    assert env.score == initial_score + 1
