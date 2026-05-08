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


def test_wall_collision_records_death_cause():
    env = SnakeEnv(grid_size=5, seed=0)
    env.reset()
    assert env.death_cause is None
    env.body = [(0, 0), (1, 0), (2, 0)]
    env.direction = "LEFT"
    _, done = env.step("LEFT")
    assert done
    assert env.death_cause == "wall"
    env.reset()
    assert env.death_cause is None


def test_self_collision_records_death_cause():
    env = SnakeEnv(grid_size=10, seed=0)
    env.reset()
    # Body configured in a loop: head at (2, 2), moving DOWN into (2, 3) where body is
    env.body = [(2, 2), (1, 2), (1, 3), (2, 3), (3, 3)]
    env.direction = "RIGHT"
    _, done = env.step("DOWN")
    assert done
    assert env.death_cause == "self"
    env.reset()
    assert env.death_cause is None


def test_worm_agent_death_causes():
    from game.snake_game import WormAgent
    from training.genetic_algorithm import DEFAULT_GENOME

    # 1. Starvation death
    w = WormAgent(
        worm_id=1,
        genome=DEFAULT_GENOME,
        color=(255, 100, 100),
        grid_size=20,
        seed=42,
        starve_limit=5,
    )
    assert w.death_cause is None
    # Step 6 times without food
    for _ in range(6):
        w.step(shared_food=(19, 19))
    assert not w.alive
    assert w.death_cause == "starved"

    # Reset clears cause
    w.reset()
    assert w.alive
    assert w.death_cause is None

    # 2. Wall collision death
    w.env.body = [(19, 10), (18, 10), (17, 10)]
    w.obs = w.env._obs()
    w.direction = "RIGHT"
    w.env.direction = "RIGHT"
    w.env.step("RIGHT")
    w.alive = False
    w.death_cause = w.env.death_cause
    assert w.death_cause == "wall"


def test_food_eaten_increases_score():
    env = SnakeEnv(grid_size=10, seed=0)
    env.reset()
    initial_score = env.score
    # Place food directly in front
    env.food = (env.body[0][0] + 1, env.body[0][1])
    env.step("RIGHT")
    assert env.score == initial_score + 1

