"""Unit tests for the 10-worm swarm simulation and multi-worm renderer."""

import os
import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg

from game.renderer import Renderer, WORM_PALETTES
from game.snake_game import WormAgent, SnakeGame
from training.genetic_algorithm import DEFAULT_GENOME


@pytest.fixture(scope="module")
def pygame_dummy():
    pg.init()
    pg.display.set_mode((1, 1))
    yield
    pg.quit()


def test_worm_agent_lifecycle(pygame_dummy):
    """Verify WormAgent initializes, steps, and updates score/fitness."""
    worm = WormAgent(worm_id=1, genome=DEFAULT_GENOME, color=(255, 175, 195), grid_size=10, seed=42)
    assert worm.alive
    assert worm.score == 0
    assert worm.steps == 0

    for _ in range(5):
        worm.step()

    assert worm.steps == 5
    assert worm.fitness > 0


def test_renderer_multi_worms(pygame_dummy):
    """Verify render_multi_worms renders all 10 worms without error."""
    renderer = Renderer(cell_size=32, grid_size=10)
    worms = [
        WormAgent(worm_id=i + 1, genome=DEFAULT_GENOME, color=WORM_PALETTES[i]["color"], grid_size=10, seed=42)
        for i in range(10)
    ]
    # Mark worm 4 as dead
    worms[3].alive = False

    surf = renderer.render_multi_worms(
        worms=worms,
        food=(5, 5),
        gen_info="GEN #1",
        focused_idx=0,
        solo_mode=False,
    )
    expected_size = (10 + 2) * 32  # 10 grid cells + 2 wall cells
    assert surf.get_width() == expected_size
    assert surf.get_height() == expected_size


def test_snake_game_10_worms_init(pygame_dummy):
    """Verify SnakeGame initializes with exactly 10 WormAgents."""
    game = SnakeGame(grid_size=10, fps=12)
    assert len(game.worms) == 10
    assert game.focused_idx == 0
    assert not game.solo_mode
    assert isinstance(game.shared_food, tuple)
    assert len(game.shared_food) == 2


def test_food_eating_and_particle_burst(pygame_dummy):
    """Verify that when a worm steps on shared food, it eats, grows, and triggers bursts."""
    game = SnakeGame(grid_size=10, fps=12)
    w = game.worms[0]
    
    # Place food directly in front of worm 0 (starts at (5, 5) heading RIGHT -> front is (6, 5))
    front_pos = (w.obs[0][0] + 1, w.obs[0][1])
    game.shared_food = front_pos
    w.env.food = front_pos
    w.direction = "RIGHT"

    initial_len = len(w.obs[2])
    ate = w.step(shared_food=front_pos)

    assert ate, "Worm should report eating food"
    assert w.score == 1, "Worm score should increment to 1"
    assert len(w.obs[2]) == initial_len + 1, "Worm body should grow by 1 segment"

    # Test burst effect
    game.renderer.add_food_burst(front_pos, w.color)
    assert len(game.renderer.burst_particles) == 16
    assert len(game.renderer.floating_popups) == 1

    surf = game.renderer.render_multi_worms(game.worms, game.shared_food)
    expected_size = (10 + 2) * 32  # 10 grid cells + 2 wall cells
    assert surf.get_width() == expected_size
    assert surf.get_height() == expected_size
