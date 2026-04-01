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
    assert surf.get_width() == 320
    assert surf.get_height() == 320


def test_snake_game_10_worms_init(pygame_dummy):
    """Verify SnakeGame initializes with exactly 10 WormAgents."""
    game = SnakeGame(grid_size=10, fps=12)
    assert len(game.worms) == 10
    assert game.focused_idx == 0
    assert not game.solo_mode
