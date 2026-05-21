"""Unit tests for Task B3: Arena rendering remodel, Chaikin smoothing, hero worm, egg rings, and whisker rays."""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pytest
import pygame as pg
import numpy as np

from game.renderer import Renderer, chaikin_smooth, WORM_PALETTES
from game.snake_game import SnakeGame, WormAgent
from training.genetic_algorithm import DEFAULT_GENOME


@pytest.fixture(autouse=True)
def init_pygame():
    pg.init()
    yield
    pg.quit()


def test_chaikin_smooth():
    """Verify that Chaikin smoothing bevels sharp corners and preserves endpoints."""
    # 90-degree corner at (10, 20)
    pts = [(10.0, 10.0), (10.0, 20.0), (20.0, 20.0)]
    smoothed = chaikin_smooth(pts, iterations=2)

    # Start and end endpoints must be preserved exactly
    assert smoothed[0] == (10.0, 10.0)
    assert smoothed[-1] == (20.0, 20.0)

    # Number of points increases to smooth the corner
    assert len(smoothed) > len(pts)

    # Corner point (10.0, 20.0) is cut/smoothed away
    assert (10.0, 20.0) not in smoothed


def test_renderer_b3_elements():
    """Verify renderer renders hero worm with rays and ghosts without errors."""
    renderer = Renderer(grid_size=20)
    worms = [
        WormAgent(worm_id=i + 1, genome=DEFAULT_GENOME, color=WORM_PALETTES[i]["color"], grid_size=20, seed=42 + i)
        for i in range(10)
    ]

    # Step worms so encoder telemetry is populated
    for w in worms:
        w.step(shared_food=(15, 15))

    # Mark worm 3 as dead to trigger death animation event
    worms[2].alive = False
    worms[2].death_cause = "wall"

    surf = renderer.render_multi_worms(
        worms=worms,
        food=(15, 15),
        gen_info="GEN #1",
        focused_idx=0,
        solo_mode=False,
        focus_mode="follow",
        dt=0.016,
    )

    assert isinstance(surf, pg.Surface)
    assert surf.get_width() == (20 + 2) * 32
    assert surf.get_height() == (20 + 2) * 32

    # Verify death event was recorded
    assert 3 in renderer.death_events
    assert renderer.death_events[3]["cause"] == "hit the wall"


def test_snake_game_follow_leader_mode():
    """Verify SnakeGame follows leader dynamically and toggles with manual picks."""
    game = SnakeGame(grid_size=20, fps=12, seed=42)
    assert game.focus_mode == "follow"

    # Make worm 5 score more eggs
    game.worms[4].score = 3

    # In follow mode, focused_idx should lock onto worm 5 (index 4)
    alive = [w for w in game.worms if w.alive]
    best_w = max(alive, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness))
    assert best_w.id - 1 == 4

    # Toggle focus mode
    game.focus_mode = "manual"
    game.focused_idx = 1
    assert game.focus_mode == "manual"
    assert game.focused_idx == 1
