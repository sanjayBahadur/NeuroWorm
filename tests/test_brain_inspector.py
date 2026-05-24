"""Unit tests for the educational BrainInspector dashboard."""

import os
import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame as pg

from game.renderer import WORM_PALETTES
from game.snake_game import WormAgent
from viz.brain_inspector import BrainInspector
from training.genetic_algorithm import DEFAULT_GENOME


@pytest.fixture(scope="module")
def pygame_dummy():
    pg.init()
    pg.display.set_mode((1, 1))
    yield
    pg.quit()


def test_brain_inspector_render_flow(pygame_dummy):
    """Verify BrainInspector renders the Live Thoughts flow tab correctly."""
    inspector = BrainInspector(width=320, height=640)
    worms = [
        WormAgent(worm_id=i + 1, genome=DEFAULT_GENOME, color=WORM_PALETTES[i]["color"], grid_size=10, seed=42)
        for i in range(10)
    ]
    worms[0].step()

    surf = inspector.render(
        worm=worms[0],
        all_worms=worms,
        generation=1,
        view_mode="flow",
        mouse_pos=(50, 50),
        solo_mode=False,
    )

    assert surf.get_width() == 320
    assert surf.get_height() == 640


def test_brain_inspector_render_synapses(pygame_dummy):
    """Verify BrainInspector renders the Synapse Learning tab correctly."""
    inspector = BrainInspector(width=320, height=640)
    worms = [
        WormAgent(worm_id=i + 1, genome=DEFAULT_GENOME, color=WORM_PALETTES[i]["color"], grid_size=10, seed=42)
        for i in range(10)
    ]

    surf = inspector.render(
        worm=worms[0],
        all_worms=worms,
        generation=2,
        view_mode="synapses",
        mouse_pos=(160, 45),
        solo_mode=False,
    )

    assert surf.get_width() == 320
    assert surf.get_height() == 640


def test_leaderboard_row_click_hit_rects(pygame_dummy):
    """Verify leaderboard row rects allow accurate click detection without magic numbers."""
    inspector = BrainInspector()
    worms = [
        WormAgent(worm_id=i + 1, genome=DEFAULT_GENOME, color=WORM_PALETTES[i]["color"], grid_size=10, seed=42)
        for i in range(10)
    ]
    # Render to populate hit rects
    inspector.render(worm=worms[0], all_worms=worms, generation=1)

    assert len(inspector.leaderboard_hit_rects) >= 5

    # Click first row
    first_rect, first_idx = inspector.leaderboard_hit_rects[0]
    clicked = inspector.get_clicked_worm_idx(first_rect.center)
    assert clicked == first_idx


def test_learning_curve_empty_and_populated(pygame_dummy):
    """Verify learning curve card handles empty history and populated history cleanly."""
    inspector = BrainInspector()
    worms = [
        WormAgent(worm_id=i + 1, genome=DEFAULT_GENOME, color=WORM_PALETTES[i]["color"], grid_size=10, seed=42)
        for i in range(10)
    ]

    # Empty history
    surf_empty = inspector.render(worm=worms[0], all_worms=worms, generation=1, history=[])
    assert isinstance(surf_empty, pg.Surface)

    # Populated history
    history = [
        {"gen": 1, "best_score": 1, "avg_score": 0.2, "best_fitness": 350.0},
        {"gen": 2, "best_score": 3, "avg_score": 1.1, "best_fitness": 920.0},
        {"gen": 3, "best_score": 4, "avg_score": 1.8, "best_fitness": 1250.0},
    ]
    surf_pop = inspector.render(worm=worms[0], all_worms=worms, generation=3, history=history)
    assert isinstance(surf_pop, pg.Surface)

