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
