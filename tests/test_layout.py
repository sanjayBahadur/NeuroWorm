"""Headless unit tests for layout constants, coordinate transforms, and wall boundary isolation."""

import os
import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from game.layout import (
    GRID_SIZE, CELL, WALL, FIELD_W, FIELD_H, ARENA_W, ARENA_H,
    cell_to_px, cell_center_px
)
from game.renderer import Renderer, WORM_PALETTES


def test_cell_to_px_monotonic_and_uniform():
    """Verify cell_to_px is strictly monotonic and advances by exactly CELL pixels."""
    for x in range(GRID_SIZE):
        px1, py1 = cell_to_px(x, 0)
        px2, py2 = cell_to_px(x + 1, 0)
        assert px2 - px1 == CELL
        assert py1 == py2

    for y in range(GRID_SIZE):
        px1, py1 = cell_to_px(0, y)
        px2, py2 = cell_to_px(0, y + 1)
        assert py2 - py1 == CELL
        assert px1 == px2


def test_edge_cells_strictly_inside_field():
    """Verify that every playable cell including corners and edges lies strictly inside field_rect."""
    field_rect = pygame.Rect(WALL * CELL, WALL * CELL, FIELD_W, FIELD_H)

    corners = [
        (0, 0),
        (GRID_SIZE - 1, 0),
        (0, GRID_SIZE - 1),
        (GRID_SIZE - 1, GRID_SIZE - 1),
    ]
    edges = (
        [(x, 0) for x in range(GRID_SIZE)]
        + [(x, GRID_SIZE - 1) for x in range(GRID_SIZE)]
        + [(0, y) for y in range(GRID_SIZE)]
        + [(GRID_SIZE - 1, y) for y in range(GRID_SIZE)]
    )

    for cx, cy in corners + edges:
        px, py = cell_to_px(cx, cy)
        cell_rect = pygame.Rect(px, py, CELL, CELL)
        assert field_rect.contains(cell_rect), f"Cell ({cx}, {cy}) rect {cell_rect} not inside field {field_rect}"


def test_adjacent_wall_cells_lie_in_wall_band():
    """Verify that outer border cells (-1, g) lie strictly outside field_rect but inside arena."""
    field_rect = pygame.Rect(WALL * CELL, WALL * CELL, FIELD_W, FIELD_H)
    arena_rect = pygame.Rect(0, 0, ARENA_W, ARENA_H)

    outer_wall_coords = (
        [(-1, y) for y in range(GRID_SIZE)]
        + [(GRID_SIZE, y) for y in range(GRID_SIZE)]
        + [(x, -1) for x in range(GRID_SIZE)]
        + [(x, GRID_SIZE) for x in range(GRID_SIZE)]
    )

    for wx, wy in outer_wall_coords:
        px, py = cell_to_px(wx, wy)
        wrect = pygame.Rect(px, py, CELL, CELL)
        assert not field_rect.colliderect(wrect), f"Wall cell ({wx}, {wy}) overlaps field_rect"
        assert arena_rect.contains(wrect), f"Wall cell ({wx}, {wy}) not inside arena_rect"


def test_edge_worm_render_does_not_bleed_into_wall():
    """Render a worm whose head is on edge cell (0, 5) and verify wall band pixels remain unpolluted."""
    renderer = Renderer(cell_size=CELL, grid_size=GRID_SIZE)

    # Edge-standing worm on cell 0
    dummy_worm = type("DummyWorm", (), {
        "obs": ((0, 5), (10, 10), [(0, 5), (0, 6), (0, 7)], "RIGHT"),
        "direction": "RIGHT",
        "alive": True,
        "score": 0,
        "fitness": 0.0,
        "id": 1,
    })()

    surf = renderer.render_multi_worms([dummy_worm], food=(10, 10), solo_mode=True)

    # Cuticle color for worm 0
    cuticle_col = WORM_PALETTES[0]["cuticle"]

    # Sample inside the playable field for cell (0, 5): x in [32, 63], y in [160, 191]
    center_px = renderer.cell_center_px(0, 5)
    field_head_color = surf.get_at((int(center_px[0]), int(center_px[1])))[:3]
    # Head center has worm cuticle/core blended with the bioluminescent neural pulse (bright pink/white)
    assert field_head_color[0] >= 200, (
        f"Expected bright worm cuticle/pulse at center {center_px}, got {field_head_color}"
    )

    # Sample left wall band x in [0, 31], y around 5*32 + 32 = 192
    # Specifically, check wall pixels at x = 16, y = 192 (mid-wall cell (-1, 5))
    left_wall_px = surf.get_at((16, int(center_px[1])))[:3]
    # Stone wall colors are dark earth tones (R < 70), untouched by worm glow/body
    assert left_wall_px[0] < 70, f"Worm color bled into left wall band: {left_wall_px}"
    assert left_wall_px != cuticle_col, "Worm cuticle bled into left wall band"
