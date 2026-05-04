"""
Central layout constants and coordinate mappings for NeuroWorm Snake.

Single source of truth for all grid, field, arena, panel, and virtual dimensions.
"""

# Grid dimensions (logical simulation cells)
GRID_SIZE: int = 20
CELL: int = 32
WALL: int = 1  # Outer border wall thickness in cells (outside the 0..GRID_SIZE-1 field)

# Playable field dimensions (inner grid)
FIELD_W: int = GRID_SIZE * CELL               # 640 px
FIELD_H: int = GRID_SIZE * CELL               # 640 px
FIELD_PX: int = FIELD_W                       # 640 px

# Arena dimensions (including the outer wall frame)
ARENA_W: int = (GRID_SIZE + 2 * WALL) * CELL  # 704 px
ARENA_H: int = (GRID_SIZE + 2 * WALL) * CELL  # 704 px
ARENA_PX: int = ARENA_W                       # 704 px
GAME_W: int = ARENA_W                         # 704 px
GAME_H: int = ARENA_H                         # 704 px

# Side dashboard panel and virtual display dimensions
PANEL_W: int = 360                            # 360 px
VIRTUAL_W: int = GAME_W + PANEL_W             # 1064 px
VIRTUAL_H: int = ARENA_H                      # 704 px


def cell_to_px(x: float, y: float, cell_size: int = CELL, wall_cells: int = WALL) -> tuple[float, float]:
    """
    Return top-left pixel coordinates (px, py) for grid cell (x, y).

    Grid cell (0, 0) starts immediately inside the outer wall at (wall_cells * cell_size).
    """
    return (x + wall_cells) * cell_size, (y + wall_cells) * cell_size


def cell_center_px(x: float, y: float, cell_size: int = CELL, wall_cells: int = WALL) -> tuple[float, float]:
    """Return center pixel coordinates (cx, cy) for grid cell (x, y)."""
    return (x + wall_cells + 0.5) * cell_size, (y + wall_cells + 0.5) * cell_size
