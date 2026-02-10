"""
Pixel art renderer for NeurWorm Snake.

Loads sprites from assets/ and renders the game board each frame.
Falls back to colored rectangles if sprite files are not yet available.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

ASSET_DIR  = Path(__file__).parent.parent / "assets"
SPRITE_DIR = ASSET_DIR / "sprites"
TILE_DIR   = ASSET_DIR / "tiles"
UI_DIR     = ASSET_DIR / "ui"

# Earthy color palette (fallback when sprites not loaded)
SOIL_DARK    = (61,  43,  31)   # #3d2b1f — soil background
SOIL_MID     = (89,  62,  43)   # #593e2b — soil mid-tone
WORM_PINK    = (255, 158, 181)  # #ff9eb5 — worm body
WORM_DARK    = (200, 100, 130)  # darker worm outline
FOOD_AMBER   = (255, 221, 136)  # #ffdd88 — food glow
FOOD_WHITE   = (240, 230, 210)  # food shell
SCORE_GOLD   = (200, 184, 126)  # #c8b87e — score text
STONE_GREY   = (80,  72,  65)   # border stones
TEXT_CREAM   = (220, 210, 180)  # UI text

FONT_NAME    = "monospace"


def _try_load(path: Path) -> Optional[pygame.Surface]:
    """Load a surface if the file exists, else return None."""
    if path.exists():
        try:
            return pygame.image.load(str(path)).convert_alpha()
        except pygame.error:
            return None
    return None


class Renderer:
    """
    Renders the Snake game board with pixel art sprites.

    Supports two rendering modes:
      - Sprite mode:   Loads PNG assets from assets/ directory
      - Fallback mode: Renders colored rectangles (dev mode)
    """

    def __init__(self, cell_size: int = 32, grid_size: int = 20):
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.W = cell_size * grid_size
        self.H = cell_size * grid_size

        self.surface = pygame.Surface((self.W, self.H))
        self.font    = pygame.font.SysFont(FONT_NAME, 16, bold=True)
        self.big_font = pygame.font.SysFont(FONT_NAME, 32, bold=True)

        self._load_sprites()
        self._build_tilemap()

    # ─────────────────────────────────────────────────────────────────────
    def _load_sprites(self) -> None:
        """Load all PNG assets. Missing files fall back to None."""
        cs = self.cell_size

        def load_scaled(path: Path) -> Optional[pygame.Surface]:
            surf = _try_load(path)
            if surf is not None:
                return pygame.transform.scale(surf, (cs, cs))
            return None

        self.sprites = {
            "head_right": load_scaled(SPRITE_DIR / "worm_head_right.png"),
            "head_left":  load_scaled(SPRITE_DIR / "worm_head_left.png"),
            "head_up":    load_scaled(SPRITE_DIR / "worm_head_up.png"),
            "head_down":  load_scaled(SPRITE_DIR / "worm_head_down.png"),
            "body":       load_scaled(SPRITE_DIR / "worm_body_straight.png"),
            "tail_right": load_scaled(SPRITE_DIR / "worm_tail_right.png"),
            "tail_left":  load_scaled(SPRITE_DIR / "worm_tail_left.png"),
            "tail_up":    load_scaled(SPRITE_DIR / "worm_tail_up.png"),
            "tail_down":  load_scaled(SPRITE_DIR / "worm_tail_down.png"),
            "food":       load_scaled(SPRITE_DIR / "food_egg.png"),
            "soil":       load_scaled(TILE_DIR   / "soil_base.png"),
            "soil_a":     load_scaled(TILE_DIR   / "soil_variant_a.png"),
            "border":     load_scaled(TILE_DIR   / "border_stone.png"),
        }

    def _build_tilemap(self) -> None:
        """
        Pre-render the background tilemap with random tile variation.
        Uses a seeded RNG so the map is consistent across frames.
        """
        rng = np.random.default_rng(1337)
        self.tilemap = rng.integers(0, 4, size=(self.grid_size, self.grid_size))

    # ─────────────────────────────────────────────────────────────────────
    def render(
        self,
        obs:       tuple,
        score:     int,
        game_over: bool,
    ) -> pygame.Surface:
        """
        Render one complete game frame.

        Args:
            obs:       (head_pos, food_pos, body, direction) from SnakeEnv.
            score:     Current score.
            game_over: Whether the game is over.

        Returns:
            Rendered pygame.Surface (W × H).
        """
        head, food, body, direction = obs
        self.surface.fill(SOIL_DARK)

        # Draw background tiles
        self._draw_background()

        # Draw food
        self._draw_food(food)

        # Draw snake body (tail → head order for correct layering)
        self._draw_snake(body, direction)

        # Draw score HUD
        self._draw_hud(score, game_over)

        return self.surface

    # ─────────────────────────────────────────────────────────────────────
    def _draw_background(self) -> None:
        cs = self.cell_size
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                rect  = pygame.Rect(gx * cs, gy * cs, cs, cs)
                tile_type = self.tilemap[gy, gx]

                # Border tiles
                is_border = (
                    gx == 0 or gy == 0
                    or gx == self.grid_size - 1
                    or gy == self.grid_size - 1
                )

                if is_border:
                    surf = self.sprites.get("border")
                    if surf:
                        self.surface.blit(surf, rect)
                    else:
                        pygame.draw.rect(self.surface, STONE_GREY, rect)
                else:
                    key  = "soil" if tile_type < 3 else "soil_a"
                    surf = self.sprites.get(key)
                    if surf:
                        self.surface.blit(surf, rect)
                    else:
                        pygame.draw.rect(self.surface, SOIL_DARK, rect)

                    # Subtle grid lines in fallback mode
                    if not surf:
                        pygame.draw.rect(
                            self.surface, SOIL_MID, rect, 1
                        )

    def _draw_food(self, food: tuple) -> None:
        fx, fy = food
        rect   = pygame.Rect(fx * self.cell_size, fy * self.cell_size,
                             self.cell_size, self.cell_size)
        surf   = self.sprites.get("food")
        if surf:
            self.surface.blit(surf, rect)
        else:
            # Fallback: glowing amber dot
            cx = fx * self.cell_size + self.cell_size // 2
            cy = fy * self.cell_size + self.cell_size // 2
            r  = self.cell_size // 2 - 4
            pygame.draw.circle(self.surface, FOOD_AMBER, (cx, cy), r)
            pygame.draw.circle(self.surface, FOOD_WHITE, (cx, cy), r - 4)

    def _draw_snake(self, body: list, direction: str) -> None:
        cs = self.cell_size
        for i, (bx, by) in enumerate(body):
            rect = pygame.Rect(bx * cs, by * cs, cs, cs)
            if i == 0:
                # Head
                key  = f"head_{direction.lower()}"
                surf = self.sprites.get(key)
                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, WORM_PINK, rect)
                    pygame.draw.rect(self.surface, WORM_DARK, rect, 2)
            elif i == len(body) - 1:
                # Tail — infer direction from previous segment
                if len(body) > 1:
                    px, py = body[-2]
                    tdx, tdy = bx - px, by - py
                    tail_dir = {
                        (1, 0): "right", (-1, 0): "left",
                        (0, 1): "down",  (0, -1): "up"
                    }.get((tdx, tdy), "right")
                    key  = f"tail_{tail_dir}"
                    surf = self.sprites.get(key)
                    if surf:
                        self.surface.blit(surf, rect)
                        continue
                pygame.draw.rect(self.surface, WORM_PINK, rect.inflate(-6, -6))
            else:
                # Body segment
                surf = self.sprites.get("body")
                if surf:
                    self.surface.blit(surf, rect)
                else:
                    inner = rect.inflate(-4, -4)
                    pygame.draw.rect(self.surface, WORM_PINK, inner)
                    pygame.draw.rect(self.surface, WORM_DARK, rect, 2)

    def _draw_hud(self, score: int, game_over: bool) -> None:
        # Score
        score_surf = self.font.render(
            f"SCORE: {score}", True, SCORE_GOLD
        )
        self.surface.blit(score_surf, (8, 8))

        if game_over:
            over_surf  = self.big_font.render("GAME OVER", True, SCORE_GOLD)
            restart    = self.font.render("Press R to restart", True, TEXT_CREAM)
            cx = self.W // 2 - over_surf.get_width()  // 2
            cy = self.H // 2 - over_surf.get_height() // 2
            self.surface.blit(over_surf, (cx, cy))
            self.surface.blit(restart, (cx + 20, cy + 40))
