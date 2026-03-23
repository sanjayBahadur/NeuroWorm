"""
Pixel art renderer for NeuroWorm Snake.

Loads authentic transparent pixel-art sprites and renders:
  - Soil tilemap with pebble & root variations
  - Directional worm head with expressive eyes
  - Smooth articulated body segments with corner turning pieces
  - Pointed directional tail segments
  - Bioluminescent pulsing amber food egg
"""

import math
from pathlib import Path
from typing import Optional
import numpy as np
import pygame

ASSET_DIR  = Path(__file__).parent.parent / "assets"
SPRITE_DIR = ASSET_DIR / "sprites"
TILE_DIR   = ASSET_DIR / "tiles"
UI_DIR     = ASSET_DIR / "ui"

SOIL_DARK    = (48,  32,  22)
SOIL_MID     = (65,  45,  30)
WORM_PINK    = (255, 175, 195)
WORM_DARK    = (45,  20,  30)
FOOD_AMBER   = (255, 190,  60)
FOOD_WHITE   = (255, 248, 225)
SCORE_GOLD   = (200, 184, 126)
STONE_GREY   = (42,  38,  35)
TEXT_CREAM   = (240, 230, 205)

FONT_NAME    = "monospace"


def _try_load(path: Path) -> Optional[pygame.Surface]:
    if path.exists():
        try:
            return pygame.image.load(str(path)).convert_alpha()
        except pygame.error:
            return None
    return None


class Renderer:
    """Renders the game board with organic pixel art sprites."""

    def __init__(self, cell_size: int = 32, grid_size: int = 20):
        if not pygame.font.get_init():
            pygame.font.init()
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.W = cell_size * grid_size
        self.H = cell_size * grid_size

        self.surface = pygame.Surface((self.W, self.H))
        self.font = pygame.font.SysFont(FONT_NAME, 14, bold=True)
        self.big_font = pygame.font.SysFont(FONT_NAME, 28, bold=True)
        self.frame_count = 0

        self._load_sprites()
        self._build_tilemap()

    def _load_sprites(self) -> None:
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
            "body_h":     load_scaled(SPRITE_DIR / "worm_body_straight.png"),
            "body_v":     load_scaled(SPRITE_DIR / "worm_body_vertical.png"),
            "turn_ne":    load_scaled(SPRITE_DIR / "worm_body_turn_ne.png"),
            "turn_nw":    load_scaled(SPRITE_DIR / "worm_body_turn_nw.png"),
            "turn_se":    load_scaled(SPRITE_DIR / "worm_body_turn_se.png"),
            "turn_sw":    load_scaled(SPRITE_DIR / "worm_body_turn_sw.png"),
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
        rng = np.random.default_rng(1337)
        self.tilemap = rng.integers(0, 5, size=(self.grid_size, self.grid_size))

    def render(self, obs: tuple, score: int, game_over: bool, gen_info: str = "") -> pygame.Surface:
        self.frame_count += 1
        head, food, body, direction = obs
        self.surface.fill(SOIL_DARK)

        self._draw_background()
        self._draw_food(food)
        self._draw_snake(body, direction)
        self._draw_hud(score, game_over, gen_info)

        return self.surface

    def _draw_background(self) -> None:
        cs = self.cell_size
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                rect = pygame.Rect(gx * cs, gy * cs, cs, cs)
                is_border = (gx == 0 or gy == 0 or gx == self.grid_size - 1 or gy == self.grid_size - 1)

                if is_border:
                    surf = self.sprites.get("border")
                    if surf:
                        self.surface.blit(surf, rect)
                    else:
                        pygame.draw.rect(self.surface, STONE_GREY, rect)
                else:
                    tt = self.tilemap[gy, gx]
                    key = "soil_a" if tt == 3 else "soil"
                    surf = self.sprites.get(key)
                    if surf:
                        self.surface.blit(surf, rect)
                    else:
                        pygame.draw.rect(self.surface, SOIL_DARK, rect)

    def _draw_food(self, food: tuple) -> None:
        fx, fy = food
        cs = self.cell_size
        rect = pygame.Rect(fx * cs, fy * cs, cs, cs)

        # Subtle bioluminescent pulse
        pulse = 1.0 + 0.15 * math.sin(self.frame_count * 0.2)
        cx = fx * cs + cs // 2
        cy = fy * cs + cs // 2
        halo_radius = int((cs // 2 + 2) * pulse)
        halo_surf = pygame.Surface((halo_radius * 2, halo_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(halo_surf, (255, 200, 80, 50), (halo_radius, halo_radius), halo_radius)
        self.surface.blit(halo_surf, (cx - halo_radius, cy - halo_radius))

        surf = self.sprites.get("food")
        if surf:
            self.surface.blit(surf, rect)
        else:
            pygame.draw.circle(self.surface, FOOD_AMBER, (cx, cy), cs // 2 - 3)
            pygame.draw.circle(self.surface, FOOD_WHITE, (cx, cy), cs // 2 - 6)

    def _draw_snake(self, body: list, direction: str) -> None:
        cs = self.cell_size
        n = len(body)
        if n == 0:
            return

        for i, (bx, by) in enumerate(body):
            rect = pygame.Rect(bx * cs, by * cs, cs, cs)

            if i == 0:
                # ── Head ──────────────────────────────────────────────
                key = f"head_{direction.lower()}"
                surf = self.sprites.get(key)
                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, WORM_PINK, rect, border_radius=6)
            elif i == n - 1:
                # ── Tail ──────────────────────────────────────────────
                prev_x, prev_y = body[i - 1]
                tdx = bx - prev_x
                tdy = by - prev_y
                tail_dir = "right"
                if tdx > 0:
                    tail_dir = "right"
                elif tdx < 0:
                    tail_dir = "left"
                elif tdy > 0:
                    tail_dir = "down"
                elif tdy < 0:
                    tail_dir = "up"

                key = f"tail_{tail_dir}"
                surf = self.sprites.get(key)
                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, WORM_PINK, rect.inflate(-6, -6), border_radius=4)
            else:
                # ── Articulated Body Segment ──────────────────────────
                prev_x, prev_y = body[i - 1]
                next_x, next_y = body[i + 1]

                v1 = (prev_x - bx, prev_y - by)
                v2 = (next_x - bx, next_y - by)

                if v1[0] == 0 and v2[0] == 0:
                    # Vertical segment
                    surf = self.sprites.get("body_v") or self.sprites.get("body_h")
                elif v1[1] == 0 and v2[1] == 0:
                    # Horizontal segment
                    surf = self.sprites.get("body_h")
                else:
                    # Corner bend
                    dirs = {v1, v2}
                    if (0, -1) in dirs and (1, 0) in dirs:
                        surf = self.sprites.get("turn_ne")
                    elif (0, -1) in dirs and (-1, 0) in dirs:
                        surf = self.sprites.get("turn_nw")
                    elif (0, 1) in dirs and (1, 0) in dirs:
                        surf = self.sprites.get("turn_se")
                    elif (0, 1) in dirs and (-1, 0) in dirs:
                        surf = self.sprites.get("turn_sw")
                    else:
                        surf = self.sprites.get("body_h")

                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, WORM_PINK, rect.inflate(-4, -4), border_radius=4)

    def _draw_hud(self, score: int, game_over: bool, gen_info: str) -> None:
        hud_bg = pygame.Surface((self.W, 26), pygame.SRCALPHA)
        hud_bg.fill((20, 12, 8, 180))
        self.surface.blit(hud_bg, (0, 0))

        score_txt = self.font.render(f"EGG SCORE: {score}", True, SCORE_GOLD)
        self.surface.blit(score_txt, (10, 5))

        if gen_info:
            info_txt = self.font.render(gen_info, True, TEXT_CREAM)
            self.surface.blit(info_txt, (self.W - info_txt.get_width() - 10, 5))

        if game_over:
            over_bg = pygame.Surface((self.W, 60), pygame.SRCALPHA)
            over_bg.fill((30, 15, 10, 220))
            self.surface.blit(over_bg, (0, self.H // 2 - 30))
            over_txt = self.big_font.render("SIMULATION RESET", True, SCORE_GOLD)
            self.surface.blit(over_txt, (self.W // 2 - over_txt.get_width() // 2, self.H // 2 - 20))
