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

# 10 Distinct Color Palettes for the 10 Worm Candidates
WORM_PALETTES = [
    {"name": "Champion 1", "tint": (255, 255, 255), "badge": "★1", "color": (255, 175, 195), "border": (255, 215, 0)},   # Gold
    {"name": "Champion 2", "tint": (150, 200, 255), "badge": "★2", "color": (160, 190, 255), "border": (200, 220, 255)}, # Silver
    {"name": "Champion 3", "tint": (255, 200, 120), "badge": "★3", "color": (255, 190, 130), "border": (205, 127, 50)},  # Bronze
    {"name": "Worm 4",     "tint": (140, 255, 180), "badge": "#4", "color": (140, 240, 180), "border": (100, 200, 140)},
    {"name": "Worm 5",     "tint": (255, 140, 180), "badge": "#5", "color": (255, 150, 180), "border": (210, 100, 140)},
    {"name": "Worm 6",     "tint": (200, 160, 255), "badge": "#6", "color": (210, 170, 255), "border": (160, 120, 220)},
    {"name": "Worm 7",     "tint": (130, 240, 240), "badge": "#7", "color": (140, 230, 230), "border": (90, 190, 190)},
    {"name": "Worm 8",     "tint": (255, 240, 130), "badge": "#8", "color": (255, 235, 140), "border": (210, 190, 80)},
    {"name": "Worm 9",     "tint": (220, 255, 140), "badge": "#9", "color": (200, 240, 130), "border": (160, 200, 80)},
    {"name": "Worm 10",    "tint": (255, 170, 130), "badge": "#10","color": (255, 170, 130), "border": (210, 120, 90)},
]


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
        self.font = pygame.font.SysFont(FONT_NAME, 13, bold=True)
        self.badge_font = pygame.font.SysFont(FONT_NAME, 9, bold=True)
        self.big_font = pygame.font.SysFont(FONT_NAME, 26, bold=True)
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

        # Pre-tint 10 sets of worm sprites
        self.worm_sprites = {}
        sprite_keys = [
            "head_right", "head_left", "head_up", "head_down",
            "body_h", "body_v", "turn_ne", "turn_nw", "turn_se", "turn_sw",
            "tail_right", "tail_left", "tail_up", "tail_down"
        ]
        for w_idx, pal in enumerate(WORM_PALETTES):
            self.worm_sprites[w_idx] = {}
            for k in sprite_keys:
                base = self.sprites.get(k)
                if base is not None:
                    tinted = base.copy()
                    tinted.fill(pal["tint"], special_flags=pygame.BLEND_RGB_MULT)
                    self.worm_sprites[w_idx][k] = tinted
                else:
                    self.worm_sprites[w_idx][k] = None

    def _build_tilemap(self) -> None:
        rng = np.random.default_rng(1337)
        self.tilemap = rng.integers(0, 5, size=(self.grid_size, self.grid_size))

    def render(self, obs: tuple, score: int, game_over: bool, gen_info: str = "") -> pygame.Surface:
        """Backward-compatible single worm render."""
        self.frame_count += 1
        head, food, body, direction = obs
        self.surface.fill(SOIL_DARK)

        self._draw_background()
        self._draw_food(food)
        self._draw_snake(body, direction, w_idx=0)
        self._draw_hud(score, game_over, gen_info)

        return self.surface

    def render_multi_worms(
        self,
        worms: list,
        food: tuple,
        gen_info: str = "",
        focused_idx: int = 0,
        solo_mode: bool = False,
    ) -> pygame.Surface:
        """Render all 10 worms simultaneously on the 20x20 arena."""
        self.frame_count += 1
        self.surface.fill(SOIL_DARK)

        self._draw_background()
        self._draw_food(food)

        # 1. Draw dead worms first (faint background trails)
        if not solo_mode:
            for i, w in enumerate(worms):
                if not w.alive and i != focused_idx:
                    self._draw_dead_worm(w, i)

        # 2. Draw alive worms (non-focused)
        for i, w in enumerate(worms):
            if solo_mode and i != focused_idx:
                continue
            if w.alive and i != focused_idx:
                self._draw_alive_worm(w, i, is_focused=False)

        # 3. Draw focused worm on top
        if focused_idx < len(worms):
            fw = worms[focused_idx]
            if fw.alive:
                self._draw_alive_worm(fw, focused_idx, is_focused=True)
            elif solo_mode or True:
                self._draw_dead_worm(fw, focused_idx, is_focused=True)

        # 4. Multi-worm HUD banner
        alive_count = sum(1 for w in worms if w.alive)
        leader = max(worms, key=lambda w: (w.score, w.fitness))
        self._draw_multi_hud(alive_count, len(worms), leader, gen_info)

        return self.surface

    def _draw_dead_worm(self, worm, w_idx: int, is_focused: bool = False) -> None:
        cs = self.cell_size
        body = worm.obs[2]
        if not body:
            return

        for bx, by in body:
            rect = pygame.Rect(bx * cs + 3, by * cs + 3, cs - 6, cs - 6)
            pygame.draw.rect(self.surface, (50, 32, 24), rect, border_radius=3)

        hx, hy = body[0]
        hcx, hcy = hx * cs + cs // 2, hy * cs + cs // 2
        pygame.draw.circle(self.surface, (150, 35, 35), (hcx, hcy), 5)
        x_txt = self.badge_font.render("x", True, (255, 200, 200))
        self.surface.blit(x_txt, (hcx - x_txt.get_width() // 2, hcy - x_txt.get_height() // 2 - 1))

    def _draw_alive_worm(self, worm, w_idx: int, is_focused: bool = False) -> None:
        cs = self.cell_size
        body = worm.obs[2]
        direction = worm.direction
        if not body:
            return

        pal = WORM_PALETTES[w_idx % len(WORM_PALETTES)]
        self._draw_snake(body, direction, w_idx=w_idx)

        hx, hy = body[0]
        hcx = hx * cs + cs // 2
        hcy = hy * cs + cs // 2

        # Focused worm halo
        if is_focused:
            pulse = 1.0 + 0.15 * math.sin(self.frame_count * 0.25)
            r_halo = int((cs // 2 + 4) * pulse)
            halo_surf = pygame.Surface((r_halo * 2, r_halo * 2), pygame.SRCALPHA)
            pygame.draw.circle(halo_surf, (255, 220, 80, 80), (r_halo, r_halo), r_halo, 2)
            self.surface.blit(halo_surf, (hcx - r_halo, hcy - r_halo))

        # Floating badge above head
        badge_str = pal["badge"]
        b_surf = self.badge_font.render(badge_str, True, (20, 12, 8))
        bw = b_surf.get_width() + 6
        bh = b_surf.get_height() + 2
        bx = hcx - bw // 2
        by = hy * cs - bh - 2

        pill_rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(self.surface, pal["border"], pill_rect, border_radius=3)
        self.surface.blit(b_surf, (bx + 3, by + 1))

    def _draw_multi_hud(self, alive_count: int, total_count: int, leader, gen_info: str) -> None:
        hud_bg = pygame.Surface((self.W, 26), pygame.SRCALPHA)
        hud_bg.fill((18, 10, 6, 210))
        self.surface.blit(hud_bg, (0, 0))

        # Left: Surving worms count
        surv_col = (0, 230, 140) if alive_count > 3 else ((255, 200, 60) if alive_count > 0 else (255, 80, 80))
        surv_txt = self.font.render(f"WORMS ALIVE: {alive_count}/{total_count}", True, surv_col)
        self.surface.blit(surv_txt, (10, 5))

        # Center: Gen Info
        if gen_info:
            g_txt = self.font.render(gen_info, True, TEXT_CREAM)
            self.surface.blit(g_txt, (self.W // 2 - g_txt.get_width() // 2, 5))

        # Right: Leader info
        lead_txt = self.font.render(f"LEADER: WORM #{leader.id} ({leader.score} 🥚)", True, SCORE_GOLD)
        self.surface.blit(lead_txt, (self.W - lead_txt.get_width() - 10, 5))

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

    def _draw_snake(self, body: list, direction: str, w_idx: int = 0) -> None:
        cs = self.cell_size
        n = len(body)
        if n == 0:
            return

        sprites = self.worm_sprites.get(w_idx) or self.sprites
        pal = WORM_PALETTES[w_idx % len(WORM_PALETTES)]
        fallback_col = pal["color"]

        for i, (bx, by) in enumerate(body):
            rect = pygame.Rect(bx * cs, by * cs, cs, cs)

            if i == 0:
                # ── Head ──────────────────────────────────────────────
                key = f"head_{direction.lower()}"
                surf = sprites.get(key)
                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, fallback_col, rect, border_radius=6)
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
                surf = sprites.get(key)
                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, fallback_col, rect.inflate(-6, -6), border_radius=4)
            else:
                # ── Articulated Body Segment ──────────────────────────
                prev_x, prev_y = body[i - 1]
                next_x, next_y = body[i + 1]

                v1 = (prev_x - bx, prev_y - by)
                v2 = (next_x - bx, next_y - by)

                if v1[0] == 0 and v2[0] == 0:
                    # Vertical segment
                    surf = sprites.get("body_v") or sprites.get("body_h")
                elif v1[1] == 0 and v2[1] == 0:
                    # Horizontal segment
                    surf = sprites.get("body_h")
                else:
                    # Corner bend
                    dirs = {v1, v2}
                    if (0, -1) in dirs and (1, 0) in dirs:
                        surf = sprites.get("turn_ne")
                    elif (0, -1) in dirs and (-1, 0) in dirs:
                        surf = sprites.get("turn_nw")
                    elif (0, 1) in dirs and (1, 0) in dirs:
                        surf = sprites.get("turn_se")
                    elif (0, 1) in dirs and (-1, 0) in dirs:
                        surf = sprites.get("turn_sw")
                    else:
                        surf = sprites.get("body_h")

                if surf:
                    self.surface.blit(surf, rect)
                else:
                    pygame.draw.rect(self.surface, fallback_col, rect.inflate(-4, -4), border_radius=4)

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
