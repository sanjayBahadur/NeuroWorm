"""
Sensory Radar & Biological Map for NeuroWorm Snake.

Replaces the abstract chart with a concrete, intuitive visual map of:
  1. The 20x20 grid with food diffusion plume (chemical field)
  2. The worm's head position and body
  3. The 3 active mechanosensory touch rays (whiskers):
     - Red ray: Front nose touch (AVM / danger ahead)
     - Yellow rays: Left & Right whiskers
  4. Real-time biological sensor gauges:
     - Nose Touch (AVM)
     - Amphid Odor Balance (ASEL vs ASER)
     - Command Status (AVB Forward vs AVA Reversal)
"""

import math
import numpy as np
import pygame as pg

RADAR_W = 252
RADAR_H = 220
GRID_BOX_SIZE = 190

# Colors
C_BG       = (18, 12, 8)
C_BORDER   = (65, 45, 28)
C_GRID_BG  = (28, 18, 12)
C_GRID_LINE= (42, 28, 18)
C_FOOD     = (255, 190, 60)
C_PLUME    = (255, 180, 40)
C_WORM_H   = (255, 175, 195)
C_WORM_B   = (210, 120, 145)
C_RAY_FRONT= (255, 60, 60)
C_RAY_SIDE = (255, 220, 60)
C_TEXT     = (220, 205, 175)
C_MUTED    = (140, 120, 95)
C_GREEN    = (0, 230, 140)
C_RED      = (255, 70, 70)


class SensoryRadar:
    """Renders a live mini-map showing what the worm's sensors see."""

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.surface = pg.Surface((RADAR_W, RADAR_H))
        self.font = pg.font.SysFont("monospace", 10)
        self.font_bold = pg.font.SysFont("monospace", 10, bold=True)
        self.cell_px = GRID_BOX_SIZE / grid_size

    def render(self, obs: tuple, telemetry: dict, direction: str) -> pg.Surface:
        self.surface.fill(C_BG)
        pg.draw.rect(self.surface, C_BORDER, (0, 0, RADAR_W, RADAR_H), 1)

        head, food, body, cur_dir = obs
        hx, hy = head
        fx, fy = food
        c_px = self.cell_px

        # ── 1. Mini-Grid Map ──────────────────────────────────────────
        map_x = 10
        map_y = 15
        map_rect = pg.Rect(map_x, map_y, GRID_BOX_SIZE, GRID_BOX_SIZE)
        pg.draw.rect(self.surface, C_GRID_BG, map_rect)
        pg.draw.rect(self.surface, C_BORDER, map_rect, 1)

        # Draw subtle grid lines
        for i in range(0, self.grid_size, 5):
            px = map_x + int(i * c_px)
            py = map_y + int(i * c_px)
            pg.draw.line(self.surface, C_GRID_LINE, (px, map_y), (px, map_y + GRID_BOX_SIZE))
            pg.draw.line(self.surface, C_GRID_LINE, (map_x, py), (map_x + GRID_BOX_SIZE, py))

        # ── Chemical Diffusion Plume around Food ──────────────────────
        food_cx = map_x + int(fx * c_px + c_px / 2)
        food_cy = map_y + int(fy * c_px + c_px / 2)

        # Concentric diffusion rings
        for r_rings in [24, 16, 9]:
            plume_surf = pg.Surface((r_rings * 2, r_rings * 2), pg.SRCALPHA)
            alpha = int(70 * (1.0 - r_rings / 30.0))
            pg.draw.circle(plume_surf, (255, 180, 40, alpha), (r_rings, r_rings), r_rings)
            self.surface.blit(plume_surf, (food_cx - r_rings, food_cy - r_rings))

        # Food pellet
        pg.draw.circle(self.surface, C_FOOD, (food_cx, food_cy), max(2, int(c_px / 2)))

        # ── Worm Body Segments ────────────────────────────────────────
        for i, (bx, by) in enumerate(body[1:]):
            bx_px = map_x + int(bx * c_px + 1)
            by_px = map_y + int(by * c_px + 1)
            pg.draw.rect(self.surface, C_WORM_B, (bx_px, by_px, int(c_px - 1), int(c_px - 1)), border_radius=2)

        # Worm Head
        head_cx = map_x + int(hx * c_px + c_px / 2)
        head_cy = map_y + int(hy * c_px + c_px / 2)
        pg.draw.circle(self.surface, C_WORM_H, (head_cx, head_cy), max(3, int(c_px / 2 + 1)))

        # ── Active Mechanosensory Whiskers (Rays) ─────────────────────
        front_hit = telemetry.get("front_hit", (hx, hy))
        left_hit  = telemetry.get("left_hit", (hx, hy))
        right_hit = telemetry.get("right_hit", (hx, hy))

        def to_map_coord(pos):
            x, y = pos
            return (
                map_x + int(np.clip(x * c_px + c_px / 2, 0, GRID_BOX_SIZE)),
                map_y + int(np.clip(y * c_px + c_px / 2, 0, GRID_BOX_SIZE))
            )

        f_end = to_map_coord(front_hit)
        l_end = to_map_coord(left_hit)
        r_end = to_map_coord(right_hit)

        # Draw Left & Right Whisker Rays (Yellow)
        pg.draw.line(self.surface, C_RAY_SIDE, (head_cx, head_cy), l_end, 1)
        pg.draw.line(self.surface, C_RAY_SIDE, (head_cx, head_cy), r_end, 1)

        # Draw Front Nose Touch Ray (Red)
        pg.draw.line(self.surface, C_RAY_FRONT, (head_cx, head_cy), f_end, 2)
        # Impact point dot
        pg.draw.circle(self.surface, C_RAY_FRONT, f_end, 3)

        # ── 2. Biological Sensory Gauges (Right column) ───────────────
        gx = map_x + GRID_BOX_SIZE + 6
        gy = map_y

        title_txt = self.font_bold.render("SENSORY RADAR", True, C_TEXT)
        self.surface.blit(title_txt, (10, 3))

        # Nose Touch (AVM) Gauge:
        f_dist = telemetry.get("front_dist", 20)
        pg.draw.rect(self.surface, (35, 20, 15), (gx, gy, 38, 55), border_radius=3)
        pg.draw.rect(self.surface, C_BORDER, (gx, gy, 38, 55), 1, border_radius=3)
        avm_label = self.font.render("NOSE", True, C_MUTED)
        self.surface.blit(avm_label, (gx + 6, gy + 3))

        # Height of danger fill (closer = fuller)
        danger_ratio = 1.0 / max(1, min(10, f_dist))
        bar_h = int(32 * danger_ratio)
        bar_col = C_RED if f_dist <= 2 else (255, 180, 40)
        pg.draw.rect(self.surface, bar_col, (gx + 10, gy + 16 + (32 - bar_h), 18, bar_h), border_radius=2)
        dist_lbl = self.font_bold.render(f"{f_dist}", True, C_TEXT)
        self.surface.blit(dist_lbl, (gx + 12, gy + 42))

        # Chemical Balance (ASEL vs ASER):
        gy2 = gy + 62
        pg.draw.rect(self.surface, (35, 20, 15), (gx, gy2, 38, 62), border_radius=3)
        pg.draw.rect(self.surface, C_BORDER, (gx, gy2, 38, 62), 1, border_radius=3)
        odor_lbl = self.font.render("ODOR", True, C_MUTED)
        self.surface.blit(odor_lbl, (gx + 6, gy2 + 3))

        delta_c = telemetry.get("delta_c", 0.0)
        l_col = C_GREEN if delta_c > 0.05 else C_MUTED
        r_col = C_GREEN if delta_c < -0.05 else C_MUTED
        l_txt = self.font_bold.render("L", True, l_col)
        r_txt = self.font_bold.render("R", True, r_col)
        self.surface.blit(l_txt, (gx + 8, gy2 + 20))
        self.surface.blit(r_txt, (gx + 24, gy2 + 20))

        # Balance indicator
        bal_x = gx + 19 + int(np.clip(delta_c * 5.0, -12, 12))
        pg.draw.circle(self.surface, C_FOOD, (bal_x, gy2 + 45), 4)
        pg.draw.line(self.surface, C_BORDER, (gx + 7, gy2 + 45), (gx + 31, gy2 + 45), 1)

        # Command Indicator (AVB vs AVA):
        gy3 = gy2 + 69
        pg.draw.rect(self.surface, (35, 20, 15), (gx, gy3, 38, 48), border_radius=3)
        pg.draw.rect(self.surface, C_BORDER, (gx, gy3, 38, 48), 1, border_radius=3)
        cmd_lbl = self.font.render("CMD", True, C_MUTED)
        self.surface.blit(cmd_lbl, (gx + 9, gy3 + 3))

        is_reversing = (f_dist == 1)
        cmd_val = self.font_bold.render("AVA" if is_reversing else "AVB", True, C_RED if is_reversing else C_GREEN)
        self.surface.blit(cmd_val, (gx + 9, gy3 + 18))
        mode_val = self.font.render("TURN" if is_reversing else "FWD", True, C_TEXT)
        self.surface.blit(mode_val, (gx + 8, gy3 + 32))

        return self.surface
