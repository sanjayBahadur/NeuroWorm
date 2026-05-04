"""
Procedural Vermiform Renderer for NeuroWorm Snake.

High-end organic dark-biology aesthetic (inspired by Rain World, Carrion, Osmos):
  - Procedural ribbon kinematics: Continuous articulated spine with smooth tapering
  - Bioluminescent translucent cuticle with glowing internal nerve cord
  - Luminous photoreceptor head eyes facing movement direction
  - Bioluminescent bacterial nutrient spore with radiant halo and particle glow
  - Ambient microscopic soil depth with drifting nutrient motes
"""

import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pygame

from game.layout import (
    GRID_SIZE, CELL, WALL, FIELD_W, FIELD_H, ARENA_W, ARENA_H,
    cell_to_px as layout_cell_to_px,
    cell_center_px as layout_cell_center_px,
)

# ─── COLOR PALETTES ───────────────────────────────────────────────────────────
SOIL_DEEP    = (14, 9, 6)       # Rich dark organic peat
SOIL_TILE    = (24, 16, 11)     # Soil substrate
BORDER_STONE = (40, 30, 24)     # Border wall substrate
BORDER_LIT   = (65, 48, 38)     # Edge highlight
TEXT_CREAM   = (240, 230, 210)
SCORE_GOLD   = (255, 215, 80)

# 10 Distinct Biological Palettes for the 10 Candidate Organisms
WORM_PALETTES = [
    {
        "name": "Champion 1", "badge": "★1",
        "cuticle": (255, 175, 195), "core": (255, 235, 245), "glow": (255, 200, 220),
        "border": (255, 215, 0), "color": (255, 180, 200)
    },
    {
        "name": "Champion 2", "badge": "★2",
        "cuticle": (160, 190, 255), "core": (230, 240, 255), "glow": (180, 210, 255),
        "border": (210, 225, 255), "color": (160, 190, 255)
    },
    {
        "name": "Champion 3", "badge": "★3",
        "cuticle": (255, 190, 130), "core": (255, 240, 215), "glow": (255, 210, 150),
        "border": (215, 140, 60), "color": (255, 190, 130)
    },
    {
        "name": "Candidate 4", "badge": "#4",
        "cuticle": (130, 235, 180), "core": (220, 255, 235), "glow": (160, 250, 200),
        "border": (90, 200, 140), "color": (130, 235, 180)
    },
    {
        "name": "Candidate 5", "badge": "#5",
        "cuticle": (255, 145, 175), "core": (255, 225, 235), "glow": (255, 170, 195),
        "border": (210, 100, 135), "color": (255, 145, 175)
    },
    {
        "name": "Candidate 6", "badge": "#6",
        "cuticle": (210, 160, 255), "core": (245, 230, 255), "glow": (225, 185, 255),
        "border": (165, 115, 225), "color": (210, 160, 255)
    },
    {
        "name": "Candidate 7", "badge": "#7",
        "cuticle": (130, 230, 240), "core": (225, 255, 255), "glow": (160, 245, 255),
        "border": (85, 190, 200), "color": (130, 230, 240)
    },
    {
        "name": "Candidate 8", "badge": "#8",
        "cuticle": (255, 235, 135), "core": (255, 255, 225), "glow": (255, 245, 165),
        "border": (215, 195, 80), "color": (255, 235, 135)
    },
    {
        "name": "Candidate 9", "badge": "#9",
        "cuticle": (205, 240, 130), "core": (245, 255, 220), "glow": (220, 250, 155),
        "border": (160, 205, 80), "color": (205, 240, 130)
    },
    {
        "name": "Candidate 10", "badge": "#10",
        "cuticle": (255, 170, 125), "core": (255, 235, 220), "glow": (255, 195, 150),
        "border": (215, 120, 80), "color": (255, 170, 125)
    },
]


class Renderer:
    """Renders the biological soil arena and procedural vermiform organisms."""

    def __init__(self, cell_size: int = CELL, grid_size: int = GRID_SIZE, wall_cells: int = WALL):
        if not pygame.font.get_init():
            pygame.font.init()
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.wall_cells = wall_cells
        self.field_w = cell_size * grid_size
        self.field_h = cell_size * grid_size
        self.W = (grid_size + 2 * wall_cells) * cell_size
        self.H = (grid_size + 2 * wall_cells) * cell_size

        self.surface = pygame.Surface((self.W, self.H))
        self.font = pygame.font.SysFont("monospace", 12, bold=True)
        self.badge_font = pygame.font.SysFont("monospace", 9, bold=True)
        self.frame_count = 0

        # Ambient floating soil spore particles inside playable field
        rng = np.random.default_rng(42)
        min_x = self.wall_cells * self.cell_size
        max_x = min_x + self.field_w
        min_y = self.wall_cells * self.cell_size
        max_y = min_y + self.field_h
        self.motes = [
            {
                "x": float(rng.uniform(min_x + 5, max_x - 5)),
                "y": float(rng.uniform(min_y + 5, max_y - 5)),
                "vx": float(rng.uniform(-0.15, 0.15)),
                "vy": float(rng.uniform(-0.15, 0.15)),
                "size": int(rng.integers(1, 3)),
                "alpha": int(rng.integers(25, 70)),
            }
            for _ in range(35)
        ]
        self.burst_particles = []
        self.floating_popups = []

    def cell_to_px(self, x: float, y: float) -> tuple[float, float]:
        """Return top-left pixel coordinates of grid cell (x, y) with outer wall band."""
        return (x + self.wall_cells) * self.cell_size, (y + self.wall_cells) * self.cell_size

    def cell_center_px(self, x: float, y: float) -> tuple[float, float]:
        """Return center pixel coordinates of grid cell (x, y) with outer wall band."""
        return (x + self.wall_cells + 0.5) * self.cell_size, (y + self.wall_cells + 0.5) * self.cell_size

    def render(self, obs: tuple, score: int, game_over: bool, gen_info: str = "") -> pygame.Surface:
        """Single-worm backward compatibility."""
        head, food, body, direction = obs
        dummy_worm = type("DummyWorm", (), {
            "obs": obs,
            "direction": direction,
            "alive": not game_over,
            "score": score,
            "fitness": float(score * 300),
            "id": 1,
        })()
        return self.render_multi_worms([dummy_worm], food, gen_info=gen_info, focused_idx=0, solo_mode=True)

    def render_multi_worms(
        self,
        worms: list,
        food: tuple,
        gen_info: str = "",
        focused_idx: int = 0,
        solo_mode: bool = False,
    ) -> pygame.Surface:
        """Render all 10 organisms simultaneously with organic procedural ribbons."""
        self.frame_count += 1
        self.surface.fill(SOIL_DEEP)

        self._draw_soil_arena()
        self._update_and_draw_motes()
        self._draw_nutrient_spore(food)

        # 1. Draw dead organisms first (faint background trails)
        if not solo_mode:
            for i, w in enumerate(worms):
                if not w.alive and i != focused_idx:
                    self._draw_dead_worm(w, i)

        # 2. Draw alive organisms (non-focused)
        for i, w in enumerate(worms):
            if solo_mode and i != focused_idx:
                continue
            if w.alive and i != focused_idx:
                self._draw_procedural_worm(w, i, is_focused=False)

        # 3. Draw focused organism on top
        if focused_idx < len(worms):
            fw = worms[focused_idx]
            if fw.alive:
                self._draw_procedural_worm(fw, focused_idx, is_focused=True)
            elif solo_mode or True:
                self._draw_dead_worm(fw, focused_idx)

        # 4. Nutrient Spore Burst Effects & Popups
        self._update_and_draw_effects()

        # 5. Arena HUD Bar
        alive_count = sum(1 for w in worms if w.alive)
        leader = max(worms, key=lambda w: (w.score, w.fitness))
        self._draw_arena_hud(alive_count, len(worms), leader, gen_info)

        return self.surface

    def add_food_burst(self, grid_pos: tuple, worm_color: tuple = (255, 215, 80)) -> None:
        """Spawn a juicy bioluminescent particle burst and floating score pop when an egg is eaten."""
        cx, cy = self.cell_center_px(*grid_pos)

        # 16 glowing particles expanding radially
        for i in range(16):
            angle = (i / 16.0) * 2.0 * math.pi + np.random.uniform(-0.2, 0.2)
            speed = float(np.random.uniform(1.8, 4.5))
            self.burst_particles.append({
                "x": float(cx),
                "y": float(cy),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "color": worm_color if i % 2 == 0 else (255, 220, 100),
                "radius": float(np.random.uniform(2.5, 4.5)),
                "alpha": 255.0,
                "decay": float(np.random.uniform(9.0, 15.0)),
            })

        # Floating popup
        self.floating_popups.append({
            "x": float(cx),
            "y": float(cy - 6),
            "text": "+1 🥚",
            "color": (255, 235, 110),
            "alpha": 255.0,
            "vy": -1.2,
            "life": 32,
        })

    def _update_and_draw_effects(self) -> None:
        """Update and draw all active particle bursts and floating text popups."""
        # 1. Spore Particles
        surviving_particles = []
        for p in self.burst_particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["alpha"] -= p["decay"]
            if p["alpha"] > 8:
                rad = max(1, int(p["radius"]))
                p_surf = pygame.Surface((rad * 2 + 2, rad * 2 + 2), pygame.SRCALPHA)
                col = (*p["color"][:3], int(p["alpha"]))
                pygame.draw.circle(p_surf, col, (rad + 1, rad + 1), rad)
                self.surface.blit(p_surf, (int(p["x"]) - rad - 1, int(p["y"]) - rad - 1))
                surviving_particles.append(p)
        self.burst_particles = surviving_particles

        # 2. Score Popups
        surviving_popups = []
        for pop in self.floating_popups:
            pop["y"] += pop["vy"]
            pop["life"] -= 1
            if pop["life"] > 0:
                alpha = int((pop["life"] / 32.0) * 255)
                txt_surf = self.badge_font.render(pop["text"], True, pop["color"])
                txt_surf.set_alpha(alpha)
                self.surface.blit(txt_surf, (int(pop["x"] - txt_surf.get_width() // 2), int(pop["y"])))
                surviving_popups.append(pop)
        self.floating_popups = surviving_popups

    def _draw_soil_arena(self) -> None:
        """Draw background soil substrate with outer border wall surrounding the playable grid."""
        cs = self.cell_size
        g = self.grid_size
        w_px = self.wall_cells * cs

        # 1. Inner Playable Soil Field (0..g-1)
        field_rect = pygame.Rect(w_px, w_px, g * cs, g * cs)
        pygame.draw.rect(self.surface, SOIL_TILE, field_rect)

        # 2. Subtle organic grid filaments inside the field
        top_y = w_px
        bot_y = w_px + g * cs
        left_x = w_px
        right_x = w_px + g * cs
        for i in range(1, g):
            px, _ = self.cell_to_px(i, 0)
            pygame.draw.line(self.surface, (32, 22, 16), (px, top_y), (px, bot_y), 1)
            _, py = self.cell_to_px(0, i)
            pygame.draw.line(self.surface, (32, 22, 16), (left_x, py), (right_x, py), 1)

        # 3. Outer border walls (strictly outside playable cells)
        # Top & Bottom outer bands
        pygame.draw.rect(self.surface, BORDER_STONE, (0, 0, self.W, w_px))
        pygame.draw.rect(self.surface, BORDER_STONE, (0, self.H - w_px, self.W, w_px))
        # Left & Right outer bands
        pygame.draw.rect(self.surface, BORDER_STONE, (0, 0, w_px, self.H))
        pygame.draw.rect(self.surface, BORDER_STONE, (self.W - w_px, 0, w_px, self.H))

        # 4. Border highlight line along the inner wall edge
        pygame.draw.rect(self.surface, BORDER_LIT, field_rect, 1)

    def _update_and_draw_motes(self) -> None:
        """Drift subtle microscopic nutrient particles across the inner playable field."""
        min_x = self.wall_cells * self.cell_size
        max_x = min_x + self.field_w
        min_y = self.wall_cells * self.cell_size
        max_y = min_y + self.field_h

        for m in self.motes:
            m["x"] += m["vx"]
            m["y"] += m["vy"]
            if m["x"] < min_x or m["x"] > max_x:
                m["vx"] *= -1
            if m["y"] < min_y or m["y"] > max_y:
                m["vy"] *= -1

            col = (200, 180, 140, m["alpha"])
            m_surf = pygame.Surface((m["size"] * 2, m["size"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(m_surf, col, (m["size"], m["size"]), m["size"])
            self.surface.blit(m_surf, (int(m["x"]), int(m["y"])))

    def _draw_nutrient_spore(self, food_pos: tuple) -> None:
        """Bioluminescent glowing bacterial egg colony."""
        cx, cy = self.cell_center_px(*food_pos)

        # Glowing pulsing aura
        pulse = 1.0 + 0.18 * math.sin(self.frame_count * 0.18)
        r_outer = int(22 * pulse)
        aura_surf = pygame.Surface((r_outer * 2, r_outer * 2), pygame.SRCALPHA)
        pygame.draw.circle(aura_surf, (255, 190, 60, 45), (r_outer, r_outer), r_outer)
        pygame.draw.circle(aura_surf, (255, 215, 100, 85), (r_outer, r_outer), int(r_outer * 0.65))
        self.surface.blit(aura_surf, (cx - r_outer, cy - r_outer))

        # Core bacterial spore nucleus
        pygame.draw.circle(self.surface, (255, 200, 70), (cx, cy), 8)
        pygame.draw.circle(self.surface, (255, 245, 200), (cx, cy), 5)
        pygame.draw.circle(self.surface, (255, 255, 255), (cx - 2, cy - 2), 2)

    def _draw_dead_worm(self, worm, w_idx: int) -> None:
        """Faint biological ghost trail of deceased organism."""
        body = worm.obs[2]
        if not body:
            return

        # Faint connected ghost trail
        pts = [self.cell_center_px(bx, by) for bx, by in body]
        for i in range(len(pts) - 1):
            pygame.draw.line(self.surface, (38, 25, 18), pts[i], pts[i + 1], 8)
        for pt in pts:
            pygame.draw.circle(self.surface, (38, 25, 18), pt, 4)

        # Red collision mark at death location
        hx, hy = pts[0]
        pygame.draw.circle(self.surface, (150, 35, 35), (hx, hy), 6)
        x_lbl = self.badge_font.render("x", True, (255, 210, 210))
        self.surface.blit(x_lbl, (hx - x_lbl.get_width() // 2, hy - x_lbl.get_height() // 2 - 1))

    def _draw_procedural_worm(self, worm, w_idx: int, is_focused: bool = False) -> None:
        """Renders an organic, smoothly tapering vermiform ribbon with bioluminescent nerve pulses."""
        body = worm.obs[2]
        direction = worm.direction
        if not body:
            return

        n = len(body)
        pal = WORM_PALETTES[w_idx % len(WORM_PALETTES)]
        cuticle_col = pal["cuticle"]
        core_col    = pal["core"]
        glow_col    = pal["glow"]

        # Center coordinates along the spine
        pts = [self.cell_center_px(bx, by) for bx, by in body]

        # ── 1. Continuous Tapered Ribbon Body ─────────────────────────
        # Draw from tail to head
        for i in range(n - 1, -1, -1):
            t = i / max(1, n - 1)
            # Head radius ~12px, tail radius ~4px
            radius = int(12.0 * (1.0 - t * 0.65))

            pt = pts[i]

            # Connect with previous point using smooth pill capsule
            if i > 0:
                prev_pt = pts[i - 1]
                prev_t = (i - 1) / max(1, n - 1)
                prev_rad = int(12.0 * (1.0 - prev_t * 0.65))
                mid_rad = (radius + prev_rad) // 2

                # Outer cuticle
                pygame.draw.line(self.surface, cuticle_col, prev_pt, pt, mid_rad * 2)
                pygame.draw.circle(self.surface, cuticle_col, pt, radius)

                # Inner nerve core
                core_rad = max(2, mid_rad - 3)
                pygame.draw.line(self.surface, core_col, prev_pt, pt, core_rad * 2)
            else:
                pygame.draw.circle(self.surface, cuticle_col, pt, radius)
                pygame.draw.circle(self.surface, core_col, pt, max(2, radius - 3))

        # ── 2. Bioluminescent Neural Pulse along Nerve Cord ───────────
        # Pulse waves down the body spine
        pulse_idx = int((self.frame_count * 0.4) % max(1, n))
        if pulse_idx < len(pts):
            p_pt = pts[pulse_idx]
            p_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, (*glow_col, 160), (10, 10), 9)
            pygame.draw.circle(p_surf, (255, 255, 255, 220), (10, 10), 4)
            self.surface.blit(p_surf, (p_pt[0] - 10, p_pt[1] - 10))

        # ── 3. Head Photoreceptor Eyes ────────────────────────────────
        hx, hy = pts[0]
        # Offset eye positions based on movement direction
        eye_offsets = {
            "RIGHT": ((5, -4), (5, 4)),
            "LEFT":  ((-5, -4), (-5, 4)),
            "UP":    ((-4, -5), (4, -5)),
            "DOWN":  ((-4, 5), (4, 5)),
        }.get(direction, ((4, -4), (4, 4)))

        for ox, oy in eye_offsets:
            eye_cx = hx + ox
            eye_cy = hy + oy
            # Dark iris
            pygame.draw.circle(self.surface, (25, 12, 18), (eye_cx, eye_cy), 3)
            # Luminous glint
            pygame.draw.circle(self.surface, (255, 255, 255), (eye_cx + (1 if ox > 0 else -1), eye_cy - 1), 1)

        # ── 4. Focused Target Halo ────────────────────────────────────
        if is_focused:
            halo_r = 18 + int(3.0 * math.sin(self.frame_count * 0.25))
            h_surf = pygame.Surface((halo_r * 2, halo_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(h_surf, (255, 220, 80, 110), (halo_r, halo_r), halo_r, 2)
            self.surface.blit(h_surf, (hx - halo_r, hy - halo_r))

        # ── 5. Sleek Identification Badge ─────────────────────────────
        badge_str = pal["badge"]
        b_txt = self.badge_font.render(badge_str, True, (14, 8, 5))
        bw = b_txt.get_width() + 6
        bh = b_txt.get_height() + 2
        bx = hx - bw // 2
        by = hy - 18

        b_rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(self.surface, pal["border"], b_rect, border_radius=3)
        self.surface.blit(b_txt, (bx + 3, by + 1))

    def _draw_arena_hud(self, alive_count: int, total_count: int, leader, gen_info: str) -> None:
        """Sleek semi-transparent HUD banner drawn strictly within the top outer wall band."""
        w_px = self.wall_cells * self.cell_size
        hud_bg = pygame.Surface((self.W, w_px), pygame.SRCALPHA)
        hud_bg.fill((12, 8, 5, 210))
        self.surface.blit(hud_bg, (0, 0))

        ty = max(2, (w_px - self.font.get_height()) // 2)

        # Left: Organism population count
        pop_col = (0, 230, 140) if alive_count > 3 else ((255, 200, 60) if alive_count > 0 else (255, 80, 80))
        pop_lbl = self.font.render(f"WORMS: {alive_count}/{total_count} ALIVE", True, pop_col)
        self.surface.blit(pop_lbl, (12, ty))

        # Center: Generation tag
        if gen_info:
            g_lbl = self.font.render(gen_info, True, TEXT_CREAM)
            self.surface.blit(g_lbl, (self.W // 2 - g_lbl.get_width() // 2, ty))

        # Right: Champion score
        champ_lbl = self.font.render(f"LEADER: #{leader.id} ({leader.score} EGGS)", True, SCORE_GOLD)
        self.surface.blit(champ_lbl, (self.W - champ_lbl.get_width() - 12, ty))
