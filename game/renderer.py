"""
Procedural Vermiform Renderer for NeuroWorm Snake.

High-end organic dark-biology aesthetic (inspired by Rain World, Carrion, Osmos):
  - Chaikin corner-cut continuous spine with smooth tapering (no 90-degree pipe corners)
  - Focused Hero Worm (full opacity, halo, whisker sensor rays, hunger arc, traveling pulse)
  - Ghost swarm (35% alpha, thinner profile, minimalist numeric badge)
  - High-contrast bacterial egg spore with 3 faint concentric scent diffusion rings
  - Dynamic death bursts, floating cause-of-death labels, and 1.5s fading trails
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
from game.theme import get_font

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
        "name": "Champion 1", "badge": "#1",
        "cuticle": (255, 175, 195), "core": (255, 235, 245), "glow": (255, 200, 220),
        "border": (255, 215, 0), "color": (255, 180, 200)
    },
    {
        "name": "Champion 2", "badge": "#2",
        "cuticle": (160, 190, 255), "core": (230, 240, 255), "glow": (180, 210, 255),
        "border": (210, 225, 255), "color": (160, 190, 255)
    },
    {
        "name": "Champion 3", "badge": "#3",
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


def chaikin_smooth(points: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
    """Chaikin's corner cutting algorithm for organic curve smoothing."""
    if len(points) < 3:
        return list(points)
    current = list(points)
    for _ in range(iterations):
        smoothed = [current[0]]
        for i in range(len(current) - 1):
            p0 = current[i]
            p1 = current[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            smoothed.extend([q, r])
        smoothed.append(current[-1])
        current = smoothed
    return current


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
        self.font = get_font("body", 12)
        self.badge_font = get_font("button", 12)
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
        self.death_events: dict[int, dict] = {}

    def cell_to_px(self, x: float, y: float) -> tuple[float, float]:
        """Return top-left pixel coordinates of grid cell (x, y) with outer wall band."""
        return (x + self.wall_cells) * self.cell_size, (y + self.wall_cells) * self.cell_size

    def cell_center_px(self, x: float, y: float) -> tuple[float, float]:
        """Return center pixel coordinates of grid cell (x, y) with outer wall band."""
        return (x + self.wall_cells + 0.5) * self.cell_size, (y + self.wall_cells + 0.5) * self.cell_size

    def render_multi_worms(
        self,
        worms: list,
        food: tuple,
        gen_info: str = "",
        focused_idx: int = 0,
        solo_mode: bool = False,
        interp_alpha: float = 1.0,
        focus_mode: str = "follow",
        dt: float = 0.016,
    ) -> pygame.Surface:
        """Render all candidate organisms simultaneously with hero focus and smooth motion."""
        self.frame_count += 1
        self.surface.fill(SOIL_DEEP)

        # 1. Background arena substrate & ambient motes
        self._draw_soil_arena()
        self._update_and_draw_motes()

        # 2. Nutrient egg with scent diffusion rings (most luminous element)
        self._draw_nutrient_spore(food)

        # 3. Dynamic death animations & fading trails
        self._update_and_draw_death_events(worms, dt)

        # 4. Ghost organisms (non-focused worms at ~35% alpha, thinner profile)
        if not solo_mode and len(worms) > 1:
            ghost_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            for i, w in enumerate(worms):
                if w.alive and i != focused_idx:
                    self._draw_procedural_worm(
                        w, i, is_focused=False, interp_alpha=interp_alpha, target_surf=ghost_surf
                    )
            ghost_surf.set_alpha(int(255 * 0.35))
            self.surface.blit(ghost_surf, (0, 0))

        # 5. Focused Hero Organism on top (full opacity, halo, rays, hunger arc)
        if focused_idx < len(worms):
            fw = worms[focused_idx]
            if fw.alive:
                self._draw_procedural_worm(
                    fw, focused_idx, is_focused=True, interp_alpha=interp_alpha, target_surf=self.surface
                )

        # 6. Spore burst particles & floating popups
        self._update_and_draw_effects()

        # 7. Arena HUD banner in outer wall band
        alive_count = sum(1 for w in worms if w.alive)
        leader = max(worms, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness))
        self._draw_arena_hud(alive_count, len(worms), leader, gen_info, focus_mode, focused_idx)

        return self.surface

    def add_food_burst(self, grid_pos: tuple, worm_color: tuple = (255, 215, 80)) -> None:
        """Spawn a bioluminescent particle burst and floating score pop when an egg is eaten."""
        cx, cy = self.cell_center_px(*grid_pos)

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

        self.floating_popups.append({
            "x": float(cx),
            "y": float(cy - 6),
            "text": "+1",
            "color": (255, 235, 110),
            "alpha": 255.0,
            "vy": -1.2,
            "life": 32,
        })

    def _update_and_draw_effects(self) -> None:
        """Update and draw active particle bursts and floating popups."""
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
        """Draw soil substrate with outer border wall surrounding the playable grid."""
        cs = self.cell_size
        g = self.grid_size
        w_px = self.wall_cells * cs

        # Inner Playable Soil Field (0..g-1)
        field_rect = pygame.Rect(w_px, w_px, g * cs, g * cs)
        pygame.draw.rect(self.surface, SOIL_TILE, field_rect)

        # Subtle organic grid filaments inside the field
        top_y = w_px
        bot_y = w_px + g * cs
        left_x = w_px
        right_x = w_px + g * cs
        for i in range(1, g):
            px, _ = self.cell_to_px(i, 0)
            pygame.draw.line(self.surface, (32, 22, 16), (px, top_y), (px, bot_y), 1)
            _, py = self.cell_to_px(0, i)
            pygame.draw.line(self.surface, (32, 22, 16), (left_x, py), (right_x, py), 1)

        # Outer border walls (strictly outside playable cells)
        pygame.draw.rect(self.surface, BORDER_STONE, (0, 0, self.W, w_px))
        pygame.draw.rect(self.surface, BORDER_STONE, (0, self.H - w_px, self.W, w_px))
        pygame.draw.rect(self.surface, BORDER_STONE, (0, 0, w_px, self.H))
        pygame.draw.rect(self.surface, BORDER_STONE, (self.W - w_px, 0, w_px, self.H))

        # Inner wall border highlight line
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
        """Bioluminescent glowing bacterial egg with scent diffusion rings."""
        cx, cy = self.cell_center_px(*food_pos)

        # 1. Three faint concentric scent diffusion rings (food smell that worms follow)
        scent_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        scent_rings = [(28, 38), (52, 24), (76, 14)]
        for r, alpha in scent_rings:
            pygame.draw.circle(scent_surf, (255, 215, 75, alpha), (int(cx), int(cy)), r, 1)
        self.surface.blit(scent_surf, (0, 0))

        # 2. Pulsing radiant aura & ring
        pulse = 1.0 + 0.18 * math.sin(self.frame_count * 0.18)
        r_outer = int(22 * pulse)
        aura_surf = pygame.Surface((r_outer * 2 + 4, r_outer * 2 + 4), pygame.SRCALPHA)
        center = (r_outer + 2, r_outer + 2)
        pygame.draw.circle(aura_surf, (255, 190, 60, 50), center, r_outer)
        pygame.draw.circle(aura_surf, (255, 220, 80, 110), center, int(15 * pulse), 2)
        self.surface.blit(aura_surf, (int(cx - center[0]), int(cy - center[1])))

        # 3. High-contrast core bacterial spore nucleus
        pygame.draw.circle(self.surface, (255, 195, 50), (int(cx), int(cy)), 10)
        pygame.draw.circle(self.surface, (255, 245, 190), (int(cx), int(cy)), 6)
        pygame.draw.circle(self.surface, (255, 255, 255), (int(cx - 2), int(cy - 2)), 2)

    def _update_and_draw_death_events(self, worms: list, dt: float) -> None:
        """Play ~0.6s death burst, floating cause label, and fading ghost trail (1.5s)."""
        for w in worms:
            w_id = getattr(w, "id", None)
            if w_id is None:
                continue
            if not w.alive:
                if w_id not in self.death_events:
                    hx, hy = w.obs[0]
                    cause = getattr(w, "death_cause", None) or "died"
                    cause_label = {
                        "wall": "hit the wall",
                        "self": "bit itself",
                        "starved": "starved",
                        "timeout": "time up",
                    }.get(cause, "died")
                    cx, cy = self.cell_center_px(hx, hy)
                    pal = WORM_PALETTES[(w_id - 1) % len(WORM_PALETTES)]
                    self.death_events[w_id] = {
                        "pos": (cx, cy),
                        "cause": cause_label,
                        "time": 0.0,
                        "pts": [self.cell_center_px(bx, by) for bx, by in w.obs[2]],
                        "color": pal["cuticle"],
                    }
            else:
                self.death_events.pop(w_id, None)

        active_events = {}
        death_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for w_id, ev in self.death_events.items():
            ev["time"] += dt
            t = ev["time"]
            if t >= 1.5:
                continue
            active_events[w_id] = ev

            # Fading body trail (1.5s)
            body_alpha = int(75.0 * max(0.0, 1.0 - t / 1.5))
            if body_alpha > 0 and len(ev["pts"]) > 1:
                smoothed = chaikin_smooth(ev["pts"], iterations=2)
                for i in range(len(smoothed) - 1):
                    p0 = (int(smoothed[i][0]), int(smoothed[i][1]))
                    p1 = (int(smoothed[i + 1][0]), int(smoothed[i + 1][1]))
                    pygame.draw.line(death_surf, (150, 40, 40, body_alpha), p0, p1, 5)
                for pt in smoothed:
                    pygame.draw.circle(death_surf, (150, 40, 40, body_alpha), (int(pt[0]), int(pt[1])), 3)

            # Red burst at head for first 0.6s
            hx, hy = ev["pos"]
            if t < 0.6:
                burst_frac = t / 0.6
                burst_rad = int(5 + burst_frac * 22)
                burst_alpha = int(240 * (1.0 - burst_frac))
                pygame.draw.circle(death_surf, (255, 50, 50, burst_alpha), (int(hx), int(hy)), burst_rad, 2)
                pygame.draw.circle(death_surf, (255, 90, 90, burst_alpha // 2), (int(hx), int(hy)), max(1, burst_rad // 2))

            # Floating cause label (up to 1.2s)
            if t < 1.2:
                lbl_frac = t / 1.2
                lbl_alpha = int(255 * (1.0 - lbl_frac))
                lbl_y = hy - 14 - int(t * 14)
                txt_surf = self.badge_font.render(ev["cause"], True, (255, 140, 140))
                txt_surf.set_alpha(lbl_alpha)
                min_px = self.wall_cells * self.cell_size
                max_px = (self.wall_cells + self.grid_size) * self.cell_size
                lx = int(min(max(min_px + 2, hx - txt_surf.get_width() // 2), max_px - txt_surf.get_width() - 2))
                death_surf.blit(txt_surf, (lx, int(lbl_y)))

        self.death_events = active_events
        self.surface.blit(death_surf, (0, 0))

    def _draw_whisker_rays(self, hero, hx: float, hy: float) -> None:
        """Draw 3 sensor rays (front, left, right) from hero head to obstacles."""
        if not hasattr(hero, "encoder") or not hasattr(hero.encoder, "telemetry"):
            return
        telemetry = hero.encoder.telemetry
        g = self.grid_size
        min_px = self.wall_cells * self.cell_size
        max_px = (self.wall_cells + g) * self.cell_size

        rays = [
            ("front", telemetry.get("front_hit", (0, 0)), telemetry.get("front_dist", 20)),
            ("left", telemetry.get("left_hit", (0, 0)), telemetry.get("left_dist", 20)),
            ("right", telemetry.get("right_hit", (0, 0)), telemetry.get("right_dist", 20)),
        ]

        ray_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for name, hit_pt, dist in rays:
            nx, ny = hit_pt
            is_wall = (nx < 0 or nx >= g or ny < 0 or ny >= g)
            if nx < 0:
                ex = float(min_px)
                ey = float(min(max(min_px, (ny + self.wall_cells + 0.5) * self.cell_size), max_px))
            elif nx >= g:
                ex = float(max_px)
                ey = float(min(max(min_px, (ny + self.wall_cells + 0.5) * self.cell_size), max_px))
            elif ny < 0:
                ex = float(min(max(min_px, (nx + self.wall_cells + 0.5) * self.cell_size), max_px))
                ey = float(min_px)
            elif ny >= g:
                ex = float(min(max(min_px, (nx + self.wall_cells + 0.5) * self.cell_size), max_px))
                ey = float(max_px)
            else:
                ex, ey = self.cell_center_px(nx, ny)

            if dist <= 2:
                col = (255, 75, 75, 210)
                lbl_col = (255, 120, 120)
            elif dist <= 4:
                col = (255, 185, 55, 170)
                lbl_col = (255, 205, 100)
            else:
                col = (100, 195, 230, 95)
                lbl_col = (140, 210, 240)

            pygame.draw.line(ray_surf, col, (int(hx), int(hy)), (int(ex), int(ey)), 1)
            pygame.draw.circle(ray_surf, col, (int(ex), int(ey)), 3)

            lbl_type = "wall" if is_wall else "self"
            lbl_txt = f"{lbl_type} {dist}"
            txt_s = self.badge_font.render(lbl_txt, True, lbl_col)
            lx = min(max(min_px + 2, ex - txt_s.get_width() // 2), max_px - txt_s.get_width() - 2)
            ly = min(max(min_px + 2, ey - txt_s.get_height() // 2), max_px - txt_s.get_height() - 2)
            ray_surf.blit(txt_s, (int(lx), int(ly)))

        self.surface.blit(ray_surf, (0, 0))

    def _draw_hunger_arc(self, hero, hx: float, hy: float) -> None:
        """Thin arc around hero head shrinking as starvation limit nears (red at <= 20%)."""
        starve_limit = getattr(hero, "starve_limit", 160)
        steps_since_food = getattr(hero, "steps_since_food", 0)
        rem = max(0.0, 1.0 - steps_since_food / max(1, starve_limit))

        if rem <= 0.20:
            arc_col = (255, 65, 65)
        elif rem <= 0.45:
            arc_col = (255, 175, 50)
        else:
            arc_col = (65, 230, 140)

        r = 16
        rect = pygame.Rect(int(hx - r), int(hy - r), r * 2, r * 2)
        start_angle = -math.pi / 2.0
        end_angle = start_angle + rem * 2.0 * math.pi
        if end_angle > start_angle + 0.05:
            pygame.draw.arc(self.surface, arc_col, rect, start_angle, end_angle, 2)

    def _draw_procedural_worm(
        self,
        worm,
        w_idx: int,
        is_focused: bool = False,
        interp_alpha: float = 1.0,
        target_surf: Optional[pygame.Surface] = None,
    ) -> None:
        """Renders an organic smoothly tapering vermiform organism with Chaikin corner smoothing."""
        if target_surf is None:
            target_surf = self.surface

        body = worm.obs[2]
        direction = worm.direction
        if not body:
            return

        n = len(body)
        pal = WORM_PALETTES[w_idx % len(WORM_PALETTES)]
        cuticle_col = pal["cuticle"]
        core_col    = pal["core"]
        glow_col    = pal["glow"]

        # 1. Sub-pixel interpolation
        prev_body = getattr(worm, "prev_body", None) or body
        raw_pts = []
        for i in range(n):
            bx, by = body[i]
            if i < len(prev_body):
                px, py = prev_body[i]
                if abs(bx - px) <= 1 and abs(by - py) <= 1:
                    ix = px + (bx - px) * interp_alpha
                    iy = py + (by - py) * interp_alpha
                else:
                    ix, iy = bx, by
            else:
                ix, iy = bx, by
            raw_pts.append(self.cell_center_px(ix, iy))

        # 2. Organic Chaikin corner-cut smoothing
        smooth_pts = chaikin_smooth(raw_pts, iterations=2)
        num_smooth = len(smooth_pts)

        # 3. Tapered Ribbon Body (hero is thicker with highlight; ghost is thinner)
        max_rad = 12.0 if is_focused else 7.0
        min_rad = 4.0 if is_focused else 2.5

        for i in range(num_smooth - 1, -1, -1):
            t = i / max(1, num_smooth - 1)
            radius = int(max_rad * (1.0 - t) + min_rad * t)
            pt = (int(smooth_pts[i][0]), int(smooth_pts[i][1]))

            if i > 0:
                prev_t = (i - 1) / max(1, num_smooth - 1)
                prev_rad = int(max_rad * (1.0 - prev_t) + min_rad * prev_t)
                mid_rad = max(2, (radius + prev_rad) // 2)
                prev_pt = (int(smooth_pts[i - 1][0]), int(smooth_pts[i - 1][1]))

                # Outer cuticle
                pygame.draw.line(target_surf, cuticle_col, prev_pt, pt, mid_rad * 2)
                pygame.draw.circle(target_surf, cuticle_col, pt, radius)

                # Inner core
                core_rad = max(1, mid_rad - (3 if is_focused else 2))
                pygame.draw.line(target_surf, core_col, prev_pt, pt, core_rad * 2)

                # Subtle highlight line for hero
                if is_focused and mid_rad > 4:
                    pygame.draw.line(target_surf, (255, 255, 255), prev_pt, pt, 1)
            else:
                pygame.draw.circle(target_surf, cuticle_col, pt, radius)
                pygame.draw.circle(target_surf, core_col, pt, max(1, radius - (3 if is_focused else 2)))

        hx, hy = smooth_pts[0]

        # 4. Ghost organism minimal numeric label
        if not is_focused:
            ghost_badge = f"#{w_idx + 1}"
            g_lbl = self.badge_font.render(ghost_badge, True, (210, 195, 180))
            target_surf.blit(g_lbl, (int(hx - g_lbl.get_width() // 2), int(hy - 14)))
            return

        # ─── HERO-SPECIFIC OVERLAYS ──────────────────────────────────────────
        # Traveling bioluminescent nerve pulse
        pulse_idx = int((self.frame_count * 0.8) % num_smooth)
        p_pt = smooth_pts[pulse_idx]
        p_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (*glow_col, 170), (10, 10), 9)
        pygame.draw.circle(p_surf, (255, 255, 255, 230), (10, 10), 4)
        target_surf.blit(p_surf, (int(p_pt[0] - 10), int(p_pt[1] - 10)))

        # Head Photoreceptor Eyes
        eye_offsets = {
            "RIGHT": ((5, -4), (5, 4)),
            "LEFT":  ((-5, -4), (-5, 4)),
            "UP":    ((-4, -5), (4, -5)),
            "DOWN":  ((-4, 5), (4, 5)),
        }.get(direction, ((4, -4), (4, 4)))

        for ox, oy in eye_offsets:
            eye_cx = int(hx + ox)
            eye_cy = int(hy + oy)
            pygame.draw.circle(target_surf, (25, 12, 18), (eye_cx, eye_cy), 3)
            pygame.draw.circle(target_surf, (255, 255, 255), (eye_cx + (1 if ox > 0 else -1), eye_cy - 1), 1)

        # Focused Halo
        halo_r = 18 + int(3.0 * math.sin(self.frame_count * 0.25))
        h_surf = pygame.Surface((halo_r * 2 + 2, halo_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(h_surf, (255, 220, 80, 120), (halo_r + 1, halo_r + 1), halo_r, 2)
        target_surf.blit(h_surf, (int(hx - halo_r - 1), int(hy - halo_r - 1)))

        # Hunger Arc
        self._draw_hunger_arc(worm, hx, hy)

        # Whisker Sensor Rays
        self._draw_whisker_rays(worm, hx, hy)

        # Hero Badge
        badge_str = f"#{w_idx + 1} HERO"
        b_txt = self.badge_font.render(badge_str, True, (14, 8, 5))
        bw = b_txt.get_width() + 8
        bh = b_txt.get_height() + 2
        bx = int(hx - bw // 2)
        by = int(hy - 24)

        b_rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(target_surf, pal["border"], b_rect, border_radius=3)
        target_surf.blit(b_txt, (bx + 4, by + 1))

    def _draw_arena_hud(
        self,
        alive_count: int,
        total_count: int,
        leader,
        gen_info: str,
        focus_mode: str = "follow",
        focused_idx: int = 0,
    ) -> None:
        """Sleek semi-transparent HUD banner drawn strictly within the top outer wall band."""
        w_px = self.wall_cells * self.cell_size
        hud_bg = pygame.Surface((self.W, w_px), pygame.SRCALPHA)
        hud_bg.fill((12, 8, 5, 225))
        self.surface.blit(hud_bg, (0, 0))

        ty = max(2, (w_px - self.font.get_height()) // 2)

        # Left: Organism population count
        pop_col = (0, 230, 140) if alive_count > 3 else ((255, 200, 60) if alive_count > 0 else (255, 80, 80))
        pop_lbl = self.font.render(f"WORMS: {alive_count}/{total_count} ALIVE", True, pop_col)
        self.surface.blit(pop_lbl, (12, ty))

        # Center: Generation tag + focus mode indicator
        mode_tag = "[FOLLOW LEADER]" if focus_mode == "follow" else f"[MANUAL #{focused_idx + 1}]"
        mid_str = f"{gen_info}  {mode_tag}".strip()
        g_lbl = self.font.render(mid_str, True, TEXT_CREAM)
        self.surface.blit(g_lbl, (self.W // 2 - g_lbl.get_width() // 2, ty))

        # Right: Champion score
        leader_id = getattr(leader, "id", 1)
        leader_score = getattr(leader, "score", 0)
        champ_lbl = self.font.render(f"LEADER: #{leader_id} ({leader_score} EGGS)", True, SCORE_GOLD)
        self.surface.blit(champ_lbl, (self.W - champ_lbl.get_width() - 12, ty))
