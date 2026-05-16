"""
High-School-Friendly Brain Activity & Synaptic Evolution Inspector.

Replaces confusing raw matrices with an intuitive, beautifully formatted
application panel that anyone with minimum high-school knowledge can read instantly:
  1. Brain Flow View (Live Thinking):
     - Stage 1: Senses (Food Odor L/R, Nose Touch Warning, Side Whiskers)
     - Stage 2: Decisions (Forward Crawl Drive vs Emergency Reversal Reflex)
     - Stage 3: Muscles (Steer Left vs Steer Right)
     - Plain-English Thought Balloon: e.g. "Smells food on Left -> Steering toward egg!"
  2. Synapses View (Genetic Evolution):
     - Human-readable cards showing what evolution just strengthened/weakened
       (e.g., "Nose Touch -> Reversal Reflex: +68% stronger across generations")
  3. Live 10-Worm Population Roster with click-to-inspect
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pygame as pg
from simulation.synaptic_circuit import SYNAPSE_DEFINITIONS
from game.layout import PANEL_W, VIRTUAL_H
from game.theme import get_font

PANEL_H = VIRTUAL_H

# Colors
C_PANEL_BG  = (18, 12, 8)
C_BOX_BG    = (28, 18, 12)
C_BOX_BORDER= (60, 42, 28)
C_HEADER    = (255, 215, 90)
C_TEXT      = (230, 215, 190)
C_MUTED     = (150, 130, 105)
C_GREEN     = (0, 230, 140)
C_RED       = (255, 75, 75)
C_AMBER     = (255, 190, 50)
C_CYAN      = (80, 210, 240)
C_BAR_BG    = (45, 30, 20)


class BrainInspector:
    """Renders the friendly, educational neuroscience laboratory panel."""

    def __init__(self, width: int = PANEL_W, height: int = PANEL_H):
        if not pg.font.get_init():
            pg.font.init()
        self.W = width
        self.H = height
        self.surface = pg.Surface((self.W, self.H))
        self.font_title = get_font("title", 14)
        self.font_bold  = get_font("card_title", 13)
        self.font_body  = get_font("body", 12)
        self.font_quote = get_font("small", 12)

        self.tab_flow_rect = pg.Rect(12, 38, (self.W - 32) // 2, 24)
        self.tab_syn_rect  = pg.Rect(16 + (self.W - 32) // 2, 38, (self.W - 32) // 2, 24)

    def render(
        self,
        worm,
        all_worms: list,
        generation: int,
        view_mode: str = "flow",   # "flow" or "synapses"
        mouse_pos: tuple = (0, 0),
        solo_mode: bool = False,
    ) -> pg.Surface:
        self.surface.fill(C_PANEL_BG)
        pg.draw.line(self.surface, C_BOX_BORDER, (0, 0), (0, self.H), 2)

        # ── 1. Top Application Header ─────────────────────────────────
        app_title = self.font_title.render("NEUROWORM LAB", True, C_HEADER)
        self.surface.blit(app_title, (14, 10))

        champ_tag = "CHAMPION 1" if worm.id == 1 else ("CHAMPION 2" if worm.id == 2 else ("CHAMPION 3" if worm.id == 3 else f"WORM #{worm.id}"))
        sub_txt = self.font_body.render(f"Inspecting: {champ_tag}", True, getattr(worm, "color", C_TEXT))
        self.surface.blit(sub_txt, (14, 25))

        # ── 2. Navigation Tabs (Brain Flow vs Synapses) ────────────────
        self._draw_tabs(view_mode, mouse_pos)

        # ── 3. Main View Area (y: 66 to 258) ──────────────────────────
        if view_mode == "flow":
            self._draw_brain_flow_view(worm)
        else:
            self._draw_synapses_view(worm)

        # ── 4. 10-Worm Swarm Roster (y: 264 to 452) ───────────────────
        self._draw_roster(all_worms, focused_id=worm.id, mouse_pos=mouse_pos, generation=generation)

        return self.surface

    def _draw_tabs(self, active_mode: str, mouse_pos: tuple) -> None:
        # Tab 1: Brain Flow
        is_hover1 = self.tab_flow_rect.collidepoint(mouse_pos)
        is_active1 = (active_mode == "flow")
        col1 = (55, 38, 22) if is_active1 else ((40, 26, 16) if is_hover1 else (26, 17, 11))
        pg.draw.rect(self.surface, col1, self.tab_flow_rect, border_radius=4)
        pg.draw.rect(self.surface, C_HEADER if is_active1 else C_BOX_BORDER, self.tab_flow_rect, 1, border_radius=4)
        t1 = self.font_bold.render("LIVE THOUGHTS", True, C_HEADER if is_active1 else C_MUTED)
        self.surface.blit(t1, (self.tab_flow_rect.centerx - t1.get_width() // 2, self.tab_flow_rect.centery - t1.get_height() // 2))

        # Tab 2: Synaptic Evolution
        is_hover2 = self.tab_syn_rect.collidepoint(mouse_pos)
        is_active2 = (active_mode == "synapses")
        col2 = (55, 38, 22) if is_active2 else ((40, 26, 16) if is_hover2 else (26, 17, 11))
        pg.draw.rect(self.surface, col2, self.tab_syn_rect, border_radius=4)
        pg.draw.rect(self.surface, C_HEADER if is_active2 else C_BOX_BORDER, self.tab_syn_rect, 1, border_radius=4)
        t2 = self.font_bold.render("SYNAPSE LEARNING", True, C_HEADER if is_active2 else C_MUTED)
        self.surface.blit(t2, (self.tab_syn_rect.centerx - t2.get_width() // 2, self.tab_syn_rect.centery - t2.get_height() // 2))

    def _draw_brain_flow_view(self, worm) -> None:
        """High-School-Friendly 3-stage live decision stream."""
        box = pg.Rect(12, 66, self.W - 24, 192)
        pg.draw.rect(self.surface, C_BOX_BG, box, border_radius=5)
        pg.draw.rect(self.surface, C_BOX_BORDER, box, 1, border_radius=5)

        s_tel = getattr(worm.encoder, "telemetry", {})
        m_tel = getattr(worm.decoder, "telemetry", {})

        food_l = s_tel.get("sense_food_l", 0)
        food_r = s_tel.get("sense_food_r", 0)
        nose   = s_tel.get("sense_nose", 0)
        f_dist = s_tel.get("front_dist", 20)

        fwd_dec = m_tel.get("decision_fwd", 85)
        rev_dec = m_tel.get("decision_rev", 15)
        turn_l  = m_tel.get("muscle_steer_l", 50)
        turn_r  = m_tel.get("muscle_steer_r", 50)
        thought = m_tel.get("thought_summary", "Crawling forward -> Seeking nutrients")

        y = 72
        # Stage 1: Senses
        s_title = self.font_bold.render("1. WHAT SENSES FEEL:", True, C_HEADER)
        self.surface.blit(s_title, (18, y))
        y += 15

        # Food Odor split bar
        odor_lbl = self.font_body.render("Food Odor:", True, C_TEXT)
        self.surface.blit(odor_lbl, (20, y))
        self._draw_split_odor_bar(110, y + 2, 175, 10, food_l, food_r)
        y += 16

        # Nose Whiskers (Danger Warning)
        nose_lbl = self.font_body.render("Nose Touch:", True, C_TEXT)
        self.surface.blit(nose_lbl, (20, y))
        if f_dist <= 2:
            w_text = f"OBSTACLE {f_dist} STEP!"
            w_col = C_RED
        elif f_dist <= 4:
            w_text = f"Wall Ahead ({f_dist})"
            w_col = C_AMBER
        else:
            w_text = "Clear Ahead"
            w_col = C_GREEN
        self._draw_progress_bar(110, y + 2, 110, 10, nose, color=C_RED if f_dist <= 2 else C_AMBER)
        w_surf = self.font_bold.render(w_text, True, w_col)
        self.surface.blit(w_surf, (226, y - 1))
        y += 18

        # Stage 2: Decisions
        d_title = self.font_bold.render("2. WHAT BRAIN DECIDES:", True, C_HEADER)
        self.surface.blit(d_title, (18, y))
        y += 15

        self.surface.blit(self.font_body.render("Forward Drive:", True, C_TEXT), (20, y))
        self._draw_progress_bar(110, y + 2, 75, 9, fwd_dec, color=C_GREEN)
        self.surface.blit(self.font_body.render("Reversal:", True, C_TEXT), (195, y))
        self._draw_progress_bar(250, y + 2, 45, 9, rev_dec, color=C_RED)
        y += 17

        # Stage 3: Muscles
        m_title = self.font_bold.render("3. WHAT MUSCLES MOVE:", True, C_HEADER)
        self.surface.blit(m_title, (18, y))
        y += 15

        self.surface.blit(self.font_body.render("Left Turn:", True, C_TEXT), (20, y))
        self._draw_progress_bar(110, y + 2, 75, 9, turn_l, color=C_CYAN)
        self.surface.blit(self.font_body.render("Right Turn:", True, C_TEXT), (195, y))
        self._draw_progress_bar(250, y + 2, 45, 9, turn_r, color=C_CYAN)
        y += 18

        # Thought Balloon
        t_box = pg.Rect(18, y, self.W - 36, 20)
        pg.draw.rect(self.surface, (36, 24, 16), t_box, border_radius=4)
        pg.draw.rect(self.surface, (70, 50, 32), t_box, 1, border_radius=4)
        th_surf = self.font_quote.render(thought, True, C_HEADER)
        self.surface.blit(th_surf, (t_box.centerx - th_surf.get_width() // 2, t_box.centery - th_surf.get_height() // 2))

    def _draw_synapses_view(self, worm) -> None:
        """High-School-Friendly view of what biological synapses evolution has altered."""
        box = pg.Rect(12, 66, self.W - 24, 192)
        pg.draw.rect(self.surface, C_BOX_BG, box, border_radius=5)
        pg.draw.rect(self.surface, C_BOX_BORDER, box, 1, border_radius=5)

        title = self.font_bold.render("TOP EVOLVED BRAIN PATHWAYS:", True, C_HEADER)
        self.surface.blit(title, (18, 72))

        # Show 4 most critical learned biological circuits
        traits = [
            ("Nose Touch -> Reversal Reflex", "AVM -> AVA", "Wall Avoidance", float(worm.genome[10])),
            ("Food Odor -> Turn Muscle",      "ASE -> AIB", "Food Tracking",  float(worm.genome[2])),
            ("Nose Touch -> Stop Forward",     "AVM -| AVB", "Braking Reflex", float(worm.genome[12])),
            ("Side Whisker -> Turn Away",      "ASH -> SMD", "Coil Avoidance", float(worm.genome[14])),
        ]

        y = 92
        for name, pathway, role, weight in traits:
            # Trait title
            n_txt = self.font_bold.render(name, True, C_TEXT)
            self.surface.blit(n_txt, (20, y))

            # Pathway label & weight
            w_str = f"{weight:+.1f} nS"
            w_col = C_GREEN if weight > 0 else C_RED
            p_txt = self.font_body.render(f"({pathway})", True, C_MUTED)
            w_txt = self.font_bold.render(w_str, True, w_col)
            self.surface.blit(p_txt, (185, y))
            self.surface.blit(w_txt, (self.W - w_txt.get_width() - 22, y))
            y += 14

            # Strength meter (-20 to +25)
            norm = int(np.clip(((weight + 20.0) / 45.0) * 100, 0, 100))
            self._draw_progress_bar(20, y, self.W - 40, 6, norm, color=C_CYAN if weight > 0 else (220, 100, 100))
            y += 12

        hint = self.font_quote.render("Values reflect physical biological conductance (nS)", True, C_MUTED)
        self.surface.blit(hint, (20, y + 2))

    def _draw_roster(self, all_worms: list, focused_id: int, mouse_pos: tuple, generation: int) -> None:
        """Live 10-worm swarm leaderboard with click selection."""
        r_box = pg.Rect(12, 264, self.W - 24, 180)
        pg.draw.rect(self.surface, C_BOX_BG, r_box, border_radius=5)
        pg.draw.rect(self.surface, C_BOX_BORDER, r_box, 1, border_radius=5)

        r_hdr = self.font_bold.render(f"POPULATION (GEN #{generation} - TOP 3 KEPT):", True, C_HEADER)
        self.surface.blit(r_hdr, (18, 270))

        ry = 286
        for i, w in enumerate(all_worms):
            row_rect = pg.Rect(16, ry - 1, self.W - 32, 15)
            is_hover = row_rect.collidepoint(mouse_pos)
            is_focused = (w.id == focused_id)

            if is_focused:
                pg.draw.rect(self.surface, (54, 36, 22), row_rect, border_radius=3)
            elif is_hover:
                pg.draw.rect(self.surface, (38, 26, 16), row_rect, border_radius=3)

            # Medal / Badge
            badge = "#1" if i == 0 else ("#2" if i == 1 else ("#3" if i == 2 else f"#{w.id:2d}"))
            b_col = (255, 215, 0) if i == 0 else ((200, 220, 255) if i == 1 else ((215, 140, 60) if i == 2 else C_MUTED))

            # Dot
            pg.draw.circle(self.surface, getattr(w, "color", C_TEXT), (45, ry + 6), 3)

            # Status
            status_str = "ALV" if w.alive else "RIP"
            status_col = C_GREEN if w.alive else (160, 80, 80)

            b_txt = self.font_bold.render(badge, True, b_col)
            s_txt = self.font_body.render(f"[{status_str}]", True, status_col)
            sc_txt = self.font_body.render(f"{w.score}", True, C_HEADER if w.score > 0 else C_MUTED)
            fit_txt = self.font_body.render(f"{w.fitness:.0f}", True, C_TEXT)

            self.surface.blit(b_txt, (20, ry))
            self.surface.blit(s_txt, (54, ry))
            self.surface.blit(sc_txt, (100, ry))
            self.surface.blit(fit_txt, (160, ry))

            if is_focused:
                # Small focus marker arrow
                mx = self.W - 24
                pts = [(mx, ry + 2), (mx - 6, ry + 7), (mx, ry + 12)]
                pg.draw.polygon(self.surface, C_GREEN, pts)

            ry += 16

    def _draw_progress_bar(self, x: int, y: int, w: int, h: int, pct: int, color: tuple) -> None:
        bg_rect = pg.Rect(x, y, w, h)
        pg.draw.rect(self.surface, C_BAR_BG, bg_rect, border_radius=2)
        fill_w = int(w * (np.clip(pct, 0, 100) / 100.0))
        if fill_w > 0:
            fill_rect = pg.Rect(x, y, fill_w, h)
            pg.draw.rect(self.surface, color, fill_rect, border_radius=2)

    def _draw_split_odor_bar(self, x: int, y: int, w: int, h: int, l_pct: int, r_pct: int) -> None:
        bg_rect = pg.Rect(x, y, w, h)
        pg.draw.rect(self.surface, C_BAR_BG, bg_rect, border_radius=2)
        mid_x = x + w // 2
        pg.draw.line(self.surface, (80, 55, 40), (mid_x, y), (mid_x, y + h), 1)

        # Left fill
        if l_pct > 0:
            fill_l = int((w // 2) * (np.clip(l_pct, 0, 100) / 100.0))
            pg.draw.rect(self.surface, C_GREEN, (mid_x - fill_l, y, fill_l, h), border_radius=2)

        # Right fill
        if r_pct > 0:
            fill_r = int((w // 2) * (np.clip(r_pct, 0, 100) / 100.0))
            pg.draw.rect(self.surface, C_GREEN, (mid_x, y, fill_r, h), border_radius=2)
