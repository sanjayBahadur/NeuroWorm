"""
High-School-Friendly Brain Activity, Leaderboard, & Learning Curve Inspector.

Structured into 3 distinct cards driven by game.layout constants:
  1. Selected Worm: Live Senses -> Decides -> Moves + Thought summary (or Synapse learning)
  2. Leaderboard: Top 5 organisms (+ selected worm if outside top 5) with status and cause
  3. Is It Learning?: Line chart of historical best and average scores across generations
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pygame as pg

from game.layout import (
    PANEL_W, VIRTUAL_H,
    CARD_X, CARD_W,
    CARD1_Y, CARD1_H,
    CARD2_Y, CARD2_H,
    CARD3_Y, CARD3_H,
)
from game.theme import (
    get_font, C_PANEL_BG, C_CARD_BG, C_CARD_HOVER, C_BORDER,
    C_HEADER, C_TEXT, C_MUTED, C_ACCENT, C_DANGER, C_STARVE,
    C_EGG_GOLD
)

C_BAR_BG = (42, 28, 18)
C_CYAN   = (80, 210, 240)


class BrainInspector:
    """Renders the educational 3-card telemetry and learning inspector panel."""

    def __init__(self, width: int = PANEL_W, height: int = VIRTUAL_H):
        if not pg.font.get_init():
            pg.font.init()
        self.W = width
        self.H = height
        self.surface = pg.Surface((self.W, self.H))
        self.font_title = get_font("title", 14)
        self.font_bold  = get_font("card_title", 13)
        self.font_body  = get_font("body", 12)
        self.font_small = get_font("small", 12)

        # Tab button rects inside Card 1
        tab_w = (CARD_W - 20) // 2
        self.tab_flow_rect = pg.Rect(CARD_X + 8, CARD1_Y + 8, tab_w, 24)
        self.tab_syn_rect  = pg.Rect(CARD_X + 12 + tab_w, CARD1_Y + 8, tab_w, 24)

        # Hit testing for leaderboard rows
        self.leaderboard_hit_rects: list[tuple[pg.Rect, int]] = []

    def get_clicked_worm_idx(self, mouse_pos: tuple[int, int]) -> Optional[int]:
        """Derive clicked worm index directly from leaderboard row rects."""
        for rect, worm_idx in self.leaderboard_hit_rects:
            if rect.collidepoint(mouse_pos):
                return worm_idx
        return None

    def render(
        self,
        worm,
        all_worms: list,
        generation: int,
        view_mode: str = "flow",   # "flow" or "synapses"
        mouse_pos: tuple = (0, 0),
        solo_mode: bool = False,
        history: Optional[list[dict]] = None,
        focus_mode: str = "follow",
    ) -> pg.Surface:
        self.surface.fill(C_PANEL_BG)
        pg.draw.line(self.surface, C_BORDER, (0, 0), (0, self.H), 1)

        # ── 1. Card 1: Selected Worm ──────────────────────────────────
        self._draw_card_selected_worm(worm, view_mode, mouse_pos)

        # ── 2. Card 2: Leaderboard (Top 5) ────────────────────────────
        self._draw_card_leaderboard(all_worms, focused_id=worm.id, mouse_pos=mouse_pos, generation=generation)

        # ── 3. Card 3: Is It Learning? (Learning Curve) ───────────────
        self._draw_card_learning_curve(history or [], generation)

        return self.surface

    def _draw_card_selected_worm(self, worm, view_mode: str, mouse_pos: tuple) -> None:
        """Card 1: Selected Worm live sensory telemetry or synaptic circuit."""
        card_rect = pg.Rect(CARD_X, CARD1_Y, CARD_W, CARD1_H)
        pg.draw.rect(self.surface, C_CARD_BG, card_rect, border_radius=6)
        pg.draw.rect(self.surface, C_BORDER, card_rect, 1, border_radius=6)

        # Tabs at top of Card 1
        is_hover1 = self.tab_flow_rect.collidepoint(mouse_pos)
        is_active1 = (view_mode == "flow")
        col1 = (52, 36, 22) if is_active1 else ((38, 26, 16) if is_hover1 else (26, 17, 11))
        pg.draw.rect(self.surface, col1, self.tab_flow_rect, border_radius=4)
        pg.draw.rect(self.surface, C_HEADER if is_active1 else C_BORDER, self.tab_flow_rect, 1, border_radius=4)
        t1 = self.font_bold.render("LIVE THOUGHTS", True, C_HEADER if is_active1 else C_MUTED)
        self.surface.blit(t1, (self.tab_flow_rect.centerx - t1.get_width() // 2, self.tab_flow_rect.centery - t1.get_height() // 2))

        is_hover2 = self.tab_syn_rect.collidepoint(mouse_pos)
        is_active2 = (view_mode == "synapses")
        col2 = (52, 36, 22) if is_active2 else ((38, 26, 16) if is_hover2 else (26, 17, 11))
        pg.draw.rect(self.surface, col2, self.tab_syn_rect, border_radius=4)
        pg.draw.rect(self.surface, C_HEADER if is_active2 else C_BORDER, self.tab_syn_rect, 1, border_radius=4)
        t2 = self.font_bold.render("SYNAPSE LEARNING", True, C_HEADER if is_active2 else C_MUTED)
        self.surface.blit(t2, (self.tab_syn_rect.centerx - t2.get_width() // 2, self.tab_syn_rect.centery - t2.get_height() // 2))

        # Selected worm quick stats
        y = CARD1_Y + 36
        starve_limit = getattr(worm, "starve_limit", 160)
        steps_food = getattr(worm, "steps_since_food", 0)
        rem_hunger = max(0, int(100.0 * (1.0 - steps_food / max(1, starve_limit))))

        name_str = f"#{worm.id} HERO" if worm.id == 1 else f"#{worm.id} Candidate"
        n_lbl = self.font_bold.render(name_str, True, getattr(worm, "color", C_TEXT))
        self.surface.blit(n_lbl, (CARD_X + 10, y))

        stats_str = f"{worm.score} eggs · {worm.steps} stp · {rem_hunger}% hunger"
        s_lbl = self.font_small.render(stats_str, True, C_MUTED)
        self.surface.blit(s_lbl, (CARD_X + CARD_W - s_lbl.get_width() - 10, y + 1))
        y += 18

        if view_mode == "flow":
            self._render_flow_strip(worm, y)
        else:
            self._render_synapses_strip(worm, y)

    def _render_flow_strip(self, worm, start_y: int) -> None:
        """Draw 3-step strip: Senses -> Decides -> Moves + Thought Balloon."""
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
        thought = m_tel.get("thought_summary", "Crawling forward — seeking food smell")

        y = start_y
        # 1. Senses
        s_title = self.font_small.render("1. Senses: Smell & Touch", True, C_HEADER)
        self.surface.blit(s_title, (CARD_X + 10, y))
        y += 14

        self.surface.blit(self.font_small.render("Odor L/R:", True, C_TEXT), (CARD_X + 12, y))
        self._draw_split_odor_bar(CARD_X + 80, y + 2, 110, 8, food_l, food_r)

        touch_text = f"Nose: {f_dist} stp" if f_dist <= 3 else "Nose: clear"
        touch_col = C_DANGER if f_dist <= 2 else (C_STARVE if f_dist <= 4 else C_ACCENT)
        self.surface.blit(self.font_small.render(touch_text, True, touch_col), (CARD_X + 200, y))
        y += 16

        # 2. Decides
        d_title = self.font_small.render("2. Decides: Forward vs Turn", True, C_HEADER)
        self.surface.blit(d_title, (CARD_X + 10, y))
        y += 14

        self.surface.blit(self.font_small.render("Crawl:", True, C_TEXT), (CARD_X + 12, y))
        self._draw_progress_bar(CARD_X + 58, y + 2, 85, 7, fwd_dec, color=C_ACCENT)
        self.surface.blit(self.font_small.render("Turn:", True, C_TEXT), (CARD_X + 155, y))
        self._draw_progress_bar(CARD_X + 195, y + 2, 85, 7, rev_dec, color=C_DANGER)
        y += 16

        # 3. Moves
        m_title = self.font_small.render("3. Moves: Steering Muscles", True, C_HEADER)
        self.surface.blit(m_title, (CARD_X + 10, y))
        y += 14

        self.surface.blit(self.font_small.render("Steer L:", True, C_TEXT), (CARD_X + 12, y))
        self._draw_progress_bar(CARD_X + 66, y + 2, 75, 7, turn_l, color=C_CYAN)
        self.surface.blit(self.font_small.render("Steer R:", True, C_TEXT), (CARD_X + 155, y))
        self._draw_progress_bar(CARD_X + 215, y + 2, 75, 7, turn_r, color=C_CYAN)
        y += 16

        # Thought Balloon
        t_box = pg.Rect(CARD_X + 10, y, CARD_W - 20, 20)
        pg.draw.rect(self.surface, (34, 23, 15), t_box, border_radius=4)
        pg.draw.rect(self.surface, C_BORDER, t_box, 1, border_radius=4)
        th_surf = self.font_small.render(thought, True, C_HEADER)
        self.surface.blit(th_surf, (t_box.centerx - th_surf.get_width() // 2, t_box.centery - th_surf.get_height() // 2))

    def _render_synapses_strip(self, worm, start_y: int) -> None:
        """Card 1 secondary tab: top evolved biological synapses."""
        traits = [
            ("Nose Touch -> Turn", "AVM -> AVA", float(worm.genome[10])),
            ("Food Odor -> Turn",  "ASE -> AIB", float(worm.genome[2])),
            ("Nose Touch -> Brake", "AVM -| AVB", float(worm.genome[12])),
            ("Side Whisk -> Steer", "ASH -> SMD", float(worm.genome[14])),
        ]
        y = start_y + 2
        for name, pathway, weight in traits:
            n_txt = self.font_small.render(f"{name} ({pathway})", True, C_TEXT)
            self.surface.blit(n_txt, (CARD_X + 12, y))

            w_str = f"{weight:+.1f} nS"
            w_col = C_ACCENT if weight > 0 else C_DANGER
            w_txt = self.font_small.render(w_str, True, w_col)
            self.surface.blit(w_txt, (CARD_X + CARD_W - w_txt.get_width() - 14, y))
            y += 15

            norm = int(np.clip(((weight + 20.0) / 45.0) * 100, 0, 100))
            self._draw_progress_bar(CARD_X + 12, y, CARD_W - 24, 6, norm, color=C_CYAN if weight > 0 else (220, 90, 90))
            y += 13

        hint = self.font_small.render("Conductance (nS) evolved via genetic selection", True, C_MUTED)
        self.surface.blit(hint, (CARD_X + 12, y + 2))

    def _draw_card_leaderboard(self, all_worms: list, focused_id: int, mouse_pos: tuple, generation: int) -> None:
        """Card 2: Top 5 leaderboard (+ selected worm if outside top 5) with status icons."""
        card_rect = pg.Rect(CARD_X, CARD2_Y, CARD_W, CARD2_H)
        pg.draw.rect(self.surface, C_CARD_BG, card_rect, border_radius=6)
        pg.draw.rect(self.surface, C_BORDER, card_rect, 1, border_radius=6)

        hdr_txt = self.font_bold.render(f"LEADERBOARD (GEN #{generation})", True, C_HEADER)
        self.surface.blit(hdr_txt, (CARD_X + 10, CARD2_Y + 8))

        # Sort all worms by (score, steps, fitness) descending
        sorted_worms = sorted(all_worms, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness), reverse=True)
        top_5 = sorted_worms[:5]

        # Check if focused worm is outside top 5
        focused_worm = next((w for w in all_worms if w.id == focused_id), None)
        display_list = list(top_5)
        if focused_worm and focused_worm not in top_5:
            display_list.append(focused_worm)

        self.leaderboard_hit_rects = []
        ry = CARD2_Y + 28

        for rank_idx, w in enumerate(display_list):
            row_rect = pg.Rect(CARD_X + 6, ry - 1, CARD_W - 12, 19)
            self.leaderboard_hit_rects.append((row_rect, w.id - 1))

            is_focused = (w.id == focused_id)
            is_hover = row_rect.collidepoint(mouse_pos)

            if is_focused:
                pg.draw.rect(self.surface, C_CARD_HOVER, row_rect, border_radius=4)
                pg.draw.rect(self.surface, (120, 85, 55), row_rect, 1, border_radius=4)
            elif is_hover:
                pg.draw.rect(self.surface, (36, 24, 16), row_rect, border_radius=4)

            # Rank badge
            rank_str = f"#{rank_idx + 1}" if rank_idx < 5 else "#sel"
            rank_col = C_EGG_GOLD if rank_idx == 0 else (C_HEADER if rank_idx < 3 else C_MUTED)
            self.surface.blit(self.font_small.render(rank_str, True, rank_col), (CARD_X + 10, ry + 1))

            # Color chip
            chip_col = getattr(w, "color", C_TEXT)
            pg.draw.circle(self.surface, chip_col, (CARD_X + 44, ry + 8), 4)

            # Worm Name
            w_name = f"Worm #{w.id}"
            self.surface.blit(self.font_small.render(w_name, True, C_TEXT), (CARD_X + 54, ry + 1))

            # Score
            sc_str = f"{w.score} eggs"
            sc_col = C_EGG_GOLD if w.score > 0 else C_MUTED
            self.surface.blit(self.font_small.render(sc_str, True, sc_col), (CARD_X + 145, ry + 1))

            # Status with cause
            if w.alive:
                st_icon = "alive"
                st_txt = "alive"
                st_col = C_ACCENT
            else:
                cause = getattr(w, "death_cause", None) or "died"
                cause_map = {"wall": "x wall", "self": "x self", "starved": "x hungry", "timeout": "x time"}
                st_icon = "x"
                st_txt = cause_map.get(cause, "x died")
                st_col = C_DANGER

            s_surf = self.font_small.render(st_txt, True, st_col)
            self.surface.blit(s_surf, (CARD_X + CARD_W - s_surf.get_width() - 12, ry + 1))

            ry += 22

    def _draw_card_learning_curve(self, history: list[dict], generation: int) -> None:
        """Card 3: Sparkline / line chart of historical best and average eggs per generation."""
        card_rect = pg.Rect(CARD_X, CARD3_Y, CARD_W, CARD3_H)
        pg.draw.rect(self.surface, C_CARD_BG, card_rect, border_radius=6)
        pg.draw.rect(self.surface, C_BORDER, card_rect, 1, border_radius=6)

        hdr_txt = self.font_bold.render("IS IT LEARNING?", True, C_HEADER)
        self.surface.blit(hdr_txt, (CARD_X + 10, CARD3_Y + 8))

        # Metrics subtitle
        if history:
            last = history[-1]
            sub_str = f"Gen {last['gen']} · Best: {last['best_score']} eggs · Avg: {last['avg_score']:.1f}"
        else:
            sub_str = f"Gen {generation} · Best: 0 eggs · Avg: 0.0"
        self.surface.blit(self.font_small.render(sub_str, True, C_MUTED), (CARD_X + 10, CARD3_Y + 24))

        # Plot boundary
        px = CARD_X + 36
        py = CARD3_Y + 44
        pw = CARD_W - 48
        ph = CARD3_H - 64
        plot_rect = pg.Rect(px, py, pw, ph)
        pg.draw.rect(self.surface, (22, 15, 10), plot_rect, border_radius=4)
        pg.draw.rect(self.surface, C_BORDER, plot_rect, 1, border_radius=4)

        if not history:
            empty_txt = self.font_small.render("Press N or F to evolve a generation.", True, C_MUTED)
            self.surface.blit(empty_txt, (plot_rect.centerx - empty_txt.get_width() // 2, plot_rect.centery - empty_txt.get_height() // 2))
            return

        # Show last 30 generations
        data = history[-30:]
        n_pts = len(data)

        max_val = max(1, max(int(d.get("best_score", 0)) for d in data))
        # Ensure y-axis ceiling has headroom
        y_ceil = max(5, int(math.ceil(max_val * 1.2)))

        # Y-axis labels & dashed gridlines
        y_ticks = [0, y_ceil // 2, y_ceil]
        for val in y_ticks:
            y_pos = py + ph - int((val / y_ceil) * ph)
            lbl = self.font_small.render(str(val), True, C_MUTED)
            self.surface.blit(lbl, (px - lbl.get_width() - 4, y_pos - lbl.get_height() // 2))
            if val > 0:
                pg.draw.line(self.surface, (40, 28, 18), (px, y_pos), (px + pw, y_pos), 1)

        # X-axis generation labels
        g_start = data[0].get("gen", 1)
        g_end = data[-1].get("gen", 1)
        lbl_s = self.font_small.render(f"G{g_start}", True, C_MUTED)
        lbl_e = self.font_small.render(f"G{g_end}", True, C_MUTED)
        self.surface.blit(lbl_s, (px, py + ph + 2))
        self.surface.blit(lbl_e, (px + pw - lbl_e.get_width(), py + ph + 2))

        # Legend at top right
        leg_b = self.font_small.render("Best", True, C_HEADER)
        leg_a = self.font_small.render("Avg", True, C_CYAN)
        lx = px + pw - leg_b.get_width() - leg_a.get_width() - 16
        self.surface.blit(leg_b, (lx, py + 4))
        self.surface.blit(leg_a, (lx + leg_b.get_width() + 8, py + 4))

        # Plot lines
        best_pts = []
        avg_pts = []
        for i, pt in enumerate(data):
            cx = px + int((i / max(1, n_pts - 1)) * pw) if n_pts > 1 else px + pw // 2
            b_val = pt.get("best_score", 0)
            a_val = pt.get("avg_score", 0.0)
            by = py + ph - int((b_val / y_ceil) * ph)
            ay = py + ph - int((a_val / y_ceil) * ph)
            best_pts.append((cx, by))
            avg_pts.append((cx, ay))

        if len(best_pts) > 1:
            pg.draw.lines(self.surface, C_HEADER, False, best_pts, 2)
            pg.draw.lines(self.surface, C_CYAN, False, avg_pts, 2)
        for pt in best_pts:
            pg.draw.circle(self.surface, C_HEADER, pt, 3)
        for pt in avg_pts:
            pg.draw.circle(self.surface, C_CYAN, pt, 2)

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
        pg.draw.line(self.surface, (70, 48, 32), (mid_x, y), (mid_x, y + h), 1)

        if l_pct > 0:
            fill_l = int((w // 2) * (np.clip(l_pct, 0, 100) / 100.0))
            pg.draw.rect(self.surface, C_ACCENT, (mid_x - fill_l, y, fill_l, h), border_radius=2)

        if r_pct > 0:
            fill_r = int((w // 2) * (np.clip(r_pct, 0, 100) / 100.0))
            pg.draw.rect(self.surface, C_ACCENT, (mid_x, y, fill_r, h), border_radius=2)
