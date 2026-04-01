"""
NeuroWorm Snake — Interactive Pygame Game Loop with 10-Worm Swarm Simulation.

Features:
  - 10 Concurrent Biological Worms on Screen:
      * All 10 candidate worms simulated and rendered simultaneously
      * Distinct color palettes for all 10 worms (Gold, Silver, Bronze for top 3 champions)
      * Floating badges above each worm's head (★1, ★2, ★3, #4..#10)
      * Dead worms leave faint trail + red collision marker while surviving worms keep foraging
  - Real-time Multi-Seed Spanning Genetic Algorithm (10 population -> keep top 3 champions)
  - Interactive Population Roster on Dashboard:
      * Real-time table showing all 10 worms, alive/dead status, egg scores, and fitness
      * Click any worm row (or press 1-9, 0) to inspect its Sensory Radar
  - Live Sensory Radar with whiskers and biological gauges for the focused worm
  - Fullscreen toggle (F11), High-DPI auto-upscaling, Swarm/Solo mode toggle (M), Fast Forward 100x (F)
"""

import sys
import copy
import time
import numpy as np
import pygame as pg

from game.snake_env import SnakeEnv
from game.renderer import Renderer, WORM_PALETTES
from connectome.loader import build_network
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from training.genetic_algorithm import GeneticOptimizer, DEFAULT_GENOME, EVAL_SEEDS
from viz.sensory_radar import SensoryRadar

try:
    from viz.neural_dashboard import NeuralDashboard
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

GRID_SIZE  = 20
CELL_SIZE  = 32
GAME_W     = GRID_SIZE * CELL_SIZE   # 640
GAME_H     = GRID_SIZE * CELL_SIZE   # 640
PANEL_W    = 280
VIRTUAL_W  = GAME_W + PANEL_W        # 920
VIRTUAL_H  = GAME_H                  # 640

# Colors
C_PANEL_BG = (22, 14, 10)
C_BORDER   = (65, 45, 28)
C_HEADER   = (255, 210, 100)
C_TEXT     = (220, 205, 175)
C_MUTED    = (150, 130, 105)
C_BTN_BG   = (48, 30, 18)
C_BTN_HOV  = (75, 48, 28)
C_BTN_TXT  = (255, 225, 140)
C_ACCENT   = (0, 230, 140)


class Button:
    def __init__(self, rect, text, hotkey=""):
        self.rect = pg.Rect(rect)
        self.text = text
        self.hotkey = hotkey

    def draw(self, surface, font, is_hovered=False):
        col = C_BTN_HOV if is_hovered else C_BTN_BG
        pg.draw.rect(surface, col, self.rect, border_radius=4)
        pg.draw.rect(surface, C_BORDER, self.rect, 1, border_radius=4)

        label = f"{self.text} [{self.hotkey}]" if self.hotkey else self.text
        txt_surf = font.render(label, True, C_BTN_TXT)
        tx = self.rect.centerx - txt_surf.get_width() // 2
        ty = self.rect.centery - txt_surf.get_height() // 2
        surface.blit(txt_surf, (tx, ty))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class WormAgent:
    """An individual biological worm agent in the population."""

    def __init__(
        self,
        worm_id: int,
        genome: np.ndarray,
        color: tuple,
        grid_size: int = 20,
        seed: int = 42,
    ):
        self.id = worm_id
        self.genome = genome.copy()
        self.color = color
        self.grid_size = grid_size
        self.env = SnakeEnv(grid_size=grid_size, seed=seed)
        self.net = build_network()
        self.encoder = SensoryEncoder(self.net, grid_size)
        self.decoder = MotorDecoder(self.net)
        self._apply_genome(genome)
        self.reset(seed=seed)

    def _apply_genome(self, genome: np.ndarray):
        p = {
            "i_food": float(genome[0]),
            "i_danger_front": float(genome[1]),
            "i_danger_side": float(genome[2]),
            "i_awa": float(genome[3]),
            "i_front_food": float(genome[4]),
            "rev_thresh": float(genome[5]),
            "turn_thresh": float(genome[6]),
            "dv_thresh": float(genome[7]),
        }
        self.encoder.update_params(p)
        self.decoder.update_params(p)

    def set_genome(self, genome: np.ndarray):
        self.genome = genome.copy()
        self._apply_genome(self.genome)

    def reset(self, seed: int = None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        self.obs = self.env.reset()
        self.direction = "RIGHT"
        self.alive = True
        self.steps = 0
        self.score = 0
        self.steps_since_food = 0
        self.min_dist_to_food = 999.0
        self.net.V[:] = -65e-3
        self.net.g_syn[:] = 0.0
        self.net.spikes[:] = False
        self.net.refractory_timer[:] = 0.0
        self.net.reset_counts()

    def step(self):
        if not self.alive:
            return

        head, food, body, cur_dir = self.obs
        dist = np.hypot(food[0] - head[0], food[1] - head[1])
        if dist < self.min_dist_to_food:
            self.min_dist_to_food = dist

        self.net.reset_counts()
        self.encoder.encode(head, food, body, self.direction)
        for _ in range(8):
            self.net.step()

        self.direction = self.decoder.decode(self.direction)
        self.obs, done = self.env.step(self.direction)
        self.steps += 1
        self.steps_since_food += 1

        if self.env.score > self.score:
            self.score = self.env.score
            self.steps_since_food = 0
            self.min_dist_to_food = 999.0

        if done or self.steps_since_food > 160:
            self.alive = False

    @property
    def fitness(self) -> float:
        return (self.score * 300.0) + (self.steps * 1.2) + max(0.0, 20.0 - self.min_dist_to_food) * 2.5


class SnakeGame:
    """Main NeuroWorm game with 10-worm swarm simulation and interactive dashboard."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        fps: int = 12,
        human_mode: bool = False,
        show_neural: bool = True,
        theta: np.ndarray = None,
        scale: float = None,
        fullscreen: bool = False,
    ):
        self.grid_size = grid_size
        self.fps = fps
        self.human_mode = human_mode
        self.show_neural = show_neural and VIZ_AVAILABLE and not human_mode
        self.scale = scale
        self.fullscreen = fullscreen
        self.paused = False
        self.fast_forwarding = False
        self.view_mode = "radar"  # "radar" or "heatmap"
        self.solo_mode = False

        self.ga = GeneticOptimizer(pop_size=10, grid_size=grid_size)

        if theta is not None and len(theta) == len(DEFAULT_GENOME):
            self.ga.best_individual.genome = np.array(theta, dtype=np.float32)
            self.ga.population[0].genome = np.array(theta, dtype=np.float32)

        self.seed_idx = 0
        init_seed = EVAL_SEEDS[self.seed_idx]

        # Initialize all 10 candidate worms
        self.worms = [
            WormAgent(
                worm_id=i + 1,
                genome=self.ga.population[i].genome,
                color=WORM_PALETTES[i]["color"],
                grid_size=grid_size,
                seed=init_seed,
            )
            for i in range(10)
        ]
        self.focused_idx = 0
        self.generation_steps = 0

        self.radar = None
        self.dashboard = None
        self.buttons = []
        self._init_ui_buttons()

    def _init_ui_buttons(self):
        bx = GAME_W + 14
        w = PANEL_W - 28
        h = 21
        self.btn_ff    = Button((bx, 444, w, h), "⏩ Fast Forward 100x", "F")
        self.btn_next  = Button((bx, 468, w, h), "⏭ Evolve Next Gen",   "N")
        self.btn_pause = Button((bx, 492, w, h), "⏯ Pause / Play",       "SPC")
        self.btn_spdup = Button((bx, 516, w // 2 - 4, h), "➕ Faster",    "+")
        self.btn_spddn = Button((bx + w // 2 + 4, 516, w // 2 - 4, h), "➖ Slower", "-")
        self.btn_mode  = Button((bx, 540, w, h), "👥 Swarm / Solo",       "M")
        self.btn_reset = Button((bx, 564, w, h), "🔄 Reset GA Population", "R")

        self.buttons = [
            self.btn_ff, self.btn_next, self.btn_pause,
            self.btn_spdup, self.btn_spddn, self.btn_mode, self.btn_reset
        ]

    def _init_pygame(self):
        pg.init()
        pg.display.set_caption("NeuroWorm Snake — 10 Biological Worms Evolving")

        info = pg.display.Info()
        # High-DPI Upscaling detection
        if self.scale is None:
            if info.current_w >= 2560:
                self.scale = 1.75   # 1610 x 1120 on 3K/4K displays
            elif info.current_w >= 1920:
                self.scale = 1.5    # 1380 x 960 on 1080p
            else:
                self.scale = 1.25

        self.win_w = int(VIRTUAL_W * self.scale)
        self.win_h = int(VIRTUAL_H * self.scale)

        flags = pg.RESIZABLE
        if self.fullscreen:
            flags |= pg.FULLSCREEN
            self.screen = pg.display.set_mode((0, 0), flags)
            self.win_w, self.win_h = self.screen.get_size()
        else:
            self.screen = pg.display.set_mode((self.win_w, self.win_h), flags)

        self.virtual_screen = pg.Surface((VIRTUAL_W, VIRTUAL_H))
        self.clock = pg.time.Clock()

        self.font_title = pg.font.SysFont("monospace", 14, bold=True)
        self.font_body  = pg.font.SysFont("monospace", 11)
        self.font_btn   = pg.font.SysFont("monospace", 10, bold=True)

        self.renderer = Renderer(cell_size=CELL_SIZE, grid_size=self.grid_size)
        self.radar = SensoryRadar(grid_size=self.grid_size)

        if self.show_neural and len(self.worms) > 0:
            self.dashboard = NeuralDashboard(self.worms[0].net)

        return pg

    def _screen_to_virtual(self, screen_pos):
        sx, sy = screen_pos
        scale = min(self.win_w / VIRTUAL_W, self.win_h / VIRTUAL_H)
        rendered_w = int(VIRTUAL_W * scale)
        rendered_h = int(VIRTUAL_H * scale)
        offset_x = (self.win_w - rendered_w) // 2
        offset_y = (self.win_h - rendered_h) // 2

        vx = (sx - offset_x) / max(0.001, scale)
        vy = (sy - offset_y) / max(0.001, scale)
        return int(vx), int(vy)

    def trigger_fast_forward(self, n=100):
        self.fast_forwarding = True
        self._draw_loading_notice("⏩ RUNNING 100 FAST GA EVALUATIONS...")
        self._present_frame()

        res = self.ga.fast_forward(n_evaluations=n)
        for i, w in enumerate(self.worms):
            w.set_genome(self.ga.population[i].genome)
        self.fast_forwarding = False
        self.seed_idx = (self.seed_idx + 1) % len(EVAL_SEEDS)
        seed = EVAL_SEEDS[self.seed_idx]
        for w in self.worms:
            w.reset(seed=seed)
        self.generation_steps = 0

    def _draw_loading_notice(self, text):
        box = pg.Rect(GAME_W // 2 - 180, GAME_H // 2 - 30, 360, 60)
        pg.draw.rect(self.virtual_screen, (30, 16, 10), box, border_radius=6)
        pg.draw.rect(self.virtual_screen, C_HEADER, box, 2, border_radius=6)
        txt = self.font_title.render(text, True, C_HEADER)
        self.virtual_screen.blit(txt, (box.centerx - txt.get_width() // 2, box.centery - txt.get_height() // 2))

    def _evolve_and_next_generation(self):
        # Update fitness for each individual
        for i, w in enumerate(self.worms):
            self.ga.population[i].fitness = w.fitness
            self.ga.population[i].score = w.score
            self.ga.population[i].steps = w.steps

        # Sort population descending by fitness
        self.ga.population.sort(key=lambda ind: ind.fitness, reverse=True)
        if self.ga.population[0].fitness > self.ga.best_individual.fitness:
            self.ga.best_individual = copy.deepcopy(self.ga.population[0])

        # Evolve (top 3 champions preserved in slots 0, 1, 2)
        self.ga.evolve_next_generation()

        # Step to next seed so foraging is tested across different food layouts
        self.seed_idx = (self.seed_idx + 1) % len(EVAL_SEEDS)
        seed = EVAL_SEEDS[self.seed_idx]

        # Reset all 10 worms
        for i, w in enumerate(self.worms):
            w.set_genome(self.ga.population[i].genome)
            w.reset(seed=seed)

        self.generation_steps = 0

    def _present_frame(self):
        scale = min(self.win_w / VIRTUAL_W, self.win_h / VIRTUAL_H)
        rendered_w = int(VIRTUAL_W * scale)
        rendered_h = int(VIRTUAL_H * scale)
        offset_x = (self.win_w - rendered_w) // 2
        offset_y = (self.win_h - rendered_h) // 2

        scaled_surf = pg.transform.smoothscale(self.virtual_screen, (rendered_w, rendered_h))
        self.screen.fill((12, 7, 4))
        self.screen.blit(scaled_surf, (offset_x, offset_y))
        pg.display.flip()

    def run(self):
        pygame = self._init_pygame()
        running = True

        while running:
            screen_mouse_pos = pg.mouse.get_pos()
            virtual_mouse_pos = self._screen_to_virtual(screen_mouse_pos)

            # ── Event Handling ────────────────────────────────────────
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.VIDEORESIZE:
                    self.win_w, self.win_h = event.w, event.h
                    self.screen = pg.display.set_mode((self.win_w, self.win_h), pg.RESIZABLE)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_F11:
                        self.fullscreen = not self.fullscreen
                        flags = pg.RESIZABLE
                        if self.fullscreen:
                            flags |= pg.FULLSCREEN
                            self.screen = pg.display.set_mode((0, 0), flags)
                        else:
                            self.screen = pg.display.set_mode((self.win_w, self.win_h), flags)
                        self.win_w, self.win_h = self.screen.get_size()
                    elif event.key == pg.K_TAB:
                        self.view_mode = "heatmap" if self.view_mode == "radar" else "radar"
                    elif event.key == pg.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pg.K_f:
                        self.trigger_fast_forward(100)
                    elif event.key == pg.K_n:
                        self._evolve_and_next_generation()
                    elif event.key == pg.K_m:
                        self.solo_mode = not self.solo_mode
                    elif event.key in (pg.K_PLUS, pg.K_EQUALS):
                        self.fps = min(60, self.fps + 4)
                    elif event.key in (pg.K_MINUS, pg.K_UNDERSCORE):
                        self.fps = max(4, self.fps - 4)
                    elif event.key == pg.K_r:
                        self.ga = GeneticOptimizer(pop_size=10, grid_size=self.grid_size)
                        for i, w in enumerate(self.worms):
                            w.set_genome(self.ga.population[i].genome)
                            w.reset(seed=EVAL_SEEDS[0])
                        self.generation_steps = 0
                    elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9):
                        self.focused_idx = event.key - pg.K_1
                    elif event.key == pg.K_0:
                        self.focused_idx = 9
                    elif event.key == pg.K_LEFT:
                        self.focused_idx = (self.focused_idx - 1) % len(self.worms)
                    elif event.key == pg.K_RIGHT:
                        self.focused_idx = (self.focused_idx + 1) % len(self.worms)

                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn_ff.is_clicked(virtual_mouse_pos):
                        self.trigger_fast_forward(100)
                    elif self.btn_next.is_clicked(virtual_mouse_pos):
                        self._evolve_and_next_generation()
                    elif self.btn_pause.is_clicked(virtual_mouse_pos):
                        self.paused = not self.paused
                    elif self.btn_spdup.is_clicked(virtual_mouse_pos):
                        self.fps = min(60, self.fps + 4)
                    elif self.btn_spddn.is_clicked(virtual_mouse_pos):
                        self.fps = max(4, self.fps - 4)
                    elif self.btn_mode.is_clicked(virtual_mouse_pos):
                        self.solo_mode = not self.solo_mode
                    elif self.btn_reset.is_clicked(virtual_mouse_pos):
                        self.ga = GeneticOptimizer(pop_size=10, grid_size=self.grid_size)
                        for i, w in enumerate(self.worms):
                            w.set_genome(self.ga.population[i].genome)
                            w.reset(seed=EVAL_SEEDS[0])
                        self.generation_steps = 0
                    else:
                        # Check roster row click:
                        rx = GAME_W + 14 + 4
                        rw = PANEL_W - 36
                        for i in range(len(self.worms)):
                            row_rect = pg.Rect(rx, 260 + i * 17 - 1, rw, 16)
                            if row_rect.collidepoint(virtual_mouse_pos):
                                self.focused_idx = i
                                break

            # ── Simulation Step (All 10 Worms) ────────────────────────
            if not self.paused:
                all_dead = True
                for w in self.worms:
                    if w.alive:
                        w.step()
                        all_dead = False

                self.generation_steps += 1
                if all_dead or self.generation_steps >= 400:
                    pg.time.wait(350)
                    self._evolve_and_next_generation()

                if self.dashboard and self.view_mode == "heatmap":
                    self.dashboard.net = self.worms[self.focused_idx].net
                    self.dashboard.update()

            # ── Drawing on Virtual Surface ────────────────────────────
            self.virtual_screen.fill(C_PANEL_BG)

            # 1. Left: Game Area with all 10 worms
            gen_tag = f"GEN #{self.ga.generation} ({'SOLO' if self.solo_mode else 'SWARM'})"
            game_surf = self.renderer.render_multi_worms(
                self.worms,
                self.worms[0].env.food,
                gen_info=gen_tag,
                focused_idx=self.focused_idx,
                solo_mode=self.solo_mode,
            )
            self.virtual_screen.blit(game_surf, (0, 0))

            # 2. Right: Interactive Dashboard
            self._draw_dashboard(virtual_mouse_pos)

            # 3. Present upscaled frame to user window
            self._present_frame()

            self.clock.tick(self.fps)

        pg.quit()
        sys.exit(0)

    def _draw_dashboard(self, mouse_pos):
        dash_rect = pg.Rect(GAME_W, 0, PANEL_W, VIRTUAL_H)
        pg.draw.rect(self.virtual_screen, C_PANEL_BG, dash_rect)
        pg.draw.line(self.virtual_screen, C_BORDER, (GAME_W, 0), (GAME_W, VIRTUAL_H), 2)

        x0 = GAME_W + 14
        fw = self.worms[self.focused_idx]
        champ_label = "🥇 CHAMP 1" if self.focused_idx == 0 else ("🥈 CHAMP 2" if self.focused_idx == 1 else ("🥉 CHAMP 3" if self.focused_idx == 2 else f"WORM #{fw.id}"))

        # Header Title
        title_txt = self.font_title.render("NEUROWORM BIO-AI", True, C_HEADER)
        self.virtual_screen.blit(title_txt, (x0, 8))

        mode_str = f"Inspecting: {champ_label}"
        subtitle = self.font_body.render(mode_str, True, fw.color)
        self.virtual_screen.blit(subtitle, (x0, 24))

        # Sensory Radar or Neural Heatmap Display
        if self.view_mode == "radar" and self.radar is not None:
            radar_surf = self.radar.render(
                fw.obs,
                fw.encoder.telemetry,
                fw.direction,
                other_worms=self.worms,
                worm_color=fw.color,
            )
            self.virtual_screen.blit(radar_surf, (x0, 40))
        elif self.dashboard:
            rgb = self.dashboard.get_rgb_array()
            dash_surf = pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
            scaled_dash = pg.transform.scale(dash_surf, (PANEL_W - 28, 215))
            self.virtual_screen.blit(scaled_dash, (x0, 40))
            pg.draw.rect(self.virtual_screen, C_BORDER, (x0, 40, PANEL_W - 28, 215), 1)

        # Population Roster Box (10 Worms)
        r_box = pg.Rect(x0, 240, PANEL_W - 28, 196)
        pg.draw.rect(self.virtual_screen, (28, 18, 12), r_box, border_radius=4)
        pg.draw.rect(self.virtual_screen, C_BORDER, r_box, 1, border_radius=4)

        r_hdr = self.font_btn.render(f"POPULATION (GEN #{self.ga.generation} - 10 WORMS):", True, C_HEADER)
        self.virtual_screen.blit(r_hdr, (x0 + 8, 244))

        ry = 260
        for i, w in enumerate(self.worms):
            row_rect = pg.Rect(x0 + 4, ry - 1, PANEL_W - 36, 17)
            is_hover = row_rect.collidepoint(mouse_pos)
            is_focused = (i == self.focused_idx)

            if is_focused:
                pg.draw.rect(self.virtual_screen, (52, 34, 20), row_rect, border_radius=3)
            elif is_hover:
                pg.draw.rect(self.virtual_screen, (36, 24, 15), row_rect, border_radius=3)

            badge = "★1" if i == 0 else ("★2" if i == 1 else ("★3" if i == 2 else f"#{i+1:2d}"))
            pal = WORM_PALETTES[i]

            # Color dot
            pg.draw.circle(self.virtual_screen, pal["color"], (x0 + 38, ry + 7), 4)

            status_str = "ALV" if w.alive else "RIP"
            status_col = C_ACCENT if w.alive else (160, 80, 80)

            b_txt = self.font_btn.render(badge, True, pal["border"])
            s_txt = self.font_body.render(f"[{status_str}]", True, status_col)
            sc_txt = self.font_body.render(f"{w.score}🥚", True, C_HEADER if w.score > 0 else C_MUTED)
            fit_txt = self.font_body.render(f"{w.fitness:.0f}", True, C_TEXT)

            self.virtual_screen.blit(b_txt, (x0 + 8, ry + 1))
            self.virtual_screen.blit(s_txt, (x0 + 48, ry + 1))
            self.virtual_screen.blit(sc_txt, (x0 + 96, ry + 1))
            self.virtual_screen.blit(fit_txt, (x0 + 155, ry + 1))

            if is_focused:
                foc_marker = self.font_btn.render("◄", True, C_ACCENT)
                self.virtual_screen.blit(foc_marker, (x0 + r_box.width - 18, ry + 1))

            ry += 17

        # Interactive Buttons
        for btn in self.buttons:
            btn.draw(self.virtual_screen, self.font_btn, is_hovered=btn.rect.collidepoint(mouse_pos))

        # Footer hints
        hint = self.font_body.render("1-0/Click: Inspect | M: Swarm/Solo | F: 100x", True, C_MUTED)
        self.virtual_screen.blit(hint, (x0, 592))
        hint2 = self.font_body.render("TAB: Map/Heatmap | F11: Fullscreen | SPC", True, C_MUTED)
        self.virtual_screen.blit(hint2, (x0, 608))
