"""
NeuroWorm Snake — Interactive Pygame Game Loop with High-DPI Upscaling & Genetic Algorithm.

Features:
  - High-Resolution Responsive Display:
      * Auto-scales up to fit big screens (1080p, 1440p, 4K/3K displays)
      * Fully resizable window (drag corners)
      * True Fullscreen toggle (<kbd>F11</kbd>)
  - Left panel:  640x640 soil-themed Snake game with pixel-art worm & glowing egg
  - Right panel: Interactive NeuroWorm Dashboard:
      * Live Neural Heatmap (302 LIF neurons firing)
      * Genetic Algorithm Telemetry (Generation, Best Score, Fitness)
      * Clickable Controls & Hotkeys:
          [Fast Forward 100x] (F key)
          [Next Generation]   (N key)
          [Pause / Play]      (SPACE key)
          [Speed +/-]         (+/- keys)
          [Reset GA]          (R key)
          [Fullscreen]        (F11 key)
"""

import sys
import time
import numpy as np
import pygame as pg

from game.snake_env import SnakeEnv
from game.renderer import Renderer
from connectome.loader import build_network
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from training.genetic_algorithm import GeneticOptimizer, DEFAULT_GENOME
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


class SnakeGame:
    """Main NeuroWorm game with high-res upscaling and evolutionary dashboard."""

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

        self.env = SnakeEnv(grid_size=grid_size)
        self.ga = GeneticOptimizer(pop_size=10, grid_size=grid_size)

        # Active worm brain
        self.net = build_network()
        self.encoder = SensoryEncoder(self.net, grid_size)
        self.decoder = MotorDecoder(self.net)

        # Apply initial best genome
        if theta is not None and len(theta) == len(DEFAULT_GENOME):
            self.ga.best_individual.genome = np.array(theta, dtype=np.float32)
            self.ga.population[0].genome = np.array(theta, dtype=np.float32)
        self._apply_genome(self.ga.best_individual.genome)

        self.radar = None
        self.dashboard = None
        self.buttons = []
        self._init_ui_buttons()

    def _init_ui_buttons(self):
        bx = GAME_W + 14
        w = PANEL_W - 28
        h = 24
        self.btn_ff    = Button((bx, 440, w, h), "⏩ Fast Forward 100x", "F")
        self.btn_next  = Button((bx, 470, w, h), "⏭ Evolve Next Gen",   "N")
        self.btn_pause = Button((bx, 500, w, h), "⏯ Pause / Play",       "SPC")
        self.btn_spdup = Button((bx, 530, w // 2 - 4, h), "➕ Faster",    "+")
        self.btn_spddn = Button((bx + w // 2 + 4, 530, w // 2 - 4, h), "➖ Slower", "-")
        self.btn_reset = Button((bx, 560, w, h), "🔄 Reset GA Population", "R")

        self.buttons = [
            self.btn_ff, self.btn_next, self.btn_pause,
            self.btn_spdup, self.btn_spddn, self.btn_reset
        ]

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

    def _init_pygame(self):
        pg.init()
        pg.display.set_caption("NeuroWorm Snake — Biological C. elegans AI")

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
        if self.show_neural and self.net is not None:
            self.dashboard = NeuralDashboard(self.net)

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
        self._apply_genome(self.ga.best_individual.genome)
        self.fast_forwarding = False
        self._reset_episode()

    def _draw_loading_notice(self, text):
        box = pg.Rect(GAME_W // 2 - 180, GAME_H // 2 - 30, 360, 60)
        pg.draw.rect(self.virtual_screen, (30, 16, 10), box, border_radius=6)
        pg.draw.rect(self.virtual_screen, C_HEADER, box, 2, border_radius=6)
        txt = self.font_title.render(text, True, C_HEADER)
        self.virtual_screen.blit(txt, (box.centerx - txt.get_width() // 2, box.centery - txt.get_height() // 2))

    def _reset_episode(self):
        self.obs = self.env.reset()
        self.direction = "RIGHT"
        self.game_over = False
        if self.net:
            self.net.V[:] = -65e-3
            self.net.g_syn[:] = 0.0
            self.net.spikes[:] = False
            self.net.refractory_timer[:] = 0.0
            self.net.reset_counts()

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
        self._reset_episode()
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
                        self.ga.evaluate_population()
                        self.ga.evolve_next_generation()
                        self._apply_genome(self.ga.best_individual.genome)
                        self._reset_episode()
                    elif event.key in (pg.K_PLUS, pg.K_EQUALS):
                        self.fps = min(60, self.fps + 4)
                    elif event.key in (pg.K_MINUS, pg.K_UNDERSCORE):
                        self.fps = max(4, self.fps - 4)
                    elif event.key == pg.K_r:
                        self.ga = GeneticOptimizer(pop_size=10, grid_size=self.grid_size)
                        self._apply_genome(self.ga.best_individual.genome)
                        self._reset_episode()
                    elif self.human_mode:
                        key_map = {
                            pg.K_UP: "UP", pg.K_w: "UP",
                            pg.K_DOWN: "DOWN", pg.K_s: "DOWN",
                            pg.K_LEFT: "LEFT", pg.K_a: "LEFT",
                            pg.K_RIGHT: "RIGHT", pg.K_d: "RIGHT",
                        }
                        if event.key in key_map:
                            self.direction = key_map[event.key]

                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn_ff.is_clicked(virtual_mouse_pos):
                        self.trigger_fast_forward(100)
                    elif self.btn_next.is_clicked(virtual_mouse_pos):
                        self.ga.evaluate_population()
                        self.ga.evolve_next_generation()
                        self._apply_genome(self.ga.best_individual.genome)
                        self._reset_episode()
                    elif self.btn_pause.is_clicked(virtual_mouse_pos):
                        self.paused = not self.paused
                    elif self.btn_spdup.is_clicked(virtual_mouse_pos):
                        self.fps = min(60, self.fps + 4)
                    elif self.btn_spddn.is_clicked(virtual_mouse_pos):
                        self.fps = max(4, self.fps - 4)
                    elif self.btn_reset.is_clicked(virtual_mouse_pos):
                        self.ga = GeneticOptimizer(pop_size=10, grid_size=self.grid_size)
                        self._apply_genome(self.ga.best_individual.genome)
                        self._reset_episode()

            # ── Simulation Step ───────────────────────────────────────
            if not self.paused and not self.game_over:
                head, food, body, cur_dir = self.obs

                if not self.human_mode:
                    self.net.reset_counts()
                    self.encoder.encode(head, food, body, self.direction)
                    for _ in range(8):
                        self.net.step()
                    self.direction = self.decoder.decode(self.direction)

                self.obs, done = self.env.step(self.direction)

                if self.dashboard:
                    self.dashboard.update()

                if done:
                    self.game_over = True

            # ── Drawing on Virtual Surface ────────────────────────────
            self.virtual_screen.fill(C_PANEL_BG)

            # 1. Left: Game Area
            gen_tag = f"GEN #{self.ga.generation} (BEST: {self.ga.best_individual.score})"
            game_surf = self.renderer.render(
                self.obs, self.env.score, self.game_over, gen_info=gen_tag
            )
            self.virtual_screen.blit(game_surf, (0, 0))

            # 2. Right: Interactive Dashboard
            self._draw_dashboard(virtual_mouse_pos)

            # 3. Present upscaled frame to user window
            self._present_frame()

            if self.game_over:
                pg.time.wait(350)
                self._reset_episode()

            self.clock.tick(self.fps)

        pg.quit()
        sys.exit(0)

    def _draw_dashboard(self, mouse_pos):
        dash_rect = pg.Rect(GAME_W, 0, PANEL_W, VIRTUAL_H)
        pg.draw.rect(self.virtual_screen, C_PANEL_BG, dash_rect)
        pg.draw.line(self.virtual_screen, C_BORDER, (GAME_W, 0), (GAME_W, VIRTUAL_H), 2)

        x0 = GAME_W + 14

        # Title
        title_txt = self.font_title.render("NEUROWORM BIO-AI", True, C_HEADER)
        self.virtual_screen.blit(title_txt, (x0, 10))

        subtitle = self.font_body.render("C. elegans Connectome (302 Neurons)", True, C_MUTED)
        self.virtual_screen.blit(subtitle, (x0, 26))

        # Sensory Radar or Neural Heatmap Display
        if self.view_mode == "radar" and self.radar is not None:
            radar_surf = self.radar.render(self.obs, self.encoder.telemetry, self.direction)
            self.virtual_screen.blit(radar_surf, (x0, 44))
        elif self.dashboard:
            rgb = self.dashboard.get_rgb_array()
            dash_surf = pg.surfarray.make_surface(rgb.transpose(1, 0, 2))
            scaled_dash = pg.transform.scale(dash_surf, (PANEL_W - 28, 220))
            self.virtual_screen.blit(scaled_dash, (x0, 44))
            pg.draw.rect(self.virtual_screen, C_BORDER, (x0, 44, PANEL_W - 28, 220), 1)

        # Telemetry Box
        t_box = pg.Rect(x0, 268, PANEL_W - 28, 162)
        pg.draw.rect(self.virtual_screen, (28, 18, 12), t_box, border_radius=4)
        pg.draw.rect(self.virtual_screen, C_BORDER, t_box, 1, border_radius=4)

        champs = sorted(self.ga.population, key=lambda ind: ind.fitness, reverse=True)[:3]
        lines = [
            ("🧬 Generation:", f"#{self.ga.generation}"),
            ("🥇 Champ 1:", f"{champs[0].score} eggs ({champs[0].fitness:.0f} fit)"),
            ("🥈 Champ 2:", f"{champs[1].score} eggs ({champs[1].fitness:.0f} fit)"),
            ("🥉 Champ 3:", f"{champs[2].score} eggs ({champs[2].fitness:.0f} fit)"),
            ("⚡ Population:", f"{self.ga.pop_size} (Keep 3)"),
            ("⏱️ Speed:", f"{self.fps} FPS"),
            ("🎮 Mode:", "PAUSED" if self.paused else "AUTONOMOUS"),
        ]

        ty = 274
        for label, val in lines:
            ltxt = self.font_body.render(label, True, C_TEXT)
            vtxt = self.font_body.render(val, True, C_ACCENT if "Champ" in label or "Gen" in label else C_HEADER)
            self.virtual_screen.blit(ltxt, (x0 + 8, ty))
            self.virtual_screen.blit(vtxt, (x0 + t_box.width - vtxt.get_width() - 8, ty))
            ty += 21

        # Interactive Buttons
        for btn in self.buttons:
            btn.draw(self.virtual_screen, self.font_btn, is_hovered=btn.rect.collidepoint(mouse_pos))

        # Footer hint
        hint = self.font_body.render("TAB: Map/Heatmap | F: 100x | Space: Pause", True, C_MUTED)
        self.virtual_screen.blit(hint, (x0, 595))
