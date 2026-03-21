"""
Pygame-based Snake game renderer with neural AI controller.

Displays:
  - Left panel:  Snake game on a soil-themed pixel-art grid
  - Right panel: Real-time neural activity heatmap (optional)
"""

import sys
import numpy as np
from game.snake_env import SnakeEnv
from connectome.loader import build_network
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from training.optimize_encoding import DEFAULT_PARAMS, STEPS_PER_FRAME
from game.renderer import Renderer

# Color palette — soil/worm theme
C_BG        = (26,  15,  10)   # dark soil background
CELL        = 32               # 32x32 pixels per grid cell (matching sprites)


class SnakeGame:
    """Full pygame Snake game with optional neural AI and live dashboard."""

    def __init__(
        self,
        grid_size: int   = 20,
        fps: int         = 10,
        human_mode: bool = False,
        show_neural: bool = True,
        theta: np.ndarray = None,
    ):
        self.grid_size   = grid_size
        self.fps         = fps
        self.human_mode  = human_mode
        self.show_neural = show_neural
        self.theta       = theta if theta is not None else DEFAULT_PARAMS

        self.game_w = grid_size * CELL
        self.game_h = grid_size * CELL

        self.env = SnakeEnv(grid_size=grid_size)

        if not human_mode:
            self.net     = build_network()
            self.encoder = SensoryEncoder(
                self.net, grid_size,
                i_food=self.theta[0],
                i_danger=self.theta[1],
                i_awa=self.theta[2],
            )
            self.decoder = MotorDecoder(
                self.net,
                rev_thresh=self.theta[3],
                dors_thresh=self.theta[4],
            )
        else:
            self.net = self.encoder = self.decoder = None

        self.dashboard = None

    def _init_pygame(self):
        import pygame
        pygame.init()
        from viz.neural_dashboard import NeuralDashboard

        dash_w = NeuralDashboard.PANEL_W if self.show_neural else 0
        total_w = self.game_w + dash_w
        self.screen = pygame.display.set_mode((total_w, self.game_h))
        pygame.display.set_caption("NeurWorm Snake — C. elegans AI")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("monospace", 14)

        if self.show_neural and self.net is not None:
            self.dashboard = NeuralDashboard(self.net)

        self.renderer = Renderer(cell_size=CELL, grid_size=self.grid_size)
        return pygame

    def _draw_grid(self, surface, pygame):
        for x in range(0, self.game_w, CELL):
            pygame.draw.line(surface, C_GRID, (x, 0), (x, self.game_h))
        for y in range(0, self.game_h, CELL):
            pygame.draw.line(surface, C_GRID, (0, y), (self.game_w, y))

    def _draw_snake(self, surface, pygame, body, dead=False):
        for i, (bx, by) in enumerate(body):
            rect = pygame.Rect(bx * CELL + 1, by * CELL + 1, CELL - 2, CELL - 2)
            color = C_DEAD if dead else (C_SNAKE_H if i == 0 else C_SNAKE_B)
            pygame.draw.rect(surface, color, rect, border_radius=4)

    def _draw_food(self, surface, pygame, food):
        fx, fy = food
        cx = fx * CELL + CELL // 2
        cy = fy * CELL + CELL // 2
        pygame.draw.circle(surface, C_FOOD, (cx, cy), CELL // 2 - 2)

    def _draw_hud(self, surface, pygame, score, episode):
        txt = self.font.render(
            f"Score: {score}   Episode: {episode}", True, C_TEXT
        )
        surface.blit(txt, (6, 4))

    def run(self):
        pygame = self._init_pygame()
        import pygame as pg

        obs      = self.env.reset()
        direction = "RIGHT"
        episode   = 1
        running   = True

        while running:
            # ── Event handling ────────────────────────────────────────
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    if self.human_mode:
                        key_map = {
                            pg.K_UP: "UP", pg.K_w: "UP",
                            pg.K_DOWN: "DOWN", pg.K_s: "DOWN",
                            pg.K_LEFT: "LEFT", pg.K_a: "LEFT",
                            pg.K_RIGHT: "RIGHT", pg.K_d: "RIGHT",
                        }
                        if event.key in key_map:
                            direction = key_map[event.key]

            # ── Neural AI step ────────────────────────────────────────
            if not self.human_mode and self.net is not None:
                self.net.reset_counts()
                self.encoder.encode(*obs, direction)
                for _ in range(STEPS_PER_FRAME):
                    self.net.step()
                direction = self.decoder.decode(direction)

            # ── Game step ─────────────────────────────────────────────
            obs, done = self.env.step(direction)
            head, food, body, direction = obs

            # ── Render ────────────────────────────────────────────────
            self.screen.fill(C_BG)
            game_surf = self.renderer.render(obs, self.env.score, done)
            self.screen.blit(game_surf, (0, 0))

            # Neural dashboard
            if self.dashboard is not None:
                self.dashboard.update()
                rgb = self.dashboard.get_rgb_array()
                dash_surf = pg.surfarray.make_surface(
                    rgb.transpose(1, 0, 2)
                )
                dash_y = (self.game_h - dash_surf.get_height()) // 2
                self.screen.blit(dash_surf, (self.game_w, max(0, dash_y)))

            pg.display.flip()
            self.clock.tick(self.fps)

            # ── Episode reset ─────────────────────────────────────────
            if done:
                pg.time.wait(600)
                obs = self.env.reset()
                direction = "RIGHT"
                episode += 1

        pg.quit()
        sys.exit(0)
