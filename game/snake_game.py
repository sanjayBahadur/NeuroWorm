"""
NeuroWorm Snake — Interactive Pygame Game Loop with 10-Worm Swarm Simulation.

Option A Implementation:
  - Direct Synaptic Weight Neuroevolution (36 biological synapses).
  - High-School-Friendly Brain Inspector panel:
      * [🧠 LIVE THOUGHTS]: 3-stage thought stream (Senses -> Decisions -> Muscles)
        with plain-English thought summary.
      * [🔬 SYNAPSE LEARNING]: Top biological pathways evolved, with conductance (nS)
        and intuitive trait descriptions.
      * Live 10-Worm Swarm Roster with click-to-inspect.
  - Procedural vermiform ribbon kinematics with bioluminescent nerve pulses.
  - Multi-seed spanning across 5 configurations + Top 3 champion elitism.
  - Sleek control deck: Fast Forward 100x (F), Evolve Next Gen (N), Pause (SPC),
    Faster/Slower (+/-), Swarm/Solo toggle (M), Reset GA (R), Fullscreen (F11).
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
from simulation.synaptic_circuit import SynapticCircuit
from training.genetic_algorithm import GeneticOptimizer, DEFAULT_GENOME, EVAL_SEEDS
from viz.brain_inspector import BrainInspector

from game.layout import (
    GRID_SIZE, CELL, WALL, FIELD_PX, ARENA_PX,
    GAME_W, GAME_H, PANEL_W, VIRTUAL_W, VIRTUAL_H,
    cell_to_px, cell_center_px
)
CELL_SIZE  = CELL

# Color Palette
C_PANEL_BG = (18, 12, 8)
C_BORDER   = (65, 45, 28)
C_HEADER   = (255, 215, 90)
C_TEXT     = (230, 215, 190)
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
    """An individual biological worm organism in the population."""

    def __init__(
        self,
        worm_id: int,
        genome: np.ndarray,
        color: tuple,
        grid_size: int = 20,
        seed: int = 42,
        starve_limit: int = 160,
    ):
        self.id = worm_id
        self.genome = genome.copy()
        self.color = color
        self.grid_size = grid_size
        self.starve_limit = starve_limit
        self.death_cause: str | None = None
        self.env = SnakeEnv(grid_size=grid_size, seed=seed)
        self.net = build_network()
        self.circuit = SynapticCircuit(self.net)
        self.encoder = SensoryEncoder(self.net, grid_size)
        self.decoder = MotorDecoder(self.net)
        self.circuit.apply_genome(genome)
        self.reset(seed=seed)

    def set_genome(self, genome: np.ndarray):
        self.genome = genome.copy()
        self.circuit.apply_genome(self.genome)

    def reset(self, seed: int = None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        self.obs = self.env.reset()
        self.direction = "RIGHT"
        self.alive = True
        self.death_cause = None
        self.steps = 0
        self.score = 0
        self.steps_since_food = 0
        self.min_dist_to_food = 999.0
        self.net.V[:] = -65e-3
        self.net.g_syn[:] = 0.0
        self.net.spikes[:] = False
        self.net.refractory_timer[:] = 0.0
        self.net.reset_counts()
        self.circuit.apply_genome(self.genome)

    def step(self, shared_food: tuple[int, int] = None) -> bool:
        if not self.alive:
            return False

        head, food, body, cur_dir = self.obs
        if shared_food is not None:
            food = shared_food
            self.env.food = shared_food

        dist = np.hypot(food[0] - head[0], food[1] - head[1])
        if dist < self.min_dist_to_food:
            self.min_dist_to_food = dist

        self.net.reset_counts()
        self.encoder.encode(head, food, body, self.direction)
        for _ in range(8):
            self.net.step()

        self.direction = self.decoder.decode(self.direction)
        self.obs, done = self.env.step(self.direction)
        self.direction = self.env.direction
        self.steps += 1
        self.steps_since_food += 1

        ate = False
        if self.env.score > self.score:
            self.score = self.env.score
            self.steps_since_food = 0
            self.min_dist_to_food = 999.0
            ate = True

        if done:
            self.alive = False
            self.death_cause = self.env.death_cause
        elif self.steps_since_food > self.starve_limit:
            self.alive = False
            self.death_cause = "starved"

        return ate

    @property
    def fitness(self) -> float:
        return (self.score * 300.0) + (self.steps * 1.2) + max(0.0, 20.0 - self.min_dist_to_food) * 2.5


class SnakeGame:
    """Main NeuroWorm game with 10-worm swarm simulation and interactive Brain Inspector."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        fps: int = 12,
        human_mode: bool = False,
        show_neural: bool = True,
        theta: np.ndarray = None,
        scale: float = None,
        fullscreen: bool = False,
        seed: int | None = None,
    ):
        pg.init()
        self.grid_size = grid_size
        self.fps = fps
        self.human_mode = human_mode
        self.show_neural = show_neural
        self.scale = scale
        self.fullscreen = fullscreen
        self.paused = False
        self.fast_forwarding = False
        self.inspector_tab = "flow"  # "flow" or "synapses"
        self.solo_mode = False
        self.human_dir = "RIGHT"
        self.rng = np.random.default_rng(seed)

        self.ga = GeneticOptimizer(pop_size=10, grid_size=grid_size)

        if theta is not None and len(theta) == len(DEFAULT_GENOME):
            self.ga.best_individual.genome = np.array(theta, dtype=np.float32)
            self.ga.population[0].genome = np.array(theta, dtype=np.float32)

        self.seed_idx = 0
        init_seed = EVAL_SEEDS[self.seed_idx]

        # Initialize all 10 candidate biological organisms
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
        self.state: str = "RUNNING"
        self.gen_end_timer: float = 0.0
        self.gen_end_duration: float = 1.2
        self.gen_end_banner_text: str = ""
        self.history: list[dict] = []
        self.shared_food = self._place_shared_food()
        for w in self.worms:
            w.env.food = self.shared_food
            w.obs = (w.obs[0], self.shared_food, w.obs[2], w.obs[3])

        self.inspector = BrainInspector(width=PANEL_W, height=VIRTUAL_H)
        self.renderer = Renderer(cell_size=CELL_SIZE, grid_size=self.grid_size)
        self.buttons = []
        self._init_ui_buttons()

    def _place_shared_food(self) -> tuple[int, int]:
        """Place shared nutrient spore across the full 0..g-1 grid, avoiding all active and drawn bodies."""
        occupied = set()
        for w in self.worms:
            if hasattr(w, "obs") and len(w.obs) > 2 and w.obs[2]:
                for pt in w.obs[2]:
                    occupied.add(pt)
        for _ in range(200):
            pos = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
            if pos not in occupied:
                return pos
        return (self.grid_size // 2, self.grid_size // 2)

    def _init_ui_buttons(self):
        bx = GAME_W + 16
        w = PANEL_W - 32
        h = 24
        self.btn_ff    = Button((bx, 480, w, h), "⏩ Fast Forward 100x", "F")
        self.btn_next  = Button((bx, 510, w, h), "⏭ Evolve Next Gen",   "N")
        self.btn_pause = Button((bx, 540, w, h), "⏯ Pause / Play",       "SPC")
        btn_w2 = (w - 8) // 2
        self.btn_spdup = Button((bx, 570, btn_w2, h), "➕ Faster", "+")
        self.btn_spddn = Button((bx + btn_w2 + 8, 570, btn_w2, h), "➖ Slower", "-")
        self.btn_mode  = Button((bx, 600, w, h), "👥 Swarm / Solo Mode", "M")
        self.btn_reset = Button((bx, 630, w, h), "🔄 Reset GA Population", "R")

        self.buttons = [
            self.btn_ff, self.btn_next, self.btn_pause,
            self.btn_spdup, self.btn_spddn, self.btn_mode, self.btn_reset
        ]

    def _init_pygame(self):
        pg.init()
        pg.display.set_caption("NeuroWorm Snake — 10 Biological Worms Evolving")

        info = pg.display.Info()
        if self.scale is None:
            if info.current_w >= 2560:
                self.scale = 1.75   # 1680 x 1120 on 3K/4K displays
            elif info.current_w >= 1920:
                self.scale = 1.5    # 1440 x 960 on 1080p
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
        self.font_body  = pg.font.SysFont("monospace", 10)
        self.font_btn   = pg.font.SysFont("monospace", 10, bold=True)

        self.renderer = Renderer(cell_size=CELL_SIZE, grid_size=self.grid_size)
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
        self.shared_food = self._place_shared_food()
        for w in self.worms:
            w.reset(seed=seed)
            w.env.food = self.shared_food
            w.obs = (w.obs[0], self.shared_food, w.obs[2], w.obs[3])
        self.generation_steps = 0
        self.state = "RUNNING"
        self.gen_end_timer = 0.0

        # Record GA fast-forward result into history (A5)
        self.history.append({
            "gen": self.ga.generation,
            "best_score": int(res["best_score"]),
            "avg_score": round(float(np.mean([ind.score for ind in self.ga.population])), 2),
            "best_fitness": round(float(res["best_fitness"]), 2),
        })

    def _draw_loading_notice(self, text):
        if not hasattr(self, "virtual_screen") or self.virtual_screen is None:
            return
        box = pg.Rect(GAME_W // 2 - 180, GAME_H // 2 - 30, 360, 60)
        pg.draw.rect(self.virtual_screen, (30, 16, 10), box, border_radius=6)
        pg.draw.rect(self.virtual_screen, C_HEADER, box, 2, border_radius=6)
        font = getattr(self, "font_title", None)
        if font is None:
            font = pg.font.SysFont("monospace", 14, bold=True)
        txt = font.render(text, True, C_HEADER)
        self.virtual_screen.blit(txt, (box.centerx - txt.get_width() // 2, box.centery - txt.get_height() // 2))

    def _record_generation_history(self):
        """Record real swarm performance at generation boundaries."""
        if self.history and self.history[-1]["gen"] == self.ga.generation:
            return
        scores = [w.score for w in self.worms]
        fits = [w.fitness for w in self.worms]
        best_score = max(scores) if scores else 0
        avg_score = float(np.mean(scores)) if scores else 0.0
        best_fitness = max(fits) if fits else 0.0
        self.history.append({
            "gen": self.ga.generation,
            "best_score": int(best_score),
            "avg_score": round(avg_score, 2),
            "best_fitness": round(best_fitness, 2),
        })

    def start_generation_end(self):
        """Non-blocking transition into GEN_END state (A4)."""
        self.state = "GEN_END"
        self.gen_end_timer = 0.0
        for w in self.worms:
            if w.alive:
                w.death_cause = "timeout"
                w.alive = False

        scores = [w.score for w in self.worms]
        best_score = max(scores) if scores else 0
        self.gen_end_banner_text = f"Generation {self.ga.generation} finished — best: {best_score} eggs"
        self._record_generation_history()

    def _draw_gen_end_banner(self):
        """Draw non-blocking generation end banner across the arena."""
        box_w = 420
        box_h = 76
        bx = (GAME_W - box_w) // 2
        by = (GAME_H - box_h) // 2
        banner_rect = pg.Rect(bx, by, box_w, box_h)

        card_surf = pg.Surface((box_w, box_h), pg.SRCALPHA)
        card_surf.fill((18, 12, 8, 235))
        self.virtual_screen.blit(card_surf, (bx, by))
        pg.draw.rect(self.virtual_screen, (200, 160, 60), banner_rect, 2, border_radius=8)

        font_title = getattr(self, "font_title", None) or pg.font.SysFont("monospace", 14, bold=True)
        font_body = getattr(self, "font_body", None) or pg.font.SysFont("monospace", 10)

        title_lbl = font_title.render(self.gen_end_banner_text, True, (255, 235, 140))
        rem_sec = max(0.0, self.gen_end_duration - self.gen_end_timer)
        sub_lbl = font_body.render(f"Evolving next generation... {rem_sec:.1f}s (press N to skip)", True, (200, 184, 150))

        self.virtual_screen.blit(title_lbl, (bx + (box_w - title_lbl.get_width()) // 2, by + 16))
        self.virtual_screen.blit(sub_lbl, (bx + (box_w - sub_lbl.get_width()) // 2, by + 44))

    def _evolve_and_next_generation(self):
        self._record_generation_history()

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

        self.shared_food = self._place_shared_food()
        # Reset all 10 worms
        for i, w in enumerate(self.worms):
            w.set_genome(self.ga.population[i].genome)
            w.reset(seed=seed)
            w.env.food = self.shared_food
            w.obs = (w.obs[0], self.shared_food, w.obs[2], w.obs[3])

        self.generation_steps = 0
        self.state = "RUNNING"
        self.gen_end_timer = 0.0

    def _present_frame(self):
        if not hasattr(self, "screen") or self.screen is None or not hasattr(self, "virtual_screen") or self.virtual_screen is None:
            return
        scale = min(self.win_w / VIRTUAL_W, self.win_h / VIRTUAL_H)
        rendered_w = int(VIRTUAL_W * scale)
        rendered_h = int(VIRTUAL_H * scale)
        offset_x = (self.win_w - rendered_w) // 2
        offset_y = (self.win_h - rendered_h) // 2

        scaled_surf = pg.transform.smoothscale(self.virtual_screen, (rendered_w, rendered_h))
        self.screen.fill((12, 7, 4))
        self.screen.blit(scaled_surf, (offset_x, offset_y))
        pg.display.flip()

    def step_simulation(self) -> bool:
        """
        Advance one simulation tick for all active organisms.
        Worms are processed in a per-tick random order to prevent bias.
        When any worm eats an egg, the egg is immediately relocated so no double-eat can occur.
        Returns True if all organisms are dead or generation cap is reached.
        """
        all_dead = True
        order = self.rng.permutation(len(self.worms))

        for idx in order:
            w = self.worms[idx]
            if not w.alive:
                continue
            if self.solo_mode and idx != self.focused_idx:
                continue

            all_dead = False
            if self.human_mode and idx == self.focused_idx:
                w.direction = self.human_dir
                w.env.food = self.shared_food
                w.obs, done = w.env.step(w.direction)
                w.direction = w.env.direction
                w.steps += 1
                if w.env.score > w.score:
                    w.score = w.env.score
                    w.steps_since_food = 0
                    self.renderer.add_food_burst(w.obs[0], w.color)
                    self.shared_food = self._place_shared_food()
                    for other in self.worms:
                        other.env.food = self.shared_food
                        other.obs = (other.obs[0], self.shared_food, other.obs[2], other.obs[3])
                if done:
                    w.alive = False
                    w.death_cause = w.env.death_cause
            else:
                ate = w.step(shared_food=self.shared_food)
                if ate:
                    self.renderer.add_food_burst(w.obs[0], w.color)
                    self.shared_food = self._place_shared_food()
                    for other in self.worms:
                        other.env.food = self.shared_food
                        other.obs = (other.obs[0], self.shared_food, other.obs[2], other.obs[3])

        self.generation_steps += 1
        return all_dead or (self.generation_steps >= 400)

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
                        self.inspector_tab = "synapses" if self.inspector_tab == "flow" else "flow"
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
                        self.shared_food = self._place_shared_food()
                        for i, w in enumerate(self.worms):
                            w.set_genome(self.ga.population[i].genome)
                            w.reset(seed=EVAL_SEEDS[0])
                            w.env.food = self.shared_food
                            w.obs = (w.obs[0], self.shared_food, w.obs[2], w.obs[3])
                        self.generation_steps = 0
                        self.state = "RUNNING"
                        self.gen_end_timer = 0.0
                        self.history.clear()
                    elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9):
                        self.focused_idx = event.key - pg.K_1
                    elif event.key == pg.K_0:
                        self.focused_idx = 9
                    elif event.key == pg.K_LEFT:
                        if self.human_mode:
                            self.human_dir = "LEFT"
                        else:
                            self.focused_idx = (self.focused_idx - 1) % len(self.worms)
                    elif event.key == pg.K_RIGHT:
                        if self.human_mode:
                            self.human_dir = "RIGHT"
                        else:
                            self.focused_idx = (self.focused_idx + 1) % len(self.worms)
                    elif event.key == pg.K_UP and self.human_mode:
                        self.human_dir = "UP"
                    elif event.key == pg.K_DOWN and self.human_mode:
                        self.human_dir = "DOWN"

                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    px = virtual_mouse_pos[0] - GAME_W
                    py = virtual_mouse_pos[1]

                    # Panel interaction
                    if virtual_mouse_pos[0] >= GAME_W:
                        # Tab clicks
                        if self.inspector.tab_flow_rect.collidepoint((px, py)):
                            self.inspector_tab = "flow"
                        elif self.inspector.tab_syn_rect.collidepoint((px, py)):
                            self.inspector_tab = "synapses"

                        # Roster row clicks
                        for i in range(len(self.worms)):
                            row_rect = pg.Rect(16, 286 + i * 16 - 1, PANEL_W - 32, 16)
                            if row_rect.collidepoint((px, py)):
                                self.focused_idx = i
                                break

                    # Button clicks
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
                        self.shared_food = self._place_shared_food()
                        for i, w in enumerate(self.worms):
                            w.set_genome(self.ga.population[i].genome)
                            w.reset(seed=EVAL_SEEDS[0])
                            w.env.food = self.shared_food
                            w.obs = (w.obs[0], self.shared_food, w.obs[2], w.obs[3])
                        self.generation_steps = 0
                        self.state = "RUNNING"
                        self.gen_end_timer = 0.0
                        self.history.clear()

            # ── Simulation Step (All 10 Organisms) ────────────────────
            if self.state == "RUNNING" and not self.paused:
                gen_over = self.step_simulation()
                if gen_over:
                    self.start_generation_end()
            elif self.state == "GEN_END":
                dt = 1.0 / max(1, self.fps)
                self.gen_end_timer += dt
                if self.gen_end_timer >= self.gen_end_duration:
                    self._evolve_and_next_generation()

            # ── Drawing on Virtual Surface ────────────────────────────
            self.virtual_screen.fill(C_PANEL_BG)

            # 1. Left: Game Area with all 10 worms
            gen_tag = f"GEN #{self.ga.generation} ({'SOLO' if self.solo_mode else 'SWARM'})"
            game_surf = self.renderer.render_multi_worms(
                self.worms,
                self.shared_food,
                gen_info=gen_tag,
                focused_idx=self.focused_idx,
                solo_mode=self.solo_mode,
            )
            self.virtual_screen.blit(game_surf, (0, 0))

            if self.state == "GEN_END":
                self._draw_gen_end_banner()

            # 2. Right: Interactive Brain Inspector Dashboard
            self._draw_dashboard(virtual_mouse_pos)

            # 3. Present upscaled frame to user window
            self._present_frame()

            self.clock.tick(self.fps)

        pg.quit()
        sys.exit(0)

    def _draw_dashboard(self, virtual_mouse_pos):
        px = virtual_mouse_pos[0] - GAME_W
        py = virtual_mouse_pos[1]

        fw = self.worms[self.focused_idx]
        inspector_surf = self.inspector.render(
            worm=fw,
            all_worms=self.worms,
            generation=self.ga.generation,
            view_mode=self.inspector_tab,
            mouse_pos=(px, py),
            solo_mode=self.solo_mode,
        )
        self.virtual_screen.blit(inspector_surf, (GAME_W, 0))

        # Interactive Buttons
        for btn in self.buttons:
            btn.draw(self.virtual_screen, self.font_btn, is_hovered=btn.rect.collidepoint(virtual_mouse_pos))

        # Footer hints
        hint1 = self.font_body.render("1-0/Click: Inspect | TAB: View Mode", True, C_MUTED)
        hint2 = self.font_body.render("M: Swarm/Solo | F11: Fullscreen | SPC", True, C_MUTED)
        self.virtual_screen.blit(hint1, (GAME_W + 16, 665))
        self.virtual_screen.blit(hint2, (GAME_W + 16, 680))
