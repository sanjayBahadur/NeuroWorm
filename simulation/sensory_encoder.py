"""
Sensory Encoder: biologically grounded sensory transduction for C. elegans.

Implements real biological foraging & avoidance mechanisms:
  1. Chemical Diffusion Plume (Chemotaxis):
     - Food sources emit a 2D diffusion field in soil.
     - Left amphid (ASEL) and right amphid (ASER) sample spatial concentration differences (Klinotaxis).
     - Temporal concentration change dC/dt controls forward drive (AWA/AVB) vs pirouette reorientation.
  2. Multi-Ray Mechanosensory Whiskers (Touch & Avoidance):
     - Anterior nose touch (AVM, FLPL/R): Ray cast straight ahead detects distance to wall/body.
     - Lateral cuticle mechanosensors (ASHL/R, ALML/R): Rays cast left and right detect flank obstacles.
"""

import math
import numpy as np
from simulation.lif_network import CElegansLIF
from connectome.neuron_groups import (
    FOOD_LEFT, FOOD_RIGHT, FOOD_BOOST, DANGER_LEFT, DANGER_RIGHT, DANGER_FRONT,
    REVERSAL_CMD, FORWARD_CMD
)

DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
DX = {"RIGHT": 1, "LEFT": -1, "UP": 0,  "DOWN": 0}
DY = {"RIGHT": 0, "LEFT": 0,  "UP": -1, "DOWN": 1}


class SensoryEncoder:
    """
    Translates chemical diffusion plumes and mechanosensory touch rays
    into micro-current injections (nA) into 302-neuron connectome.
    """

    def __init__(
        self,
        network: CElegansLIF,
        grid_size: int = 20,
        i_food: float = 4.5,
        i_danger_front: float = 14.0,
        i_danger_side: float = 6.0,
        i_awa: float = 2.0,
        i_front_food: float = 4.0,
    ):
        self.net = network
        self.grid_size = grid_size
        self.i_food = i_food
        self.i_danger_front = i_danger_front
        self.i_danger_side = i_danger_side
        self.i_awa = i_awa
        self.i_front_food = i_front_food

        self.last_concentration = 0.0

        # Telemetry for the live Sensory Radar display
        self.telemetry = {
            "front_dist": 20,
            "left_dist": 20,
            "right_dist": 20,
            "c_left": 0.0,
            "c_right": 0.0,
            "c_center": 0.0,
            "dc_dt": 0.0,
            "front_hit": (0, 0),
            "left_hit": (0, 0),
            "right_hit": (0, 0),
        }

    def update_params(self, params: dict):
        if "i_food" in params:
            self.i_food = float(params["i_food"])
        if "i_danger_front" in params:
            self.i_danger_front = float(params["i_danger_front"])
        if "i_danger_side" in params:
            self.i_danger_side = float(params["i_danger_side"])
        if "i_awa" in params:
            self.i_awa = float(params["i_awa"])
        if "i_front_food" in params:
            self.i_front_food = float(params["i_front_food"])

    def _cast_ray(self, start_pos, dir_name, body_set, max_range=15):
        """Cast a ray along a direction and return distance and hit coordinate."""
        x, y = start_pos
        dx, dy = DX[dir_name], DY[dir_name]
        g = self.grid_size

        for dist in range(1, max_range + 1):
            nx, ny = x + dx * dist, y + dy * dist
            # Wall boundary check
            if nx < 0 or ny < 0 or nx >= g or ny >= g:
                return dist, (nx, ny)
            # Self-body collision check
            if (nx, ny) in body_set:
                return dist, (nx, ny)

        return max_range, (x + dx * max_range, y + dy * max_range)

    def _concentration_at(self, pos, food_pos):
        """
        Simulate bacterial chemical diffusion field:
        Concentration C(x, y) = 100 / (1 + 0.4 * distance)
        """
        px, py = pos
        fx, fy = food_pos
        dist = math.hypot(fx - px, fy - py)
        return 100.0 / (1.0 + 0.45 * dist)

    def encode(
        self,
        snake_head: tuple[int, int],
        food_pos:   tuple[int, int],
        snake_body: list[tuple[int, int]],
        direction:  str,
        *extra,
    ) -> None:
        if extra:
            direction = extra[0]
        self.net.clear_external()

        hx, hy = snake_head
        fx, fy = food_pos
        g = self.grid_size
        body_set = set(map(tuple, snake_body[1:]))

        cur_idx = DIRECTIONS.index(direction)
        left_dir  = DIRECTIONS[(cur_idx - 1) % 4]
        right_dir = DIRECTIONS[(cur_idx + 1) % 4]

        # ── 1. Mechanosensory Whiskers (Touch Rays) ───────────────────
        front_dist, front_hit = self._cast_ray(snake_head, direction, body_set, max_range=g)
        left_dist,  left_hit  = self._cast_ray(snake_head, left_dir,  body_set, max_range=g)
        right_dist, right_hit = self._cast_ray(snake_head, right_dir, body_set, max_range=g)

        # Immediate danger ahead (Nose touch):
        if front_dist == 1:
            # Physical collision imminent: extreme nose touch (AVM) + reversal command (AVA)
            for n in DANGER_FRONT:
                self.net.inject(n, self.i_danger_front * 1.5)
            for n in REVERSAL_CMD:
                self.net.inject(n, self.i_danger_front * 1.2)
        elif front_dist <= 3:
            # Approaching obstacle: warning touch gradient
            warn = self.i_danger_front * (1.0 / front_dist)
            for n in DANGER_FRONT:
                self.net.inject(n, warn)

        # Flank obstacles:
        if left_dist <= 2:
            l_curr = self.i_danger_side * (1.0 / left_dist)
            for n in DANGER_LEFT:
                self.net.inject(n, l_curr)

        if right_dist <= 2:
            r_curr = self.i_danger_side * (1.0 / right_dist)
            for n in DANGER_RIGHT:
                self.net.inject(n, r_curr)

        # ── 2. Biological Chemical Diffusion Plume (Chemotaxis) ───────
        # Amphid sensors are located bilaterally on the head tip:
        # Left amphid position (offset half a cell to the left)
        # Right amphid position (offset half a cell to the right)
        l_pos = (hx + DX[left_dir] * 0.7,  hy + DY[left_dir] * 0.7)
        r_pos = (hx + DX[right_dir] * 0.7, hy + DY[right_dir] * 0.7)

        c_center = self._concentration_at(snake_head, food_pos)
        c_left   = self._concentration_at(l_pos, food_pos)
        c_right  = self._concentration_at(r_pos, food_pos)

        delta_c = c_left - c_right
        dc_dt   = c_center - self.last_concentration
        self.last_concentration = c_center

        # Klinotaxis steering:
        if delta_c > 0.05:
            # Odor concentration higher on left -> stimulate ASEL
            scale = min(3.0, delta_c * 0.8)
            for n in FOOD_LEFT:
                self.net.inject(n, self.i_food * scale)
        elif delta_c < -0.05:
            # Odor concentration higher on right -> stimulate ASER
            scale = min(3.0, abs(delta_c) * 0.8)
            for n in FOOD_RIGHT:
                self.net.inject(n, self.i_food * scale)

        # If food is directly ahead along forward line:
        f_vec = (fx - hx, fy - hy)
        f_dir = (DX[direction], DY[direction])
        dot   = f_vec[0] * f_dir[0] + f_vec[1] * f_dir[1]
        cross = f_dir[0] * f_vec[1] - f_dir[1] * f_vec[0]
        if dot > 0 and abs(cross) <= 1:
            for n in FORWARD_CMD:
                self.net.inject(n, self.i_front_food)

        # Ambient chemotaxis motivation (AWA)
        for n in FOOD_BOOST:
            self.net.inject(n, self.i_awa)

        # Save telemetry for sensory radar
        self.telemetry = {
            "front_dist": front_dist,
            "left_dist": left_dist,
            "right_dist": right_dist,
            "c_left": c_left,
            "c_right": c_right,
            "c_center": c_center,
            "dc_dt": dc_dt,
            "front_hit": front_hit,
            "left_hit": left_hit,
            "right_hit": right_hit,
            "delta_c": delta_c,
        }
