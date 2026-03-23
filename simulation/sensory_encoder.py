"""
Sensory Encoder: translates Snake game state into C. elegans neuron injections.

Biological rationale:
  - Egocentric spatial encoding: C. elegans senses danger and food relative to its body.
  - ASEL/ASER: Chemosensory amphid neurons detecting attractants to the left/right.
  - ASHL/ASHR: Polymodal nociceptors detecting aversive stimuli (walls, body) on the left/right.
  - AVM, ALML, ALMR: Mechanosensory touch neurons (anterior nose collision).
  - AWAL/AWAR/AWACON: Chemotaxis forward drive amplifiers.
"""

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
    Translates game state into targeted current injections (nA) into
    sensory neurons and command interneurons.
    """

    def __init__(
        self,
        network: CElegansLIF,
        grid_size: int = 20,
        i_food: float = 3.0,
        i_danger_front: float = 8.0,
        i_danger_side: float = 4.0,
        i_awa: float = 1.5,
        i_front_food: float = 2.0,
    ):
        self.net = network
        self.grid_size = grid_size
        self.i_food = i_food
        self.i_danger_front = i_danger_front
        self.i_danger_side = i_danger_side
        self.i_awa = i_awa
        self.i_front_food = i_front_food

    def update_params(self, params: dict):
        """Update encoding strengths from genetic algorithm genome."""
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

    def encode(
        self,
        snake_head: tuple[int, int],
        food_pos:   tuple[int, int],
        snake_body: list[tuple[int, int]],
        direction:  str,
        *extra,
    ) -> None:
        """
        Inject currents into sensory and command neurons based on
        egocentric game state.
        """
        if extra:
            direction = extra[0]
        self.net.clear_external()

        hx, hy = snake_head
        fx, fy = food_pos
        g = self.grid_size
        body_set = set(map(tuple, snake_body[1:]))

        cur_idx = DIRECTIONS.index(direction)
        left_dir = DIRECTIONS[(cur_idx - 1) % 4]
        right_dir = DIRECTIONS[(cur_idx + 1) % 4]

        # ── 1. Egocentric Danger Proximity (Walls & Body) ─────────────
        def is_blocked(pos):
            x, y = pos
            return (x < 0 or y < 0 or x >= g or y >= g or (x, y) in body_set)

        front_pos = (hx + DX[direction], hy + DY[direction])
        left_pos  = (hx + DX[left_dir],  hy + DY[left_dir])
        right_pos = (hx + DX[right_dir], hy + DY[right_dir])

        front_blocked = is_blocked(front_pos)
        left_blocked  = is_blocked(left_pos)
        right_blocked = is_blocked(right_pos)

        # Danger directly ahead: Nose touch (AVM) + strong avoidance
        if front_blocked:
            for n in DANGER_FRONT:
                self.net.inject(n, self.i_danger_front)
            for n in REVERSAL_CMD:
                self.net.inject(n, self.i_danger_front * 0.8)

        # Danger on the left: left nociceptors (ASHL, ALML)
        if left_blocked:
            for n in DANGER_LEFT:
                self.net.inject(n, self.i_danger_side)

        # Danger on the right: right nociceptors (ASHR, ALMR)
        if right_blocked:
            for n in DANGER_RIGHT:
                self.net.inject(n, self.i_danger_side)

        # ── 2. Egocentric Food Gradient ───────────────────────────────
        fx_diff = fx - hx
        fy_diff = fy - hy

        # Vector math relative to current heading
        # Forward vector (fx_dir, fy_dir)
        f_dx = DX[direction]
        f_dy = DY[direction]

        # 2D cross product: f_dx * fy_diff - f_dy * fx_diff
        # Positive = food is to relative right; Negative = food is to relative left
        cross = f_dx * fy_diff - f_dy * fx_diff
        # Dot product: positive = food is in forward hemisphere
        dot = f_dx * fx_diff + f_dy * fy_diff

        # Scale intensity by normalized displacement
        norm_dist = np.clip(np.hypot(fx_diff, fy_diff) / g, 0.05, 1.0)
        proximity_boost = (1.0 / norm_dist) * 0.3

        if cross > 0:
            # Food is to the right
            for n in FOOD_RIGHT:
                self.net.inject(n, self.i_food * (1.0 + proximity_boost))
        elif cross < 0:
            # Food is to the left
            for n in FOOD_LEFT:
                self.net.inject(n, self.i_food * (1.0 + proximity_boost))

        # Food directly ahead in front
        if dot > 0 and abs(cross) <= 2:
            for n in FORWARD_CMD:
                self.net.inject(n, self.i_front_food)

        # Ambient food seeking motivation
        for n in FOOD_BOOST:
            self.net.inject(n, self.i_awa)
