"""
Sensory Encoder: translates Snake game state into C. elegans neuron injections.

Neuron selection rationale:
  ASEL/ASER   — chemosensory neurons for attractive stimuli (food gradient)
  ASHL/ASHR   — polymodal nociceptors for aversive stimuli (walls, body)
  AWACON/AWAL/AWAR — attractive odor amplifiers (general food-seeking drive)
  AVM/ALML/ALMR   — anterior touch neurons (close-range forward danger)

These mappings are NOT arbitrary: each neuron class evolved to detect
stimuli that correspond to a plausible game-world analog.
"""

import numpy as np
from simulation.lif_network import CElegansLIF
from connectome.neuron_groups import (
    FOOD_LEFT, FOOD_RIGHT, FOOD_BOOST, DANGER_LEFT, DANGER_RIGHT, DANGER_FRONT
)


class SensoryEncoder:
    """
    Translates Snake game state to sensory neuron input currents.

    All current magnitudes are tunable via the constructor — these are the
    parameters optimized by the Evolution Strategy in training/optimize_encoding.py.
    """

    def __init__(
        self,
        network: CElegansLIF,
        grid_size: int,
        i_food: float   = 2.0,
        i_danger: float = 3.5,
        i_awa: float    = 1.0,
    ):
        """
        Args:
            network:    The CElegansLIF instance to inject into.
            grid_size:  Game grid dimension (assumes square grid).
            i_food:     Base current (nA) for food-direction signal.
            i_danger:   Base current (nA) for danger/avoidance signal.
            i_awa:      Constant ambient food-seeking drive current (nA).
        """
        self.net       = network
        self.grid_size = grid_size
        self.i_food    = i_food
        self.i_danger  = i_danger
        self.i_awa     = i_awa

    def encode(
        self,
        snake_head: tuple[int, int],
        food_pos:   tuple[int, int],
        snake_body: list[tuple[int, int]],
        direction:  str,
        *extra,
    ) -> None:
        """
        Compute and inject sensory currents for one game frame.

        Args:
            snake_head:  (x, y) grid coords of snake head.
            food_pos:    (x, y) grid coords of food pellet.
            snake_body:  Full list of (x, y) body segment positions.
            direction:   Current heading: 'UP', 'DOWN', 'LEFT', or 'RIGHT'.
        """
        if extra:
            # Handle callers passing (*obs, direction) where obs has 4 elements
            direction = extra[0]
        self.net.clear_external()

        hx, hy = snake_head
        fx, fy = food_pos
        g      = self.grid_size

        # ── FOOD SIGNALS ─────────────────────────────────────────────────
        # Normalize displacement to [-1, 1]
        dx = np.clip((fx - hx) / g, -1.0, 1.0)
        dy = np.clip((fy - hy) / g, -1.0, 1.0)

        # Horizontal food direction
        if dx > 0:    # food is to the right
            for n in FOOD_RIGHT:
                self.net.inject(n, self.i_food * abs(dx))
        elif dx < 0:  # food is to the left
            for n in FOOD_LEFT:
                self.net.inject(n, self.i_food * abs(dx))

        # Vertical food direction (half weight — less primary)
        if dy > 0:    # food is below
            for n in FOOD_RIGHT:
                self.net.inject(n, self.i_food * abs(dy) * 0.5)
        elif dy < 0:  # food is above
            for n in FOOD_LEFT:
                self.net.inject(n, self.i_food * abs(dy) * 0.5)

        # Ambient food-seeking drive (always on when food exists)
        for n in FOOD_BOOST:
            self.net.inject(n, self.i_awa)

        # ── DANGER SIGNALS ───────────────────────────────────────────────
        body_set = set(map(tuple, snake_body[1:]))  # body without head

        adjacent = {
            'UP':    (hx,     hy - 1),
            'DOWN':  (hx,     hy + 1),
            'LEFT':  (hx - 1, hy),
            'RIGHT': (hx + 1, hy),
        }

        for dir_name, (nx, ny) in adjacent.items():
            is_wall = (nx < 0 or ny < 0 or nx >= g or ny >= g)
            is_body = (nx, ny) in body_set

            if not (is_wall or is_body):
                continue

            # Determine which hemisphere neurons to stimulate
            if dir_name in ("LEFT", "UP"):
                neurons = DANGER_LEFT
            else:
                neurons = DANGER_RIGHT

            for n in neurons:
                self.net.inject(n, self.i_danger)

            # Anterior touch: extra strong signal when danger is straight ahead
            if dir_name == direction:
                for n in DANGER_FRONT:
                    self.net.inject(n, self.i_danger * 1.5)
