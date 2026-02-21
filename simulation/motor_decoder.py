"""
Motor Decoder: reads C. elegans motor neuron firing rates → snake direction.

Decision hierarchy:
  1. Reversal check:     AVA >> AVB  →  force 90° turn (can't go backward)
  2. Dorsal/ventral:     DB >> VB    →  turn UP
                         VB >> DB    →  turn DOWN
  3. Head L/R:           SMDDL > SMDDR →  turn LEFT
                         SMDDR > SMDDL →  turn RIGHT
  4. Default:            continue in current direction
"""

import numpy as np
from simulation.lif_network import CElegansLIF
from connectome.neuron_groups import (
    FORWARD_CMD, REVERSAL_CMD, DORSAL_MOTOR, VENTRAL_MOTOR,
    HEAD_LEFT, HEAD_RIGHT
)

DIRECTION_LIST = ["UP", "RIGHT", "DOWN", "LEFT"]
DIRECTION_MAP  = {d: i for i, d in enumerate(DIRECTION_LIST)}


class MotorDecoder:
    """
    Decodes motor neuron ensemble activity into a discrete game action.

    All thresholds are tunable — these are optimized by the ES in training.
    """

    def __init__(
        self,
        network: CElegansLIF,
        rev_thresh:  float = 0.60,
        dors_thresh: float = 0.55,
    ):
        """
        Args:
            network:      The CElegansLIF instance to read from.
            rev_thresh:   If AVA/(AVA+AVB) > this, trigger a turn (reversal).
            dors_thresh:  If DB/(DB+VB) > this, turn UP.
                          If VB/(DB+VB) > this, turn DOWN.
        """
        self.net         = network
        self.rev_thresh  = rev_thresh
        self.dors_thresh = dors_thresh

    def _rate(self, names: list[str]) -> float:
        """Average firing rate (Hz) across a list of neuron names."""
        return self.net.group_firing_rate(names)

    def decode(self, current_direction: str) -> str:
        """
        Read motor state and return the next movement direction.

        Args:
            current_direction: Current snake heading ('UP','DOWN','LEFT','RIGHT').
        Returns:
            Next direction string.
        """
        fwd_r  = self._rate(FORWARD_CMD)
        rev_r  = self._rate(REVERSAL_CMD)
        db_r   = self._rate(DORSAL_MOTOR)
        vb_r   = self._rate(VENTRAL_MOTOR)
        hl_r   = self._rate(HEAD_LEFT)
        hr_r   = self._rate(HEAD_RIGHT)

        cur_idx = DIRECTION_MAP[current_direction]

        # ── 1. Reversal check ─────────────────────────────────────────
        fwd_rev_total = fwd_r + rev_r
        if fwd_rev_total > 1e-4 and (rev_r / fwd_rev_total) > self.rev_thresh:
            # Worm wants to reverse — map to 90° turn instead
            turn = -1 if hl_r > hr_r else 1  # left or right
            return DIRECTION_LIST[(cur_idx + turn) % 4]

        # ── 2. Dorsal / ventral ───────────────────────────────────────
        dv_total = db_r + vb_r
        if dv_total > 1e-4:
            d_ratio = db_r / dv_total
            if d_ratio > self.dors_thresh:
                return "UP"
            if (1.0 - d_ratio) > self.dors_thresh:
                return "DOWN"

        # ── 3. Head L / R ─────────────────────────────────────────────
        lr_total = hl_r + hr_r
        if lr_total > 1e-4:
            if hl_r / lr_total > 0.6:
                return "LEFT"
            if hr_r / lr_total > 0.6:
                return "RIGHT"

        # ── 4. No strong signal — hold course ─────────────────────────
        return current_direction
