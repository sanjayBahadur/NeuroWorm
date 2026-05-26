"""
Motor Decoder: translates C. elegans motor neuron activity into game actions.

Reads firing rates from:
  - AVBL / AVBR: Forward locomotion command
  - AVAL / AVAR: Reversal / avoidance command
  - SMDDL / SMDDR: Head motor neurons (lateral turns)
  - DB01–DB07: Dorsal body motor neurons
  - VB01–VB11: Ventral body motor neurons
  - ASHL / ASHR: Lateral danger discrimination
"""

import numpy as np
from simulation.lif_network import CElegansLIF
from connectome.neuron_groups import (
    FORWARD_CMD, REVERSAL_CMD, DORSAL_MOTOR, VENTRAL_MOTOR,
    HEAD_LEFT, HEAD_RIGHT, DANGER_LEFT, DANGER_RIGHT, FOOD_LEFT, FOOD_RIGHT
)

DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
DIRECTION_MAP = {d: i for i, d in enumerate(DIRECTIONS)}


class MotorDecoder:
    """
    Decodes motor and command interneuron firing rates into discrete
    heading decisions.
    """

    def __init__(
        self,
        network: CElegansLIF,
        rev_thresh: float = 0.50,
        turn_thresh: float = 0.55,
        dv_thresh: float = 0.55,
    ):
        self.net = network
        self.rev_thresh = rev_thresh
        self.turn_thresh = turn_thresh
        self.dv_thresh = dv_thresh

    def update_params(self, params: dict):
        """Update decoding thresholds from genetic algorithm genome."""
        if "rev_thresh" in params:
            self.rev_thresh = float(params["rev_thresh"])
        if "turn_thresh" in params:
            self.turn_thresh = float(params["turn_thresh"])
        if "dv_thresh" in params:
            self.dv_thresh = float(params["dv_thresh"])

    def _rate(self, names: list[str]) -> float:
        return self.net.group_firing_rate(names)

    def decode(self, current_direction: str, context: dict = None) -> str:
        """
        Read neural ensemble state and return next movement direction.
        
        Args:
            current_direction: Current snake heading ('UP','DOWN','LEFT','RIGHT').
            context: Optional dictionary containing sensory hints (e.g. egocentric obstacles).
        """
        fwd_r = self._rate(FORWARD_CMD)
        rev_r = self._rate(REVERSAL_CMD)
        db_r  = self._rate(DORSAL_MOTOR)
        vb_r  = self._rate(VENTRAL_MOTOR)
        hl_r  = self._rate(HEAD_LEFT)
        hr_r  = self._rate(HEAD_RIGHT)
        dl_r  = self._rate(DANGER_LEFT)
        dr_r  = self._rate(DANGER_RIGHT)
        fl_r  = self._rate(FOOD_LEFT)
        fr_r  = self._rate(FOOD_RIGHT)

        cur_idx = DIRECTION_MAP[current_direction]
        left_dir  = DIRECTIONS[(cur_idx - 1) % 4]
        right_dir = DIRECTIONS[(cur_idx + 1) % 4]

        total_fwd_rev = fwd_r + rev_r

        # ── 1. Reversal / Emergency Evasion ───────────────────────────
        # When AVAL/AVAR is activated (e.g. obstacle detected ahead via evolved synapses)
        if total_fwd_rev > 1e-4 and (rev_r / total_fwd_rev) > self.rev_thresh:
            left_penalty  = dl_r * 2.0 - fl_r
            right_penalty = dr_r * 2.0 - fr_r

            if left_penalty > right_penalty:
                next_dir = right_dir
            elif right_penalty > left_penalty:
                next_dir = left_dir
            else:
                next_dir = left_dir if hl_r >= hr_r else right_dir

            thought = "Wall Alert! Emergency Avoidance Turn"
            self._record_telemetry(fwd_r, rev_r, hl_r, hr_r, next_dir, thought)
            return next_dir

        # ── 2. Head Motor Lateral Steering (SMDDL vs SMDDR) ───────────
        total_lr = hl_r + hr_r + fl_r + fr_r
        if total_lr > 1e-4:
            left_drive  = hl_r + fl_r * 1.5 - dl_r * 2.0
            right_drive = hr_r + fr_r * 1.5 - dr_r * 2.0

            ratio = (left_drive + 1e-4) / (abs(left_drive) + abs(right_drive) + 2e-4)
            if ratio > self.turn_thresh:
                thought = "Food Scent on Left — Steering Left"
                self._record_telemetry(fwd_r, rev_r, hl_r, hr_r, left_dir, thought)
                return left_dir
            if (1.0 - ratio) > self.turn_thresh:
                thought = "Food Scent on Right — Steering Right"
                self._record_telemetry(fwd_r, rev_r, hl_r, hr_r, right_dir, thought)
                return right_dir

        # ── 3. Dorsal vs Ventral Motor Activation ─────────────────────
        total_dv = db_r + vb_r
        if total_dv > 1e-4:
            d_ratio = db_r / total_dv
            if d_ratio > self.dv_thresh and current_direction != "DOWN":
                thought = "Dorsal Wave Active — Steering UP"
                self._record_telemetry(fwd_r, rev_r, hl_r, hr_r, "UP", thought)
                return "UP"
            elif (1.0 - d_ratio) > self.dv_thresh and current_direction != "UP":
                thought = "Ventral Wave Active — Steering DOWN"
                self._record_telemetry(fwd_r, rev_r, hl_r, hr_r, "DOWN", thought)
                return "DOWN"

        # ── 4. Default: Forward Crawling ──────────────────────────────
        thought = "Forward Crawl — Tracking Bacterial Trail"
        self._record_telemetry(fwd_r, rev_r, hl_r, hr_r, current_direction, thought)
        return current_direction

    def _record_telemetry(self, fwd_r, rev_r, hl_r, hr_r, action, thought):
        tot_cmd = max(1e-4, fwd_r + rev_r)
        tot_head = max(1e-4, hl_r + hr_r)
        self.telemetry = {
            "decision_fwd": int(np.clip((fwd_r / tot_cmd) * 100, 0, 100)),
            "decision_rev": int(np.clip((rev_r / tot_cmd) * 100, 0, 100)),
            "muscle_steer_l": int(np.clip((hl_r / tot_head) * 100, 0, 100)),
            "muscle_steer_r": int(np.clip((hr_r / tot_head) * 100, 0, 100)),
            "action": action,
            "thought_summary": thought,
        }
