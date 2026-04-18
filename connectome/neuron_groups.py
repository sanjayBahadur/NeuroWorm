"""
Named neuron group definitions for C. elegans connectome.

Biological classifications based on WormAtlas and published literature.
Neuron roles are approximations used for game sensory/motor mapping.
"""

# ─── SENSORY NEURONS ────────────────────────────────────────────────────────
# Food-detecting (attractive chemosensory)
FOOD_LEFT   = ["ASEL"]
FOOD_RIGHT  = ["ASER"]
FOOD_BOOST  = ["AWAL", "AWAR"]  # attractive-odor amplifiers

# Danger-detecting (nociceptive / avoidance)
DANGER_LEFT  = ["ASHL"]
DANGER_RIGHT = ["ASHR"]
DANGER_FRONT = ["AVM", "ALML", "ALMR"]    # anterior mechanosensory

# ─── COMMAND INTERNEURONS ────────────────────────────────────────────────────
FORWARD_CMD  = ["AVBL", "AVBR"]           # drive forward locomotion
REVERSAL_CMD = ["AVAL", "AVAR"]           # drive backward / reversal

# ─── MOTOR NEURONS ───────────────────────────────────────────────────────────
# Dorsal body motor neurons (forward circuit)
DORSAL_MOTOR = ["DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"]

# Ventral body motor neurons (forward circuit)
VENTRAL_MOTOR = [
    "VB1", "VB2", "VB3", "VB4", "VB5",
    "VB6", "VB7", "VB8", "VB9", "VB10", "VB11",
]

# Head motor neurons — directional
HEAD_LEFT  = ["SMDDL"]
HEAD_RIGHT = ["SMDDR"]
HEAD_DORSAL   = ["SMDVL", "SMDVR"]

# Neck motor — used as a tiebreaker
NECK_MOTOR = ["RIVL", "RIVR"]

# ─── ALL GROUPS (for UI / heatmap display) ───────────────────────────────────
GROUP_LABELS = {
    "Food Sensors":       FOOD_LEFT + FOOD_RIGHT + FOOD_BOOST,
    "Danger Sensors":     DANGER_LEFT + DANGER_RIGHT + DANGER_FRONT,
    "Forward Command":    FORWARD_CMD,
    "Reversal Command":   REVERSAL_CMD,
    "Dorsal Motor":       DORSAL_MOTOR,
    "Ventral Motor":      VENTRAL_MOTOR,
    "Head Motor L/R":     HEAD_LEFT + HEAD_RIGHT,
}
