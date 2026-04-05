"""
Biological Synaptic Circuit for C. elegans Neuroevolution.

Maps the core navigation, foraging, and obstacle-avoidance subcircuit:
  1. Sensory Input (Amphids ASEL/ASER, Nose Touch AVM, Nociceptors ASHL/ASHR, Odor AWAL/AWAR)
  2. Interneuron Decision (Chemotaxis AIY/AIB, Forward AVB, Reversal AVA)
  3. Motor Outputs (Head Steering SMDDL/SMDDR, Neck RIVL/RIVR, Body DB/VB)

The Genetic Algorithm evolves the actual synaptic weights (nS) in this circuit,
enabling true neuroevolution of biological neural dynamics.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# ─── CORE EVOLVABLE SYNAPSES (36 PATHWAYS) ───────────────────────────────────
SYNAPSE_DEFINITIONS: List[Dict[str, Any]] = [
    # ── 1. Chemotaxis (Food Odor & Taste Steering) ───────────────────────────
    {
        "pre": "ASEL", "post": "AIYL",
        "name": "Left Odor -> Forward Drive",
        "label": "Left Food Scent -> Crawl Drive",
        "category": "foraging", "default": 6.5, "min": -5.0, "max": 18.0
    },
    {
        "pre": "ASER", "post": "AIYR",
        "name": "Right Odor -> Forward Drive",
        "label": "Right Food Scent -> Crawl Drive",
        "category": "foraging", "default": 6.5, "min": -5.0, "max": 18.0
    },
    {
        "pre": "ASEL", "post": "AIBL",
        "name": "Left Odor -> Turn Muscle",
        "label": "Left Food Scent -> Left Turn",
        "category": "foraging", "default": 8.0, "min": -5.0, "max": 18.0
    },
    {
        "pre": "ASER", "post": "AIBR",
        "name": "Right Odor -> Turn Muscle",
        "label": "Right Food Scent -> Right Turn",
        "category": "foraging", "default": 8.0, "min": -5.0, "max": 18.0
    },
    {
        "pre": "AIYL", "post": "SMDDL",
        "name": "Chemotaxis -> Left Head Muscle",
        "label": "Left Food Decision -> Steer Left",
        "category": "foraging", "default": 7.0, "min": -5.0, "max": 16.0
    },
    {
        "pre": "AIYR", "post": "SMDDR",
        "name": "Chemotaxis -> Right Head Muscle",
        "label": "Right Food Decision -> Steer Right",
        "category": "foraging", "default": 7.0, "min": -5.0, "max": 16.0
    },
    {
        "pre": "AIBL", "post": "SMDDL",
        "name": "Turn Interneuron -> Left Steering",
        "label": "Turn Circuit -> Steer Left",
        "category": "steering", "default": 8.5, "min": -5.0, "max": 18.0
    },
    {
        "pre": "AIBR", "post": "SMDDR",
        "name": "Turn Interneuron -> Right Steering",
        "label": "Turn Circuit -> Steer Right",
        "category": "steering", "default": 8.5, "min": -5.0, "max": 18.0
    },
    {
        "pre": "AIBL", "post": "RIVL",
        "name": "Turn Circuit -> Left Neck Flex",
        "label": "Sharp Pivot -> Left Neck",
        "category": "steering", "default": 6.0, "min": -5.0, "max": 15.0
    },
    {
        "pre": "AIBR", "post": "RIVR",
        "name": "Turn Circuit -> Right Neck Flex",
        "label": "Sharp Pivot -> Right Neck",
        "category": "steering", "default": 6.0, "min": -5.0, "max": 15.0
    },

    # ── 2. Obstacle & Wall Collision Avoidance (Reflexes) ────────────────────
    {
        "pre": "AVM", "post": "AVAL",
        "name": "Nose Touch -> Left Reversal Command",
        "label": "Nose Touch -> Emergency Reversal",
        "category": "defense", "default": 14.0, "min": 0.0, "max": 25.0
    },
    {
        "pre": "AVM", "post": "AVAR",
        "name": "Nose Touch -> Right Reversal Command",
        "label": "Nose Touch -> Emergency Reversal",
        "category": "defense", "default": 14.0, "min": 0.0, "max": 25.0
    },
    {
        "pre": "AVM", "post": "AVBL",
        "name": "Nose Touch -> Inhibit Forward",
        "label": "Nose Touch -> Stop Forward Drive",
        "category": "defense", "default": -10.0, "min": -20.0, "max": 2.0
    },
    {
        "pre": "AVM", "post": "AVBR",
        "name": "Nose Touch -> Inhibit Forward",
        "label": "Nose Touch -> Stop Forward Drive",
        "category": "defense", "default": -10.0, "min": -20.0, "max": 2.0
    },
    {
        "pre": "ASHL", "post": "SMDDR",
        "name": "Left Whisker -> Reflexive Right Turn",
        "label": "Left Wall Detected -> Steer Right",
        "category": "defense", "default": 11.0, "min": -5.0, "max": 22.0
    },
    {
        "pre": "ASHR", "post": "SMDDL",
        "name": "Right Whisker -> Reflexive Left Turn",
        "label": "Right Wall Detected -> Steer Left",
        "category": "defense", "default": 11.0, "min": -5.0, "max": 22.0
    },
    {
        "pre": "ASHL", "post": "AVAL",
        "name": "Left Danger -> Reversal",
        "label": "Side Danger -> Trigger Turn Reflex",
        "category": "defense", "default": 8.0, "min": -5.0, "max": 18.0
    },
    {
        "pre": "ASHR", "post": "AVAR",
        "name": "Right Danger -> Reversal",
        "label": "Side Danger -> Trigger Turn Reflex",
        "category": "defense", "default": 8.0, "min": -5.0, "max": 18.0
    },

    # ── 3. Locomotion Propulsion & Forward Crawl ─────────────────────────────
    {
        "pre": "AWAL", "post": "AVBL",
        "name": "Odor Drive -> Forward Command",
        "label": "Food Motivation -> Forward Engine",
        "category": "propulsion", "default": 5.0, "min": -2.0, "max": 15.0
    },
    {
        "pre": "AWAR", "post": "AVBR",
        "name": "Odor Drive -> Forward Command",
        "label": "Food Motivation -> Forward Engine",
        "category": "propulsion", "default": 5.0, "min": -2.0, "max": 15.0
    },
    {
        "pre": "AIYL", "post": "AVBL",
        "name": "Food Tracking -> Forward Crawl",
        "label": "Tracking Odor -> Crawl Toward Food",
        "category": "propulsion", "default": 6.0, "min": -2.0, "max": 16.0
    },
    {
        "pre": "AIYR", "post": "AVBR",
        "name": "Food Tracking -> Forward Crawl",
        "label": "Tracking Odor -> Crawl Toward Food",
        "category": "propulsion", "default": 6.0, "min": -2.0, "max": 16.0
    },
    {
        "pre": "AVBL", "post": "DB1",
        "name": "Forward Drive -> Dorsal Body Muscle",
        "label": "Forward Command -> Body Muscle Wave",
        "category": "propulsion", "default": 8.0, "min": 0.0, "max": 18.0
    },
    {
        "pre": "AVBR", "post": "VB1",
        "name": "Forward Drive -> Ventral Body Muscle",
        "label": "Forward Command -> Body Muscle Wave",
        "category": "propulsion", "default": 8.0, "min": 0.0, "max": 18.0
    },

    # ── 4. Reciprocal Inhibition (Mutual Decision Balance) ───────────────────
    {
        "pre": "AVAL", "post": "AVBL",
        "name": "Reversal Inhibits Forward Drive",
        "label": "Reversal Shuts Down Forward Crawl",
        "category": "coordination", "default": -9.0, "min": -20.0, "max": 0.0
    },
    {
        "pre": "AVAR", "post": "AVBR",
        "name": "Reversal Inhibits Forward Drive",
        "label": "Reversal Shuts Down Forward Crawl",
        "category": "coordination", "default": -9.0, "min": -20.0, "max": 0.0
    },
    {
        "pre": "AVBL", "post": "AVAL",
        "name": "Forward Inhibits Reversal",
        "label": "Forward Crawl Suppresses Reversals",
        "category": "coordination", "default": -6.0, "min": -18.0, "max": 0.0
    },
    {
        "pre": "AVBR", "post": "AVAR",
        "name": "Forward Inhibits Reversal",
        "label": "Forward Crawl Suppresses Reversals",
        "category": "coordination", "default": -6.0, "min": -18.0, "max": 0.0
    },
    {
        "pre": "SMDDL", "post": "SMDDR",
        "name": "Left Steering Inhibits Right",
        "label": "Left Turn Inhibits Right Turn",
        "category": "coordination", "default": -7.0, "min": -16.0, "max": 0.0
    },
    {
        "pre": "SMDDR", "post": "SMDDL",
        "name": "Right Steering Inhibits Left",
        "label": "Right Turn Inhibits Left Turn",
        "category": "coordination", "default": -7.0, "min": -16.0, "max": 0.0
    },

    # ── 5. Reorientation (Pirouette Steering) ────────────────────────────────
    {
        "pre": "AVAL", "post": "SMDDL",
        "name": "Reversal -> Pivot Left",
        "label": "Emergency Turn -> Pivot Left",
        "category": "steering", "default": 8.0, "min": -5.0, "max": 18.0
    },
    {
        "pre": "AVAR", "post": "SMDDR",
        "name": "Reversal -> Pivot Right",
        "label": "Emergency Turn -> Pivot Right",
        "category": "steering", "default": 8.0, "min": -5.0, "max": 18.0
    },
    {
        "pre": "AIYL", "post": "AIBL",
        "name": "Food Tracking Suppresses Turns",
        "label": "Holding Scent Trail -> Smooth Crawl",
        "category": "coordination", "default": -5.0, "min": -15.0, "max": 2.0
    },
    {
        "pre": "AIYR", "post": "AIBR",
        "name": "Food Tracking Suppresses Turns",
        "label": "Holding Scent Trail -> Smooth Crawl",
        "category": "coordination", "default": -5.0, "min": -15.0, "max": 2.0
    },
    {
        "pre": "ASHL", "post": "AVBL",
        "name": "Left Obstacle Dampens Forward",
        "label": "Side Collision Warning -> Brake",
        "category": "defense", "default": -6.0, "min": -18.0, "max": 2.0
    },
    {
        "pre": "ASHR", "post": "AVBR",
        "name": "Right Obstacle Dampens Forward",
        "label": "Side Collision Warning -> Brake",
        "category": "defense", "default": -6.0, "min": -18.0, "max": 2.0
    },
]

GENE_COUNT = len(SYNAPSE_DEFINITIONS)
GENE_BOUNDS = np.array([[s["min"], s["max"]] for s in SYNAPSE_DEFINITIONS], dtype=np.float32)
DEFAULT_SYNAPTIC_GENOME = np.array([s["default"] for s in SYNAPSE_DEFINITIONS], dtype=np.float32)


class SynapticCircuit:
    """Manages the biological evolvable synaptic circuit on a CElegansLIF network."""

    def __init__(self, network):
        self.net = network
        self._indices: List[Tuple[int, int]] = []
        for s in SYNAPSE_DEFINITIONS:
            pre_idx = self.net.name_to_idx.get(s["pre"])
            post_idx = self.net.name_to_idx.get(s["post"])
            self._indices.append((pre_idx, post_idx))

    def apply_genome(self, genome: np.ndarray) -> None:
        """Apply the evolved synaptic weights directly into the connectome matrix W_syn."""
        for i, (pre_idx, post_idx) in enumerate(self._indices):
            if pre_idx is not None and post_idx is not None:
                # W_syn[post, pre] = synaptic conductance
                self.net.W_syn[post_idx, pre_idx] = float(genome[i])

    def get_summary(self, genome: np.ndarray) -> List[Dict[str, Any]]:
        """Return human-readable trait cards suitable for high-school level understanding."""
        summary = []
        for i, s in enumerate(SYNAPSE_DEFINITIONS):
            val = float(genome[i])
            default_val = float(s["default"])
            pct_change = int(((val - default_val) / (abs(default_val) + 1e-6)) * 100)

            # Trait strength
            if val > 12.0:
                strength = "VERY STRONG"
            elif val > 6.0:
                strength = "ACTIVE"
            elif val > 1.0:
                strength = "MODERATE"
            elif val < -5.0:
                strength = "STRONG INHIBITION"
            else:
                strength = "WEAK"

            summary.append({
                "label": s["label"],
                "pathway": f"{s['pre']} -> {s['post']}",
                "category": s["category"],
                "value": round(val, 2),
                "strength": strength,
                "pct_change": pct_change,
            })
        return summary
