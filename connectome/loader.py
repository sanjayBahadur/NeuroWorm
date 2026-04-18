"""
Load and parse the C. elegans connectome from CSV data files.

Data source: OpenWorm connectome (Varshney et al. 2011 + White et al. 1986)
CSV format mirrors the cect (C. elegans Connectome Toolbox) output.
The connectome data files are bundled in connectome/data/.
"""

import csv
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
NEURON_COUNT = 302


def load_connectome() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the C. elegans connectome.

    Returns:
        syn_weights:  (302, 302) ndarray — chemical synapse weights (excitatory).
        gap_weights:  (302, 302) ndarray — symmetric gap junction weights.
        neuron_names: list of 302 neuron name strings.
    """
    meta_path = DATA_DIR / "neuron_metadata.csv"
    syn_path  = DATA_DIR / "chemical_synapses.csv"
    gap_path  = DATA_DIR / "gap_junctions.csv"

    if not (syn_path.exists() and gap_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"Connectome CSV data files missing in {DATA_DIR}. "
            "Expected neuron_metadata.csv, chemical_synapses.csv, and gap_junctions.csv."
        )

    # Load neuron names
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    neuron_names = [r["name"] for r in rows]
    N = len(neuron_names)
    name_to_idx = {n: i for i, n in enumerate(neuron_names)}

    # Load chemical synapses
    syn_weights = np.zeros((N, N), dtype=np.float32)
    with open(syn_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre  = row["pre"]
            post = row["post"]
            w    = float(row["weight"])
            if pre in name_to_idx and post in name_to_idx:
                syn_weights[name_to_idx[post], name_to_idx[pre]] += w

    # Load gap junctions
    gap_weights = np.zeros((N, N), dtype=np.float32)
    with open(gap_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n1 = row["neuron1"]
            n2 = row["neuron2"]
            w  = float(row["weight"])
            if n1 in name_to_idx and n2 in name_to_idx:
                i1, i2 = name_to_idx[n1], name_to_idx[n2]
                gap_weights[i1, i2] += w
                gap_weights[i2, i1] += w   # enforce symmetry

    return syn_weights, gap_weights, neuron_names


def build_network():
    """Convenience: load connectome and return a CElegansLIF instance."""
    from simulation.lif_network import CElegansLIF
    syn, gap, names = load_connectome()
    return CElegansLIF(syn, gap, names)
