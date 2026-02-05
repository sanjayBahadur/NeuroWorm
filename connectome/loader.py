"""
Load and parse the C. elegans connectome from CSV data files.

Data source: OpenWorm connectome (Varshney et al. 2011 + White et al. 1986)
CSV format mirrors the cect (C. elegans Connectome Toolbox) output.

The raw data is downloaded from the OpenWorm GitHub on first run and
cached locally in connectome/data/.
"""

import os
import csv
import numpy as np
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Public OpenWorm connectome CSVs (hosted on GitHub)
SYN_URL = (
    "https://raw.githubusercontent.com/openworm/"
    "c302/master/c302/NeuroML2/CElegansConnectome.csv"
)

# Fallback: embedded minimal connectome if download fails
NEURON_COUNT = 302


def _download_if_missing(url: str, dest: Path) -> None:
    """Download a file from url to dest if dest doesn't exist."""
    if dest.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading connectome data to {dest}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print("Download complete.")


def load_connectome() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the C. elegans connectome.

    Returns:
        syn_weights:  (302, 302) ndarray — signed chemical synapse weights.
                      Positive = excitatory, Negative = inhibitory.
        gap_weights:  (302, 302) ndarray — symmetric gap junction weights.
        neuron_names: list of 302 neuron name strings.
    """
    meta_path = DATA_DIR / "neuron_metadata.csv"
    syn_path  = DATA_DIR / "chemical_synapses.csv"
    gap_path  = DATA_DIR / "gap_junctions.csv"

    # If data directory doesn't exist or files are missing, generate synthetic
    # placeholder data so the simulation can still run during development.
    if not (syn_path.exists() and gap_path.exists() and meta_path.exists()):
        print(
            "[WARNING] Connectome data files not found. "
            "Generating placeholder data for development. "
            "Run `python -m connectome.download` to fetch real data."
        )
        return _make_placeholder_connectome()

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


def _make_placeholder_connectome(
    N: int = 302,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate a synthetic 302-neuron connectome for development / CI testing.
    NOT biologically accurate — use only when real data is unavailable.
    """
    rng = np.random.default_rng(42)

    # Synthetic neuron names (realistic format)
    prefixes = [
        "ASEL", "ASER", "ASHL", "ASHR", "AVM", "ALML", "ALMR",
        "AWACON", "AWAL", "AWAR", "AVBL", "AVBR", "AVAL", "AVAR",
        "SMDDL", "SMDDR", "SMDVL", "SMDVR",
        "DB01", "DB02", "DB03", "DB04", "DB05", "DB06", "DB07",
        "VB01", "VB02", "VB03", "VB04", "VB05", "VB06",
        "VB07", "VB08", "VB09", "VB10", "VB11",
        "RIVL", "RIVR",
    ]
    neuron_names = prefixes[:]
    for i in range(N - len(prefixes)):
        neuron_names.append(f"N{i:03d}")
    neuron_names = neuron_names[:N]

    # Sparse random synaptic weights (~5% connectivity)
    syn = rng.random((N, N)).astype(np.float32)
    mask = rng.random((N, N)) < 0.05
    syn_weights = np.where(mask, (syn - 0.3) * 3.0, 0.0).astype(np.float32)
    np.fill_diagonal(syn_weights, 0.0)  # no self-connections

    # Sparse symmetric gap junctions
    gap_upper = np.where(rng.random((N, N)) < 0.03, rng.random((N, N)), 0.0)
    gap_weights = (gap_upper + gap_upper.T).astype(np.float32)
    np.fill_diagonal(gap_weights, 0.0)

    return syn_weights, gap_weights, neuron_names


def build_network():
    """Convenience: load connectome and return a CElegansLIF instance."""
    from simulation.lif_network import CElegansLIF
    syn, gap, names = load_connectome()
    return CElegansLIF(syn, gap, names)
