# 🪱 NeurWorm Snake

> **C. elegans connectome plays Snake** — a bio-inspired AI game powered by a real 302-neuron worm nervous system simulation.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

NeurWorm Snake simulates the **complete nervous system of *C. elegans*** — a 302-neuron nematode worm — and uses it to play the classic Snake game. The worm's sensory neurons detect food and danger; its motor neurons steer the snake.

No backprop. No gradient descent. Just a biophysically-grounded **Leaky Integrate-and-Fire** network and an **Evolution Strategy** that tunes how game state maps to neuron currents.

---

## Architecture

```
Game State
    │
    ▼
SensoryEncoder          ← maps food/danger positions to neuron currents
    │  injects into
    ▼
CElegansLIF (302 neurons)  ← vectorized numpy LIF simulation
    │  fire rates read by
    ▼
MotorDecoder            ← maps AVA/AVB/DB/VB/SMDD firing to direction
    │
    ▼
Snake Direction (UP / DOWN / LEFT / RIGHT)
```

### Key Neuron Mappings

| Game Event       | Neuron(s)         | Biological Role        |
|------------------|-------------------|------------------------|
| Food to the right | ASER             | Chemosensory (attractive) |
| Food to the left  | ASEL             | Chemosensory (attractive) |
| Wall/body ahead   | AVM, ALML, ALMR  | Anterior mechanosensory |
| Turn UP           | DB01–DB07         | Dorsal body motor      |
| Turn DOWN         | VB01–VB11         | Ventral body motor     |
| Reverse → Turn    | AVAL, AVAR        | Reversal command       |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Run the demo (worm brain controls snake)
python demo.py

# 3. Human keyboard control
python demo.py --human

# 4. Train encoding parameters (ES optimizer)
python train.py --generations 50

# 5. Benchmark worm brain performance
python benchmark.py --episodes 20
```

---

## Project Structure

```
neurworm-snake/
├── connectome/
│   ├── loader.py          # load/generate connectome data
│   ├── neuron_groups.py   # named neuron group definitions
│   └── data/              # CSV data files (downloaded separately)
├── simulation/
│   ├── lif_network.py     # vectorized LIF simulation (core)
│   ├── sensory_encoder.py # game state → neuron currents
│   └── motor_decoder.py   # firing rates → game direction
├── game/
│   ├── snake_env.py       # headless Snake environment (no pygame)
│   └── snake_game.py      # full pygame renderer + AI loop
├── training/
│   └── optimize_encoding.py  # (1+λ)-ES parameter optimizer
├── viz/
│   └── neural_dashboard.py   # real-time firing rate heatmap
├── tests/                    # pytest unit tests
├── demo.py                   # main entry point
├── train.py                  # training entry point
└── benchmark.py              # worm vs DQN comparison
```

---

## The Science

*C. elegans* has exactly **302 neurons** and **~7,000 synaptic connections**, fully mapped since 1986 (White et al.). The connectome is biologically fixed — we only optimize 5 scalar parameters that control **how strongly** game events stimulate specific sensory neurons.

The LIF model uses parameters adapted from Kunert et al. (2014):
- τ_m = 10 ms (membrane time constant)
- V_thresh = −55 mV, V_reset = −70 mV
- τ_syn = 5 ms (synaptic conductance decay)
- Refractory period = 2 ms

---

## References

- White et al. (1986) — *The structure of the nervous system of the nematode C. elegans*
- Varshney et al. (2011) — *Structural properties of the C. elegans neuronal network*
- Kunert et al. (2014) — *Multistability and long-timescale transients encoded by network structure in a model of C. elegans connectome dynamics*
- Gerstner & Kistler (2002) — *Spiking Neuron Models*

---

## License

MIT © NeurWorm Dev
