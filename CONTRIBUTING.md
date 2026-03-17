# Contributing to NeurWorm Snake

Thank you for your interest in contributing to **NeurWorm Snake**! We welcome contributions spanning computational neuroscience, game development, reinforcement learning baselines, and pixel art.

---

## 🛠️ Development Setup

```bash
# 1. Clone repository
git clone https://github.com/neurworm/neurworm-snake.git
cd neurworm-snake

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## 🧪 Testing

We require all pull requests to pass the test suite:

```bash
pytest tests/ -v
```

To run a headless benchmark of the worm's performance:

```bash
python benchmark.py --episodes 20
```

---

## 📝 Commit Conventions

We adhere to conventional commit messages:
- `feat:` New feature (e.g. `feat(simulation): add gap junction ohmic current`)
- `fix:` Bug fix (e.g. `fix(decoder): guard against division by zero at rest`)
- `docs:` Documentation improvements
- `perf:` Performance optimizations
- `test:` Unit and integration test additions
- `chore:` Build system or tooling maintenance

---

## 🔬 Biological Accuracy Philosophy

1. **Connectome Topology:** Synaptic and electrical connectivity should remain faithful to published *C. elegans* wiring diagrams (e.g., White et al. 1986, Varshney et al. 2011, Cook et al. 2019).
2. **Zero In-Circuit Training:** We do not modify internal synaptic weights through backpropagation or gradient descent. Evolution configured the connectome; our task is calibration of the sensory-motor interface.
3. **Reproducibility:** Experiments should support reproducible random seeds.
