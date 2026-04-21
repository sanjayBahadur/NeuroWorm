# Contributing to NeuroWorm Snake

Thank you for your interest in contributing to **NeuroWorm Snake**! We welcome contributions spanning computational neuroscience, bio-inspired AI, game simulation, and testing.

---

## 🛠️ Development Setup

```bash
# 1. Clone repository
git clone https://github.com/neurworm/neurworm-snake.git
cd neurworm-snake

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 🧪 Testing & Evaluation

All pull requests must pass the test suite and code quality checks:

```bash
# Run unit tests
pytest -q

# Check formatting and linting
black --check .
ruff check .
```

### Reproducible Evaluation Harness

To run the standard 30-seed evaluation suite and ablation controls:

```bash
python eval.py
```

For a rapid smoke check (5 seeds):

```bash
python eval.py --quick
```

### Headless Genetic Algorithm Training

To evolve new biological synaptic weights in the headless CLI:

```bash
python train_ga.py --generations 20 --pop-size 10
```

Results are saved to `runs/best_genome.npy` without modifying tracked repository files.

---

## 📝 Commit Conventions

We adhere to conventional commit messages (`feat:`, `fix:`, `docs:`, `perf:`, `test:`, `chore:`, `ci:`):
- `feat:` New feature (e.g. `feat(simulation): add degree-preserving connectome shuffling`)
- `fix:` Bug fix (e.g. `fix(decoder): guard against division by zero at rest`)
- `docs:` Documentation improvements
- `perf:` Performance optimizations
- `test:` Unit and integration test additions
- `chore:` Build system or tooling maintenance

---

## 🔬 Biological Accuracy Philosophy

1. **Connectome Topology:** Synaptic and electrical connectivity should remain faithful to published *C. elegans* wiring diagrams (e.g., White et al. 1986, Varshney et al. 2011, Cook et al. 2019).
2. **Reproducibility & Honesty:** All claims regarding performance must be verified against the reproducible evaluation harness (`eval.py`) across held-out random seeds.
3. **Controlled Baselines:** Changes to the neural dynamics or architecture should be ablated against random walk and heuristic controls to confirm empirical contribution.
