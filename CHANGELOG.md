# Changelog

All notable changes to the **NeuroWorm Snake** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-10-15

### Added
- **Reproducible Evaluation Harness (`eval.py`)**: Standardized 30-seed evaluation suite testing random walk, greedy heuristic, default and evolved genomes, isolated 36-synapse subcircuit, and degree-preserving shuffled connectome wiring. Outputs Markdown and JSON reports (`docs/results.md`, `docs/results.json`).
- **Headless GA Training CLI (`train_ga.py`)**: Dedicated command-line interface for off-screen neuroevolution that saves outputs to `runs/` without modifying tracked repository files.
- **Connectome Loader Validation & Tests**: Added assertions verifying that 100% of neuron names referenced in neuron groups and evolvable circuits exist in `neuron_metadata.csv`. Added test verifying `FileNotFoundError` on missing CSV data.
- **Detailed Plain-Language Architecture Guide**: Added `docs/how-it-works.md` with 4-box pipeline diagram and neuroevolution explanation.

### Changed
- **Honest Documentation Rewrite**: Updated `README.md` to present empirical ablation benchmark numbers, plain-language explanations for recruiters and reviewers with zero biology background, and a 4-term glossary.
- **Isolate Pretrained Genome**: Renamed tracked weights to `training/pretrained_genome.npy` and set `GeneticOptimizer` to load read-only.
- **Unified Simulation Constants**: Consolidated `STEPS_PER_FRAME = 8` in `simulation/lif_network.py`.
- **Dependency Pruning**: Removed unused `torch`, `scipy`, and `requests` dependencies from `pyproject.toml` and `requirements.txt`.

### Fixed
- **Cleaned Neuron Definitions**: Removed non-existent biological identifiers `AWACON` and `RIV` from `connectome/neuron_groups.py`, replacing with canonical `AWAL`/`AWAR` and `RIVL`/`RIVR`.
- **Stale Scripts Removal**: Deleted obsolete `benchmark.py`, `train.py`, `training/optimize_encoding.py`, and `training/dqn_baseline.py`.

---

## [1.1.0] - 2026-10-11

### Fixed
- **Shared Nutrient Spore Synchronization**: Resolved critical bug where candidate worms in the 10-worm swarm would pass through the visible egg without eating it or incrementing their score. Previously, each worm possessed an unsynchronized private food position while the renderer only displayed Worm 0's food. The arena now manages a synchronized `shared_food` state consumed universally by all active worms.
- **Egg Collision & Re-Spawning**: Guaranteed that whenever any worm's head reaches the egg, the score increments immediately, the worm body grows by one segment, and a fresh spore spawns in unoccupied soil.

### Added
- **Bioluminescent Spore Consumption VFX**: Added a 16-particle golden radial burst effect whenever an egg is devoured by any worm.
- **Floating Score Indicator**: Added animated `+1 🥚` floating popup text above the eaten spore that drifts upward and fades smoothly.
- **Neuroevolution Benchmark**: Benchmarked evolved elite genome across 5 multi-seed configurations (achieving a mean score of 1.70 ± 0.90, peak score of 5, and 119.2 mean survival steps). *Note: Rigorous evaluation on 30 held-out seeds in v1.2.0 measures the evolved genome at 2.47 ± 0.77 (max 8); see docs/results.md.*

---

## [1.0.0] - 2026-10-11

### Added
- **Option A Biological Synaptic Neuroevolution**: Direct evolution of 36 chemical synaptic pathways ($W_{\text{syn}}$ in nS) inside the canonical 302-neuron *C. elegans* connectome matrix.
- **High-School-Friendly Brain Flow Inspector**:
  - `[🧠 LIVE THOUGHTS]`: 3-stage live decision stream (Stage 1 Senses: Food Odor Split Bar & Nose Touch Warning -> Stage 2 Decisions: Forward Drive vs Reversal Reflex -> Stage 3 Muscles: Left vs Right Turn) with plain-English thought summary.
  - `[🔬 SYNAPSE LEARNING]`: Real-time cards explaining evolved biological circuits with conductance meters in nS.
  - Interactive 10-Worm Swarm Roster with click-to-inspect and champion badges (`★1`, `★2`, `★3`).
- **Procedural Vermiform Ribbon Kinematics**: Continuous articulated pill capsules with smooth tapering, bioluminescent nerve pulses, and dynamic photoreceptor eyes replacing disjoint tile sprites.
- **Multi-Seed Spanning (5 Seeds)** with Top 3 champion elitism.
- **Controls & Interaction**: Fast Forward 100x (`F`), Evolve Next Gen (`N`), Pause (`SPC`), Speed controls (`+`/`-`), Swarm/Solo mode (`M`), GA Reset (`R`), Fullscreen (`F11`).
