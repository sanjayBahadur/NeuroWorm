# Changelog

All notable changes to the **NeuroWorm Snake** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-05

### Fixed
- **Shared Nutrient Spore Synchronization**: Resolved critical bug where candidate worms in the 10-worm swarm would pass through the visible egg without eating it or incrementing their score. Previously, each worm possessed an unsynchronized private food position while the renderer only displayed Worm 0's food. The arena now manages a synchronized `shared_food` state consumed universally by all active worms.
- **Egg Collision & Re-Spawning**: Guaranteed that whenever any worm's head reaches the egg, the score increments immediately, the worm body grows by one segment, and a fresh spore spawns in unoccupied soil.

### Added
- **Bioluminescent Spore Consumption VFX**: Added a 16-particle golden radial burst effect whenever an egg is devoured by any worm.
- **Floating Score Indicator**: Added animated `+1 🥚` floating popup text above the eaten spore that drifts upward and fades smoothly.
- **Neuroevolution Benchmark**: Benchmarked evolved elite genome across 5 multi-seed configurations (achieving a mean score of 1.70 ± 0.90, peak score of 5, and 119.2 mean survival steps).

---

## [1.0.0] - 2026-08-29

### Added
- **Option A Biological Synaptic Neuroevolution**: Direct evolution of 36 chemical synaptic pathways ($W_{\text{syn}}$ in nS) inside the canonical 302-neuron *C. elegans* connectome matrix.
- **High-School-Friendly Brain Flow Inspector**:
  - `[🧠 LIVE THOUGHTS]`: 3-stage live decision stream (Stage 1 Senses: Food Odor Split Bar & Nose Touch Warning -> Stage 2 Decisions: Forward Drive vs Reversal Reflex -> Stage 3 Muscles: Left vs Right Turn) with plain-English thought summary.
  - `[🔬 SYNAPSE LEARNING]`: Real-time cards explaining evolved biological circuits with conductance meters in nS.
  - Interactive 10-Worm Swarm Roster with click-to-inspect and champion badges (`★1`, `★2`, `★3`).
- **Procedural Vermiform Ribbon Kinematics**: Continuous articulated pill capsules with smooth tapering, bioluminescent nerve pulses, and dynamic photoreceptor eyes replacing disjoint tile sprites.
- **Multi-Seed Spanning (5 Seeds)** with Top 3 champion elitism.
- **Controls & Interaction**: Fast Forward 100x (`F`), Evolve Next Gen (`N`), Pause (`SPC`), Speed controls (`+`/`-`), Swarm/Solo mode (`M`), GA Reset (`R`), Fullscreen (`F11`).
