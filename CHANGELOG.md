# Changelog

All notable changes to the **NeuroWorm Snake** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] - 2026-05-28

### Added
- **Decoupled 60 FPS Render Loop & Sub-Pixel Interpolation**: Render rate is separated from the biological simulation rate with fractional cell interpolation (`prev_body` to `body`), ensuring fluid vermiform gliding motion across all simulation speeds (4–60 ticks/s).
- **Organic Vermiform Kinematics**: Implemented continuous Chaikin corner-cutting smoothing that rounds 90-degree pipe corners into organic curves, complete with continuous tapering (12 px head to 4 px tail), subtle outline, highlight ridge, and travelling bioluminescent pulses.
- **Hero Worm & Ghost Swarm Hierarchy**: Active focus mode spotlights the current generation leader (or manually selected worm via click / `0`–`9` keys) with full opacity, glowing cyan halo, dynamic photoreceptor eyes, whisker proximity rays, and a depleting hunger ring. The other 9 swarm candidates recede as translucent ghosts (~35% opacity) to eliminate visual noise.
- **Interactive Proximity Whiskers & Dynamic Death Bursts**: Whisker rays project from the hero head to obstacle hit points clamped to the inner arena wall, color-coded by proximity distance (`wall 2`, `self 1`). On termination, worms play a radial burst animation and float an explicit cause label (`hit the wall`, `bit itself`, `starved`, `timeout`).
- **Learning Curve Visualization ("Is It Learning?")**: Real-time line chart tracking generational peak (yellow) and swarm average (cyan) egg scores over the last 30 generations, complete with labeled axes and empty state guidance.
- **Bundled OFL Typography & Vector Icons**: Integrated SIL OFL-licensed JetBrains Mono font (`assets/fonts/`) and procedural vector icons in `game/ui.py`, eliminating missing-glyph tofu boxes (`▯`) and unicode emojis across the UI. Added non-blocking hotkey help overlay (`?`).
- **Headless Screenshot & Animation Tooling (`tools/capture.py`)**: Added reproducible headless capture script producing full frames, arena closeups, panel telemetry, death states, generation transitions, and optimized animated demo GIF (< 1 MB).

### Fixed
- **Wall Collision & Geometry Alignment**: Centralized arena metrics in `game/layout.py` with an outer wall border band. Fixed off-by-one rendering offsets where alive worms were drawn on top of stone wall tiles (reduced border-tile collisions from 55.4% to 0.0%).
- **Shared Nutrient Spore Race Condition**: Ensured instantaneous food relocation upon consumption so multiple worms cannot eat the same egg on the same tick. Randomized worm stepping sequence per frame using permutations.
- **Accurate Death Cause Telemetry**: Added explicit `death_cause` tracking (`"wall"`, `"self"`, `"starved"`, `"timeout"`) across `SnakeEnv` and `WormAgent`, resolving silent starvation where starving worms were misattributed to wall collisions. Synchronized agent heading with environment heading.
- **Non-Blocking Generation Transition**: Replaced blocking `pygame.time.wait` / `time.sleep` calls with a clean 1.2s non-blocking `RUNNING -> GEN_END -> RUNNING` state machine with instant skip (`N`).
- **Swarm Evolutionary History Preservation**: Resolved bug where generational history was wiped upon starting new generations; wired history directly into the persistent genetic optimizer.

---

## [1.2.0] - 2026-05-01

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

## [1.1.0] - 2026-04-09

### Fixed
- **Shared Nutrient Spore Synchronization**: Resolved critical bug where candidate worms in the 10-worm swarm would pass through the visible egg without eating it or incrementing their score. Previously, each worm possessed an unsynchronized private food position while the renderer only displayed Worm 0's food. The arena now manages a synchronized `shared_food` state consumed universally by all active worms.
- **Egg Collision & Re-Spawning**: Guaranteed that whenever any worm's head reaches the egg, the score increments immediately, the worm body grows by one segment, and a fresh spore spawns in unoccupied soil.

### Added
- **Bioluminescent Spore Consumption VFX**: Added a 16-particle golden radial burst effect whenever an egg is devoured by any worm.
- **Floating Score Indicator**: Added animated `+1 🥚` floating popup text above the eaten spore that drifts upward and fades smoothly.
- **Neuroevolution Benchmark**: Benchmarked evolved elite genome across 5 multi-seed configurations (achieving a mean score of 1.70 ± 0.90, peak score of 5, and 119.2 mean survival steps). *Note: Rigorous evaluation on 30 held-out seeds in v1.2.0 measures the evolved genome at 2.47 ± 0.77 (max 8); see docs/results.md.*

---

## [1.0.0] - 2026-04-05

### Added
- **Option A Biological Synaptic Neuroevolution**: Direct evolution of 36 chemical synaptic pathways ($W_{\text{syn}}$ in nS) inside the canonical 302-neuron *C. elegans* connectome matrix.
- **High-School-Friendly Brain Flow Inspector**:
  - `[🧠 LIVE THOUGHTS]`: 3-stage live decision stream (Stage 1 Senses: Food Odor Split Bar & Nose Touch Warning -> Stage 2 Decisions: Forward Drive vs Reversal Reflex -> Stage 3 Muscles: Left vs Right Turn) with plain-English thought summary.
  - `[🔬 SYNAPSE LEARNING]`: Real-time cards explaining evolved biological circuits with conductance meters in nS.
  - Interactive 10-Worm Swarm Roster with click-to-inspect and champion badges (`★1`, `★2`, `★3`).
- **Procedural Vermiform Ribbon Kinematics**: Continuous articulated pill capsules with smooth tapering, bioluminescent nerve pulses, and dynamic photoreceptor eyes replacing disjoint tile sprites.
- **Multi-Seed Spanning (5 Seeds)** with Top 3 champion elitism.
- **Controls & Interaction**: Fast Forward 100x (`F`), Evolve Next Gen (`N`), Pause (`SPC`), Speed controls (`+`/`-`), Swarm/Solo mode (`M`), GA Reset (`R`), Fullscreen (`F11`).
