# Benchmark & Connectome Ablation Results

Evaluated on 30 held-out seeds (`1000` to `1029`) on a 20x20 grid.

| Agent | Mean Score | 95% CI | Max | Zero-Score (%) | Mean Steps |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random walk | 0.10 ± 0.11 | ±0.11 | 1 | 90.0% | 49.8 |
| Greedy rule (no brain) | 34.60 ± 4.20 | ±4.20 | 60 | 0.0% | 519.1 |
| Full connectome + default genome | 2.07 ± 0.54 | ±0.54 | 7 | 3.3% | 146.0 |
| Full connectome + evolved genome | 2.47 ± 0.77 | ±0.77 | 8 | 10.0% | 167.7 |
| Connectome removed (36 isolated synapses) | 3.60 ± 1.08 | ±1.08 | 13 | 0.0% | 77.5 |
| Shuffled connectome wiring | 1.50 ± 0.26 | ±0.26 | 3 | 3.3% | 93.3 |

### Key Findings
1. **Beats random**: The worm brain achieves a score of ~2.5, significantly outperforming random walk (~0.1).
2. **Simple rule beats it**: A 15-line greedy heuristic achieves a mean score of ~34.6 (peak >60), highlighting the ceiling of the task.
3. **Connectome wiring does not drive performance**: Removing the 302-neuron background connectome entirely (`Connectome removed`) performs as well or better than the full wiring (~3.6 vs ~2.5), demonstrating that behavior is primarily driven by the 36 hand-tuned circuits and sensory/motor transduction rules rather than the mapped connectome topology.
