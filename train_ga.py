#!/usr/bin/env python3
"""
Headless Genetic Algorithm Training CLI for NeuroWorm Snake.

Evolves biological synaptic weights directly in the connectome across
multi-seed evaluation environments.
"""

import argparse
from pathlib import Path
import numpy as np

from training.genetic_algorithm import GeneticOptimizer, DEFAULT_RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description="Headless Genetic Algorithm Neuroevolution")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations to evolve (default: 20)")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size (default: 10)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npy path for evolved champion genome (default: runs/best_genome.npy)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else DEFAULT_RUNS_DIR / "best_genome.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Genetic Optimizer (Pop: {args.pop_size}, Target: {args.generations} generations)...")
    ga = GeneticOptimizer(pop_size=args.pop_size, save_path=out_path)

    res = ga.fast_forward(n_generations=args.generations)
    print(f"Evolution complete in {res['elapsed_sec']:.2f}s ({res['evals_per_sec']:.1f} evals/sec)")
    print(f"Best Fitness: {res['best_fitness']:.1f} | Best Score: {res['best_score']} eggs | Generation: {res['current_gen']}")

    np.save(out_path, ga.best_individual.genome)
    print(f"Saved evolved champion genome to {out_path}")


if __name__ == "__main__":
    main()
