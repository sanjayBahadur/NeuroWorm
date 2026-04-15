from pathlib import Path
import numpy as np
import pytest
from training.genetic_algorithm import GeneticOptimizer, DEFAULT_GENOME, EVAL_SEEDS


def test_ga_initialization():
    """Verify GA initializes with population of 10 and default/random genomes."""
    ga = GeneticOptimizer(pop_size=10, grid_size=10, max_eval_steps=20)
    assert len(ga.population) == 10
    assert ga.pop_size == 10
    assert ga.generation == 1
    assert ga.best_individual is not None
    assert len(ga.best_individual.genome) == len(DEFAULT_GENOME)


def test_multi_seed_spanning():
    """Verify evaluation spans across the 5 predefined evaluation seeds."""
    assert len(EVAL_SEEDS) == 5
    ga = GeneticOptimizer(pop_size=2, grid_size=10, max_eval_steps=10)
    ind = ga.population[0]
    initial_total_evals = ga.total_evaluations
    ga.evaluate_individual_multi_seed(ind)
    assert ga.total_evaluations == initial_total_evals + 5
    assert ind.fitness >= 0.0


def test_top_3_champions_preservation():
    """Verify evolution retains top 3 champions intact in slots 1, 2, and 3."""
    ga = GeneticOptimizer(pop_size=10, grid_size=10, max_eval_steps=10)
    # Manually set arbitrary fitnesses to verify sorting & elitism
    for i, ind in enumerate(ga.population):
        ind.fitness = float(100 - i * 10)
        ind.genome[0] = float(i + 1)

    c1_genome = ga.population[0].genome.copy()
    c2_genome = ga.population[1].genome.copy()
    c3_genome = ga.population[2].genome.copy()

    ga.evolve_next_generation()

    assert len(ga.population) == 10
    assert ga.generation == 2
    # Check top 3 are preserved
    assert np.allclose(ga.population[0].genome, c1_genome)
    assert np.allclose(ga.population[1].genome, c2_genome)
    assert np.allclose(ga.population[2].genome, c3_genome)


def test_fast_forward():
    """Verify fast_forward advances generations and returns telemetry."""
    ga = GeneticOptimizer(pop_size=4, grid_size=8, max_eval_steps=10)
    stats = ga.fast_forward(n_generations=2)
    assert stats["gens_run"] == 2
    assert stats["current_gen"] == 3
    assert stats["evals_per_sec"] > 0


def test_ga_does_not_modify_shipped_genome(tmp_path):
    """Verify that running a GA optimization does not touch training/pretrained_genome.npy."""
    pretrained_path = Path(__file__).parent.parent / "training" / "pretrained_genome.npy"
    assert pretrained_path.exists()
    mtime_before = pretrained_path.stat().st_mtime
    data_before = np.load(pretrained_path)

    custom_save = tmp_path / "custom_run.npy"
    ga = GeneticOptimizer(pop_size=4, grid_size=8, max_eval_steps=10, save_path=custom_save)
    ga.fast_forward(n_generations=2)

    assert pretrained_path.stat().st_mtime == mtime_before
    data_after = np.load(pretrained_path)
    assert np.array_equal(data_before, data_after)
