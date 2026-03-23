"""
Genetic Algorithm Optimizer for NeuroWorm Snake.

Evolves the sensory-motor interface parameters of the C. elegans connectome:
  - Food steering gain
  - Anterior collision touch sensitivity (AVM / reversal)
  - Lateral danger avoidance gain (ASHL / ASHR)
  - Forward chemotaxis drive (AWA / AVB)
  - Lateral turn threshold
  - Reversal threshold
  - Dorsal/ventral balance

Provides off-screen ultra-fast batch evaluation (100+ iterations/sec)
and step-by-step live generational evolution.
"""

import time
import copy
import numpy as np
from pathlib import Path
from connectome.loader import build_network
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from game.snake_env import SnakeEnv

PARAMS_PATH = Path(__file__).parent / "best_params.npy"

# Parameter gene bounds: [min, max]
GENE_BOUNDS = np.array([
    [1.0, 15.0],   # i_food (food attraction current)
    [3.0, 25.0],   # i_danger_front (anterior collision nociception)
    [1.0, 15.0],   # i_danger_side (lateral edge avoidance)
    [0.2,  6.0],   # i_awa (chemotaxis forward crawl motivation)
    [0.5, 10.0],   # i_front_food (straight food approach current)
    [0.25, 0.75],  # rev_thresh (reversal / turn sensitivity)
    [0.45, 0.75],  # turn_thresh (head motor turn sensitivity)
    [0.45, 0.75],  # dv_thresh (dorsal/ventral turn sensitivity)
])

PARAM_KEYS = [
    "i_food", "i_danger_front", "i_danger_side", "i_awa",
    "i_front_food", "rev_thresh", "turn_thresh", "dv_thresh"
]

DEFAULT_GENOME = np.array([4.5, 12.0, 6.0, 1.8, 3.5, 0.45, 0.52, 0.55])


class Individual:
    def __init__(self, genome: np.ndarray = None):
        if genome is None:
            # Initialize with Gaussian perturbation around sensible defaults
            self.genome = np.clip(
                DEFAULT_GENOME + np.random.randn(len(DEFAULT_GENOME)) * 0.5,
                GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
            )
        else:
            self.genome = np.clip(genome, GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])

        self.fitness = 0.0
        self.score = 0
        self.steps = 0

    def to_dict(self) -> dict:
        return {k: float(self.genome[i]) for i, k in enumerate(PARAM_KEYS)}


class GeneticOptimizer:
    """
    Manages population, evaluation, selection, crossover, and mutation.
    """

    def __init__(self, pop_size: int = 16, grid_size: int = 20, max_eval_steps: int = 800):
        self.pop_size = pop_size
        self.grid_size = grid_size
        self.max_eval_steps = max_eval_steps

        # Load best params if existing
        if PARAMS_PATH.exists():
            try:
                saved = np.load(PARAMS_PATH)
                if len(saved) == len(DEFAULT_GENOME):
                    init_g = saved
                else:
                    init_g = DEFAULT_GENOME
            except Exception:
                init_g = DEFAULT_GENOME
        else:
            init_g = DEFAULT_GENOME

        self.population = [Individual(init_g.copy() if i == 0 else None) for i in range(pop_size)]
        self.generation = 1
        self.current_ind_idx = 0
        self.best_individual = self.population[0]
        self.history_best_fitness = []
        self.history_avg_fitness = []
        self.total_evaluations = 0

    def evaluate_genome(self, genome: np.ndarray, seed: int = 42, max_steps: int = None) -> tuple[float, int, int]:
        """
        Evaluate a single genome headlessly in pure Python/NumPy.
        Returns: (fitness, score, steps_survived)
        """
        if max_steps is None:
            max_steps = self.max_eval_steps

        env = SnakeEnv(grid_size=self.grid_size, seed=seed)
        net = build_network()
        p = {k: float(genome[i]) for i, k in enumerate(PARAM_KEYS)}

        encoder = SensoryEncoder(
            net, self.grid_size,
            i_food=p["i_food"],
            i_danger_front=p["i_danger_front"],
            i_danger_side=p["i_danger_side"],
            i_awa=p["i_awa"],
            i_front_food=p["i_front_food"],
        )
        decoder = MotorDecoder(
            net,
            rev_thresh=p["rev_thresh"],
            turn_thresh=p["turn_thresh"],
            dv_thresh=p["dv_thresh"],
        )

        obs = env.reset()
        direction = "RIGHT"
        score = 0
        steps = 0
        steps_since_food = 0
        min_dist_to_food = 999.0

        for s in range(max_steps):
            head, food, body, cur_dir = obs
            dist = np.hypot(food[0] - head[0], food[1] - head[1])
            if dist < min_dist_to_food:
                min_dist_to_food = dist

            net.reset_counts()
            encoder.encode(head, food, body, direction)
            # 8 LIF timesteps (4ms of simulated biological dynamics)
            for _ in range(8):
                net.step()

            direction = decoder.decode(direction)
            obs, done = env.step(direction)
            steps += 1
            steps_since_food += 1

            if env.score > score:
                score = env.score
                steps_since_food = 0
                min_dist_to_food = 999.0

            # Prevent endless circular loops without progress
            if steps_since_food > 180:
                break

            if done:
                break

        # Dense fitness function: reward food pellets heavily, reward survival,
        # reward closing the distance to food
        fitness = (score * 250.0) + (steps * 1.5) + max(0.0, 20.0 - min_dist_to_food) * 2.0
        return float(fitness), int(score), int(steps)

    def evaluate_population(self):
        """Evaluate all individuals in current population."""
        for i, ind in enumerate(self.population):
            # Test across 2 distinct random seeds for stability
            fit1, sc1, st1 = self.evaluate_genome(ind.genome, seed=self.generation * 10 + 1)
            fit2, sc2, st2 = self.evaluate_genome(ind.genome, seed=self.generation * 10 + 2)
            ind.fitness = (fit1 + fit2) / 2.0
            ind.score = max(sc1, sc2)
            ind.steps = (st1 + st2) // 2
            self.total_evaluations += 1

        # Sort population by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        if self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
            np.save(PARAMS_PATH, self.best_individual.genome)

        avg_fit = float(np.mean([ind.fitness for ind in self.population]))
        self.history_best_fitness.append(self.best_individual.fitness)
        self.history_avg_fitness.append(avg_fit)

    def evolve_next_generation(self):
        """Run selection, crossover, and mutation to produce the next generation."""
        # 1. Elitism: preserve top 2 individuals
        new_pop = [copy.deepcopy(self.population[0]), copy.deepcopy(self.population[1])]

        # 2. Tournament selection + crossover
        while len(new_pop) < self.pop_size:
            # Tournament selection (size 3)
            p1 = max(np.random.choice(self.population, 3, replace=False), key=lambda x: x.fitness)
            p2 = max(np.random.choice(self.population, 3, replace=False), key=lambda x: x.fitness)

            # Uniform arithmetic crossover
            alpha = np.random.uniform(0.1, 0.9, size=len(DEFAULT_GENOME))
            child_g = alpha * p1.genome + (1.0 - alpha) * p2.genome

            # Gaussian mutation (20% mutation probability per gene)
            mutation_mask = np.random.rand(len(DEFAULT_GENOME)) < 0.25
            mutation_scale = (GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]) * 0.08
            mutations = np.random.randn(len(DEFAULT_GENOME)) * mutation_scale
            child_g += mutation_mask * mutations

            child = Individual(child_g)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        self.current_ind_idx = 0

    def fast_forward(self, n_evaluations: int = 100) -> dict:
        """
        Fast forward off-screen at maximum CPU throughput.
        Performs n_evaluations iterations of individual assessments,
        evolving generations as needed.
        """
        t0 = time.time()
        evals_done = 0
        gens_started = self.generation

        while evals_done < n_evaluations:
            # Evaluate remaining individuals in this generation
            self.evaluate_population()
            evals_done += self.pop_size
            self.evolve_next_generation()

        elapsed = time.time() - t0
        rate = evals_done / max(0.001, elapsed)

        return {
            "evals_done": evals_done,
            "elapsed_sec": elapsed,
            "rate_per_sec": rate,
            "current_gen": self.generation,
            "best_fitness": self.best_individual.fitness,
            "best_score": self.best_individual.score,
        }
