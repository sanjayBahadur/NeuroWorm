"""
Genetic Algorithm Optimizer for NeuroWorm Snake.

Evolutionary Strategy:
  - Population size: 10 candidate worms
  - Multi-Seed Spanning: Each worm is evaluated across 5 random seeds to ensure
    robust foraging regardless of starting layout
  - Truncation Selection: Top 3 champions are preserved (pure elitism) and replicated
    with calibrated mutations to fill the remaining 7 slots
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
    [1.0, 16.0],   # i_food (lateral chemotaxis steering)
    [5.0, 30.0],   # i_danger_front (anterior collision / nose touch)
    [1.0, 18.0],   # i_danger_side (lateral whisker touch)
    [0.2,  6.0],   # i_awa (chemotaxis forward crawl motivation)
    [0.5, 10.0],   # i_front_food (straight food approach current)
    [0.20, 0.70],  # rev_thresh (reversal / turn sensitivity)
    [0.45, 0.75],  # turn_thresh (head motor turn sensitivity)
    [0.45, 0.75],  # dv_thresh (dorsal/ventral turn sensitivity)
])

PARAM_KEYS = [
    "i_food", "i_danger_front", "i_danger_side", "i_awa",
    "i_front_food", "rev_thresh", "turn_thresh", "dv_thresh"
]

DEFAULT_GENOME = np.array([5.5, 16.0, 7.5, 2.0, 4.0, 0.40, 0.52, 0.55])
EVAL_SEEDS = [42, 107, 219, 333, 512]  # Multi-seed spanning across 5 configurations


class Individual:
    def __init__(self, genome: np.ndarray = None):
        if genome is None:
            self.genome = np.clip(
                DEFAULT_GENOME + np.random.randn(len(DEFAULT_GENOME)) * 0.6,
                GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
            )
        else:
            self.genome = np.clip(genome, GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])

        self.fitness = 0.0
        self.score = 0
        self.steps = 0


class GeneticOptimizer:
    """
    Population of 10 candidates, multi-seed spanning, keeping top 3 champions.
    """

    def __init__(self, pop_size: int = 10, grid_size: int = 20, max_eval_steps: int = 600):
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
        self.best_individual = self.population[0]
        self.history_best_fitness = []
        self.history_avg_fitness = []
        self.total_evaluations = 0

    def evaluate_genome_on_seed(self, genome: np.ndarray, seed: int) -> tuple[float, int, int]:
        """Evaluate a single genome on one seed."""
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

        for s in range(self.max_eval_steps):
            head, food, body, cur_dir = obs
            dist = np.hypot(food[0] - head[0], food[1] - head[1])
            if dist < min_dist_to_food:
                min_dist_to_food = dist

            net.reset_counts()
            encoder.encode(head, food, body, direction)
            # 8 LIF steps (4ms of biological dynamics)
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

            # Prevent endless idle circling
            if steps_since_food > 150:
                break

            if done:
                break

        # Fitness: heavy reward for eating food, reward survival, reward closing distance
        fitness = (score * 300.0) + (steps * 1.2) + max(0.0, 20.0 - min_dist_to_food) * 2.5
        return float(fitness), int(score), int(steps)

    def evaluate_individual_multi_seed(self, ind: Individual):
        """Evaluate one candidate across 5 distinct seeds and average results."""
        fits = []
        scores = []
        steps = []
        for seed in EVAL_SEEDS:
            f, sc, st = self.evaluate_genome_on_seed(ind.genome, seed)
            fits.append(f)
            scores.append(sc)
            steps.append(st)

        ind.fitness = float(np.mean(fits))
        ind.score = int(max(scores))
        ind.steps = int(np.mean(steps))
        self.total_evaluations += len(EVAL_SEEDS)

    def evaluate_population(self):
        """Evaluate all 10 individuals across all seeds and rank them."""
        for ind in self.population:
            self.evaluate_individual_multi_seed(ind)

        # Sort descending by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        if self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
            np.save(PARAMS_PATH, self.best_individual.genome)

        avg_fit = float(np.mean([ind.fitness for ind in self.population]))
        self.history_best_fitness.append(self.best_individual.fitness)
        self.history_avg_fitness.append(avg_fit)

    def evolve_next_generation(self):
        """
        Keep top 3 champions (pure elitism).
        Replicate the top 3 with calibrated mutations to fill slots 4 to 10.
        """
        # Top 3 champions preserved intact
        c1 = copy.deepcopy(self.population[0])
        c2 = copy.deepcopy(self.population[1])
        c3 = copy.deepcopy(self.population[2])

        def mutate(genome, scale=0.08):
            mask = np.random.rand(len(genome)) < 0.3
            noise = np.random.randn(len(genome)) * (GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]) * scale
            return np.clip(genome + mask * noise, GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])

        def crossover(g1, g2):
            alpha = np.random.uniform(0.2, 0.8, size=len(g1))
            return alpha * g1 + (1.0 - alpha) * g2

        new_pop = [
            c1,                                         # Slot 1: Champion 1
            c2,                                         # Slot 2: Champion 2
            c3,                                         # Slot 3: Champion 3
            Individual(mutate(c1.genome, scale=0.05)),  # Slot 4: C1 slight tweak
            Individual(mutate(c1.genome, scale=0.12)),  # Slot 5: C1 wider search
            Individual(mutate(crossover(c1.genome, c2.genome), scale=0.06)), # Slot 6: C1 x C2
            Individual(mutate(c2.genome, scale=0.06)),  # Slot 7: C2 slight tweak
            Individual(mutate(c2.genome, scale=0.12)),  # Slot 8: C2 wider search
            Individual(mutate(crossover(c1.genome, c3.genome), scale=0.06)), # Slot 9: C1 x C3
            Individual(mutate(c3.genome, scale=0.10)),  # Slot 10: C3 exploration
        ]

        self.population = new_pop
        self.generation += 1

    def fast_forward(self, n_generations: int = None, n_evaluations: int = None) -> dict:
        """
        Fast forward off-screen through n_generations or n_evaluations.
        """
        evals_per_gen = self.pop_size * len(EVAL_SEEDS)
        if n_evaluations is not None:
            n_generations = max(1, int(np.ceil(n_evaluations / evals_per_gen)))
        elif n_generations is None:
            n_generations = 10

        t0 = time.time()
        start_gen = self.generation

        for _ in range(n_generations):
            self.evaluate_population()
            self.evolve_next_generation()

        elapsed = time.time() - t0
        rate = (n_generations * evals_per_gen) / max(0.001, elapsed)

        return {
            "gens_run": n_generations,
            "elapsed_sec": elapsed,
            "evals_per_sec": rate,
            "current_gen": self.generation,
            "best_fitness": self.best_individual.fitness,
            "best_score": self.best_individual.score,
        }
