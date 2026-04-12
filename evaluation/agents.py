"""
Agents and environment runners for benchmark evaluation and ablation testing.
"""

from typing import Tuple, Optional
import numpy as np

from connectome.loader import load_connectome
from simulation.lif_network import CElegansLIF
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from simulation.synaptic_circuit import SynapticCircuit
from game.snake_env import SnakeEnv

# Cache raw connectome data once to avoid repeated disk reads
_CACHED_CONNECTOME: Optional[Tuple[np.ndarray, np.ndarray, list[str]]] = None


def get_cached_connectome() -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Return the cached canonical C. elegans connectome dataset."""
    global _CACHED_CONNECTOME
    if _CACHED_CONNECTOME is None:
        _CACHED_CONNECTOME = load_connectome()
    return _CACHED_CONNECTOME


def make_evaluation_network(wiring: str = "full", shuffle_seed: int = 0) -> CElegansLIF:
    """
    Construct a CElegansLIF instance with the requested wiring ablation mode.

    Args:
        wiring: 'full' (canonical connectome),
                'isolated' (zero connectome, only the 36 circuit pathways),
                'shuffled' (degree-preserving shuffled synaptic wiring).
        shuffle_seed: Random seed for deterministic wiring shuffling.
    """
    syn0, gap0, names = get_cached_connectome()
    syn = syn0.copy()
    gap = gap0.copy()

    if wiring == "isolated":
        # Delete all background connectome connections; only the 36 hand-wired synapses exist
        syn.fill(0.0)
        gap.fill(0.0)
    elif wiring == "shuffled":
        # Out-degree preserving shuffle: each presynaptic neuron preserves its number
        # of outgoing synapses and weight values, but targets randomly chosen postsynaptic neurons.
        rng = np.random.default_rng(shuffle_seed)
        shuffled_syn = np.zeros_like(syn)
        N = syn.shape[0]

        for pre_col in range(N):
            post_rows = np.where(syn[:, pre_col] != 0.0)[0]
            if len(post_rows) == 0:
                continue
            weights = syn[post_rows, pre_col].copy()
            # Candidate targets (avoid self-synapses if original had no self-synapse)
            candidates = [r for r in range(N) if r != pre_col]
            if len(candidates) >= len(post_rows):
                new_post = rng.choice(candidates, size=len(post_rows), replace=False)
            else:
                new_post = rng.choice(N, size=len(post_rows), replace=False)
            shuffled_syn[new_post, pre_col] = weights

        syn = shuffled_syn
    elif wiring != "full":
        raise ValueError(f"Unknown wiring mode: {wiring}. Expected 'full', 'isolated', or 'shuffled'.")

    return CElegansLIF(syn, gap, names)


def run_random_walk(seed: int, grid_size: int = 20, max_steps: int = 1000) -> Tuple[int, int]:
    """Uniformly random action baseline."""
    env = SnakeEnv(grid_size=grid_size, seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()

    for _ in range(max_steps):
        action = rng.choice(env.DIRECTIONS)
        obs, done = env.step(action)
        if done:
            break

    return int(env.score), int(env.steps)


def run_greedy_rule(
    seed: int,
    grid_size: int = 20,
    max_steps: int = 1000,
    starve_limit: int = 150,
) -> Tuple[int, int]:
    """
    Full-board greedy heuristic baseline (~15 lines).
    Moves towards the food pellet while avoiding immediate collisions.
    Serves as an upper-bound reference for the game environment.
    """
    env = SnakeEnv(grid_size=grid_size, seed=seed)
    obs = env.reset()
    steps_since_food = 0
    last_score = 0

    for _ in range(max_steps):
        head, food, body, cur_dir = obs
        best_action = None
        min_dist = float("inf")

        for action in env.DIRECTIONS:
            # Cannot reverse direction into neck
            if action == env.OPPOSITES[cur_dir]:
                continue
            nx = head[0] + env.DX[action]
            ny = head[1] + env.DY[action]
            next_pos = (nx, ny)

            # Avoid out of bounds and self-body collisions
            if not (0 <= nx < grid_size and 0 <= ny < grid_size) or next_pos in body[:-1]:
                continue

            dist = abs(nx - food[0]) + abs(ny - food[1])
            if dist < min_dist:
                min_dist = dist
                best_action = action

        # If trapped or no legal safe move, terminate
        if best_action is None:
            break

        obs, done = env.step(best_action)
        steps_since_food += 1

        if env.score > last_score:
            last_score = env.score
            steps_since_food = 0

        if done or steps_since_food > starve_limit:
            break

    return int(env.score), int(env.steps)


def run_worm_brain(
    seed: int,
    genome: np.ndarray,
    wiring: str = "full",
    grid_size: int = 20,
    max_steps: int = 1000,
    starve_limit: int = 150,
) -> Tuple[int, int]:
    """
    Simulates the biological connectome agent across an evaluation episode.

    Args:
        seed: Random seed initializing snake and food layout.
        genome: 36-element array of synaptic conductances (nS).
        wiring: 'full', 'isolated', or 'shuffled'.
        grid_size: Dimension of square grid (default 20).
        max_steps: Maximum step cutoff.
        starve_limit: Maximum steps allowed without consuming an egg before terminating.
    """
    env = SnakeEnv(grid_size=grid_size, seed=seed)
    net = make_evaluation_network(wiring=wiring)
    SynapticCircuit(net).apply_genome(genome)

    encoder = SensoryEncoder(net, grid_size=grid_size)
    decoder = MotorDecoder(net)

    obs = env.reset()
    direction = "RIGHT"
    steps_since_food = 0
    last_score = 0

    for _ in range(max_steps):
        head, food, body, cur_dir = obs
        net.reset_counts()
        encoder.encode(head, food, body, direction)
        for _ in range(8):
            net.step()

        direction = decoder.decode(direction)
        obs, done = env.step(direction)
        steps_since_food += 1

        if env.score > last_score:
            last_score = env.score
            steps_since_food = 0

        if done or steps_since_food > starve_limit:
            break

    return int(env.score), int(env.steps)
