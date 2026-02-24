"""
Evolution Strategy (1+λ)-ES to optimize sensory/motor encoding parameters.

Optimizes 5 parameters that control how game state maps to neuron currents:
  θ[0] = i_food      (food attraction strength)
  θ[1] = i_danger    (danger repulsion strength)
  θ[2] = i_awa       (ambient food-seeking drive)
  θ[3] = rev_thresh  (reversal command sensitivity)
  θ[4] = dors_thresh (dorsal/ventral split sensitivity)

No neural network weights are modified — the connectome is fixed.
"""

import numpy as np
from pathlib import Path
from connectome.loader import build_network
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from game.snake_env import SnakeEnv

OUTPUT_PATH = Path(__file__).parent / "best_params.npy"

PARAM_BOUNDS = np.array([
    [0.5,  5.0],   # i_food
    [1.0,  8.0],   # i_danger
    [0.0,  3.0],   # i_awa
    [0.4,  0.8],   # rev_thresh
    [0.45, 0.75],  # dors_thresh
])

DEFAULT_PARAMS = np.array([2.0, 3.5, 1.0, 0.60, 0.55])

LAMBDA     = 10
SIGMA      = 0.1
N_GEN      = 50
EVAL_STEPS = 2000
SEEDS      = [42, 7, 99]
STEPS_PER_FRAME = 10   # LIF timesteps per game frame (5 ms of simulation)


def _clip(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def evaluate(theta: np.ndarray, verbose: bool = False) -> float:
    """
    Run one episode per seed and return mean score.

    Args:
        theta:    Parameter vector (5,).
        verbose:  Print per-seed scores if True.
    Returns:
        Mean food-eating score across seeds.
    """
    scores = []
    for seed in SEEDS:
        env = SnakeEnv(grid_size=20, seed=seed)
        net = build_network()
        encoder = SensoryEncoder(
            net, grid_size=20,
            i_food=theta[0], i_danger=theta[1], i_awa=theta[2]
        )
        decoder = MotorDecoder(
            net, rev_thresh=theta[3], dors_thresh=theta[4]
        )

        obs = env.reset()
        direction = "RIGHT"

        for _ in range(EVAL_STEPS):
            net.reset_counts()
            encoder.encode(*obs, direction)
            for _ in range(STEPS_PER_FRAME):
                net.step()
            direction = decoder.decode(direction)
            obs, done = env.step(direction)
            if done:
                break

        scores.append(env.score)
        if verbose:
            print(f"  seed={seed}: score={env.score}")

    return float(np.mean(scores))


def run_es(
    n_gen: int = N_GEN,
    lambda_: int = LAMBDA,
    sigma: float = SIGMA,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run the (1+λ)-Evolution Strategy optimization.

    Returns best parameter vector found.
    """
    theta_best = DEFAULT_PARAMS.copy()
    score_best = evaluate(theta_best)
    if verbose:
        print(f"Gen   0 | Score: {score_best:.2f} | θ={theta_best.round(3)}")

    for gen in range(1, n_gen + 1):
        # Generate λ mutants
        mutants = [
            _clip(theta_best + np.random.randn(5) * sigma)
            for _ in range(lambda_)
        ]
        mut_scores = [evaluate(m) for m in mutants]
        best_idx   = int(np.argmax(mut_scores))

        if mut_scores[best_idx] > score_best:
            theta_best = mutants[best_idx]
            score_best = mut_scores[best_idx]

        if verbose and gen % 5 == 0:
            print(
                f"Gen {gen:3d} | Score: {score_best:.2f} "
                f"| θ={theta_best.round(3)}"
            )

    np.save(OUTPUT_PATH, theta_best)
    if verbose:
        print(f"\n✅ Best params saved to {OUTPUT_PATH}")

    return theta_best


if __name__ == "__main__":
    run_es()
