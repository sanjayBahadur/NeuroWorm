"""
Benchmark: Worm Brain vs DQN Baseline.

Runs both agents for N episodes and compares:
  - Mean food eaten per episode
  - Mean survival steps per episode
  - Score distribution (printed as ASCII histogram)

Usage:
    python benchmark.py --episodes 50
"""

import argparse
import numpy as np
from pathlib import Path
from connectome.loader import build_network
from simulation.sensory_encoder import SensoryEncoder
from simulation.motor_decoder import MotorDecoder
from game.snake_env import SnakeEnv
from training.optimize_encoding import DEFAULT_PARAMS, STEPS_PER_FRAME


def run_worm_episode(theta, grid_size=20, max_steps=5000, seed=None):
    env     = SnakeEnv(grid_size=grid_size, seed=seed)
    net     = build_network()
    encoder = SensoryEncoder(net, grid_size, *theta[:3])
    decoder = MotorDecoder(net, *theta[3:])
    obs     = env.reset()
    direction = "RIGHT"
    for _ in range(max_steps):
        net.reset_counts()
        encoder.encode(*obs, direction)
        for _ in range(STEPS_PER_FRAME):
            net.step()
        direction = decoder.decode(direction)
        obs, done = env.step(direction)
        if done:
            break
    return env.score, env.steps


def print_histogram(scores, label, width=40):
    print(f"\n{label} score distribution:")
    if not scores:
        print("  (no data)")
        return
    max_s = max(scores) + 1
    bins  = range(max_s + 1)
    counts = [scores.count(v) for v in bins]
    max_c  = max(counts) if counts else 1
    for v, c in zip(bins, counts):
        bar = "█" * int(c / max_c * width)
        print(f"  {v:3d} | {bar} ({c})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--grid",     type=int, default=20)
    parser.add_argument("--params",   type=str, default=None)
    args = parser.parse_args()

    if args.params and Path(args.params).exists():
        theta = np.load(args.params)
    else:
        theta = DEFAULT_PARAMS

    print("=" * 50)
    print("  NeurWorm Benchmark: Worm Brain vs DQN")
    print("=" * 50)

    # ── Worm brain ────────────────────────────────────────────────────
    worm_scores, worm_steps = [], []
    for ep in range(args.episodes):
        s, t = run_worm_episode(theta, grid_size=args.grid, seed=ep)
        worm_scores.append(s)
        worm_steps.append(t)
        print(f"  Worm ep {ep+1:3d}: score={s}, steps={t}")

    print(f"\nWorm  — mean score: {np.mean(worm_scores):.2f} "
          f"± {np.std(worm_scores):.2f}")
    print_histogram(worm_scores, "Worm")

    # ── DQN placeholder ───────────────────────────────────────────────
    print("\n[DQN comparison requires running training/dqn_baseline.py first]")
    print("Run: python training/dqn_baseline.py --train")
    print("Then: python benchmark.py")


if __name__ == "__main__":
    main()
