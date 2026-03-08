"""
NeurWorm Snake — Main Demo Entry Point

Runs the Snake game controlled by the C. elegans connectome simulation.
Opens two windows:
  Left:  The Snake game (soil-themed pixel art)
  Right: Real-time neural firing rate heatmap

Usage:
    python demo.py              # worm brain controls the snake
    python demo.py --human      # human keyboard control
    python demo.py --params training/best_params.npy  # use trained params
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="NeurWorm Snake Demo")
    parser.add_argument("--human",  action="store_true",
                        help="Human keyboard control instead of worm brain")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to best_params.npy for worm encoding")
    parser.add_argument("--grid",   type=int, default=20,
                        help="Grid size (default 20)")
    parser.add_argument("--fps",    type=int, default=10,
                        help="Game FPS (default 10)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable neural dashboard (faster)")
    args = parser.parse_args()

    # Load encoding parameters
    if args.params and Path(args.params).exists():
        theta = np.load(args.params)
        print(f"Loaded params from {args.params}: {theta.round(3)}")
    else:
        from training.optimize_encoding import DEFAULT_PARAMS
        theta = DEFAULT_PARAMS
        print("Using default encoding parameters.")

    # Import game (pygame) only here to keep imports clean
    from game.snake_game import SnakeGame
    game = SnakeGame(
        grid_size   = args.grid,
        fps         = args.fps,
        human_mode  = args.human,
        show_neural = not args.no_viz,
        theta       = theta,
    )
    game.run()


if __name__ == "__main__":
    main()
