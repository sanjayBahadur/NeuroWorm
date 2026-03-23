"""
NeuroWorm Snake — Main Demo Entry Point

Runs the Snake game controlled by the C. elegans connectome simulation.
Features:
  Left:  Soil-themed pixel-art game with articulated worm and pulsing egg
  Right: Interactive Dashboard with live neural heatmap and GA evolution controls

Usage:
    python demo.py              # biological connectome brain plays snake
    python demo.py --human      # human keyboard control (WASD / Arrows)
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="NeuroWorm Snake Demo")
    parser.add_argument("--human",  action="store_true",
                        help="Human keyboard control instead of worm brain")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to best_params.npy for worm encoding")
    parser.add_argument("--grid",   type=int, default=20,
                        help="Grid size (default 20)")
    parser.add_argument("--fps",    type=int, default=12,
                        help="Game FPS (default 12)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable neural dashboard (faster)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Window scale factor (e.g. 1.5, 1.75, 2.0; auto-detected if not specified)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Launch in true fullscreen mode (toggle with F11 anytime)")
    args = parser.parse_args()

    # Load encoding parameters
    if args.params and Path(args.params).exists():
        theta = np.load(args.params)
        print(f"Loaded params from {args.params}: {theta.round(3)}")
    else:
        from training.genetic_algorithm import DEFAULT_GENOME
        theta = DEFAULT_GENOME
        print("Using default encoding parameters.")

    # Import game (pygame) only here to keep imports clean
    from game.snake_game import SnakeGame
    game = SnakeGame(
        grid_size   = args.grid,
        fps         = args.fps,
        human_mode  = args.human,
        show_neural = not args.no_viz,
        theta       = theta,
        scale       = args.scale,
        fullscreen  = args.fullscreen,
    )
    game.run()


if __name__ == "__main__":
    main()
