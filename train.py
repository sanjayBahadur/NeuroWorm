"""
Run the Evolution Strategy optimization to find best encoding parameters.

Usage:
    python train.py
    python train.py --generations 100 --lambda 20
"""

import argparse
from training.optimize_encoding import run_es


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--lambda",      type=int, default=10, dest="lambda_")
    parser.add_argument("--sigma",       type=float, default=0.1)
    args = parser.parse_args()
    run_es(n_gen=args.generations, lambda_=args.lambda_, sigma=args.sigma)


if __name__ == "__main__":
    main()
