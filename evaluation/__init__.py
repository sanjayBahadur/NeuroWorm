"""Evaluation module for reproducible benchmarking and ablation testing."""

from evaluation.agents import (
    run_random_walk,
    run_greedy_rule,
    run_worm_brain,
    make_evaluation_network,
)

__all__ = [
    "run_random_walk",
    "run_greedy_rule",
    "run_worm_brain",
    "make_evaluation_network",
]
