"""Unit tests for the reproducible evaluation harness and ablation suite."""

import pytest
import numpy as np

from evaluation.agents import run_random_walk, run_greedy_rule, run_worm_brain, make_evaluation_network
from simulation.synaptic_circuit import DEFAULT_SYNAPTIC_GENOME
from training.genetic_algorithm import EVAL_SEEDS
from eval import format_markdown_table, evaluate_agent


def test_seed_determinism():
    """Verify evaluation runners are strictly deterministic given the same seed."""
    seed = 1005
    score1, steps1 = run_random_walk(seed=seed, max_steps=100)
    score2, steps2 = run_random_walk(seed=seed, max_steps=100)
    assert score1 == score2
    assert steps1 == steps2

    g_score1, g_steps1 = run_greedy_rule(seed=seed, max_steps=100)
    g_score2, g_steps2 = run_greedy_rule(seed=seed, max_steps=100)
    assert g_score1 == g_score2
    assert g_steps1 == g_steps2

    w_score1, w_steps1 = run_worm_brain(seed=seed, genome=DEFAULT_SYNAPTIC_GENOME, wiring="full", max_steps=100)
    w_score2, w_steps2 = run_worm_brain(seed=seed, genome=DEFAULT_SYNAPTIC_GENOME, wiring="full", max_steps=100)
    assert w_score1 == w_score2
    assert w_steps1 == w_steps2


def test_greedy_beats_random():
    """Verify greedy baseline consistently outperforms uniform random walk."""
    seeds = [1001, 1002, 1003, 1004, 1005]
    random_scores = [run_random_walk(s, max_steps=200)[0] for s in seeds]
    greedy_scores = [run_greedy_rule(s, max_steps=200)[0] for s in seeds]

    assert np.mean(greedy_scores) > np.mean(random_scores)
    assert np.mean(greedy_scores) > 5.0


def test_held_out_seeds_disjoint_from_training():
    """Verify held-out evaluation seeds do not overlap genetic algorithm training seeds."""
    held_out_seeds = list(range(1000, 1030))
    training_seeds = set(EVAL_SEEDS)
    overlap = set(held_out_seeds).intersection(training_seeds)
    assert len(overlap) == 0, f"Found overlapping seeds: {overlap}"


def test_wiring_modes_construct():
    """Verify all 3 wiring modes construct valid LIF networks."""
    for mode in ["full", "isolated", "shuffled"]:
        net = make_evaluation_network(wiring=mode, shuffle_seed=42)
        assert net.W_syn.shape == (302, 302)
        if mode == "isolated":
            assert np.count_nonzero(net.W_syn) == 0
        elif mode in ["full", "shuffled"]:
            assert np.count_nonzero(net.W_syn) > 0


def test_format_markdown_table():
    """Verify format_markdown_table renders expected markdown structure."""
    dummy_results = [
        {
            "agent": "Test Agent",
            "mean_score": 2.50,
            "ci_95": 0.40,
            "max_score": 5,
            "zero_score_pct": 10.0,
            "mean_steps": 120.0,
        }
    ]
    md = format_markdown_table(dummy_results)
    assert "| Agent | Mean Score | 95% CI | Max | Zero-Score (%) | Mean Steps |" in md
    assert "| Test Agent | 2.50 ± 0.40 | ±0.40 | 5 | 10.0% | 120.0 |" in md
