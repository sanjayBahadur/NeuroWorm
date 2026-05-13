"""Unit tests for generation end state machine (A4) and learning history recording (A5)."""

import os
import re
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pytest
from game.snake_game import SnakeGame


def test_no_blocking_wait_or_sleep_in_game_package():
    """Grep guard: assert no pg.time.wait or time.sleep calls exist in game/ source files."""
    game_dir = Path(__file__).parent.parent / "game"
    pattern = re.compile(r"\b(time\.sleep|time\.wait|pg\.time\.wait|pygame\.time\.wait)\b")

    violations = []
    for py_file in game_dir.glob("*.py"):
        content = py_file.read_text()
        for idx, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{py_file.name}:{idx} {line.strip()}")

    assert not violations, f"Found blocking sleep/wait in game/ package: {violations}"


def test_generation_cap_triggers_gen_end_and_timeout():
    """Verify state transitions: RUNNING -> GEN_END (survivors get death_cause='timeout') -> RUNNING."""
    game = SnakeGame(grid_size=15, fps=30, seed=42)
    assert game.state == "RUNNING"
    assert len(game.history) == 0

    # Advance steps to 399
    game.generation_steps = 399
    # Ensure at least 1 worm is still alive
    assert any(w.alive for w in game.worms)

    # 400th step triggers generation cap
    gen_over = game.step_simulation()
    assert gen_over, "400-tick cap should signal generation completion"
    assert game.generation_steps >= 400

    # Trigger generation end
    game.start_generation_end()
    assert game.state == "GEN_END"
    assert "Generation 1 finished" in game.gen_end_banner_text

    # Surviving worms must now have death_cause='timeout'
    timeout_worms = [w for w in game.worms if w.death_cause == "timeout"]
    assert len(timeout_worms) > 0, "Surviving worms at cap must be marked with death_cause='timeout'"

    # History must record generation 1
    assert len(game.history) == 1
    h = game.history[0]
    assert h["gen"] == 1
    assert "best_score" in h
    assert "avg_score" in h
    assert "best_fitness" in h

    # Advancing timer past duration transitions to next generation and back to RUNNING
    game.gen_end_timer = game.gen_end_duration + 0.1
    game._evolve_and_next_generation()
    assert game.state == "RUNNING"
    assert game.ga.generation == 2
    assert game.generation_steps == 0
    # Living worms reset for gen 2
    assert all(w.alive for w in game.worms)
    assert all(w.death_cause is None for w in game.worms)
