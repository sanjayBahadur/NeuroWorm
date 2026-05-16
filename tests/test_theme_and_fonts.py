"""Tests for typography, OFL font bundling, vector icons, and emoji prohibition (B1)."""

import os
import re
from pathlib import Path
import pytest
import pygame as pg

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from game.theme import get_font, C_TEXT, C_MUTED, C_CARD_BG
from game.ui import draw_icon, Button


def test_zero_emojis_in_game_and_viz_source():
    """Fail if any unicode emoji code points appear in game/ or viz/ source files."""
    repo_root = Path(__file__).parent.parent
    emoji_pattern = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F]"
    )

    violations = []
    for pkg in ("game", "viz"):
        for py_path in (repo_root / pkg).rglob("*.py"):
            text = py_path.read_text(encoding="utf-8")
            for idx, line in enumerate(text.splitlines(), start=1):
                matches = emoji_pattern.findall(line)
                if matches:
                    violations.append(f"{py_path.name}:{idx} {matches} -> {line.strip()}")

    assert not violations, f"Found emoji code points: {violations}"


def test_bundled_ofl_font_exists_and_loads():
    """Verify that bundled OFL font files exist and load correctly via get_font."""
    pg.font.init()
    fonts_dir = Path(__file__).parent.parent / "assets" / "fonts"
    assert (fonts_dir / "OFL.txt").exists(), "OFL.txt license missing"
    assert (fonts_dir / "JetBrainsMono-Regular.ttf").exists(), "Regular TTF missing"
    assert (fonts_dir / "JetBrainsMono-Bold.ttf").exists(), "Bold TTF missing"

    # Test loading different roles
    font_title = get_font("title", 14)
    font_body = get_font("body", 12)
    font_btn = get_font("button", 12)

    assert font_title is not None
    assert font_body is not None
    assert font_btn is not None

    # Text rendering sanity
    surf = font_body.render("NeuroWorm Clean Typography", True, (255, 255, 255))
    assert surf.get_width() > 50
    assert surf.get_height() >= 12


def test_vector_icons_render_without_error():
    """Verify all procedural vector icons render onto a surface without errors."""
    pg.init()
    surf = pg.Surface((200, 200))
    icons = [
        "play", "pause", "fast_forward", "skip", "reset",
        "skull", "x", "egg", "clock", "wall", "alive", "check", "starved"
    ]
    for idx, name in enumerate(icons):
        draw_icon(surf, name, 10 + (idx % 5) * 20, 10 + (idx // 5) * 20, size=14)

    btn = Button((10, 150, 100, 24), "Test", "T", icon="play")
    font = get_font("button", 12)
    btn.draw(surf, font, is_hovered=False)
    assert btn.is_clicked((15, 155))
    assert not btn.is_clicked((2, 2))
