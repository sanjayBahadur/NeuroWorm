"""
Headless Screenshot & Animated GIF Capture Tool for NeuroWorm Snake.

Generates documentation assets:
  - assets/screenshot-arena.png (focused hero worm, ghosts, scent rings, whisker rays)
  - assets/screenshot-panel.png (3 cards: Selected worm, Leaderboard, Learning curve)
  - assets/screenshot-full.png (combined 1064x704 virtual frame)
  - assets/screenshot-death.png (death burst and cause label)
  - assets/screenshot-gen-end.png (non-blocking generation-end banner)
  - assets/demo.gif (smooth 60->15fps capture, < 5MB)
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
from pathlib import Path
import pygame as pg
from PIL import Image

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.layout import GAME_W, GAME_H, VIRTUAL_W, VIRTUAL_H
from game.snake_game import SnakeGame

ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def surface_to_image(surf: pg.Surface) -> Image.Image:
    """Convert a pygame Surface to a PIL Image."""
    raw = pg.image.tobytes(surf, "RGB")
    return Image.frombytes("RGB", (surf.get_width(), surf.get_height()), raw)


def capture_all() -> None:
    print("Initializing headless NeuroWorm Snake simulation...")
    game = SnakeGame(grid_size=20, fps=15, seed=42)
    game._init_pygame()

    # 1. Fast forward 8 generations so the learning curve has real historical data
    print("Fast-forwarding 8 generations to build learning curve data...")
    for _ in range(8):
        game.trigger_fast_forward(10)
    assert len(game.history) >= 8, f"Expected >= 8 history entries, got {len(game.history)}"

    # 2. Run simulation steps to let hero worm move, sense, and approach food
    print("Simulating active generation...")
    frames_for_gif = []
    death_captured = False
    gen_end_captured = False

    # Render loop
    for tick in range(120):
        # Step simulation
        for w in game.worms:
            if hasattr(w, "obs") and len(w.obs) > 2:
                w.prev_body = list(w.obs[2])

        gen_over = game.step_simulation()

        # Follow leader focus
        if game.focus_mode == "follow":
            alive = [w for w in game.worms if w.alive]
            best_w = max(alive, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness)) if alive else max(game.worms, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness))
            game.focused_idx = best_w.id - 1

        # Draw frame
        game.virtual_screen.fill((20, 14, 10))
        gen_tag = f"GEN #{game.ga.generation} ({'SOLO' if game.solo_mode else 'SWARM'})"
        game_surf = game.renderer.render_multi_worms(
            game.worms,
            game.shared_food,
            gen_info=gen_tag,
            focused_idx=game.focused_idx,
            solo_mode=game.solo_mode,
            interp_alpha=1.0,
            focus_mode=game.focus_mode,
            dt=0.066,
        )
        game.virtual_screen.blit(game_surf, (0, 0))

        if game.state == "GEN_END":
            game._draw_gen_end_banner()

        game._draw_dashboard((0, 0))

        img = surface_to_image(game.virtual_screen)

        # Record frame for GIF (sample every 2 ticks -> ~15 FPS)
        if tick % 2 == 0:
            # Scale down slightly for optimized GIF file size (532x352)
            gif_frame = img.resize((VIRTUAL_W // 2, VIRTUAL_H // 2), Image.Resampling.BILINEAR)
            frames_for_gif.append(gif_frame)

        # Capture mid-run hero shot around tick 35
        if tick == 35:
            full_path = ASSETS_DIR / "screenshot-full.png"
            arena_path = ASSETS_DIR / "screenshot-arena.png"
            panel_path = ASSETS_DIR / "screenshot-panel.png"

            img.save(full_path)
            arena_img = img.crop((0, 0, GAME_W, GAME_H))
            arena_img.save(arena_path)
            panel_img = img.crop((GAME_W, 0, VIRTUAL_W, VIRTUAL_H))
            panel_img.save(panel_path)
            print(f"Saved: {full_path}")
            print(f"Saved: {arena_path}")
            print(f"Saved: {panel_path}")

        # Capture death event if active
        if not death_captured and len(game.renderer.death_events) > 0:
            death_path = ASSETS_DIR / "screenshot-death.png"
            img.save(death_path)
            print(f"Saved: {death_path}")
            death_captured = True

        if gen_over and not gen_end_captured:
            game.start_generation_end()
            game._draw_gen_end_banner()
            end_img = surface_to_image(game.virtual_screen)
            end_path = ASSETS_DIR / "screenshot-gen-end.png"
            end_img.save(end_path)
            print(f"Saved: {end_path}")
            gen_end_captured = True

    # If death wasn't naturally captured in 120 ticks, trigger one artificially for screenshot
    if not death_captured:
        print("Capturing death event...")
        game.worms[2].alive = False
        game.worms[2].death_cause = "wall"
        # Render frame with death event
        game.renderer.render_multi_worms(
            game.worms,
            game.shared_food,
            focused_idx=game.focused_idx,
            dt=0.1,
        )
        game.virtual_screen.blit(game.renderer.surface, (0, 0))
        game._draw_dashboard((0, 0))
        death_img = surface_to_image(game.virtual_screen)
        death_path = ASSETS_DIR / "screenshot-death.png"
        death_img.save(death_path)
        print(f"Saved: {death_path}")

    # If gen_end wasn't naturally captured, trigger banner
    if not gen_end_captured:
        print("Capturing generation-end banner...")
        game.start_generation_end()
        game.virtual_screen.blit(game.renderer.surface, (0, 0))
        game._draw_gen_end_banner()
        game._draw_dashboard((0, 0))
        end_img = surface_to_image(game.virtual_screen)
        end_path = ASSETS_DIR / "screenshot-gen-end.png"
        end_img.save(end_path)
        print(f"Saved: {end_path}")

    # 3. Save animated GIF
    if frames_for_gif:
        gif_path = ASSETS_DIR / "demo.gif"
        print(f"Compiling animated GIF ({len(frames_for_gif)} frames)...")
        # Quantize to adaptive web palette for crisp pixels and small size
        quantized_frames = [
            f.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
            for f in frames_for_gif
        ]
        quantized_frames[0].save(
            gif_path,
            save_all=True,
            append_images=quantized_frames[1:],
            duration=66,  # ~15 FPS
            loop=0,
            optimize=True,
        )
        file_size_mb = gif_path.stat().st_size / (1024 * 1024)
        print(f"Saved: {gif_path} ({file_size_mb:.2f} MB)")
        assert file_size_mb < 5.0, f"GIF size {file_size_mb:.2f} MB exceeds 5 MB limit!"

    pg.quit()
    print("Capture complete!")


if __name__ == "__main__":
    capture_all()
