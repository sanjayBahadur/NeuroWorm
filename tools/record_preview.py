"""
Screen Recording & Preview Generator for NeuroWorm Snake.

Provides high-definition screen recording and an automated, choreographed
preview tour exploring all major features and components of the project:
  1. Subterranean Soil Arena & Nutrient Spore Scent Diffusion
  2. Hero Organism & Dynamic Sensory Whisker Proximity Rays
  3. Ghost Organism Swarm Exploration (50% Alpha)
  4. Real-time Brain Flow Inspector (Sensory -> Command -> Motor Neurons)
  5. Direct Biological Synaptic Wiring Matrix Tab (Option A Neuroevolution)
  6. Multi-Seed Spanning GA Learning Curve & Fast-Forward Engine (100x)
  7. Solo Mode vs Swarm Mode Dynamic Toggling
  8. Interactive Leaderboard & Candidate Selection
  9. Eating VFX (+1 EGG Particles) & Attribution Death Events (#X wall, #X starved)
  10. Keyboard Shortcut Overlay Modal

Usage:
    python tools/record_preview.py --tour                  # Generate full automated component tour (assets/preview.mp4)
    python tools/record_preview.py --tour --gif            # Also export animated preview GIF
    python tools/record_preview.py --interactive           # Play live while recording full session to MP4
    python tools/record_preview.py --x11 --duration 15     # Record X11 desktop display directly via ffmpeg
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import numpy as np
import pygame as pg

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.layout import VIRTUAL_W, VIRTUAL_H, GAME_W, GAME_H
from game.snake_game import SnakeGame

ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


class VideoRecorder:
    """Streams Pygame surfaces directly into an ffmpeg subprocess pipe."""

    def __init__(self, output_path: Path, width: int = VIRTUAL_W, height: int = VIRTUAL_H, fps: int = 30):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libopenh264",
            "-b:v", "2M",
            "-pix_fmt", "yuv420p",
            str(self.output_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_frame(self, surf: pg.Surface):
        if self.proc and self.proc.stdin:
            raw = pg.image.tobytes(surf, "RGB")
            try:
                self.proc.stdin.write(raw)
            except BrokenPipeError:
                pass

    def close(self):
        if self.proc:
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            self.proc.wait()
            self.proc = None
            print(f"Recorded video saved: {self.output_path} ({self.output_path.stat().st_size / (1024 * 1024):.2f} MB)")


def convert_mp4_to_gif(mp4_path: Path, gif_path: Path, fps: int = 15, scale_w: int = 640):
    """Generate high-quality palette-optimized GIF from MP4 using ffmpeg."""
    print(f"Generating palette-optimized GIF at {gif_path}...")
    vf = f"fps={fps},scale={scale_w}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vf", vf,
        str(gif_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"GIF saved: {gif_path} ({size_mb:.2f} MB)")


def run_choreographed_tour(output_mp4: Path, fps: int = 30) -> None:
    """
    Executes a structured component walkthrough showcasing all capabilities of NeuroWorm Snake.
    Total runtime: ~28 seconds (840 frames at 30 fps).
    """
    print("=" * 60)
    print("  LAUNCHING AUTOMATED COMPONENT PREVIEW TOUR")
    print("=" * 60)

    # Use headless dummy video driver so tour can run headlessly and deterministically
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

    game = SnakeGame(grid_size=20, fps=12, seed=42)
    game._init_pygame()

    # Pre-populate history with 6 generations so learning curve begins populated
    print("[1/6] Pre-training GA history to initialize learning curve...")
    for _ in range(6):
        game.trigger_fast_forward(10)

    recorder = VideoRecorder(output_mp4, width=VIRTUAL_W, height=VIRTUAL_H, fps=fps)

    # Timeline choreography (840 frames = 28 seconds at 30 fps)
    total_frames = 840
    font_banner = pg.font.SysFont("monospace", 14, bold=True)
    font_sub = pg.font.SysFont("monospace", 11)

    print(f"[2/6] Recording choreographed tour ({total_frames} frames)...")

    for f in range(total_frames):
        # ── Scripted Actions along the Timeline ──────────────────────
        # 0s - 5s: Multi-Worm Swarm & Scent Diffusion
        if f == 0:
            tour_title = "FEATURE 1: 10-WORM SWARM FORAGING & SCENT DIFFUSION"
            tour_desc = "Hero organism with proximity whiskers + 50% alpha ghost organisms"
            game.focus_mode = "follow"
            game.solo_mode = False
            game.inspector_tab = "flow"

        # 5s - 9s: Candidate Selection & Hero Focusing
        elif f == 150:
            tour_title = "FEATURE 2: CANDIDATE ROSTER SELECTION & FOCUS TRACKING"
            tour_desc = "Inspecting Worm #4 and Worm #7 with dedicated telemetry"
            game.focus_mode = "manual"
            game.focused_idx = 3  # Worm #4

        elif f == 220:
            game.focused_idx = 6  # Worm #7

        # 9s - 14s: Direct Synaptic Weight Matrix Tab (Option A Neuroevolution)
        elif f == 270:
            tour_title = "FEATURE 3: SYNAPTIC CIRCUIT MATRIX (36 EVOLVED WEIGHTS)"
            tour_desc = "Biological connection weights on canonical C. elegans connectome"
            game.inspector_tab = "synapses"

        # 14s - 19s: Solo Mode Isolation
        elif f == 420:
            tour_title = "FEATURE 4: SOLO MODE ISOLATION vs SWARM TOGGLING"
            tour_desc = "Inspect single champion worm navigation with unobstructed rays"
            game.solo_mode = True
            game.inspector_tab = "flow"

        # 19s - 22s: Fast Forward 100x & Learning Curve Growth
        elif f == 570:
            tour_title = "FEATURE 5: FAST-FORWARD 100x & REAL-TIME GA LEARNING CURVE"
            tour_desc = "Evaluating 100 generations across 5-seed multi-seed spanning"
            game.solo_mode = False
            game.trigger_fast_forward(30)

        # 22s - 25s: Dynamic Death Events (Wall Collision & Starvation)
        elif f == 660:
            tour_title = "FEATURE 6: CAUSE-ATTRIBUTED DEATH VFX & HUNGER ARCS"
            tour_desc = "Red impact bursts (#X wall) vs amber metabolic depletion (#X starved)"
            # Trigger clean visual demonstration of wall and starvation events
            game.worms[2].alive = False
            game.worms[2].death_cause = "wall"
            game.worms[4].alive = False
            game.worms[4].death_cause = "starved"

        # 25s - 28s: In-Game Help & Shortcuts Overlay
        elif f == 750:
            tour_title = "FEATURE 7: INTERACTIVE HELP & KEYBOARD CONTROLS MODAL"
            tour_desc = "Full keyboard shortcut legend (Space, +/-, N, F, M, L, 1-0, R, ?)"
            game.show_help = True

        # ── Simulation Step (Decoupled ticks) ────────────────────────
        if f % 2 == 0 and not game.paused and not game.show_help:
            for w in game.worms:
                if hasattr(w, "obs") and len(w.obs) > 2:
                    w.prev_body = list(w.obs[2])
            gen_over = game.step_simulation()
            if gen_over:
                game.start_generation_end()

        # Follow-leader focus if in follow mode
        if game.focus_mode == "follow":
            alive = [w for w in game.worms if w.alive]
            best_w = max(alive, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness)) if alive else max(game.worms, key=lambda w: (w.score, getattr(w, "steps", 0), w.fitness))
            game.focused_idx = best_w.id - 1

        # ── Render Frame ─────────────────────────────────────────────
        dt = 1.0 / fps
        game.renderer.render_multi_worms(
            worms=game.worms,
            food=game.shared_food,
            gen_info=f"GEN #{game.ga.generation}",
            focused_idx=game.focused_idx,
            solo_mode=game.solo_mode,
            interp_alpha=0.5,
            focus_mode=game.focus_mode,
            dt=dt,
        )

        game.virtual_screen.blit(game.renderer.surface, (0, 0))
        game._draw_dashboard(pg.mouse.get_pos())

        if game.state == "GEN_END":
            game._draw_gen_end_banner()

        if game.show_help:
            game._draw_help_overlay()

        # ── Draw Tour Title Banner Overlay at Bottom ─────────────────
        banner_rect = pg.Rect(16, VIRTUAL_H - 52, VIRTUAL_W - 32, 42)
        banner_surf = pg.Surface((banner_rect.w, banner_rect.h), pg.SRCALPHA)
        banner_surf.fill((20, 12, 8, 220))
        pg.draw.rect(banner_surf, (200, 184, 126, 200), (0, 0, banner_rect.w, banner_rect.h), 1, border_radius=4)

        # Title text
        t_title = font_banner.render(tour_title, True, (255, 220, 110))
        banner_surf.blit(t_title, (12, 6))

        # Subtitle text
        t_sub = font_sub.render(tour_desc, True, (200, 190, 175))
        banner_surf.blit(t_sub, (12, 23))

        game.virtual_screen.blit(banner_surf, (banner_rect.x, banner_rect.y))

        # Write frame to video stream
        recorder.write_frame(game.virtual_screen)

        if f % 120 == 0:
            print(f"  Progress: frame {f}/{total_frames} ({f / total_frames * 100:.0f}%)")

    recorder.close()
    print("[3/6] Tour recording completed successfully!")


def run_interactive_recording(output_mp4: Path, fps: int = 30) -> None:
    """Launches interactive game on display and records player gameplay directly to MP4."""
    print("=" * 60)
    print("  LAUNCHING INTERACTIVE GAME WITH REAL-TIME RECORDER")
    print("  Play normally — full video will be saved when you exit.")
    print("=" * 60)

    game = SnakeGame(grid_size=20, fps=12)
    recorder = VideoRecorder(output_mp4, width=VIRTUAL_W, height=VIRTUAL_H, fps=fps)

    # Monkey patch _present_frame to record every virtual frame
    orig_present = game._present_frame

    def hooked_present():
        recorder.write_frame(game.virtual_screen)
        orig_present()

    game._present_frame = hooked_present

    try:
        game.run()
    finally:
        recorder.close()


def run_x11_screen_record(output_mp4: Path, duration_s: int = 15, display: str = ":0.0") -> None:
    """Captures desktop screen via ffmpeg x11grab."""
    print(f"Recording desktop screen on {display} for {duration_s} seconds...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "x11grab",
        "-draw_mouse", "1",
        "-framerate", "30",
        "-i", display,
        "-t", str(duration_s),
        "-c:v", "mpeg4",
        "-q:v", "3",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"X11 screen recording saved: {output_mp4}")


def main():
    parser = argparse.ArgumentParser(description="NeuroWorm Snake — Screen Recording & Preview Tool")
    parser.add_argument("--tour", action="store_true", help="Generate automated 28s feature walkthrough tour")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive game and record gameplay")
    parser.add_argument("--x11", action="store_true", help="Record X11 desktop directly via ffmpeg x11grab")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds for x11 mode (default: 15s)")
    parser.add_argument("--out", type=str, default=str(ASSETS_DIR / "preview.mp4"), help="Output video path")
    parser.add_argument("--gif", action="store_true", help="Also generate an animated GIF preview (preview.gif)")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.interactive:
        run_interactive_recording(out_path)
    elif args.x11:
        run_x11_screen_record(out_path, duration_s=args.duration)
    else:
        # Default: run the full automated component tour
        run_choreographed_tour(out_path)
        if args.gif:
            gif_path = out_path.with_suffix(".gif")
            convert_mp4_to_gif(out_path, gif_path)


if __name__ == "__main__":
    main()
