"""Procedural UI components and geometric vector icons for NeuroWorm Snake.

Renders sharp, deterministic icons (14-16 px) using pygame vector primitives
to completely eliminate missing glyph tofu boxes across all operating systems.
"""

import math
import pygame as pg
from game.theme import (
    C_BTN_BG, C_BTN_HOV, C_BTN_TXT, C_BORDER, C_ACCENT, C_DANGER,
    C_EGG_GOLD, C_STARVE, C_MUTED
)


def draw_icon(surface: pg.Surface, name: str, x: int, y: int, size: int = 14, color: tuple = None) -> None:
    """
    Draw a clean 14-16 px geometric vector icon onto the target surface.

    Supported icons:
      - 'play': right-pointing triangle
      - 'pause': two vertical bars
      - 'fast_forward': dual right-pointing triangles
      - 'skip': triangle + vertical end bar
      - 'reset': circular arrow / refresh loop
      - 'skull' / 'x': crisp cross lines (dead status)
      - 'egg': glowing oval spore
      - 'clock': circle with clock hands
      - 'wall': stone masonry block
      - 'alive' / 'check': green active dot / pulse
      - 'starved': hunger flame / warning triangle
    """
    s = size
    col = color or (230, 220, 205)

    if name == "play":
        # Right pointing equilateral triangle
        pts = [(x + 2, y + 1), (x + s - 2, y + s // 2), (x + 2, y + s - 1)]
        pg.draw.polygon(surface, col, pts)

    elif name == "pause":
        # Two vertical bars
        bw = max(2, s // 4)
        pg.draw.rect(surface, col, (x + 2, y + 2, bw, s - 4))
        pg.draw.rect(surface, col, (x + s - bw - 2, y + 2, bw, s - 4))

    elif name in ("fast_forward", "ff"):
        # Dual triangles
        mid_x = x + s // 2
        pts1 = [(x + 1, y + 2), (mid_x, y + s // 2), (x + 1, y + s - 2)]
        pts2 = [(mid_x, y + 2), (x + s - 1, y + s // 2), (mid_x, y + s - 2)]
        pg.draw.polygon(surface, col, pts1)
        pg.draw.polygon(surface, col, pts2)

    elif name == "skip":
        # Triangle + end bar
        mid_x = x + s - 4
        pts = [(x + 2, y + 2), (mid_x, y + s // 2), (x + 2, y + s - 2)]
        pg.draw.polygon(surface, col, pts)
        pg.draw.rect(surface, col, (mid_x + 1, y + 2, 2, s - 4))

    elif name == "reset":
        # Circular arc with arrow head
        cx, cy = x + s // 2, y + s // 2
        rad = s // 2 - 2
        arc_rect = pg.Rect(cx - rad, cy - rad, rad * 2, rad * 2)
        pg.draw.arc(surface, col, arc_rect, 0.4, 2 * math.pi - 0.5, 2)
        # Arrowhead at top
        arr_pts = [(cx + 2, cy - rad - 2), (cx + 5, cy - rad + 1), (cx + 1, cy - rad + 3)]
        pg.draw.polygon(surface, col, arr_pts)

    elif name in ("skull", "x", "dead"):
        # Crisp X
        col = color or C_DANGER
        pg.draw.line(surface, col, (x + 3, y + 3), (x + s - 3, y + s - 3), 2)
        pg.draw.line(surface, col, (x + s - 3, y + 3), (x + 3, y + s - 3), 2)

    elif name == "egg":
        # Glowing oval spore
        col = color or C_EGG_GOLD
        egg_rect = pg.Rect(x + 2, y + 1, s - 4, s - 2)
        pg.draw.ellipse(surface, col, egg_rect)
        # Bright nucleus
        pg.draw.circle(surface, (255, 255, 230), (x + s // 2, y + s // 2), max(1, s // 5))

    elif name == "clock":
        # Clock circle + hands
        cx, cy = x + s // 2, y + s // 2
        rad = s // 2 - 1
        pg.draw.circle(surface, col, (cx, cy), rad, 1)
        pg.draw.line(surface, col, (cx, cy), (cx, cy - rad + 3), 1)
        pg.draw.line(surface, col, (cx, cy), (cx + rad - 3, cy), 1)

    elif name == "wall":
        # Masonry stone block
        col = color or (170, 150, 130)
        pg.draw.rect(surface, col, (x + 1, y + 2, s - 2, s - 4), 1)
        pg.draw.line(surface, col, (x + 1, y + s // 2), (x + s - 1, y + s // 2), 1)
        pg.draw.line(surface, col, (x + s // 2, y + 2), (x + s // 2, y + s // 2), 1)

    elif name in ("alive", "check"):
        # Green luminous status dot
        col = color or C_ACCENT
        cx, cy = x + s // 2, y + s // 2
        pg.draw.circle(surface, col, (cx, cy), s // 3)

    elif name in ("starved", "hungry"):
        # Starvation warning symbol (small amber triangle)
        col = color or C_STARVE
        pts = [(x + s // 2, y + 2), (x + 2, y + s - 2), (x + s - 2, y + s - 2)]
        pg.draw.polygon(surface, col, pts, 1)
        pg.draw.line(surface, col, (x + s // 2, y + 5), (x + s // 2, y + s - 5), 1)

    else:
        # Default dot
        cx, cy = x + s // 2, y + s // 2
        pg.draw.circle(surface, col, (cx, cy), 3)


class Button:
    """High-contrast UI button with vector icon, text label, and keycap hint."""

    def __init__(self, rect, text: str, hotkey: str = "", icon: str = None):
        self.rect = pg.Rect(rect)
        self.text = text
        self.hotkey = hotkey
        self.icon = icon

    def draw(self, surface: pg.Surface, font: pg.font.Font, is_hovered: bool = False):
        bg_col = C_BTN_HOV if is_hovered else C_BTN_BG
        pg.draw.rect(surface, bg_col, self.rect, border_radius=4)
        border_col = (140, 105, 75) if is_hovered else C_BORDER
        pg.draw.rect(surface, border_col, self.rect, 1, border_radius=4)

        cy = self.rect.centery
        icon_s = 12 if self.icon else 0
        txt_surf = font.render(self.text, True, C_BTN_TXT)
        key_surf = font.render(f"[{self.hotkey}]", True, C_MUTED) if self.hotkey else None

        needed_w = (icon_s + 6 if self.icon else 0) + txt_surf.get_width() + (key_surf.get_width() + 6 if key_surf else 0)

        if needed_w + 12 > self.rect.width:
            # Compact mode: fit gracefully without overlapping
            if self.icon and key_surf:
                tot_w = icon_s + 4 + key_surf.get_width()
                sx = self.rect.centerx - tot_w // 2
                draw_icon(surface, self.icon, sx, cy - icon_s // 2, size=icon_s, color=C_BTN_TXT)
                surface.blit(key_surf, (sx + icon_s + 4, cy - key_surf.get_height() // 2))
            else:
                surface.blit(txt_surf, (self.rect.centerx - txt_surf.get_width() // 2, cy - txt_surf.get_height() // 2))
        else:
            # Spacious mode: icon + text on left, hotkey on right
            x_cursor = self.rect.x + 8
            if self.icon:
                draw_icon(surface, self.icon, x_cursor, cy - icon_s // 2, size=icon_s, color=C_BTN_TXT)
                x_cursor += icon_s + 6
            surface.blit(txt_surf, (x_cursor, cy - txt_surf.get_height() // 2))
            if key_surf:
                kx = self.rect.right - key_surf.get_width() - 8
                surface.blit(key_surf, (kx, cy - key_surf.get_height() // 2))

    def is_clicked(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)
