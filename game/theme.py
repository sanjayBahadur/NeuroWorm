"""Central design system tokens and typography for NeuroWorm Snake.

Enforces high-contrast readability (>= 4.5:1 on dark soil substrate)
and bundles OFL-licensed typography with fallback protection.
"""

from pathlib import Path
import pygame as pg

# ─── COLOR PALETTE TOKENS ─────────────────────────────────────────────────────
# Dark Soil Substrate & Structure
C_BG            = (14, 9, 6)        # Deepest underground soil
C_ARENA_SOIL    = (22, 14, 9)       # Playable soil field
C_PANEL_BG      = (20, 14, 10)      # Right-hand telemetry & dashboard panel
C_CARD_BG       = (28, 19, 13)      # Container cards
C_CARD_HOVER    = (38, 26, 18)      # Interactive card hover
C_BORDER        = (62, 44, 30)      # Subtle structural border
C_BORDER_LIGHT  = (125, 92, 60)     # Focused / active border outline
C_WALL_STONE    = (18, 12, 8)       # Impassable border stone band

# Typography & Text Contrast (minimum 12 px virtual resolution)
C_TEXT          = (245, 235, 220)   # High-contrast primary cream text (>12:1 ratio)
C_MUTED         = (165, 148, 132)   # Secondary guidance / label text (>4.5:1 ratio)
C_HEADER        = (255, 225, 130)   # Bright golden title & section heading

# Semantic States & Indicators
C_ACCENT        = (0, 230, 140)     # Bioluminescent emerald (success / alive / high)
C_DANGER        = (255, 80, 80)     # Nociceptive crimson (wall / collision / dead)
C_STARVE        = (255, 160, 50)    # Amber warning (hunger / starving)
C_EGG_GOLD      = (255, 215, 75)    # Nutrient spore glow
C_EGG_CORE      = (255, 250, 210)   # Nutrient spore nucleus

# Interactive Button Palette
C_BTN_BG        = (36, 24, 16)      # Idle button background
C_BTN_HOV       = (54, 36, 24)      # Hover button background
C_BTN_TXT       = (240, 230, 215)   # Button text

# ─── FONT LOADER WITH OFL FALLBACK ───────────────────────────────────────────
_FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"
_FONT_REGULAR_PATH = _FONTS_DIR / "JetBrainsMono-Regular.ttf"
_FONT_BOLD_PATH = _FONTS_DIR / "JetBrainsMono-Bold.ttf"

_FONT_CACHE: dict[tuple[str, int], pg.font.Font] = {}


def get_font(role: str = "body", size: int | None = None) -> pg.font.Font:
    """
    Retrieve cached pygame Font for a given typography role.

    Roles:
      - 'title': bold header (default 15 px)
      - 'card_title': bold card section header (default 13 px)
      - 'body': regular readable text (default 12 px)
      - 'button': bold control text (default 12 px)
      - 'small': compact metric text (default 12 px)
      - 'mono': regular monospace text (default 12 px)

    Falls back safely to pygame system font if bundled files are inaccessible.
    """
    if not pg.font.get_init():
        pg.font.init()

    # Determine default size and weight
    is_bold = role in ("title", "card_title", "button", "bold")
    if size is None:
        if role == "title":
            size = 15
        elif role == "card_title":
            size = 13
        else:
            size = 12

    # Ensure minimum rendered text height is at least 12 px
    size = max(12, size)
    cache_key = (role, size)

    if cache_key in _FONT_CACHE:
        cached = _FONT_CACHE[cache_key]
        try:
            cached.get_height()
            return cached
        except Exception:
            _FONT_CACHE.pop(cache_key, None)

    target_path = _FONT_BOLD_PATH if is_bold else _FONT_REGULAR_PATH
    font_obj = None

    if target_path.exists():
        try:
            font_obj = pg.font.Font(str(target_path), size)
        except Exception:
            font_obj = None

    if font_obj is None:
        # Fallback to default pygame font
        font_obj = pg.font.Font(None, size)
        if is_bold:
            font_obj.set_bold(True)

    _FONT_CACHE[cache_key] = font_obj
    return font_obj
