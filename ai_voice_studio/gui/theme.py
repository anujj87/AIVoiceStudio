"""Theme manager (SPEC: General tab -> Appearance).

The application ships three built-in appearances that must never change:

* ``system`` follows the OS appearance on Windows via
  ``wx.SystemSettings.GetAppearance()``,
* ``light`` and
* ``dark``.

On top of those it offers twenty *predefined dark themes*, each with its own
sincere colour palette and its own painted welcome-screen look, and any number
of user-created *custom themes* (Settings -> Appearance -> Create custom
theme...).  The first ten are the original 2027 set; :data:`THEME_PRESETS_VIBRANT`
adds ten more built around saturated, vivid accents.  None of the original ten
changed when the new ones arrived.

A theme is a plain ``#rrggbb`` palette.  :func:`apply_theme` paints explicit
colours onto container/text/input controls (buttons keep the native look so
they stay accessible); the welcome screen additionally uses
:func:`paint_background` to draw a wallpaper - the user's own image when one
is chosen, otherwise the theme's built-in pattern.

Wallpapers and custom themes live in the application settings so they travel
with the rest of the user's configuration:

* ``custom_themes``: a list of ``{id, name, colors, pattern}`` records,
* ``wallpapers``: ``{theme id: {enabled, image, pattern}}`` for any theme,
  preset or custom.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading

import wx

from ..constants import THEME_DARK, THEME_LIGHT, THEME_SYSTEM
from . import contrast

log = logging.getLogger(__name__)

#: The two built-in palettes.  They are deliberately frozen: the light and
#: dark themes are part of the accessible baseline (WCAG AA contrast, see
#: tests/test_a11y_access_keys.py) and must not drift when themes are added.
_LIGHT: dict[str, str] = {
    "bg": "#ffffff",
    "fg": "#1a1a1a",
    "control_bg": "#ffffff",
    "control_fg": "#1a1a1a",
}
_DARK: dict[str, str] = {
    "bg": "#1e1e1e",
    "fg": "#e8e8e8",
    "control_bg": "#2d2d2d",
    "control_fg": "#e8e8e8",
}

#: Painted-background styles.  A theme picks one; the user may override it.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("solid", "Solid colour"),
    ("gradient_v", "Vertical gradient"),
    ("gradient_h", "Horizontal gradient"),
    ("diagonal", "Diagonal gradient"),
    ("vignette", "Soft vignette"),
    ("grid", "Grid"),
    ("dots", "Dots"),
    ("stripes", "Stripes"),
    ("rays", "Corner glow"),
    ("waves", "Waves"),
)
PATTERN_IDS = tuple(value for value, _label in PATTERNS)


class ThemePreset:
    """A predefined theme: a name, a palette and a signature wallpaper style."""

    __slots__ = ("colors", "name", "note", "pattern", "theme_id")

    def __init__(self, theme_id: str, name: str, colors: dict[str, str],
                 pattern: str, note: str = "") -> None:
        self.theme_id = theme_id
        self.name = name
        self.colors = colors
        self.pattern = pattern
        self.note = note


def _preset(theme_id: str, name: str, bg: str, fg: str, control_bg: str,
            control_fg: str, accent: str, heading: str, pattern: str,
            note: str) -> ThemePreset:
    """Build a preset palette, keeping the four baseline keys first."""
    return ThemePreset(
        theme_id, name,
        {
            "bg": bg,
            "fg": fg,
            "control_bg": control_bg,
            "control_fg": control_fg,
            "accent": accent,
            "heading": heading,
        },
        pattern,
        note,
    )


#: The ten original predefined dark themes.  Every palette is dark-on-light
#: text with a distinct accent and its own painted look, so no two feel alike.
#: This tuple is frozen: these ten are what shipped first, and their ids, names
#: and colours must not drift (a user's saved theme selection names one by id).
THEME_PRESETS: tuple[ThemePreset, ...] = (
    _preset(
        "graphite_midnight", "Midnight Graphite",
        "#16181c", "#e6e9ef", "#22262d", "#e6e9ef", "#8ab4f8", "#ffffff",
        "gradient_v",
        "Neutral charcoal with a cool slate-blue accent.",
    ),
    _preset(
        "obsidian_gold", "Obsidian Gold",
        "#141210", "#f0e6d2", "#241f18", "#f5ecd9", "#d4af37", "#ffffff",
        "vignette",
        "Near-black with a warm antique-gold accent.",
    ),
    _preset(
        "deep_navy", "Deep Navy",
        "#0d1b2a", "#dce6f2", "#16283d", "#e3ecf7", "#4c8bf5", "#ffffff",
        "gradient_h",
        "Dark navy with a royal-blue accent.",
    ),
    _preset(
        "forest_pine", "Forest Pine",
        "#0f1a14", "#ddebe0", "#1b2e22", "#e4f0e4", "#7bc47f", "#ffffff",
        "dots",
        "Deep pine green with a soft leaf-green accent.",
    ),
    _preset(
        "plum_royale", "Plum Royale",
        "#1a1220", "#eedff2", "#2a1b33", "#f1e4f4", "#b57edc", "#ffffff",
        "gradient_v",
        "Royal plum with an orchid accent.",
    ),
    _preset(
        "crimson_slate", "Crimson Slate",
        "#1b1416", "#f2e2e4", "#2c1e21", "#f5e7e9", "#e5484d", "#ffffff",
        "stripes",
        "Slate charcoal with a firm crimson accent.",
    ),
    _preset(
        "ocean_teal", "Ocean Teal",
        "#0c1a1c", "#d8eeee", "#143034", "#e0f2f2", "#2dd4bf", "#ffffff",
        "waves",
        "Deep ocean teal with a bright aqua accent.",
    ),
    _preset(
        "amber_dusk", "Amber Dusk",
        "#1a130c", "#f3e5d3", "#2c2014", "#f6ebda", "#f59e0b", "#ffffff",
        "rays",
        "Warm dusk brown with a golden-amber accent.",
    ),
    _preset(
        "carbon_lime", "Carbon Lime",
        "#121512", "#e4efdc", "#1e251c", "#eaf2e2", "#a3e635", "#ffffff",
        "grid",
        "Carbon black with a bold lime accent.",
    ),
    _preset(
        "blue_steel", "Blue Steel",
        "#10151c", "#dde5ee", "#1c2530", "#e5ebf2", "#9fb3c8", "#ffffff",
        "diagonal",
        "Gunmetal blue-grey with a steel accent.",
    ),
)

#: Ten more dark themes, added for the 2027 release on request.  Each one is
#: still dark-on-light text (the WCAG AA contrast of every palette is pinned by
#: tests/test_themes.py) but its accent is a *vivid* saturated hue rather than
#: the muted one the original set uses, and its background and controls are
#: tinted with that hue instead of being a neutral grey.
THEME_PRESETS_VIBRANT: tuple[ThemePreset, ...] = (
    _preset(
        "neon_magenta", "Neon Magenta",
        "#1a0614", "#ffd6f2", "#2b0a21", "#ffdff5", "#ff00c8", "#ffffff",
        "rays",
        "Deep magenta-black with a piercing pink accent.",
    ),
    _preset(
        "inferno_red", "Inferno Red",
        "#1c0507", "#ffd6da", "#2e0a0e", "#ffdee1", "#ff1f3d", "#ffffff",
        "stripes",
        "Ember-black with a vivid signal-red accent.",
    ),
    _preset(
        "solar_flare", "Solar Flare",
        "#1d0c02", "#ffddc2", "#2f1505", "#ffe4cc", "#ff6a00", "#ffffff",
        "gradient_h",
        "Burnt-orange black with a hot molten-orange accent.",
    ),
    _preset(
        "voltage_yellow", "Voltage Yellow",
        "#1a1703", "#fff6c2", "#2b2708", "#fffbcf", "#ffe600", "#ffffff",
        "grid",
        "Olive-black with an electric-yellow accent.",
    ),
    _preset(
        "laser_lime", "Laser Lime",
        "#111a02", "#ecffc2", "#1e2b08", "#f1ffcf", "#b6ff00", "#ffffff",
        "dots",
        "Moss-black with a searing lime-green accent.",
    ),
    _preset(
        "emerald_surge", "Emerald Surge",
        "#04170c", "#c9ffdd", "#0a2b18", "#d4ffe6", "#00ff6a", "#ffffff",
        "waves",
        "Deep forest black with a brilliant emerald accent.",
    ),
    _preset(
        "cyber_cyan", "Cyber Cyan",
        "#031418", "#c6f5ff", "#0a2a33", "#d2f8ff", "#00e5ff", "#ffffff",
        "diagonal",
        "Ink-black with a bright cyan accent.",
    ),
    _preset(
        "azure_rush", "Azure Rush",
        "#050f1f", "#cfe2ff", "#0c2038", "#d8e8ff", "#0091ff", "#ffffff",
        "gradient_v",
        "Midnight-blue black with a vivid azure accent.",
    ),
    _preset(
        "electric_indigo", "Electric Indigo",
        "#0a0820", "#d9d6ff", "#161244", "#e2dfff", "#6c4dff", "#ffffff",
        "vignette",
        "Indigo-black with a saturated electric-violet accent.",
    ),
    _preset(
        "ultraviolet", "Ultraviolet",
        "#13061f", "#ecd9ff", "#221040", "#f0e0ff", "#b026ff", "#ffffff",
        "vignette",
        "Near-black violet with a glowing ultraviolet accent.",
    ),
)

#: Every predefined theme: the original ten first, then the vibrant ten, so the
#: Theme list keeps the order a user of the 2026 series already knows.
ALL_THEME_PRESETS: tuple[ThemePreset, ...] = THEME_PRESETS + THEME_PRESETS_VIBRANT

_PRESET_BY_ID = {preset.theme_id: preset for preset in ALL_THEME_PRESETS}

#: Prefix that marks a user-created theme id (e.g. ``custom:blush``).
CUSTOM_PREFIX = "custom:"

_state = {"theme": THEME_SYSTEM, "colors": _LIGHT}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public palette/theme API
# ---------------------------------------------------------------------------
def current_theme() -> str:
    with _lock:
        return _state["theme"]


def current_theme_colors() -> dict[str, str]:
    with _lock:
        return dict(_state["colors"])


def _resolve(theme: str) -> str:
    if theme == THEME_SYSTEM:
        try:
            appearance = wx.SystemSettings.GetAppearance()
            return THEME_DARK if appearance.IsDark() else THEME_LIGHT
        except Exception:
            return THEME_LIGHT
    return theme


def custom_themes(settings) -> list[dict]:
    """Every user-created theme, newest last (``[]`` without settings)."""
    if settings is None:
        return []
    records = settings.get("custom_themes", []) or []
    out: list[dict] = []
    for record in records:
        if isinstance(record, dict) and record.get("id"):
            out.append(record)
    return out


def get_custom_theme(settings, theme_id: str) -> dict | None:
    for record in custom_themes(settings):
        if record.get("id") == theme_id:
            return record
    return None


def save_custom_theme(settings, record: dict) -> str:
    """Store ``record`` (adding or replacing it); return its id."""
    if settings is None:
        return record.get("id", "")
    theme_id = record.get("id") or new_theme_id(settings, record.get("name", ""))
    record = dict(record)
    record["id"] = theme_id
    records = [r for r in custom_themes(settings) if r.get("id") != theme_id]
    records.append(record)
    settings.set("custom_themes", records)
    settings.save()
    return theme_id


def delete_custom_theme(settings, theme_id: str) -> None:
    if settings is None:
        return
    settings.set(
        "custom_themes",
        [r for r in custom_themes(settings) if r.get("id") != theme_id],
    )
    wallpapers = dict(settings.get("wallpapers", {}) or {})
    if theme_id in wallpapers:
        del wallpapers[theme_id]
        settings.set("wallpapers", wallpapers)
    settings.save()


def new_theme_id(settings, name: str) -> str:
    """A unique ``custom:`` id derived from ``name``."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "theme").strip().lower()).strip("-")
    slug = slug or "theme"
    existing = {r.get("id") for r in custom_themes(settings)}
    candidate = CUSTOM_PREFIX + slug
    counter = 2
    while candidate in existing:
        candidate = f"{CUSTOM_PREFIX}{slug}-{counter}"
        counter += 1
    return candidate


def theme_label(theme: str, settings=None) -> str:
    """Human-readable name for a theme id."""
    if theme == THEME_SYSTEM:
        return "System default"
    if theme == THEME_LIGHT:
        return "Light"
    if theme == THEME_DARK:
        return "Dark"
    preset = _PRESET_BY_ID.get(theme)
    if preset is not None:
        return preset.name
    record = get_custom_theme(settings, theme)
    if record is not None:
        return str(record.get("name") or "Custom theme")
    return str(theme)


def all_theme_choices(settings=None) -> list[tuple[str, str]]:
    """``(value, label)`` pairs for a theme selector, built-ins first."""
    choices: list[tuple[str, str]] = [
        (THEME_SYSTEM, "System default"),
        (THEME_LIGHT, "Light"),
        (THEME_DARK, "Dark (original)"),
    ]
    choices.extend((preset.theme_id, preset.name) for preset in ALL_THEME_PRESETS)
    choices.extend(
        (record["id"], str(record.get("name") or "Custom theme"))
        for record in custom_themes(settings)
    )
    return choices


def palette_from_record(record: dict) -> dict[str, str]:
    """Normalise a custom theme record into a complete palette."""
    colors = dict(record.get("colors", {}) or {})
    palette = {
        "bg": colors.get("bg", _DARK["bg"]),
        "fg": colors.get("fg", _DARK["fg"]),
        "control_bg": colors.get("control_bg", _DARK["control_bg"]),
        "control_fg": colors.get("control_fg", _DARK["control_fg"]),
        "accent": colors.get("accent", colors.get("fg", _DARK["fg"])),
        "heading": colors.get("heading", "#ffffff"),
    }
    return {key: _clean_colour(value) for key, value in palette.items()}


def resolve_palette(theme: str, settings=None) -> dict[str, str]:
    """The palette a theme id resolves to (never raises).

    While Windows high contrast is on this is the user's *own* scheme, for
    every theme id - a theme is the application's choice of colours, and high
    contrast is the user saying they have already made that choice.  The id is
    left alone in the settings, so switching high contrast off restores the
    theme that was chosen.
    """
    if contrast.enabled():
        return contrast.palette()
    if theme == THEME_SYSTEM:
        theme = _resolve(theme)
    if theme == THEME_DARK:
        return dict(_DARK)
    if theme == THEME_LIGHT:
        return dict(_LIGHT)
    preset = _PRESET_BY_ID.get(theme)
    if preset is not None:
        return dict(preset.colors)
    if isinstance(theme, str) and theme.startswith(CUSTOM_PREFIX):
        record = get_custom_theme(settings, theme)
        if record is not None:
            return palette_from_record(record)
        return dict(_DARK)
    # An unknown id (e.g. a removed custom theme) falls back to dark, which is
    # legible, instead of the old implicit light default.
    return dict(_DARK)


def theme_pattern(theme: str, settings=None) -> str:
    """The painted-background style to draw for ``theme``.

    A high-contrast scheme always gets a flat fill: a gradient, a grid or a
    vignette is decoration drawn over the colours the user relies on.
    """
    if contrast.enabled():
        return "solid"
    if theme == THEME_SYSTEM:
        theme = _resolve(theme)
    wallpaper = get_wallpaper(settings, theme)
    override = wallpaper.get("pattern") or ""
    if override in PATTERN_IDS:
        return override
    preset = _PRESET_BY_ID.get(theme)
    if preset is not None:
        return preset.pattern
    record = get_custom_theme(settings, theme)
    if record is not None:
        pattern = record.get("pattern")
        if pattern in PATTERN_IDS:
            return pattern
    return "solid"


# ---------------------------------------------------------------------------
# Wallpaper
# ---------------------------------------------------------------------------
def get_wallpaper(settings, theme: str) -> dict:
    """The wallpaper record for ``theme`` (defaults when unset)."""
    default = {"enabled": False, "image": "", "pattern": ""}
    if settings is None:
        return default
    wallpapers = settings.get("wallpapers", {}) or {}
    record = wallpapers.get(theme)
    if not isinstance(record, dict):
        return default
    return {
        "enabled": bool(record.get("enabled", False)),
        "image": str(record.get("image") or ""),
        "pattern": str(record.get("pattern") or ""),
    }


def set_wallpaper(settings, theme: str, *, enabled: bool | None = None,
                  image: str | None = None, pattern: str | None = None) -> None:
    """Update the wallpaper record for ``theme`` and persist it."""
    if settings is None:
        return
    record = dict(get_wallpaper(settings, theme))
    if enabled is not None:
        record["enabled"] = bool(enabled)
    if image is not None:
        record["image"] = str(image)
    if pattern is not None:
        record["pattern"] = pattern if pattern in PATTERN_IDS else ""
    wallpapers = dict(settings.get("wallpapers", {}) or {})
    wallpapers[theme] = record
    settings.set("wallpapers", wallpapers)
    settings.save()


def is_wallpaper_enabled(theme: str, settings) -> bool:
    if settings is None or contrast.enabled():
        # Under high contrast the welcome screen keeps its words: a wallpaper
        # is both unreadable and, for a magnifier user, a source of noise.
        return False
    record = get_wallpaper(settings, theme)
    return bool(record.get("enabled"))


# ---------------------------------------------------------------------------
# Applying a theme
# ---------------------------------------------------------------------------
def apply_theme(top: wx.Window, theme: str, settings=None) -> None:
    """Apply ``theme`` to ``top`` and all of its descendants.

    ``settings`` is optional and only needed to resolve user-created themes
    and to paint the welcome wallpaper; the built-in themes work without it.
    """
    colors = resolve_palette(theme, settings)
    with _lock:
        _state["theme"] = theme
        _state["colors"] = colors

    _paint(top, colors)
    for child in _descendants(top):
        _paint(child, colors)
    top.Refresh()


def _descendants(window: wx.Window):
    stack = list(window.GetChildren())
    while stack:
        child = stack.pop()
        yield child
        stack.extend(child.GetChildren())


def _paint(window: wx.Window, colors: dict[str, str]) -> None:
    try:
        if isinstance(window, (wx.Panel, wx.Dialog, wx.Frame, wx.ScrolledWindow)):
            window.SetBackgroundColour(colors["bg"])
        elif isinstance(
            window,
            (wx.StaticText, wx.CheckBox, wx.RadioButton, wx.StaticBox, wx.RadioBox),
        ):
            window.SetForegroundColour(colors["fg"])
        elif isinstance(
            window,
            (wx.TextCtrl, wx.ListBox, wx.ComboBox, wx.ListCtrl, wx.Choice, wx.SpinCtrl),
        ):
            window.SetBackgroundColour(colors["control_bg"])
            window.SetForegroundColour(colors["control_fg"])
    except Exception:
        log.debug("Ignoring a failure in _paint", exc_info=True)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _clean_colour(value: str) -> str:
    """Coerce ``value`` to a ``#rrggbb`` string (falls back to black)."""
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
        return "#" + "".join(ch * 2 for ch in text[1:].lower())
    return "#000000"


def _colour(value: str) -> wx.Colour:
    return wx.Colour(_clean_colour(value))


def _rgb(colour: wx.Colour) -> tuple[int, int, int]:
    return colour.Red(), colour.Green(), colour.Blue()


def mix(colour_a: str, colour_b: str, t: float) -> wx.Colour:
    """Blend two ``#rrggbb`` colours; ``t`` is the weight of ``colour_b``."""
    t = max(0.0, min(1.0, t))
    ra, ga, ba = _rgb(_colour(colour_a))
    rb, gb, bb = _rgb(_colour(colour_b))
    return wx.Colour(
        round(ra + (rb - ra) * t),
        round(ga + (gb - ga) * t),
        round(ba + (bb - ba) * t),
    )


def paint_background(dc: wx.DC, size: tuple[int, int], colors: dict[str, str],
                     pattern: str = "solid", image_path: str = "") -> None:
    """Draw the welcome-screen background (image first, else a pattern).

    Called from the welcome panel's ``EVT_PAINT``; ``size`` is the panel's
    client size in pixels.

    High contrast is handled before anything else: the panel gets the user's
    own background colour, with no wallpaper image and no pattern.
    """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return

    if contrast.enabled():
        flat = _colour(colors.get("bg") or contrast.palette()["bg"])
        dc.SetBrush(wx.Brush(flat))
        dc.SetPen(wx.Pen(flat))
        dc.DrawRectangle(0, 0, width, height)
        return

    if image_path and os.path.isfile(image_path) and _draw_image(
            dc, (width, height), image_path):
        return

    bg = colors.get("bg", _DARK["bg"])
    accent = colors.get("accent") or colors.get("fg") or _DARK["fg"]
    try:
        drawer = _PATTERN_DRAWERS.get(pattern, _draw_solid)
        drawer(dc, width, height, bg, accent)
    except Exception:
        dc.SetBrush(wx.Brush(_colour(bg)))
        dc.SetPen(wx.Pen(_colour(bg)))
        dc.DrawRectangle(0, 0, width, height)


def _draw_image(dc: wx.DC, size: tuple[int, int], image_path: str) -> bool:
    """Aspect-fill the wallpaper image; False when it cannot be loaded."""
    width, height = size
    try:
        image = wx.Image(image_path, wx.BITMAP_TYPE_ANY)
        if not image.IsOk() or not image.GetWidth() or not image.GetHeight():
            return False
        iw, ih = image.GetWidth(), image.GetHeight()
        scale = max(width / iw, height / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        image = image.Scale(nw, nh, wx.IMAGE_QUALITY_HIGH)
        if nw > width or nh > height:
            x, y = (nw - width) // 2, (nh - height) // 2
            image = image.GetSubImage(wx.Rect(x, y, min(width, nw), min(height, nh)))
        dc.DrawBitmap(wx.Bitmap(image), 0, 0, False)
        return True
    except Exception:
        return False


def _draw_solid(dc: wx.DC, width: int, height: int, bg: str, accent: str) -> None:
    dc.SetPen(wx.Pen(_colour(bg)))
    dc.SetBrush(wx.Brush(_colour(bg)))
    dc.DrawRectangle(0, 0, width, height)


def _draw_gradient_v(dc, width, height, bg, accent):
    for y in range(height):
        dc.SetPen(wx.Pen(mix(bg, accent, (y / max(1, height - 1)) * 0.55)))
        dc.DrawLine(0, y, width, y)


def _draw_gradient_h(dc, width, height, bg, accent):
    for x in range(width):
        dc.SetPen(wx.Pen(mix(bg, accent, (x / max(1, width - 1)) * 0.55)))
        dc.DrawLine(x, 0, x, height)


def _draw_diagonal(dc, width, height, bg, accent):
    _draw_solid(dc, width, height, bg, accent)
    total = width + height
    for offset in range(-height, width):
        t = (offset + height) / max(1, total)
        dc.SetPen(wx.Pen(mix(bg, accent, t * 0.5)))
        dc.DrawLine(offset, 0, offset + height, height)


def _draw_vignette(dc, width, height, bg, accent):
    dark = mix(bg, "#000000", 0.55)
    steps = 14
    dc.SetBrush(wx.Brush(_colour(dark)))
    dc.SetPen(wx.Pen(_colour(dark)))
    dc.DrawRectangle(0, 0, width, height)
    for i in range(steps):
        t = i / max(1, steps - 1)
        inset = round(min(width, height) * 0.5 * (1 - t))
        colour = mix(dark, bg, t)
        dc.SetBrush(wx.Brush(colour))
        dc.SetPen(wx.Pen(colour))
        dc.DrawRectangle(inset, inset, max(1, width - 2 * inset),
                         max(1, height - 2 * inset))


def _draw_grid(dc, width, height, bg, accent):
    _draw_solid(dc, width, height, bg, accent)
    line = mix(bg, accent, 0.18)
    dc.SetPen(wx.Pen(line))
    step = 34
    for x in range(0, width + 1, step):
        dc.DrawLine(x, 0, x, height)
    for y in range(0, height + 1, step):
        dc.DrawLine(0, y, width, y)


def _draw_dots(dc, width, height, bg, accent):
    _draw_solid(dc, width, height, bg, accent)
    dot = mix(bg, accent, 0.28)
    dc.SetPen(wx.Pen(dot))
    dc.SetBrush(wx.Brush(dot))
    step = 30
    radius = 3
    for y in range(step, height, step):
        for x in range(step, width, step):
            dc.DrawCircle(x, y, radius)


def _draw_stripes(dc, width, height, bg, accent):
    _draw_solid(dc, width, height, bg, accent)
    stripe = mix(bg, accent, 0.16)
    dc.SetPen(wx.Pen(stripe, 14))
    for offset in range(-height, width + 1, 46):
        dc.DrawLine(offset, 0, offset + height, height)


def _draw_rays(dc, width, height, bg, accent):
    _draw_solid(dc, width, height, bg, accent)
    cx, cy = int(width * 0.82), int(height * 0.16)
    max_r = int(math.hypot(max(cx, width - cx), max(cy, height - cy)))
    steps = 22
    for i in range(steps, 0, -1):
        t = i / steps
        colour = mix(bg, accent, 0.16 * (1 - t))
        radius = int(max_r * t)
        dc.SetBrush(wx.Brush(colour))
        dc.SetPen(wx.Pen(colour))
        dc.DrawCircle(cx, cy, radius)


def _draw_waves(dc, width, height, bg, accent):
    _draw_solid(dc, width, height, bg, accent)
    bands = 7
    for band in range(bands):
        base_y = int(height * (band + 1) / (bands + 1))
        amplitude = max(8, height // 24)
        phase = band * 0.9
        colour = mix(bg, accent, 0.10 + 0.05 * band)
        dc.SetPen(wx.Pen(colour, 3))
        points = []
        for x in range(0, width + 8, 8):
            y = base_y + int(amplitude * math.sin(x / max(1.0, width / 6.0) + phase))
            points.append((x, y))
        if len(points) > 1:
            dc.DrawLines(points)


_PATTERN_DRAWERS = {
    "solid": _draw_solid,
    "gradient_v": _draw_gradient_v,
    "gradient_h": _draw_gradient_h,
    "diagonal": _draw_diagonal,
    "vignette": _draw_vignette,
    "grid": _draw_grid,
    "dots": _draw_dots,
    "stripes": _draw_stripes,
    "rays": _draw_rays,
    "waves": _draw_waves,
}
