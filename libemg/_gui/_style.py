"""How the LibEMG window looks, and the pieces a panel lays its controls out with.

DearPyGui's defaults are built for debugging tools: a 13 px bitmap font, tight
padding, and every label drawn to the *right* of its control, so a row of
inputs reads as "1 Up 0 Down 4 Left" and nobody can tell which number is whose.
This module replaces the font and the spacing for the whole window once, at
startup, and gives panels a form layout -- a label column on the left, the
control on the right, an optional (?) beside the label for the long
explanation -- so a setup screen reads like a form rather than a list of
widgets.

Everything here only decides how things look. Panels still build their own
items; they call :func:`form` and :func:`row` where they used to call
``add_input_int(label=...)``.

Examples
---------
>>> with form():
...     with row("Dwell time", help="How long the cursor has to stay in the target."):
...         dpg.add_input_float(default_value=1.0, step=0, width=120)
...         unit("s")
"""

import os
from contextlib import contextmanager
from pathlib import Path

import dearpygui.dearpygui as dpg

#: Text size in pixels. DearPyGui's own font is 13 px and a bitmap.
FONT_SIZE = 17

# The palette. Text colours are passed straight to add_text(color=...).
TEXT = (226, 229, 234)
MUTED = (146, 152, 163)
HEADING = (150, 186, 238)
ACCENT = (96, 165, 250)
SUCCESS = (110, 204, 145)
WARNING = (236, 186, 94)
ERROR = (240, 113, 113)

_BACKGROUND = (27, 29, 33)
_PANEL = (33, 36, 41)
_FRAME = (46, 50, 58)
_FRAME_HOVER = (57, 62, 72)
_FRAME_ACTIVE = (66, 72, 84)
_BORDER = (58, 63, 72)

# Tried in order; the first that exists is used. Nothing is bundled, so a machine
# with none of these keeps DearPyGui's own font and everything still works.
_FONTS = {
    "regular": [
        "{windows}/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "{windows}/Fonts/segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}

# Fonts and themes are DearPyGui items, and items die with their context, so a
# GUI started twice in one process -- or a test that makes a context per test --
# has to make them again. Each is made under a fixed alias and looked up by it:
# aliases belong to a context, so a new context simply does not have them yet.
# Caching the item ids instead does not work, because a new context hands out
# the same ids again and a stale theme would look alive.
_FONT_REGISTRY = "__libemg_fonts"
# Fonts that were looked for and not found, per context: (kind, size) pairs.
# Cleared whenever the font registry is new, which is whenever the context is.
_missing = set()


def apply():
    """Give the whole window its font and look. Call once, after ``dpg.create_context()``."""
    regular = font("regular")
    if regular is not None:
        dpg.bind_font(regular)
    dpg.bind_theme(_global_theme())


def symbols():
    """Whether arrows and the like can be drawn. DearPyGui's own font has none of them."""
    return font("regular") is not None


def arrow(direction, fallback=""):
    """An arrow for 'up', 'down', 'left' or 'right', or ``fallback`` when the font has none."""
    if not symbols():
        return fallback
    return {"up": "\u2191", "down": "\u2193", "left": "\u2190", "right": "\u2192"}.get(direction, fallback)


def font(kind="regular", size=None):
    """The font of ``kind`` ('regular' or 'bold') at ``size`` px, or None if none was found.

    ``size`` defaults to the window's text size; a larger one is for text that
    has to be read from a distance, such as the prompt during collection.
    """
    size = int(size or FONT_SIZE)
    alias = f"__libemg_font_{kind}_{size}"
    if not dpg.does_item_exist(_FONT_REGISTRY):
        _missing.clear()
        dpg.add_font_registry(tag=_FONT_REGISTRY)
    if dpg.does_item_exist(alias):
        return alias
    if (kind, size) in _missing:
        return None
    loaded = _load_font(kind, size, alias)
    if loaded is None:
        _missing.add((kind, size))
    return loaded


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------
@contextmanager
def form(label_width=170, tag=None, parent=0):
    """A two-column layout: labels on the left, at one width, controls on the right.

    Use :func:`row` inside it for each line.
    """
    kwargs = {"tag": tag} if tag else {}
    if parent:
        kwargs["parent"] = parent
    with dpg.table(header_row=False, resizable=False, borders_innerH=False,
                   borders_outerH=False, borders_innerV=False, borders_outerV=False,
                   policy=dpg.mvTable_SizingFixedFit, **kwargs) as table:
        dpg.add_table_column(width_fixed=True, init_width_or_weight=label_width)
        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
        yield table


@contextmanager
def row(label, help=None, tag=None, color=None):
    """One line of a :func:`form`. Whatever is added inside goes beside the label, left to right.

    ``help`` is shown when the label is hovered. A (?) on every line of a long
    form is more noise than help; say once, near the top, that names can be
    hovered (see :func:`hover_hint`).
    """
    kwargs = {"tag": tag} if tag else {}
    with dpg.table_row(**kwargs):
        name = dpg.add_text(label, color=color or TEXT)
        if help:
            with dpg.tooltip(name):
                dpg.add_text(help, wrap=380)
        with dpg.group(horizontal=True) as cell:
            yield cell


def hover_hint(text="Hover over a setting's name to see what it does.", wrap=0):
    """The one line that tells people the names in a form explain themselves."""
    return note(text, wrap=wrap)


def section(title, first=False, tag=None):
    """A titled rule that starts a section: the heading of a group of rows.

    Returns the heading's text item, tagged ``tag`` if given, so a panel can
    change what it says.
    """
    if not first:
        dpg.add_spacer(height=6)
    # Text over a rule rather than DearPyGui's labelled separator, which takes
    # neither a font nor a colour of its own.
    heading = dpg.add_text(title, color=HEADING, **({"tag": tag} if tag else {}))
    bold = font("bold")
    if bold is not None:
        dpg.bind_item_font(heading, bold)
    dpg.add_separator()
    return heading


def help_marker(text, wrap=380):
    """A small (?) that explains its neighbour when hovered."""
    marker = dpg.add_text("(?)", color=MUTED)
    with dpg.tooltip(marker):
        dpg.add_text(text, wrap=wrap)
    return marker


def note(text, color=MUTED, wrap=0, tag=None, show=True):
    """A line of explanatory text, quieter than a label."""
    kwargs = {"tag": tag} if tag else {}
    return dpg.add_text(text, color=color, wrap=wrap, show=show, **kwargs)


def unit(text):
    """The unit after a number, such as 's' or 'px'."""
    return dpg.add_text(text, color=MUTED)


def button(label, kind="default", **kwargs):
    """A button. ``kind`` is 'primary' (the action the screen is for), 'danger', 'subtle' or 'default'."""
    item = dpg.add_button(label=label, **kwargs)
    if kind != "default":
        dpg.bind_item_theme(item, _button_theme(kind))
    return item


def set_enabled(item, enabled):
    """Enable or disable an item if it exists. Disabled items are drawn dimmed by the theme."""
    if dpg.does_item_exist(item):
        (dpg.enable_item if enabled else dpg.disable_item)(item)


def set_status(item, text, kind="info"):
    """Show ``text`` in a status line, coloured by ``kind``: info, ok, busy, warn or error."""
    if dpg.does_item_exist(item):
        dpg.set_value(item, text)
        dpg.configure_item(item, color={"info": MUTED, "ok": SUCCESS, "busy": ACCENT,
                                        "warn": WARNING, "error": ERROR}.get(kind, MUTED))


# ----------------------------------------------------------------------
# what apply() binds
# ----------------------------------------------------------------------
def _load_font(kind, size, alias):
    windows = os.environ.get("WINDIR", "C:/Windows").replace("\\", "/")
    for candidate in _FONTS[kind]:
        path = Path(candidate.format(windows=windows))
        if not path.exists():
            continue
        try:
            # DearPyGui 2 loads whatever characters are drawn, arrows included.
            dpg.add_font(str(path), size, parent=_FONT_REGISTRY, tag=alias)
            return alias
        except Exception:
            continue
    return None


def _global_theme():
    alias = "__libemg_theme_global"
    if dpg.does_item_exist(alias):
        return alias
    with dpg.theme(tag=alias) as theme:
        with dpg.theme_component(dpg.mvAll):
            for var, value in (
                    (dpg.mvStyleVar_WindowPadding, (14, 12)),
                    (dpg.mvStyleVar_FramePadding, (8, 4)),
                    (dpg.mvStyleVar_ItemSpacing, (10, 7)),
                    (dpg.mvStyleVar_ItemInnerSpacing, (6, 4)),
                    (dpg.mvStyleVar_CellPadding, (6, 3)),
                    (dpg.mvStyleVar_SeparatorTextPadding, (0, 5)),
                    (dpg.mvStyleVar_SeparatorTextAlign, (0.0, 0.5)),
            ):
                dpg.add_theme_style(var, *value, category=dpg.mvThemeCat_Core)
            for var, value in (
                    (dpg.mvStyleVar_WindowRounding, 6),
                    (dpg.mvStyleVar_ChildRounding, 6),
                    (dpg.mvStyleVar_FrameRounding, 5),
                    (dpg.mvStyleVar_PopupRounding, 5),
                    (dpg.mvStyleVar_GrabRounding, 5),
                    (dpg.mvStyleVar_TabRounding, 5),
                    (dpg.mvStyleVar_ScrollbarSize, 14),
                    (dpg.mvStyleVar_ScrollbarRounding, 6),
                    (dpg.mvStyleVar_SeparatorTextBorderSize, 1),
            ):
                dpg.add_theme_style(var, value, category=dpg.mvThemeCat_Core)
            for colour, value in (
                    (dpg.mvThemeCol_Text, TEXT),
                    (dpg.mvThemeCol_TextDisabled, MUTED),
                    (dpg.mvThemeCol_WindowBg, _PANEL),
                    (dpg.mvThemeCol_ChildBg, _PANEL),
                    (dpg.mvThemeCol_PopupBg, _BACKGROUND),
                    (dpg.mvThemeCol_Border, _BORDER),
                    (dpg.mvThemeCol_Separator, _BORDER),
                    (dpg.mvThemeCol_FrameBg, _FRAME),
                    (dpg.mvThemeCol_FrameBgHovered, _FRAME_HOVER),
                    (dpg.mvThemeCol_FrameBgActive, _FRAME_ACTIVE),
                    (dpg.mvThemeCol_TitleBg, _BACKGROUND),
                    (dpg.mvThemeCol_TitleBgActive, (40, 46, 58)),
                    (dpg.mvThemeCol_TitleBgCollapsed, _BACKGROUND),
                    (dpg.mvThemeCol_MenuBarBg, _BACKGROUND),
                    (dpg.mvThemeCol_Button, _FRAME),
                    (dpg.mvThemeCol_ButtonHovered, _FRAME_HOVER),
                    (dpg.mvThemeCol_ButtonActive, _FRAME_ACTIVE),
                    (dpg.mvThemeCol_Header, (49, 58, 74)),
                    (dpg.mvThemeCol_HeaderHovered, (59, 70, 90)),
                    (dpg.mvThemeCol_HeaderActive, (68, 82, 106)),
                    (dpg.mvThemeCol_CheckMark, ACCENT),
                    (dpg.mvThemeCol_SliderGrab, ACCENT),
                    (dpg.mvThemeCol_SliderGrabActive, (130, 185, 255)),
                    (dpg.mvThemeCol_Tab, _FRAME),
                    (dpg.mvThemeCol_TabHovered, _FRAME_HOVER),
                    (dpg.mvThemeCol_TabActive, (59, 70, 90)),
                    (dpg.mvThemeCol_ScrollbarBg, _PANEL),
                    (dpg.mvThemeCol_ScrollbarGrab, _FRAME_HOVER),
                    (dpg.mvThemeCol_TableBorderLight, _BORDER),
                    (dpg.mvThemeCol_TableBorderStrong, _BORDER),
                    (dpg.mvThemeCol_TableRowBgAlt, (38, 41, 47)),
            ):
                dpg.add_theme_color(colour, value, category=dpg.mvThemeCat_Core)
        # DearPyGui draws a disabled control exactly like an enabled one unless
        # told otherwise, so a greyed-out option would look clickable.
        for component in (dpg.mvInputInt, dpg.mvInputFloat, dpg.mvInputText, dpg.mvCombo,
                          dpg.mvCheckbox, dpg.mvButton, dpg.mvRadioButton, dpg.mvColorEdit):
            with dpg.theme_component(component, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (105, 110, 120), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (38, 41, 47), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (38, 41, 47), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (38, 41, 47), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (38, 41, 47), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (38, 41, 47), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (105, 110, 120), category=dpg.mvThemeCat_Core)
    return theme


def _button_theme(kind):
    alias = f"__libemg_theme_button_{kind}"
    if dpg.does_item_exist(alias):
        return alias
    colours = {
        "primary": ((37, 99, 235), (59, 116, 245), (29, 78, 196), (255, 255, 255)),
        "danger": ((150, 45, 50), (178, 58, 64), (128, 36, 42), (255, 255, 255)),
        "subtle": ((0, 0, 0, 0), _FRAME_HOVER, _FRAME_ACTIVE, MUTED),
    }[kind]
    # Wider than a plain button but the same height, so a row mixing the two
    # lines up without every panel giving its buttons explicit heights.
    padding = (14, 4)
    with dpg.theme(tag=alias) as theme:
        with dpg.theme_component(dpg.mvButton):
            for colour, value in zip((dpg.mvThemeCol_Button, dpg.mvThemeCol_ButtonHovered,
                                      dpg.mvThemeCol_ButtonActive, dpg.mvThemeCol_Text), colours):
                dpg.add_theme_color(colour, value, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, *padding, category=dpg.mvThemeCat_Core)
        # Disabled, a primary or danger button is grey like any other: its colour
        # says "this is what to press", which is not true while it can't be.
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            for colour in (dpg.mvThemeCol_Button, dpg.mvThemeCol_ButtonHovered,
                           dpg.mvThemeCol_ButtonActive):
                dpg.add_theme_color(colour, (42, 45, 52), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (105, 110, 120), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, *padding, category=dpg.mvThemeCat_Core)
    return theme
