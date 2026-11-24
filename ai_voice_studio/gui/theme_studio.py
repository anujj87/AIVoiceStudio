"""Custom theme creator (Settings -> Appearance -> Create custom theme...).

Users design their own theme - a name, six colours and a painted style - and
may give it a wallpaper of their own image.  Any number of themes can be
created, edited and deleted; they are stored in the application settings
(``custom_themes``) and appear in the Theme list beside the twenty predefined
dark themes.

The dialog is deliberately a small form plus a live preview: every colour
picker and the style box redraw the preview immediately, so the effect of a
choice is visible before the theme is saved.
"""

from __future__ import annotations

import logging

import wx

from ..settings import Settings
from . import announce, dpi
from .a11y import add_labeled, finalize_accessibility, set_accessible_name
from .theme import (
    PATTERNS,
    custom_themes,
    delete_custom_theme,
    get_wallpaper,
    mix,
    new_theme_id,
    paint_background,
    palette_from_record,
    save_custom_theme,
    set_wallpaper,
)

log = logging.getLogger(__name__)

#: The colour fields the form edits (attribute, palette key, label).
_COLOUR_FIELDS = (
    ("bg_picker", "bg", "Background"),
    ("fg_picker", "fg", "Text"),
    ("control_bg_picker", "control_bg", "Control background"),
    ("control_fg_picker", "control_fg", "Control text"),
    ("accent_picker", "accent", "Accent"),
    ("heading_picker", "heading", "Heading"),
)

_DEFAULT_COLORS = {
    "bg": "#1b1b24",
    "fg": "#e8e8f0",
    "control_bg": "#2a2a38",
    "control_fg": "#e8e8f0",
    "accent": "#7aa2f7",
    "heading": "#ffffff",
}


class _ThemePreview(wx.Panel):
    """A small canvas that paints a candidate palette and style."""

    def __init__(self, parent: wx.Window):
        super().__init__(parent, size=(260, 130))
        self.SetMinSize((200, 110))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetName("Theme preview")
        self._colors: dict[str, str] = dict(_DEFAULT_COLORS)
        self._pattern = "gradient_v"
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda _evt: None)

    def update_preview(self, colors: dict[str, str], pattern: str) -> None:
        self._colors = dict(colors)
        self._pattern = pattern
        self.Refresh()

    def _on_paint(self, _evt) -> None:
        dc = wx.AutoBufferedPaintDC(self)
        width, height = self.GetClientSize()
        paint_background(dc, (width, height), self._colors, self._pattern)
        # A little sample text so the contrast of the choices is visible.
        dc.SetTextForeground(wx.Colour(self._colors.get("fg", "#ffffff")))
        dc.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL,
                           wx.FONTWEIGHT_BOLD))
        dc.DrawText("AI Voice Studio", 14, 16)
        accent = self._colors.get("accent", "#7aa2f7")
        dc.SetPen(wx.Pen(wx.Colour(accent), 3))
        dc.DrawLine(14, 40, 140, 40)
        box = mix(self._colors.get("control_bg", "#2a2a38"),
                  self._colors.get("accent", "#7aa2f7"), 0.10)
        dc.SetBrush(wx.Brush(box))
        dc.SetPen(wx.Pen(box))
        dc.DrawRectangle(14, 56, 220, 26)


class ThemeStudioDialog(wx.Dialog):
    """Create, edit and delete custom themes (unlimited in number)."""

    def __init__(self, parent: wx.Window, settings: Settings):
        super().__init__(
            parent,
            title="Custom theme studio",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX,
            size=(720, 560),
        )
        self.settings = settings
        self._current_id: str = ""
        self.last_saved_theme_id: str = ""

        outer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(wx.StaticText(self, label="Your custom themes:"), 0, wx.ALL, 8)

        body = wx.BoxSizer(wx.HORIZONTAL)

        # -- theme list -----------------------------------------------------
        left = wx.BoxSizer(wx.VERTICAL)
        self.theme_list = wx.ListBox(self, style=wx.LB_SINGLE, size=(200, 260))
        self.theme_list.SetName("Custom themes")
        self.theme_list.SetToolTip("Select a theme to edit it")
        self.theme_list.Bind(wx.EVT_LISTBOX, self._on_select)
        left.Add(self.theme_list, 1, wx.EXPAND | wx.ALL, 4)
        self.new_btn = wx.Button(self, label="&New theme")
        self.new_btn.SetToolTip("Start a new custom theme")
        self.new_btn.Bind(wx.EVT_BUTTON, lambda _evt: self._new_theme())
        left.Add(self.new_btn, 0, wx.EXPAND | wx.ALL, 4)
        self.delete_btn = wx.Button(self, label="&Delete theme")
        self.delete_btn.SetToolTip("Delete the selected custom theme")
        self.delete_btn.Bind(wx.EVT_BUTTON, lambda _evt: self._delete_theme())
        left.Add(self.delete_btn, 0, wx.EXPAND | wx.ALL, 4)
        body.Add(left, 0, wx.EXPAND | wx.ALL, 6)

        # -- form -----------------------------------------------------------
        right = wx.BoxSizer(wx.VERTICAL)

        form = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        form.AddGrowableCol(1)
        self.name_ctrl = wx.TextCtrl(self)
        self.name_ctrl.SetToolTip("A name for this theme")
        add_labeled(self, form, "Theme name", self.name_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self._pickers: dict[str, wx.ColourPickerCtrl] = {}
        for attr, key, label in _COLOUR_FIELDS:
            picker = wx.ColourPickerCtrl(self, colour=wx.Colour(_DEFAULT_COLORS[key]))
            set_accessible_name(picker, label)
            picker.Bind(wx.EVT_COLOURPICKER_CHANGED, lambda _evt: self._refresh_preview())
            setattr(self, attr, picker)
            self._pickers[key] = picker
            form.Add(wx.StaticText(self, label=label + ":"), 0,
                     wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
            form.Add(picker, 0, wx.LEFT | wx.RIGHT, 2)

        self.style_choice = wx.Choice(self, choices=[label for _v, label in PATTERNS])
        self.style_choice.SetSelection(0)
        self.style_choice.Bind(wx.EVT_CHOICE, lambda _evt: self._refresh_preview())
        add_labeled(self, form, "Style", self.style_choice,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        right.Add(form, 0, wx.EXPAND | wx.ALL, 6)

        self.preview = _ThemePreview(self)
        right.Add(self.preview, 0, wx.EXPAND | wx.ALL, 6)

        # -- wallpaper ------------------------------------------------------
        right.Add(wx.StaticText(self, label="Wallpaper (on the welcome screen)"),
                  0, wx.ALL, 6)
        self.wallpaper_cb = wx.CheckBox(
            self, label="Show a wallpaper in the empty area of the welcome screen")
        self.wallpaper_cb.SetName("Wallpaper")
        self.wallpaper_cb.SetToolTip(
            "Fills the empty area of the welcome screen - the region beside "
            "the project sidebar - with a painted background. The buttons "
            "and the recent projects list keep the theme colour.")
        right.Add(self.wallpaper_cb, 0, wx.LEFT | wx.RIGHT, 6)

        wall_row = wx.BoxSizer(wx.HORIZONTAL)
        self.image_ctrl = wx.TextCtrl(self, style=wx.TE_READONLY)
        self.image_ctrl.SetName("Wallpaper image")
        wall_row.Add(self.image_ctrl, 1, wx.EXPAND | wx.RIGHT, 6)
        self.browse_btn = wx.Button(self, label="Choose &image...")
        self.browse_btn.Bind(wx.EVT_BUTTON, lambda _evt: self._choose_image())
        wall_row.Add(self.browse_btn, 0, wx.RIGHT, 4)
        self.clear_btn = wx.Button(self, label="C&lear")
        self.clear_btn.Bind(wx.EVT_BUTTON, lambda _evt: self._clear_image())
        wall_row.Add(self.clear_btn, 0)
        right.Add(wall_row, 0, wx.EXPAND | wx.ALL, 6)
        self.pattern_choice = wx.Choice(
            self, choices=["(use this theme's style)"] + [label for _v, label in PATTERNS])
        self.pattern_choice.SetSelection(0)
        self.pattern_choice.Bind(wx.EVT_CHOICE, lambda _evt: self._refresh_preview())
        pattern_row = wx.BoxSizer(wx.HORIZONTAL)
        pattern_label = wx.StaticText(
            self, label="Background style when no image is chosen:")
        pattern_row.Add(pattern_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        set_accessible_name(self.pattern_choice,
                            "Background style when no image is chosen")
        pattern_row.Add(self.pattern_choice, 1, wx.EXPAND)
        right.Add(pattern_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.status = wx.StaticText(self, label="")
        right.Add(self.status, 0, wx.ALL, 6)

        body.Add(right, 1, wx.EXPAND | wx.ALL, 6)
        outer.Add(body, 1, wx.EXPAND | wx.ALL, 6)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.save_btn = wx.Button(self, label="&Save theme")
        self.save_btn.SetToolTip("Save this theme and make it available in the Theme list")
        self.save_btn.Bind(wx.EVT_BUTTON, lambda _evt: self._save())
        buttons.Add(self.save_btn, 0, wx.RIGHT, 6)
        close_btn = wx.Button(self, wx.ID_CLOSE, label="Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda _evt: self.EndModal(wx.ID_CLOSE))
        buttons.Add(close_btn, 0)
        outer.Add(buttons, 0, wx.ALL | wx.ALIGN_RIGHT, 10)

        self.SetSizer(outer)
        self.SetMinSize((640, 480))
        self._reload_list()
        self._new_theme(focus_name=False)
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)

    # -- list/form ----------------------------------------------------------
    def _reload_list(self, select_id: str = "") -> None:
        self.theme_list.Clear()
        self._records = list(custom_themes(self.settings))
        for index, record in enumerate(self._records):
            self.theme_list.Append(str(record.get("name") or "Custom theme"))
            if record.get("id") == select_id:
                self.theme_list.SetSelection(index)
        self.delete_btn.Enable(bool(self._records))

    def _on_select(self, _evt) -> None:
        index = self.theme_list.GetSelection()
        if 0 <= index < len(self._records):
            self._load_record(self._records[index])

    def _load_record(self, record: dict) -> None:
        self._current_id = str(record.get("id") or "")
        self.name_ctrl.SetValue(str(record.get("name") or ""))
        palette = palette_from_record(record)
        for _attr, key, _label in _COLOUR_FIELDS:
            self._pickers[key].SetColour(wx.Colour(palette[key]))
        pattern = record.get("pattern") if record.get("pattern") in dict(PATTERNS) else "gradient_v"
        self.style_choice.SetSelection(list(dict(PATTERNS)).index(pattern))
        wallpaper = get_wallpaper(self.settings, self._current_id)
        self.wallpaper_cb.SetValue(bool(wallpaper["enabled"]))
        self.image_ctrl.SetValue(wallpaper["image"])
        pattern_override = wallpaper.get("pattern") or ""
        ids = ["", *[value for value, _label in PATTERNS]]
        if pattern_override in ids:
            self.pattern_choice.SetSelection(ids.index(pattern_override))
        else:
            self.pattern_choice.SetSelection(0)
        self._refresh_preview()

    def _new_theme(self, focus_name: bool = True) -> None:
        self._current_id = ""
        self.theme_list.SetSelection(-1)
        self.name_ctrl.SetValue("")
        for _attr, key, _label in _COLOUR_FIELDS:
            self._pickers[key].SetColour(wx.Colour(_DEFAULT_COLORS[key]))
        self.style_choice.SetSelection(1)  # vertical gradient
        self.wallpaper_cb.SetValue(False)
        self.image_ctrl.SetValue("")
        self.pattern_choice.SetSelection(0)
        announce.status_text(self.status, "Fill in a name and colours, then Save theme.")
        self._refresh_preview()
        if focus_name:
            self.name_ctrl.SetFocus()

    def _current_colors(self) -> dict[str, str]:
        return {
            key: self._pickers[key].GetColour().GetAsString(wx.C2S_HTML_SYNTAX)
            for _attr, key, _label in _COLOUR_FIELDS
        }

    def _current_pattern(self) -> str:
        index = self.style_choice.GetSelection()
        if 0 <= index < len(PATTERNS):
            return PATTERNS[index][0]
        return "gradient_v"

    def _refresh_preview(self) -> None:
        colors = self._current_colors()
        pattern = self._current_pattern()
        index = self.pattern_choice.GetSelection()
        if index > 0 and index - 1 < len(PATTERNS):
            pattern = PATTERNS[index - 1][0]
        self.preview.update_preview(colors, pattern)

    # -- wallpaper ----------------------------------------------------------
    def _choose_image(self) -> None:
        wildcard = (
            "Images (*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp)|"
            "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp|All files (*.*)|*.*"
        )
        with wx.FileDialog(
            self, "Choose a wallpaper image", wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.image_ctrl.SetValue(dlg.GetPath())
                self.wallpaper_cb.SetValue(True)

    def _clear_image(self) -> None:
        self.image_ctrl.SetValue("")

    def _save(self) -> None:
        name = self.name_ctrl.GetValue().strip()
        if not name:
            wx.MessageBox("Please give the theme a name.", "Custom theme",
                          style=wx.OK | wx.ICON_INFORMATION, parent=self)
            self.name_ctrl.SetFocus()
            return
        theme_id = self._current_id or new_theme_id(self.settings, name)
        record = {
            "id": theme_id,
            "name": name,
            "colors": self._current_colors(),
            "pattern": self._current_pattern(),
        }
        save_custom_theme(self.settings, record)
        pattern_index = self.pattern_choice.GetSelection()
        override = ""
        if 0 < pattern_index <= len(PATTERNS):
            override = PATTERNS[pattern_index - 1][0]
        set_wallpaper(
            self.settings, theme_id,
            enabled=self.wallpaper_cb.GetValue(),
            image=self.image_ctrl.GetValue().strip(),
            pattern=override,
        )
        self.last_saved_theme_id = theme_id
        self._current_id = theme_id
        self._reload_list(select_id=theme_id)
        announce.status_text(self.status, f"Saved '{name}'.")

    def _delete_theme(self) -> None:
        if not self._current_id:
            wx.MessageBox("Select a custom theme to delete first.",
                          "Custom theme", style=wx.OK | wx.ICON_INFORMATION,
                          parent=self)
            return
        record = next((r for r in custom_themes(self.settings)
                       if r.get("id") == self._current_id), None)
        name = str(record.get("name")) if record else self._current_id
        if wx.MessageBox(
            f"Delete the custom theme '{name}'?",
            "Custom theme", style=wx.YES_NO | wx.ICON_QUESTION, parent=self,
        ) != wx.YES:
            return
        delete_custom_theme(self.settings, self._current_id)
        self._reload_list()
        self._new_theme(focus_name=False)
        announce.status_text(self.status, f"Deleted '{name}'.")
