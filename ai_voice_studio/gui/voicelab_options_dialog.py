"""Per-engine tuning dialog for the Voice Lab engines.

One dialog serves both places the tuning options can be set:

* **Settings > Voice Clone** stores the *defaults* of an engine (used by every
  new project), and
* the **Recording window** stores *per-project overrides*, exactly like the
  OmniVoice options dialog does for the OmniVoice engines.

The fields come straight from ``voicelab.options`` (one widget per declared
option), so adding a knob to the engine data is enough for it to appear here.
A control always shows the engine's own default as its starting point, and only
values that differ from that default are kept - "Reset to engine defaults" is
therefore the same as "send nothing and let the engine decide".

Every control carries a real MSAA accessible name (see ``gui.a11y``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import wx

from ..voicelab import engines as voice_lab
from ..voicelab import options as voice_options
from .a11y import add_labeled, finalize_accessibility, set_accessible_name

#: Text shown above the controls, per scope.
_SCOPE_INTRO = {
    "project": "These tuning options are saved with this project only.",
    "default": "These tuning options are the defaults for every new project.",
}


class VoiceLabOptionsDialog(wx.Dialog):
    """Tune one Voice Lab engine (per project, or as the saved default)."""

    def __init__(
        self,
        parent,
        engine_id: str,
        values: Optional[Dict[str, Any]] = None,
        project_name: str = "",
        scope: str = "project",
    ):
        engine_label = voice_lab.engine_name(engine_id)
        title = f"{engine_label} options"
        if project_name:
            title = f"{title} - {project_name}"
        super().__init__(
            parent,
            title=title,
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.engine_id = engine_id
        self.engine_label = engine_label
        self.scope = scope if scope in _SCOPE_INTRO else "project"
        self._specs = voice_options.specs(engine_id)
        # Start from the engine defaults and overlay the stored overrides, so
        # every control shows a real value even when nothing was set yet.
        self._values: Dict[str, Any] = dict(voice_options.defaults(engine_id))
        self._values.update(voice_options.clean(engine_id, values))
        self._controls: Dict[str, wx.Window] = {}
        self._build_ui()
        self._load_values(self._values)
        finalize_accessibility(self)
        wx.CallAfter(self._focus_first)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label=_SCOPE_INTRO[self.scope]),
            0, wx.ALL, 8,
        )

        if not self._specs:
            sizer.Add(
                wx.StaticText(
                    self,
                    label="This engine has no tuning options.",
                ),
                0, wx.ALL, 8,
            )
        else:
            box = wx.StaticBoxSizer(
                wx.StaticBox(self, label="Generation settings"), wx.VERTICAL
            )
            holder = box.GetStaticBox()
            grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
            grid.AddGrowableCol(1)
            for option in self._specs:
                # The controls live inside the group box, so the label text
                # next to each one is its sibling (that is what a screen reader
                # and the label-adjacency audit both look at).
                control = self._make_control(option, holder)
                self._controls[option.key] = control
                label = option.label
                if option.optional and option.default in (None, ""):
                    label = f"{label} (optional)"
                add_labeled(holder, grid, label, control,
                            flag=wx.LEFT | wx.RIGHT, border=2)
                if option.help:
                    help_text = wx.StaticText(holder, label="    " + option.help)
                    help_text.SetName(f"{option.label} help")
                    grid.Add((0, 0))
                    grid.Add(help_text, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)
            box.Add(grid, 0, wx.EXPAND | wx.ALL, 4)
            note = voice_options.max_seconds_note(self.engine_id)
            if note:
                box.Add(
                    wx.StaticText(holder, label=note),
                    0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
                )
            sizer.Add(box, 0, wx.EXPAND | wx.ALL, 8)

        self.summary = wx.StaticText(self, label="")
        self.summary.SetName("Active tuning overrides")
        sizer.Add(self.summary, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        sizer.Add(
            wx.StaticText(
                self,
                label=(
                    "Only the values that differ from the engine defaults are "
                    "stored and sent. Parameters the installed engine version "
                    "does not support are skipped and reported, never fatal."
                ),
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10,
        )

        self.reset_btn = wx.Button(self, label="Reset to engine defaults")
        set_accessible_name(self.reset_btn, "Reset to engine defaults")
        self.reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)
        sizer.Add(self.reset_btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # The buttons are created explicitly instead of through
        # ``CreateSeparatedButtonSizer``: wx re-creates the stock buttons while
        # laying the dialog out, so a name or a handler attached to the button
        # found by id can end up on a window that is not the one on screen.
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        buttons.AddStretchSpacer(1)
        self.save_btn = wx.Button(self, wx.ID_OK, label="Save options")
        set_accessible_name(self.save_btn, "Save options")
        self.save_btn.SetDefault()
        self.save_btn.Bind(wx.EVT_BUTTON, self._on_ok)
        self.cancel_btn = wx.Button(self, wx.ID_CANCEL, label="Cancel")
        set_accessible_name(self.cancel_btn, "Cancel")
        self.cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        buttons.Add(self.save_btn, 0, wx.ALL, 4)
        buttons.Add(self.cancel_btn, 0, wx.ALL, 4)
        sizer.Add(buttons, 0, wx.EXPAND | wx.ALL, 6)
        self.SetSizerAndFit(sizer)
        self.SetMinSize((560, -1))

    def _make_control(self, option: voice_options.Option,
                      parent: wx.Window) -> wx.Window:
        """Create the widget the option's kind asks for."""
        kind = option.kind
        if kind == voice_options.KIND_BOOL:
            control: wx.Window = wx.CheckBox(parent, label="")
            control.Bind(wx.EVT_CHECKBOX, self._on_changed)
        elif kind == voice_options.KIND_INT:
            control = wx.SpinCtrl(
                parent,
                min=int(option.minimum or 0),
                max=int(option.maximum or 999),
            )
            control.Bind(wx.EVT_SPINCTRL, self._on_changed)
        elif kind == voice_options.KIND_FLOAT:
            control = wx.SpinCtrlDouble(
                parent,
                min=float(option.minimum or 0.0),
                max=float(option.maximum or 1.0),
                inc=float(option.increment or 0.1),
            )
            control.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_changed)
        elif kind == voice_options.KIND_CHOICE:
            control = wx.ComboBox(parent, style=wx.CB_READONLY)
            for choice in option.choices:
                control.Append(choice, choice)
            control.Bind(wx.EVT_COMBOBOX, self._on_changed)
        elif kind == voice_options.KIND_SEED:
            control = wx.TextCtrl(parent)
            control.SetToolTip(option.help or "")
            control.Bind(wx.EVT_TEXT, self._on_changed)
        else:  # KIND_TEXT
            control = wx.TextCtrl(parent)
            control.SetToolTip(option.help or "")
            control.Bind(wx.EVT_TEXT, self._on_changed)
        set_accessible_name(control, option.label)
        return control

    def _focus_first(self):
        for option in self._specs:
            control = self._controls.get(option.key)
            if control is not None:
                control.SetFocus()
                return
        self.reset_btn.SetFocus()

    # -- state -------------------------------------------------------------
    def _load_values(self, values: Dict[str, Any]):
        for option in self._specs:
            control = self._controls.get(option.key)
            if control is None:
                continue
            value = values.get(option.key, option.default)
            if option.kind == voice_options.KIND_BOOL:
                control.SetValue(bool(value))
            elif option.kind == voice_options.KIND_INT:
                control.SetValue(int(value if value is not None else 0))
            elif option.kind == voice_options.KIND_FLOAT:
                control.SetValue(float(value if value is not None else 0.0))
            elif option.kind == voice_options.KIND_CHOICE:
                index = control.FindString(str(value)) if value else wx.NOT_FOUND
                control.SetSelection(index if index != wx.NOT_FOUND else 0)
            else:
                control.ChangeValue("" if value is None else str(value))
        self._refresh_summary()

    def _read_values(self) -> Dict[str, Any]:
        """The raw control values, as typed."""
        values: Dict[str, Any] = {}
        for option in self._specs:
            control = self._controls.get(option.key)
            if control is None:
                continue
            if option.kind == voice_options.KIND_BOOL:
                values[option.key] = bool(control.GetValue())
            elif option.kind in (voice_options.KIND_INT,
                                 voice_options.KIND_FLOAT):
                values[option.key] = control.GetValue()
            elif option.kind == voice_options.KIND_CHOICE:
                selection = control.GetSelection()
                values[option.key] = (
                    str(control.GetClientData(selection))
                    if selection != wx.NOT_FOUND else ""
                )
            else:
                values[option.key] = control.GetValue().strip()
        return values

    def _refresh_summary(self):
        """Show which overrides are currently active."""
        try:
            cleaned = voice_options.clean(self.engine_id, self._read_values())
        except ValueError:
            self.summary.SetLabel("Active overrides: fix the highlighted value.")
            return
        text = voice_options.describe_overrides(self.engine_id, cleaned)
        self.summary.SetLabel(
            f"Active overrides: {text}" if text
            else "Active overrides: none - the engine defaults are used."
        )

    def _on_changed(self, _evt):
        self._refresh_summary()

    def _on_reset(self, _evt):
        self._load_values(voice_options.defaults(self.engine_id))

    def _on_ok(self, _evt):
        try:
            voice_options.clean(self.engine_id, self._read_values())
        except ValueError as exc:
            # Do not close: the user has to fix the value first.
            wx.MessageBox(str(exc), "Voice Lab options",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        self.EndModal(wx.ID_OK)

    def _on_cancel(self, _evt):
        self.EndModal(wx.ID_CANCEL)

    # -- result ------------------------------------------------------------
    def get_options(self) -> Dict[str, Any]:
        """The stored overrides: only values that differ from the defaults."""
        return voice_options.clean(self.engine_id, self._read_values())

    def has_overrides(self) -> bool:
        return bool(self.get_options())

    def summary_text(self) -> str:
        return voice_options.describe_overrides(self.engine_id, self.get_options())


def tune_options(parent, engine_id: str, values: Optional[Dict[str, Any]],
                 project_name: str = "", scope: str = "project"):
    """Show the dialog; return the overrides, or ``None`` when cancelled."""
    engine_label = voice_lab.engine_name(engine_id)
    dlg = VoiceLabOptionsDialog(
        parent,
        engine_id=engine_id,
        values=values,
        project_name=project_name if scope == "project" else "",
        scope=scope,
    )
    try:
        if dlg.ShowModal() != wx.ID_OK:
            return None
        return dlg.get_options()
    finally:
        dlg.Destroy()


def engine_choices() -> List[str]:
    """Engines that have tuning options (handy for tests and for the GUI)."""
    return [engine_id for engine_id in voice_lab.engine_ids()
            if voice_options.has_options(engine_id)]
