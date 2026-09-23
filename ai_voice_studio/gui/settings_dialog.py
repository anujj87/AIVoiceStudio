"""Settings dialog (SPEC 3.1, NVDA-style).

The dialog follows the NVDA Settings dialog design: a list of categories on
the left, the selected category's panel on the right, and OK / Cancel / Apply
buttons underneath. Keyboard behaviour matches NVDA too:

* Up / Down arrows move through the categories (the panel changes on focus).
* Ctrl+Tab / Ctrl+Shift+Tab switch categories, wrapping around.
* Enter activates OK; Ctrl+S activates Apply (a guarded, screen-reader
  friendly variant of NVDA's ``_enterActivatesOk_ctrlSActivatesApply``).
* Focus starts on the category list.

Categories:
1. General                -- theme (System / Light / Dark)
2. Download and remove    -- model manager (DownloadPanel)
3. Available TTS          -- downloaded voices (AvailablePanel)
4. Voice Clone            -- CPU/GPU voice-clone engines (VoiceClonePanel)
5. OmniVoice engines      -- GPU TTS engine variants (Server, Triton, Hybrid)
6. Recording settings     -- speed, pitch, volume, preview
7. Punctuation            -- default punctuation mode (spoken-word expansion)
8. Audio file creation    -- audio modes with descriptions + pages per file
9. DAISY settings         -- DAISY 2.02 audio book defaults
10. Compute               -- optional GPU (CUDA) runtime + OmniVoice
11. Developer             -- addon management, pip, diagnostics
12. OmniVoice Server      -- network TTS server configuration
13. Reset                 -- restore defaults
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import wx
from wx.lib import scrolledpanel

from .. import paths, runtime
from ..constants import (
    AUDIO_MODE_CHOICES,
    AUDIO_MODE_DESCRIPTIONS,
    DAISY_SPLIT_ALL_HEADINGS,
    DAISY_SPLIT_CHOICES,
    DAISY_SPLIT_DESCRIPTIONS,
    DAISY_SPLIT_H1,
    MODE_PAGE_ONLY,
    MODE_PAGE_WITH_H1,
    PAGES_PER_FILE_MAX,
    PAGES_PER_FILE_MIN,
    PITCH_MAX,
    PITCH_MIN,
    PROJECT_TYPE_DAISY_AUDIO,
    PUNCTUATION_CHOICES,
    PUNCTUATION_DEFAULT,
    RATE_MAX,
    RATE_MIN,
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    VOLUME_MAX,
    VOLUME_MIN,
)
from .. import venv_packages
from ..settings import Settings
from ..tts import catalog, windows_tts
from ..tts.downloader import ModelDownloader
from ..tts.models import ModelStore
from .a11y import (
    add_labeled,
    finalize_accessibility,
    set_accessible_name,
    update_accessible_name,
)
from .events import (
    DownloadFinishedEvent,
    DownloadProgressEvent,
    EVT_DOWNLOAD_FINISHED,
    EVT_DOWNLOAD_PROGRESS,
)
from . import access_keys, dialogs, language_choice
from .clone_engines_panel import VoiceClonePanel
from .model_panels import AvailablePanel, DownloadPanel
from .theme import apply_theme

log = logging.getLogger(__name__)

_THEME_CHOICES = [
    (THEME_SYSTEM, "System default"),
    (THEME_LIGHT, "Light"),
    (THEME_DARK, "Dark"),
]

# Managed-environment status line, keyed by env dir: probing it runs a
# subprocess, so the answer is reused until an install changes it.
_PIP_STATUS_CACHE: dict = {}


class _SettingsPanel(access_keys.AccessKeyHints, wx.Panel):
    """Base class for a settings category (NVDA ``SettingsPanel`` pattern).

    Mirrors NVDA's ``SettingsPanel`` interface:
    - ``title``: shown in the category list.
    - ``panelDescription``: spoken description for screen readers.
    - ``on_activated()`` / ``on_deactivated()``: called on category switch.
    - ``apply_to_settings()``: called on OK / Apply.
    - ``isValid()``: validation hook (return False to block save).
    """

    title = ""
    description = ""
    panelDescription = ""

    def on_activated(self):
        """Called when this category is selected (NVDA onPanelActivated)."""
        self.Show()
        self.Layout()

    def on_deactivated(self):
        """Called when another category is selected (NVDA onPanelDeactivated)."""
        self.Hide()

    def apply_to_settings(self):
        """Save this panel's settings (NVDA onSave)."""
        pass

    def isValid(self) -> bool:
        """Validate this panel's settings (NVDA isValid).
        Return False to block saving.
        """
        return True

    # Access-key feedback (access_key_hint / show_access_key_hint) comes from
    # the AccessKeyHints mixin below, so every settings category can explain a
    # disabled access key the same way.

    # -- background voice discovery ---------------------------------------
    def _on_voices_ready(self, _voices=None):
        """A background voice probe finished: refresh this panel's cascade.

        Used by the managed-venv package probe and the Windows voice
        enumeration so that building the panel never waits for a subprocess
        (that wait is what made opening Settings feel slow).
        """
        populate = getattr(self, "_populate_voices", None)
        if populate is None:
            return
        try:
            wx.CallAfter(populate)
        except Exception:  # noqa: BLE001
            pass

    def _on_package_probe(self, package: str, engine: str | None = None):
        """A managed-venv probe finished; re-add voices when it is installed."""
        try:
            if venv_packages.version(package, engine=engine):
                self._on_voices_ready()
        except Exception:  # noqa: BLE001
            pass


class _SettingsPanelAccessible(wx.Accessible):
    """Report a settings panel as a property page with a spoken description
    (mirrors NVDA's ``SettingsPanelAccessible``)."""

    def __init__(self, panel):
        super().__init__(panel)
        self._desc = panel.description

    def GetRole(self, childId):
        return (wx.ACC_OK, wx.ROLE_SYSTEM_PROPERTYPAGE)

    def GetDescription(self, childId):
        return (wx.ACC_OK, self._desc)


class SettingsDialog(wx.Dialog):
    """NVDA-style multi-category settings dialog.

    Layout: category list (left) + settings panel (right) + OK / Cancel / Apply.
    Keyboard: arrows move between categories, Ctrl+Tab wraps, Enter = OK,
    Ctrl+S = Apply.
    """

    # Order shown in the category list (resolved in __init__: the panel
    # classes are defined further down this module). Panels are constructed
    # eagerly so that every category's controls exist (and can be saved)
    # regardless of whether the user visits it.
    CATEGORIES = []

    def __init__(self, parent, settings: Settings, store: ModelStore):
        super().__init__(
            parent,
            title="Settings - AI Voice Studio",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX
            | wx.MINIMIZE_BOX,
            size=(820, 560),
        )
        self.settings = settings
        self.store = store
        self.downloader = ModelDownloader(store)
        self._orig = settings.as_dict()
        self._panels: list[_SettingsPanel] = []
        self._current: int = 0
        self.CATEGORIES = [
            _GeneralPanel,
            DownloadPanel,
            AvailablePanel,
            VoiceClonePanel,
            _OmniVoiceEnginesPanel,
            _OmniVoiceServerPanel,
            _RecordingSettingsPanel,
            _PunctuationPanel,
            _AudioModePanel,
            _DaisySettingsPanel,
            _ComputePanel,
            _DeveloperPanel,
            _ResetPanel,
        ]

        self._build_ui()
        self._bind_events()

        self._show_category(0)
        apply_theme(self, self.settings.theme)
        # Real MSAA accNames for every labelled control (SetName alone is
        # ignored by MSAA on this wx build).
        finalize_accessibility(self)
        # NVDA-style postInit: focus lands on the category list.
        wx.CallAfter(self.cat_list.SetFocus)

    # -- construction -------------------------------------------------------
    def _build_ui(self):
        """Build the dialog using NVDA's layout patterns:
        - GridBagSizer for the 2-column layout (list left, panel right)
        - ScrolledPanel for the settings panel (handles overflow)
        - Freeze/Thaw during category changes (prevents flicker)
        """
        main = wx.BoxSizer(wx.VERTICAL)

        # Category list (NVDA uses AutoWidthColumnListCtrl; we use ListCtrl
        # with the same style flags).
        self.cat_list = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_NO_HEADER,
            size=(200, 10),  # minimal height; grid grows it
        )
        self.cat_list.InsertColumn(0, "Categories:")
        self.cat_list.SetName("Settings categories")

        # ScrolledPanel: content scrolls when it exceeds the visible area
        # (NVDA pattern — prevents controls from being clipped).
        self.container = scrolledpanel.ScrolledPanel(
            self, style=wx.TAB_TRAVERSAL | wx.BORDER_THEME,
        )
        self.container.SetName("Settings panel")
        self.container.SetMinSize((1, 1))
        self.container_sizer = wx.BoxSizer(wx.VERTICAL)
        self.container.SetSizer(self.container_sizer)

        for cls in self.CATEGORIES:
            self.cat_list.Append((cls.title,))
            panel = cls(self.container, *self._panel_args(cls))
            panel.Hide()
            panel.description = cls.description
            try:
                panel.SetAccessible(_SettingsPanelAccessible(panel))
            except Exception:  # noqa: BLE001
                pass
            self.container_sizer.Add(
                panel, 1, wx.ALL | wx.EXPAND,
                border=10,  # NVDA BORDER_FOR_DIALOGS
            )
            self._panels.append(panel)

        # NVDA uses GridBagSizer with 1:3 proportion (list:panel).
        grid = wx.GridBagSizer(
            hgap=7,   # SPACE_BETWEEN_BUTTONS_HORIZONTAL
            vgap=5,   # SPACE_BETWEEN_BUTTONS_VERTICAL
        )
        categories_label = wx.StaticText(self, label="&Categories:")
        grid.Add(categories_label, pos=(0, 0), span=(1, 2))
        grid.Add(self.cat_list, pos=(1, 0), flag=wx.EXPAND)
        grid.Add(self.container, pos=(1, 1), flag=wx.EXPAND)
        grid.AddGrowableRow(1)
        grid.AddGrowableCol(0, proportion=1)
        grid.AddGrowableCol(1, proportion=3)
        main.Add(grid, 1, wx.EXPAND | wx.ALL, border=10)

        # Separated button sizer (NVDA pattern).
        main.Add(
            self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL | wx.APPLY),
            0, wx.EXPAND | wx.ALL, border=10,
        )
        self.SetSizer(main)
        self.SetMinSize((700, 420))
        self.container.SetupScrolling()

    def _panel_args(self, cls):
        """Extra constructor arguments per panel class (NVDA panels take a
        parent plus category-specific dependencies)."""
        if cls is _ComputePanel:
            return ()
        if cls in (DownloadPanel,):
            return (self.store, self.downloader)
        if cls in (AvailablePanel,):
            return (self.store, self.settings)
        if cls in (_OmniVoiceEnginesPanel, VoiceClonePanel):
            return (self.settings, self.store)
        if cls in (_OmniVoiceServerPanel,):
            return (self.settings,)
        if cls in (_RecordingSettingsPanel, _PunctuationPanel):
            return (self.settings, self.store)
        return (self.settings,)

    # -- events -------------------------------------------------------------
    def _bind_events(self):
        self.cat_list.Bind(wx.EVT_LIST_ITEM_FOCUSED, self._on_category_focus)
        self.Bind(wx.EVT_BUTTON, self._on_apply, id=wx.ID_APPLY)
        self.Bind(wx.EVT_BUTTON, self._on_ok, id=wx.ID_OK)
        self.Bind(wx.EVT_BUTTON, self._on_cancel, id=wx.ID_CANCEL)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        self.Bind(wx.EVT_CLOSE, self._on_close)

    def _on_category_focus(self, evt):
        self._show_category(evt.GetIndex(), focus_panel=False)
        evt.Skip()

    def _on_close(self, evt):
        """Handle the window close button (X) and Alt+F4."""
        self._on_cancel(None)

    def _show_category(self, index: int, *, focus_panel: bool = True):
        """Show the panel for ``index`` and hide every other (NVDA pattern).

        NVDA's ``MultiCategorySettingsDialog._doCategoryChange`` uses
        Freeze/Thaw to prevent visual artifacts during the switch.

        When *focus_panel* is True (Ctrl+Tab or initial open) focus moves
        into the panel so the screen reader announces the first control.
        When False (arrow keys in the category list) focus stays in the
        list so subsequent Up/Down keys keep navigating categories.
        """
        if not (0 <= index < len(self._panels)):
            return
        # NVDA pattern: Freeze the container during the switch to prevent
        # controls from briefly appearing in wrong positions.
        self.container.Freeze()
        try:
            for i, panel in enumerate(self._panels):
                if i == index:
                    panel.on_activated()
                else:
                    panel.on_deactivated()
            self._current = index
            self.container.Layout()
            self.container.SetupScrolling()
            self.container.Refresh()
        finally:
            self.container.Thaw()
        if focus_panel:
            # Move focus into the panel so NVDA/JAWS announce the first control.
            panel = self._panels[index]
            first = self._find_first_focusable(panel)
            if first:
                def _focus():
                    first.SetFocus()
                    # Ensure the focused control is visible in the scrolled panel.
                    if hasattr(self.container, "ScrollChildIntoView"):
                        self.container.ScrollChildIntoView(first)
                wx.CallAfter(_focus)
            else:
                wx.CallAfter(panel.SetFocus)

    @staticmethod
    def _find_first_focusable(panel: wx.Window) -> wx.Window | None:
        """Find the first enabled, visible interactive child control
        (NVDA-style depth-first search)."""
        _INTERACTIVE = (wx.TextCtrl, wx.ComboBox, wx.Choice, wx.CheckBox,
                        wx.RadioButton, wx.ListBox, wx.Button, wx.Slider,
                        wx.SpinCtrl)
        stack = list(panel.GetChildren())
        while stack:
            child = stack.pop(0)
            if (isinstance(child, _INTERACTIVE)
                    and child.IsShown() and child.IsEnabled()):
                return child
            stack.extend(child.GetChildren())
        return None

    def _on_char_hook(self, evt):
        """NVDA-style keyboard handling.

        Mirrors NVDA's ``_enterActivatesOk_ctrlSActivatesApply`` plus
        category switching.  Escape and Alt+F4 are handled by wx's default
        dialog close mechanism (``evt.Skip()``).

        - Ctrl+Tab / Ctrl+Shift+Tab: switch category (wraps around).
        - Enter: activate OK (but skip if a button/combo has focus).
        - Ctrl+S: activate Apply.
        - Escape / Alt+F4: handled by ``EVT_CLOSE`` binding.
        """
        key = evt.GetKeyCode()
        control = self.FindFocus()
        if evt.ControlDown() and key == wx.WXK_TAB:
            # NVDA pattern: focus the category list first so the panel
            # hides correctly, then switch.
            list_had_focus = self.cat_list.HasFocus()
            if not list_had_focus:
                self.cat_list.SetFocus()
            index = self.cat_list.GetFirstSelected()
            if index < 0:
                index = self._current
            step = -1 if evt.ShiftDown() else 1
            new_index = (index + step) % len(self._panels)
            self.cat_list.Select(new_index)
            self.cat_list.Focus(new_index)
            # NVDA pattern: restore focus to panel after category switch
            # if the list didn't originally have focus.
            if not list_had_focus and self._panels[new_index].IsShown():
                self._show_category(new_index, focus_panel=True)
            else:
                self._show_category(new_index, focus_panel=False)
            return
        multiline = isinstance(control, wx.TextCtrl) and (
            control.GetWindowStyle() & wx.TE_MULTILINE
        )
        if key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER) and not multiline:
            # Controls that consume Enter themselves (buttons, comboboxes and
            # their popups) must keep it: otherwise Enter on e.g. the Download
            # button would close the dialog instead of starting the download.
            if isinstance(control, (wx.Button, wx.Choice, wx.ComboBox, wx.ListBox)):
                evt.Skip()
                return
            self.ProcessEvent(
                wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED, wx.ID_OK)
            )
        elif evt.ControlDown() and evt.GetUnicodeKey() == ord("S"):
            self.ProcessEvent(
                wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED, wx.ID_APPLY)
            )
        else:
            # NVDA pattern: let wx handle Escape, Alt+F4, and all other keys
            # via its default dialog behaviour.
            evt.Skip()

    # -- callbacks ----------------------------------------------------------
    def _refresh_available(self):
        self.available_panel.refresh()
        self.download_panel._refresh_buttons()

    def _validate_all(self) -> bool:
        """Check all panels are valid before saving (NVDA _validateAllPanels)."""
        for panel in self._panels:
            try:
                if not hasattr(panel, "isValid"):
                    log.warning("Panel %s has no isValid() method — skipping validation", panel.title)
                    continue
                if panel.isValid() is False:
                    log.warning("Panel %s failed validation", panel.title)
                    return False
            except Exception:  # noqa: BLE001
                log.exception("Panel %s raised during validation", panel.title)
                return False
        return True

    def _save_from_ui(self):
        """Validate all panels, then save (NVDA _doSave pattern)."""
        if not self._validate_all():
            return False
        for panel in self._panels:
            try:
                panel.apply_to_settings()
            except Exception:  # noqa: BLE001
                log.exception("Panel %s failed to save", panel.title)
        self.settings.save()
        return True

    def _on_apply(self, _):
        log.info("Settings dialog: Apply button clicked")
        if self._save_from_ui():
            log.info("Settings dialog: Apply succeeded")
            apply_theme(self, self.settings.theme)
            # Stay open — move focus back to the category list so the
            # user can continue configuring (matches NVDA pattern).
            wx.CallAfter(self.cat_list.SetFocus)
        else:
            log.warning("Settings dialog: Apply failed (validation or save error)")

    def _on_ok(self, _):
        log.info("Settings dialog: OK button clicked")
        if self._save_from_ui():
            log.info("Settings dialog: OK succeeded — closing")
            self.EndModal(wx.ID_OK)
        else:
            log.warning("Settings dialog: OK failed (validation or save error)")

    def _on_cancel(self, _):
        log.info("Settings dialog: Cancel button clicked — discarding changes")
        # Discard any in-flight changes by restoring the pre-open snapshot.
        self.settings.set_many({k: v for k, v in self._orig.items()})
        self.EndModal(wx.ID_CANCEL)

    # Convenience attributes kept for tests / external code.
    def _panel_by_title(self, title: str):
        """Look a category up by its title (keeps the accessors honest)."""
        for panel in self._panels:
            if panel.title == title:
                return panel
        raise KeyError(title)

    @property
    def general_panel(self):
        return self._panels[0]

    @property
    def download_panel(self):
        return self._panels[1]

    @property
    def available_panel(self):
        return self._panels[2]

    @property
    def voice_clone_panel(self):
        """The 'Voice Clone' category: Pocket TTS / Bark / F5-TTS."""
        return self._panel_by_title("Voice Clone")

    @property
    def omnivoice_engines_panel(self):
        """The 'OmniVoice engines' category (voice library + engine cards).

        This used to be a separate 'Voice clone' category; the clone/design
        studio now lives inside the OmniVoice engines category, so the old
        ``voice_clone_panel`` alias was renamed to match what it returns.
        """
        return self._panel_by_title("OmniVoice engines")

    @property
    def omnivoice_server_panel(self):
        return self._panel_by_title("OmniVoice Server")

    @property
    def recording_panel(self):
        return self._panel_by_title("Recording settings")

    @property
    def punctuation_panel(self):
        return self._panel_by_title("Punctuation")

    @property
    def audio_mode_panel(self):
        return self._panel_by_title("Audio file creation")

    @property
    def daisy_panel(self):
        return self._panel_by_title("DAISY settings")

    @property
    def compute_panel(self):
        return self._panel_by_title("Compute")

    @property
    def developer_panel(self):
        return self._panel_by_title("Developer")

    @property
    def reset_panel(self):
        return self._panel_by_title("Reset")


# ---------------------------------------------------------------------------
# General category
# ---------------------------------------------------------------------------
class _GeneralPanel(_SettingsPanel):
    title = "General"

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Appearance"),
            0, wx.ALL, 6,
        )
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.theme_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Theme selector")
        for value, label in _THEME_CHOICES:
            self.theme_combo.Append(label, value)
        index = next((i for i, (v, _) in enumerate(_THEME_CHOICES) if v == settings.theme), 0)
        self.theme_combo.SetSelection(index)
        add_labeled(self, grid, "Theme", self.theme_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        # Developer Mode checkbox
        self.dev_mode_cb = wx.CheckBox(self, label="Enable Developer Mode")
        self.dev_mode_cb.SetName("Developer Mode")
        self.dev_mode_cb.SetValue(settings.get("developer_mode", False))
        self.dev_mode_cb.SetToolTip(
            "Enables advanced features: heavy logging to diagnostic files, "
            "addon management, and additional developer tools. "
            "Requires restarting the application for full effect."
        )
        grid.Add((1, 1))  # spacer
        grid.Add(self.dev_mode_cb, 0, wx.ALL, 2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        self.preview = wx.StaticText(
            self,
            label="Note: the theme applies immediately to this dialog when you "
                  "press Apply or OK.",
        )
        sizer.Add(self.preview, 0, wx.ALL, 6)

        sizer.Add(
            wx.StaticText(self, label="File locations"),
            0, wx.ALL, 6,
        )
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.recordings_ctrl = self._path_row(
            grid, "Recorded files location",
            settings.get("paths.recordings_dir") or paths.recordings_dir(),
        )
        self.models_ctrl = self._path_row(
            grid, "Model files location",
            settings.get("paths.models_dir") or paths.models_dir(),
        )
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="New recordings and downloads are saved to these folders. "
                      "Existing files are not moved.",
            ),
            0, wx.ALL, 6,
        )

        sizer.Add(wx.StaticText(self, label="Updates"), 0, wx.ALL, 6)
        self.update_auto_cb = wx.CheckBox(
            self,
            label="Check for updates automatically when the application starts",
        )
        self.update_auto_cb.SetName("Check for updates automatically")
        self.update_auto_cb.SetValue(bool(settings.get("updates.auto_check", True)))
        self.update_auto_cb.SetToolTip(
            "Once a day the application asks GitHub whether a newer version "
            "has been released. Nothing is ever downloaded or installed "
            "without asking you first."
        )
        sizer.Add(self.update_auto_cb, 0, wx.LEFT | wx.RIGHT, 6)
        self.update_now_btn = wx.Button(self, label="Check for updates &now")
        self.update_now_btn.SetName("Check for updates now")
        self.update_now_btn.SetToolTip(
            "Ask GitHub for the newest release right now"
        )
        self.update_now_btn.Bind(wx.EVT_BUTTON, self._on_check_updates_now)
        update_row = wx.BoxSizer(wx.HORIZONTAL)
        update_row.Add(self.update_now_btn, 0, wx.ALL, 6)
        sizer.Add(update_row, 0, wx.EXPAND)
        sizer.Add(
            wx.StaticText(
                self,
                label="Updates are downloaded from the project's GitHub releases "
                      "page, and only after you agree.",
            ),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)
        self.theme_combo.Bind(wx.EVT_COMBOBOX, self._on_theme)

    def _path_row(self, grid, name: str, current: str) -> wx.TextCtrl:
        """Add ``name`` + read-only path field + Browse... button to the grid."""
        text = wx.TextCtrl(self, style=wx.TE_READONLY)
        text.SetName(name)
        text.SetValue(current)
        browse = wx.Button(self, label="Browse...")
        browse.SetName(name + " browse")
        browse.Bind(wx.EVT_BUTTON, lambda _evt, c=text, n=name: self._on_browse(c, n))
        label = wx.StaticText(self, label=name + ":")
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(text, 1, wx.EXPAND)
        row.Add(browse, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6)
        grid.Add(row, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        return text

    def _on_browse(self, ctrl: wx.TextCtrl, name: str):
        with wx.DirDialog(
            self,
            message="Choose folder for " + name.lower(),
            defaultPath=ctrl.GetValue() or paths.user_data_dir(),
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                ctrl.SetValue(dlg.GetPath())

    def _on_check_updates_now(self, _evt=None):
        """Run a check straight from Settings (Help has the same action)."""
        from .update_dialog import check_for_updates_interactive  # noqa: PLC0415

        check_for_updates_interactive(self, self.settings)

    def _on_theme(self, _):
        apply_theme(self, self.selected())

    def selected(self) -> str:
        sel = self.theme_combo.GetSelection()
        return self.theme_combo.GetClientData(sel) if sel >= 0 else THEME_SYSTEM

    def apply_to_settings(self):
        self.settings.set("theme", self.selected())
        self.settings.set("paths.recordings_dir", self.recordings_ctrl.GetValue().strip())
        self.settings.set("paths.models_dir", self.models_ctrl.GetValue().strip())
        self.settings.set("developer_mode", self.dev_mode_cb.GetValue())
        self.settings.set("updates.auto_check", self.update_auto_cb.GetValue())


# ---------------------------------------------------------------------------
# Recording settings category
# ---------------------------------------------------------------------------
class _RecordingSettingsPanel(_SettingsPanel):
    title = "Recording settings"

    def __init__(self, parent, settings: Settings, store: ModelStore):
        super().__init__(parent)
        self.settings = settings
        self.store = store
        self._voices: list = []
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Voice cascade: choose which downloaded/ready TTS engine these
        # Speed/Pitch/Volume defaults apply to (same cascade as the
        # Punctuation panel).
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.tts_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                     name="TTS engine for recording defaults")
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Variant for recording defaults")
        self.voice_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Voice for recording defaults")
        add_labeled(self, grid, "TTS engine", self.tts_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Variant", self.variant_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Voice", self.voice_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        # -- TTS and voice preview ------------------------------------------
        # The same idea as the Punctuation category: every voice the selected
        # TTS engine can speak with is listed here, so a long list (Kokoro's
        # multilingual speakers, the built-in Windows voices, ...) stays easy
        # to browse.  Choosing a row fills the Variant/Voice boxes above; the
        # Preview button then speaks it with the current Speed/Pitch/Volume.
        voices_box = wx.StaticBox(self, label="TTS and voice preview")
        voices_sizer = wx.StaticBoxSizer(voices_box, wx.VERTICAL)
        voices_sizer.Add(
            wx.StaticText(
                voices_box,
                label="Voices available for the selected TTS engine:",
            ),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 4,
        )
        self.voices_list = wx.ListBox(
            voices_box, size=(-1, 110),
            name="Voices for the selected TTS engine",
        )
        voices_sizer.Add(self.voices_list, 1, wx.EXPAND | wx.ALL, 4)
        self.voice_count = wx.StaticText(voices_box, label="")
        self.voice_count.SetName("Voice count")
        voices_sizer.Add(self.voice_count, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(voices_sizer, 1, wx.EXPAND | wx.ALL, 6)
        self._voice_list_entries: list = []

        self.rate = self._slider_row(sizer, "Speed", 1.0, RATE_MIN, RATE_MAX)
        self.pitch = self._slider_row(sizer, "Pitch", 1.0, PITCH_MIN, PITCH_MAX)
        self.volume = self._slider_row(sizer, "Volume", 1.0, VOLUME_MIN, VOLUME_MAX)

        sizer.Add(
            wx.StaticText(self, label="Preview text (edit it, then press Preview):"),
            0, wx.ALL, 4,
        )
        self.sample_text = wx.TextCtrl(
            self, value="Welcome to AI Voice Studio. This is a preview of how "
                        "the selected voice will sound.",
            style=wx.TE_MULTILINE, size=(-1, 90),
        )
        self.sample_text.SetName("Preview text")
        sizer.Add(self.sample_text, 0, wx.EXPAND | wx.ALL, 6)

        # Alt+P previews (the & is the Windows access-key marker; the
        # accessible name stays "Preview").
        self.preview_btn = wx.Button(self, label="&Preview")
        self.preview_btn.SetName("Preview")
        # Which back-end the Preview speaks with (CPU, GPU when detected,
        # Auto); remembered per category in Settings.
        from .compute_choice import make_compute_row  # noqa: PLC0415

        preview_row = wx.BoxSizer(wx.HORIZONTAL)
        compute_label, self.compute_combo = make_compute_row(
            self, self.settings, "recording_settings"
        )
        preview_row.Add(compute_label, 0,
                        wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        preview_row.Add(self.compute_combo, 0,
                        wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        preview_row.Add(self.preview_btn, 0, wx.ALL, 4)
        sizer.Add(preview_row, 0, wx.LEFT, 2)
        self.preview_status = wx.StaticText(self, label="")
        sizer.Add(self.preview_status, 0, wx.ALL, 4)
        self.SetSizer(sizer)

        self.tts_combo.Bind(wx.EVT_COMBOBOX, self._on_tts)
        self.variant_combo.Bind(wx.EVT_COMBOBOX, self._on_variant)
        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
        self.voices_list.Bind(wx.EVT_LISTBOX, self._on_voice_list_select)
        self._populate_voices()

    def on_activated(self):
        """Refresh the voice cascade when the category is opened so voices
        created elsewhere (e.g. the OmniVoice voice library) appear without
        closing the dialog."""
        super().on_activated()
        self._populate_voices()

    # -- voice cascade ------------------------------------------------------
    def _populate_voices(self):
        self._voices = self.store.installed_voices()
        self._inject_pip_installed_voices()
        # Built-in Windows voices (SAPI5 / Windows Core): always available,
        # no download and no package needed.
        windows_tts.add_installed_voices(self._voices, self._on_voices_ready)
        # Universal OmniVoice voice library: created voices are engine
        # agnostic, so register them under every installed OmniVoice engine.
        try:
            from ..omnivoice import voice_store  # noqa: PLC0415
            if voice_store.omni_custom_voices(self.store):
                installed = voice_store.engine_ids_installed()
                self._voices.extend(
                    voice_store.consumer_entries(self.store, installed)
                )
        except Exception:  # noqa: BLE001
            pass
        tts_ids = sorted({v["tts"] for v in self._voices})
        self.tts_combo.Clear()
        for tts_id in tts_ids:
            tts = catalog.find_tts(tts_id)
            self.tts_combo.Append(tts["name"] if tts else tts_id, tts_id)
        if self.tts_combo.GetCount():
            self.tts_combo.SetSelection(0)
            self._on_tts(None)
        else:
            self.variant_combo.Clear()
            self.voice_combo.Clear()
            self.preview_btn.Disable()
            self.preview_status.SetLabel(
                "No voices downloaded. Use the 'Download and remove' tab first."
            )
            self._load_defaults_for_tts(None)
        self._refresh_voice_list()

    def _inject_pip_installed_voices(self):
        """Inject voices for TTS engines installed via pip (e.g. OmniVoice)."""
        try:
            # Voice Lab engines (Pocket TTS, Bark, F5-TTS): their pre-made
            # voices are provided by the installed package rather than by a
            # catalog variant.
            from ..voicelab import builtin_voice_entries  # noqa: PLC0415

            self._voices.extend(builtin_voice_entries())
            for tts_entry in catalog.get_tts_list():
                pkg = tts_entry.get("requires_package")
                if not pkg:
                    continue
                # Every pip-installed TTS engine lives in its *own* Python
                # environment, so its catalog id names the environment.
                engine = catalog.engine_env_id(tts_entry)
                if not venv_packages.installed(pkg, engine=engine):
                    # Not installed yet (or the background probe is still
                    # running): ask to be told when the answer arrives.
                    if not venv_packages.is_known(pkg, engine=engine):
                        venv_packages.request(
                            pkg, engine=engine,
                            on_ready=lambda _v, p=pkg, e=engine:
                                self._on_package_probe(p, e),
                        )
                    continue
                for lang in tts_entry.get("languages", []):
                    for variant in lang.get("variants", []):
                        for voice in variant.get("voices", []):
                            self._voices.append(
                                {
                                    "tts": tts_entry["id"],
                                    "tts_name": tts_entry["name"],
                                    "language": lang["code"],
                                    "variant": variant["id"],
                                    "voice": voice["id"],
                                    "voice_name": voice.get("name", voice["id"]),
                                    "sid": voice.get("sid", 0),
                                    "engine": tts_entry.get("engine", "vits"),
                                    "dir": "",
                                    "requires_gpu": tts_entry.get("requires_gpu", False),
                                    "requires_package": pkg,
                                }
                            )
        except Exception:  # noqa: BLE001
            pass

    def _on_tts(self, _):
        tts_sel = self.tts_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        tts = catalog.find_tts(tts_id) if tts_id else None
        languages = sorted({v["language"] for v in self._voices if v["tts"] == tts_id})
        keys = sorted({
            (v["language"], v["variant"])
            for v in self._voices if v["tts"] == tts_id
        })
        self.variant_combo.Clear()
        for lang, vid in keys:
            variant = catalog.find_variant(tts, lang, vid) if tts else None
            label = variant["name"] if variant else vid
            if len(languages) > 1:
                lang_label = catalog.language_display_name(tts, lang) if tts else lang
                label = f"{label} ({lang_label})"
            self.variant_combo.Append(label, (lang, vid))
        if self.variant_combo.GetCount():
            self.variant_combo.SetSelection(0)
        self._on_variant(None)
        self._load_defaults_for_tts(tts_id)
        self._refresh_voice_list()

    def _on_variant(self, _):
        key = self.variant_combo.GetClientData(self.variant_combo.GetSelection()) \
            if self.variant_combo.GetSelection() >= 0 else None
        self.voice_combo.Clear()
        if not key:
            return
        lang, vid = key
        tts_id = self.tts_combo.GetClientData(self.tts_combo.GetSelection())
        for v in self._voices:
            if (v["tts"], v["language"], v["variant"]) == (tts_id, lang, vid):
                self.voice_combo.Append(v["voice_name"], v)
        if self.voice_combo.GetCount():
            self.voice_combo.SetSelection(0)

    def selected_voice(self):
        sel = self.voice_combo.GetSelection()
        if sel < 0:
            return None
        return self.voice_combo.GetClientData(sel)

    def _selected_tts_id(self):
        sel = self.tts_combo.GetSelection()
        return self.tts_combo.GetClientData(sel) if sel >= 0 else None

    # -- TTS and voice preview list -----------------------------------------
    def _refresh_voice_list(self):
        """List every voice of the selected TTS engine (all variants)."""
        tts_id = self._selected_tts_id()
        tts = catalog.find_tts(tts_id) if tts_id else None
        self.voices_list.Clear()
        self._voice_list_entries = []
        for voice in self._voices:
            if voice["tts"] != tts_id:
                continue
            variant = (
                catalog.find_variant(tts, voice["language"], voice["variant"])
                if tts else None
            )
            variant_label = variant["name"] if variant else voice["variant"]
            self._voice_list_entries.append(voice)
            self.voices_list.Append(
                f"{voice.get('voice_name', voice['voice'])} - {variant_label}"
            )
        if self._voice_list_entries:
            self.voices_list.SetSelection(0)
        count = len(self._voice_list_entries)
        if not tts_id:
            self.voice_count.SetLabel("No TTS engine available.")
        elif count:
            self.voice_count.SetLabel(
                f"{count} voice{'s' if count != 1 else ''} available for this "
                "TTS engine."
            )
        else:
            self.voice_count.SetLabel(
                "No voices available for this TTS engine."
            )

    def _on_voice_list_select(self, _evt=None):
        """Choosing a row in the list moves the Variant/Voice boxes above."""
        sel = self.voices_list.GetSelection()
        if sel < 0 or sel >= len(self._voice_list_entries):
            return
        voice = self._voice_list_entries[sel]
        for index in range(self.variant_combo.GetCount()):
            if self.variant_combo.GetClientData(index) != (
                voice["language"], voice["variant"]
            ):
                continue
            if self.variant_combo.GetSelection() != index:
                self.variant_combo.SetSelection(index)
                self._on_variant(None)
            break
        for index in range(self.voice_combo.GetCount()):
            entry = self.voice_combo.GetClientData(index)
            if entry and entry.get("voice") == voice.get("voice"):
                self.voice_combo.SetSelection(index)
                break
        self.preview_status.SetLabel(
            f"Selected: {voice.get('voice_name', voice['voice'])}."
        )

    # -- per-TTS defaults ---------------------------------------------------
    def _per_tts(self) -> dict:
        return self.settings.get("recording.per_tts", {}) or {}

    def _tts_defaults(self, tts_id):
        per = self._per_tts()
        if tts_id and per.get(tts_id):
            return per[tts_id]
        return {
            "rate": self.settings.get("recording.rate", 1.0),
            "pitch": self.settings.get("recording.pitch", 1.0),
            "volume": self.settings.get("recording.volume", 1.0),
        }

    def _load_defaults_for_tts(self, tts_id):
        """Move the stored per-TTS Speed/Pitch/Volume into the sliders."""
        defaults = self._tts_defaults(tts_id)
        for slider, name, key in (
            (self.rate, "Speed", "rate"),
            (self.pitch, "Pitch", "pitch"),
            (self.volume, "Volume", "volume"),
        ):
            value = float(defaults.get(key, 1.0))
            self._set_slider_value(slider, value, name)

    def _set_slider_value(self, slider, value: float, name: str):
        slider.SetValue(int(round(float(value) * 100)))
        label = getattr(slider, "value_label", None)
        if label is not None:
            label.SetLabel(f"{name}: {value:.2f}")
        update_accessible_name(slider, f"{name}: {value:.2f}")

    def _slider_row(self, sizer, name: str, value: float, lo: float, hi: float):
        """A labelled slider row.

        The value lives in the row label's own text ("Speed: 1.00"), NOT in a
        separate bare-number static: screen readers announce static text by
        its visible text, so a standalone "1.00" node would be read with no
        context ("1.0", "1.0", ...).  One text node carries the word and the
        number together.
        """
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        label = wx.StaticText(self, label=f"{name}: {value:.2f}")
        label.SetName(f"{name} value")
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        slider = wx.Slider(self, minValue=int(lo * 100), maxValue=int(hi * 100),
                           value=int(value * 100))
        set_accessible_name(slider, f"{name}: {value:.2f}")
        grid.Add(slider, 1, wx.EXPAND)
        slider.value_label = label  # type: ignore[attr-defined]
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 4)

        def _on_change(_evt, s=slider, lb=label, n=name):
            lb.SetLabel(f"{n}: {s.GetValue() / 100.0:.2f}")
            update_accessible_name(s, f"{n}: {s.GetValue() / 100.0:.2f}")

        slider.Bind(wx.EVT_SLIDER, _on_change)
        return slider

    def apply_to_settings(self):
        tts_id = self._selected_tts_id()
        values = {
            "rate": self.rate.GetValue() / 100.0,
            "pitch": self.pitch.GetValue() / 100.0,
            "volume": self.volume.GetValue() / 100.0,
        }
        if tts_id:
            # Per-TTS defaults: each engine remembers its own sliders.
            per_tts = dict(self._per_tts())
            per_tts[tts_id] = values
            self.settings.set("recording.per_tts", per_tts)
        else:
            # No TTS selected: adjust the global defaults instead.
            self.settings.set("recording.rate", values["rate"])
            self.settings.set("recording.pitch", values["pitch"])
            self.settings.set("recording.volume", values["volume"])

    def _on_preview(self, evt=None):
        # One Alt+P press reaches this handler more than once (see
        # access_keys.once): the guard keeps it to one preview.
        if not access_keys.once(evt if evt is not None else self.preview_btn):
            return
        voice = self.selected_voice()
        if not voice:
            wx.MessageBox(
                "Select a TTS engine and voice first. Use the 'Download and "
                "remove' tab to install voices if none are ready.",
                "Preview unavailable",
                style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        # Snapshot the UI state on the UI thread; the job runs on a worker.
        text = self.sample_text.GetValue() or "Hello."
        punct = self.settings.get("recording.punctuation", "default")
        rate = self.rate.GetValue() / 100.0
        pitch = self.pitch.GetValue() / 100.0
        volume = self.volume.GetValue() / 100.0
        self.preview_btn.Disable()
        self.preview_status.SetLabel("Synthesizing preview...")
        # Snapshot the compute choice on the UI thread; the job runs on a worker.
        from .compute_choice import _combo_value  # noqa: PLC0415

        choice = _combo_value(self.compute_combo)
        threading.Thread(
            target=self._preview_job,
            args=(voice, text, punct, rate, pitch, volume, choice),
            daemon=True,
        ).start()

    def _preview_job(self, voice, text, punct, rate, pitch, volume, choice="cpu"):
        from ..audio.output import write_wav
        from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation
        from .compute_choice import provider_for_preview

        try:
            engine = get_engine(
                voice, provider=provider_for_preview(choice, voice.get("engine"))
            )
            text = process_punctuation(text, punct)
            samples = engine.synthesize(
                text,
                sid=voice.get("sid", 0),
                speed=rate,
                pitch=pitch,
                volume=volume,
            )
            import tempfile  # noqa: PLC0415

            fd, tmp = tempfile.mkstemp(prefix="aivs_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}")

    def _preview_done(self, tmp, error):
        import wx.adv  # noqa: PLC0415

        self.preview_btn.Enable()
        if error:
            self.preview_status.SetLabel(error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()
        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            # Keep a reference: wxSound must outlive Play(SOUND_ASYNC) or the
            # preview is cut off before it is heard (garbage collection bug).
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            self.preview_status.SetLabel("Playing preview.")
        else:
            self.preview_status.SetLabel("Preview file could not be played.")

    def _stop_preview_sound(self):
        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:  # noqa: BLE001
                pass
            self._preview_sound = None


# ---------------------------------------------------------------------------
# Punctuation category (default mode for new projects)
# ---------------------------------------------------------------------------
class _PunctuationPanel(_SettingsPanel):
    title = "Punctuation"

    """Default punctuation mode used by the New Project wizard.

    'All' expands every punctuation mark into its spoken word (quote, dot,
    left paren, tic, ...) so TTS voices that cannot pronounce punctuation
    still read text correctly. A voice cascade (TTS -> variant -> voice) and
    a Preview button let you hear exactly how the chosen mode will sound.
    """

    _EXAMPLE = 'My name is "Anuj Sharma". print(\'hello\')'

    def __init__(self, parent, settings: Settings, store: ModelStore):
        super().__init__(parent)
        self.settings = settings
        self.store = store
        self._voices: list = []
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="'All' reads every punctuation mark as a word "
                                      "(quote, dot, left paren, tic)."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.punct_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                        name="Punctuation mode")
        for value, label in PUNCTUATION_CHOICES:
            self.punct_combo.Append(label, value)
        current = settings.get("recording.punctuation", PUNCTUATION_DEFAULT)
        idx = next(
            (i for i, (v, _) in enumerate(PUNCTUATION_CHOICES) if v == current), 0
        )
        self.punct_combo.SetSelection(idx)
        add_labeled(self, grid, "Punctuation mode", self.punct_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        # Voice selection: TTS -> variant -> voice (same cascade as the
        # Recording window, without a separate language combo; the language is
        # shown in the variant label when a TTS has more than one).
        voice_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        voice_grid.AddGrowableCol(1)
        self.tts_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                     name="TTS engine for preview")
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Variant for preview")
        self.voice_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Voice for preview")
        add_labeled(self, voice_grid, "TTS engine", self.tts_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, voice_grid, "Variant", self.variant_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, voice_grid, "Voice", self.voice_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(voice_grid, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(
            wx.StaticText(self, label="Example text (edit it to try your own):"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 4,
        )
        self.example_text = wx.TextCtrl(self, value=self._EXAMPLE,
                                        style=wx.TE_MULTILINE, size=(-1, 60))
        self.example_text.SetName("Punctuation example text")
        sizer.Add(self.example_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        sizer.Add(
            wx.StaticText(self, label="Will be spoken as:"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 4,
        )
        self.example_out = wx.StaticText(self, label="")
        self.example_out.SetName("Punctuation example output")
        self.example_out.Wrap(700)
        sizer.Add(self.example_out, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # Alt+P previews (the & is the Windows access-key marker; the
        # accessible name stays "Preview").
        self.preview_btn = wx.Button(self, label="&Preview")
        self.preview_btn.SetName("Preview")
        # Which back-end the Preview speaks with (CPU, GPU when detected,
        # Auto); remembered per category in Settings.
        from .compute_choice import make_compute_row  # noqa: PLC0415

        preview_row = wx.BoxSizer(wx.HORIZONTAL)
        compute_label, self.compute_combo = make_compute_row(
            self, settings, "punctuation"
        )
        preview_row.Add(compute_label, 0,
                        wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        preview_row.Add(self.compute_combo, 0,
                        wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        preview_row.Add(self.preview_btn, 0, wx.ALL, 4)
        sizer.Add(preview_row, 0, wx.LEFT, 2)
        self.preview_status = wx.StaticText(self, label="")
        sizer.Add(self.preview_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        self.SetSizer(sizer)

        self.punct_combo.Bind(wx.EVT_COMBOBOX, self._update_example)
        self.example_text.Bind(wx.EVT_TEXT, self._update_example)
        self.tts_combo.Bind(wx.EVT_COMBOBOX, self._on_tts)
        self.variant_combo.Bind(wx.EVT_COMBOBOX, self._on_variant)
        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
        self._populate_voices()
        self._update_example()

    def on_activated(self):
        """Refresh the preview voice cascade when the category is opened so
        voices created elsewhere (e.g. the OmniVoice voice library) appear
        without closing the dialog."""
        super().on_activated()
        self._populate_voices()
        self._update_example()

    # -- voice cascade ------------------------------------------------------
    def _populate_voices(self):
        self._voices = self.store.installed_voices()
        # Inject pip-installed OmniVoice voices (not in artifact system)
        self._inject_pip_installed_voices()
        # Built-in Windows voices (SAPI5 / Windows Core): always available,
        # no download and no package needed.
        windows_tts.add_installed_voices(self._voices, self._on_voices_ready)
        # Universal OmniVoice voice library: created voices are engine
        # agnostic, so register them under every installed OmniVoice engine.
        try:
            from ..omnivoice import voice_store  # noqa: PLC0415
            if voice_store.omni_custom_voices(self.store):
                installed = voice_store.engine_ids_installed()
                self._voices.extend(
                    voice_store.consumer_entries(self.store, installed)
                )
        except Exception:  # noqa: BLE001
            pass
        tts_ids = sorted({v["tts"] for v in self._voices})
        self.tts_combo.Clear()
        for tts_id in tts_ids:
            tts = catalog.find_tts(tts_id)
            self.tts_combo.Append(tts["name"] if tts else tts_id, tts_id)
        if self.tts_combo.GetCount():
            self.tts_combo.SetSelection(0)
            self._on_tts(None)
        else:
            self.preview_btn.Disable()
            self.preview_status.SetLabel(
                "No voices downloaded. Use the 'Download and remove' tab first."
            )

    def _inject_pip_installed_voices(self):
        """Inject voices for TTS engines installed via pip (e.g. OmniVoice).

        These engines don't use the artifact download system, so their
        voices never appear in ``store.installed_voices()``.  We check if
        the package is installed in the *managed venv* (not the main
        process) and, if so, create voice entries from the catalog.
        """
        try:
            # Voice Lab engines (Pocket TTS, Bark, F5-TTS): their pre-made
            # voices are provided by the installed package rather than by a
            # catalog variant.
            from ..voicelab import builtin_voice_entries  # noqa: PLC0415

            self._voices.extend(builtin_voice_entries())
            for tts_entry in catalog.get_tts_list():
                pkg = tts_entry.get("requires_package")
                if not pkg:
                    continue
                # Every pip-installed TTS engine lives in its *own* Python
                # environment, so its catalog id names the environment.
                engine = catalog.engine_env_id(tts_entry)
                if not venv_packages.installed(pkg, engine=engine):
                    # Not installed yet (or the background probe is still
                    # running): ask to be told when the answer arrives.
                    if not venv_packages.is_known(pkg, engine=engine):
                        venv_packages.request(
                            pkg, engine=engine,
                            on_ready=lambda _v, p=pkg, e=engine:
                                self._on_package_probe(p, e),
                        )
                    continue
                for lang in tts_entry.get("languages", []):
                    for variant in lang.get("variants", []):
                        for voice in variant.get("voices", []):
                            self._voices.append(
                                {
                                    "tts": tts_entry["id"],
                                    "tts_name": tts_entry["name"],
                                    "language": lang["code"],
                                    "variant": variant["id"],
                                    "voice": voice["id"],
                                    "voice_name": voice.get("name", voice["id"]),
                                    "sid": voice.get("sid", 0),
                                    "engine": tts_entry.get("engine", "vits"),
                                    "dir": "",
                                    "requires_gpu": tts_entry.get("requires_gpu", False),
                                    "requires_package": pkg,
                                }
                            )
        except Exception:  # noqa: BLE001
            pass

    def _on_tts(self, _):
        tts_sel = self.tts_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        tts = catalog.find_tts(tts_id) if tts_id else None
        languages = sorted({v["language"] for v in self._voices if v["tts"] == tts_id})
        keys = sorted({
            (v["language"], v["variant"])
            for v in self._voices if v["tts"] == tts_id
        })
        self.variant_combo.Clear()
        for lang, vid in keys:
            variant = catalog.find_variant(tts, lang, vid) if tts else None
            label = variant["name"] if variant else vid
            if len(languages) > 1:
                lang_label = catalog.language_display_name(tts, lang) if tts else lang
                label = f"{label} ({lang_label})"
            self.variant_combo.Append(label, (lang, vid))
        if self.variant_combo.GetCount():
            self.variant_combo.SetSelection(0)
        self._on_variant(None)

    def _on_variant(self, _):
        key = self.variant_combo.GetClientData(self.variant_combo.GetSelection()) \
            if self.variant_combo.GetSelection() >= 0 else None
        self.voice_combo.Clear()
        if not key:
            return
        lang, vid = key
        tts_id = self.tts_combo.GetClientData(self.tts_combo.GetSelection())
        for v in self._voices:
            if (v["tts"], v["language"], v["variant"]) == (tts_id, lang, vid):
                self.voice_combo.Append(v["voice_name"], v)
        if self.voice_combo.GetCount():
            self.voice_combo.SetSelection(0)

    def selected_voice(self):
        sel = self.voice_combo.GetSelection()
        if sel < 0:
            return None
        return self.voice_combo.GetClientData(sel)

    # -- punctuation --------------------------------------------------------
    def _update_example(self, _evt=None):
        from ..tts.engine import process_punctuation  # noqa: PLC0415

        mode = self.selected()
        text = self.example_text.GetValue() or "Hello."
        self.example_out.SetLabel(process_punctuation(text, mode))

    def selected(self) -> str:
        sel = self.punct_combo.GetSelection()
        return self.punct_combo.GetClientData(sel) if sel >= 0 else PUNCTUATION_DEFAULT

    def apply_to_settings(self):
        self.settings.set("recording.punctuation", self.selected())

    # -- preview ------------------------------------------------------------
    def _on_preview(self, evt=None):
        # One Alt+P press reaches this handler more than once (see
        # access_keys.once): the guard keeps it to one preview.
        if not access_keys.once(evt if evt is not None else self.preview_btn):
            return
        voice = self.selected_voice()
        if not voice:
            wx.MessageBox("Select a voice first.", "Preview",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        self.preview_btn.Disable()
        self.preview_status.SetLabel("Synthesizing preview...")
        mode = self.selected()
        text = self.example_text.GetValue() or "Hello."
        # Snapshot the compute choice on the UI thread; the job runs on a worker.
        from .compute_choice import _combo_value  # noqa: PLC0415

        choice = _combo_value(self.compute_combo)
        threading.Thread(
            target=self._preview_job, args=(voice, text, mode, choice), daemon=True
        ).start()

    def _preview_job(self, voice, text, mode, choice="cpu"):
        from ..audio.output import write_wav
        from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation
        from .compute_choice import provider_for_preview

        try:
            engine = get_engine(
                voice, provider=provider_for_preview(choice, voice.get("engine"))
            )
            text = process_punctuation(text, mode)
            samples = engine.synthesize(text, sid=voice.get("sid", 0), speed=1.0)
            import tempfile  # noqa: PLC0415

            fd, tmp = tempfile.mkstemp(prefix="aivs_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}")

    def _preview_done(self, tmp, error):
        import wx.adv  # noqa: PLC0415

        self.preview_btn.Enable()
        if error:
            self.preview_status.SetLabel(error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()

        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            self.preview_status.SetLabel("Playing preview.")
        else:
            self.preview_status.SetLabel("Preview file could not be played.")

    def _stop_preview_sound(self):
        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:  # noqa: BLE001
                pass
            self._preview_sound = None


# ---------------------------------------------------------------------------
# Audio file creation category
# ---------------------------------------------------------------------------
class _AudioModePanel(_SettingsPanel):
    title = "Audio file creation"

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.radios = []
        self.description = wx.StaticText(self, label="")
        sizer.Add(self.description, 0, wx.ALL, 6)
        current = settings.get("audio_mode", MODE_PAGE_WITH_H1)
        first = True
        for value, label in AUDIO_MODE_CHOICES:
            radio = wx.RadioButton(
                self, label=label,
                style=wx.RB_GROUP if first else 0,
            )
            radio.SetName(label)
            radio.SetValue(value == current)
            sizer.Add(radio, 0, wx.ALL, 4)
            radio.Bind(
                wx.EVT_RADIOBUTTON,
                lambda _evt, v=value: self._on_select(v),
            )
            self.radios.append((radio, value))
            first = False

        # Default pages per audio file (used by the wizard's "Page by page
        # only" mode). Enabled only while that mode is selected.
        pages_row = wx.BoxSizer(wx.HORIZONTAL)
        pages_label = wx.StaticText(self, label="Pages per audio file:")
        pages_label.SetName("Pages per audio file")
        pages_row.Add(pages_label, 0,
                      wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 4)
        pages_default = int(settings.get("audio_mode_pages_per_file", 1) or 1)
        pages_default = max(PAGES_PER_FILE_MIN,
                           min(PAGES_PER_FILE_MAX, pages_default))
        self.pages_spin = wx.SpinCtrl(
            self, min=PAGES_PER_FILE_MIN, max=PAGES_PER_FILE_MAX,
            initial=pages_default, name="Pages per audio file",
        )
        self.pages_spin.SetToolTip(
            "Default number of pages recorded into one audio file when the "
            "'Page by page only' mode is used in the New Project wizard."
        )
        pages_row.Add(self.pages_spin, 0, wx.RIGHT, 4)
        sizer.Add(pages_row, 0, wx.ALL, 4)

        self._on_select(self.selected())
        self._update_description()
        self.SetSizer(sizer)

    def _on_select(self, value: str):
        self.pages_spin.Enable(value == MODE_PAGE_ONLY)
        self._update_description()

    def _update_description(self):
        mode = self.selected()
        self.description.SetLabel(AUDIO_MODE_DESCRIPTIONS.get(mode, ""))
        self.description.Wrap(700)

    def selected(self) -> str:
        for radio, value in self.radios:
            if radio.GetValue():
                return value
        return MODE_PAGE_WITH_H1

    def apply_to_settings(self):
        mode = self.selected()
        self.settings.set("audio_mode", mode)
        if mode == MODE_PAGE_ONLY:
            self.settings.set("audio_mode_pages_per_file", self.pages_spin.GetValue())


# ---------------------------------------------------------------------------
# DAISY book settings category
# ---------------------------------------------------------------------------
class _DaisySettingsPanel(_SettingsPanel):
    title = "DAISY settings"

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(
            wx.StaticText(self, label="Chapter splitting"),
            0, wx.ALL, 6,
        )
        sizer.Add(
            wx.StaticText(
                self,
                label="How the document is split into DAISY chapters when a "
                      "DAISY project is created.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )
        self.radios = []
        self.description = wx.StaticText(self, label="")
        sizer.Add(self.description, 0, wx.ALL, 6)
        current = settings.get("daisy.splitting", DAISY_SPLIT_H1)
        first = True
        for value, label in DAISY_SPLIT_CHOICES:
            radio = wx.RadioButton(
                self, label=label, style=wx.RB_GROUP if first else 0
            )
            radio.SetName(label)
            radio.SetValue(value == current)
            sizer.Add(radio, 0, wx.ALL, 4)
            radio.Bind(
                wx.EVT_RADIOBUTTON,
                lambda _evt, v=value: self._update_description(),
            )
            self.radios.append((radio, value))
            first = False
        self._update_description()

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)

        self.lang_ctrl = wx.TextCtrl(self)
        self.lang_ctrl.SetName("DAISY language code")
        self.lang_ctrl.SetValue(settings.get("daisy.language", "en"))
        add_labeled(self, grid, "Language code (ISO 639)", self.lang_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.publisher_ctrl = wx.TextCtrl(self)
        self.publisher_ctrl.SetName("DAISY publisher")
        self.publisher_ctrl.SetValue(settings.get("daisy.publisher", ""))
        add_labeled(self, grid, "Publisher (optional)", self.publisher_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(
            wx.StaticText(
                self,
                label="Language code (en, hi, fr, de, es) and optional publisher "
                      "name are written into the DAISY book metadata (dc:language, "
                      "dc:publisher).",
            ),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)

    def selected(self) -> str:
        for radio, value in self.radios:
            if radio.GetValue():
                return value
        return DAISY_SPLIT_H1

    def _update_description(self):
        self.description.SetLabel(
            DAISY_SPLIT_DESCRIPTIONS.get(self.selected(), "")
        )
        self.description.Wrap(680)

    def apply_to_settings(self):
        self.settings.set("daisy.splitting", self.selected())
        self.settings.set("daisy.language", self.lang_ctrl.GetValue().strip() or "en")
        self.settings.set("daisy.publisher", self.publisher_ctrl.GetValue().strip())


# ---------------------------------------------------------------------------
# OmniVoice Engines category (GPU TTS variants)
# ---------------------------------------------------------------------------
class _OmniVoiceEnginesPanel(_SettingsPanel):
    title = "OmniVoice engines"

    """OmniVoice Engines panel.

    Two parts:

    1. **Voice library** - create reusable voices by *cloning* a reference
       sample or by *describing* the voice, then rename / delete / preview
       them.  The library is universal: one created voice is usable through
       every OmniVoice engine (direct ``omnivoice`` and ``omnivoice_server``)
       and appears in Available TTS, Punctuation and the Recording window.

    2. **Engine cards** - the OmniVoice engine variants (Server, Triton,
       Hybrid) with their features and install status.
    """

    _PREVIEW_DEFAULT = "Welcome to AI Voice Studio. This is a preview of your created voice."

    def __init__(self, parent, settings: Settings, store: ModelStore):
        super().__init__(parent)
        self.settings = settings
        self.store = store
        self._thread: threading.Thread | None = None
        self._preview_sound = None
        self._library: list = []  # ModelStore custom voices (omni kinds)
        self._lib_names: list[str] = []  # parallel display list
        self._installed: list = []  # engine ids with packages installed

        sizer = wx.BoxSizer(wx.VERTICAL)
        self._engines: list = []  # list of dicts for each engine variant

        # -- Dependency shortcut -------------------------------------------
        # OmniVoice is not downloadable from this category (it is installed
        # from Compute), so a missing dependency gets one button that takes
        # the user straight there.
        self._build_dependency_hint(sizer)

        # -- Voice library (universal, both engines) -----------------------
        self._build_voice_library(sizer)

        # -- Engine cards -------------------------------------------------
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 6)
        self._build_engine_cards(sizer)
        self._refresh_engine_status()

        # -- Features overview --------------------------------------------
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(self, label="Engine features"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        features = [
            "• 600+ languages with auto-detection",
            "• Voice design from text descriptions",
            "• Zero-shot voice cloning (3-15 seconds reference audio)",
            "• OpenAI-compatible API (Server mode)",
            "• Streaming audio output",
            "• Network access for other apps (Server mode)",
        ]
        for feat in features:
            sizer.Add(
                wx.StaticText(self, label=feat),
                0, wx.LEFT | wx.RIGHT | wx.TOP, 12,
            )

        self.SetSizer(sizer)
        self._refresh_library()

    # ------------------------------------------------------------------ UI
    def _build_voice_library(self, sizer):
        """The voice clone / voice design studio with the universal voice list."""
        from ..omnivoice import voice_store  # noqa: PLC0415

        title = wx.StaticText(self, label="Create a reusable voice")
        title.SetFont(title.GetFont().Bold())
        sizer.Add(title, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)

        # -- engine + mode selectors --------------------------------------
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.engine_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                        name="TTS version for voices")
        for label, engine_id in (
            ("OmniVoice Server", "omnivoice_server"),
            ("OmniVoice (direct, Triton)", "omnivoice"),
        ):
            self.engine_combo.Append(label, engine_id)
        self.engine_combo.SetSelection(0)
        add_labeled(self, grid, "1. TTS version", self.engine_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.mode_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Voice creation mode")
        self.mode_combo.Append("Clone a voice from a sample", "clone")
        self.mode_combo.Append("Describe a new voice", "design")
        self.mode_combo.SetSelection(0)
        add_labeled(self, grid, "2. Creation mode", self.mode_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        # 3) language: Auto (the engine detects it) or one of the 646
        # languages OmniVoice was trained on, pinned into the created voice so
        # every recording that uses it stays on that language.
        self.omni_language_combo = wx.ComboBox(
            self, style=wx.CB_READONLY,
            name="Language for the created voice",
        )
        language_choice.fill(self.omni_language_combo)
        self.omni_language_combo.SetToolTip(
            f"{language_choice.label(None)} keeps OmniVoice's own detection. "
            f"Pick one of the {language_choice.COUNT} languages the "
            "model was trained on to pin the created voice to it."
        )
        add_labeled(self, grid, "3. Language", self.omni_language_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        self.mode_tip = wx.StaticText(self, label="")
        self.mode_tip.Wrap(640)
        sizer.Add(self.mode_tip, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # -- clone group (sample -> name -> transcript -> create) ----------
        # Rows follow the order the user asked for: voice name right after
        # the sample, then the optional transcript, and the Create voice
        # button last so each labelled box is unambiguous; every row uses
        # one label directly to the left of its box.
        self.clone_panel = wx.Panel(self)
        c_sizer = wx.BoxSizer(wx.VERTICAL)

        # 1) voice sample (file + Browse) ---------------------------------
        sample_row = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        sample_row.AddGrowableCol(1)
        self.sample_ctrl = wx.TextCtrl(self.clone_panel)
        self.sample_ctrl.SetName("Voice sample")
        sample_box = wx.BoxSizer(wx.HORIZONTAL)
        sample_box.Add(self.sample_ctrl, 1, wx.EXPAND)
        browse_btn = wx.Button(self.clone_panel, label="Browse...")
        browse_btn.SetName("Browse voice sample")
        sample_box.Add(browse_btn, 0, wx.LEFT, 4)
        sample_row.Add(
            wx.StaticText(
                self.clone_panel,
                label="Voice sample:",
            ),
            0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2,
        )
        sample_row.Add(sample_box, 1, wx.EXPAND)
        c_sizer.Add(sample_row, 0, wx.EXPAND | wx.ALL, 4)
        sample_hint = wx.StaticText(
            self.clone_panel,
            label="3-15 second recording of the voice to copy (WAV/MP3/OGG/FLAC).",
        )
        c_sizer.Add(sample_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # 2) voice name ----------------------------------------------------
        name_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        name_grid.AddGrowableCol(1)
        self.clone_name_ctrl = wx.TextCtrl(self.clone_panel)
        self.clone_name_ctrl.SetName("Voice name")
        add_labeled(self.clone_panel, name_grid, "Voice name", self.clone_name_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        c_sizer.Add(name_grid, 0, wx.EXPAND | wx.ALL, 4)

        # 3) optional transcript (before the Create button) -----------------
        ref_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        ref_grid.AddGrowableCol(1)
        self.ref_text_ctrl = wx.TextCtrl(self.clone_panel)
        self.ref_text_ctrl.SetName("Transcript of the sample (optional)")
        add_labeled(
            self.clone_panel, ref_grid,
            "Transcript of the sample (optional)", self.ref_text_ctrl,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )
        c_sizer.Add(ref_grid, 0, wx.EXPAND | wx.ALL, 4)

        # 4) create button (last) ------------------------------------------
        self.clone_create_btn = wx.Button(self.clone_panel, label="Create voice")
        self.clone_create_btn.SetName("Create clone voice")
        c_sizer.Add(self.clone_create_btn, 0, wx.ALL, 4)
        self.clone_status = wx.StaticText(self.clone_panel, label="")
        self.clone_status.SetName("Clone voice status")
        c_sizer.Add(self.clone_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        self.clone_panel.SetSizer(c_sizer)
        sizer.Add(self.clone_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)

        # -- design group (description -> name -> create) ------------------
        self.design_panel = wx.Panel(self)
        d_sizer = wx.BoxSizer(wx.VERTICAL)
        desc_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        desc_grid.AddGrowableCol(1)
        self.design_desc_ctrl = wx.TextCtrl(
            self.design_panel, style=wx.TE_MULTILINE, size=(-1, 70),
            value="female, young adult, british accent",
        )
        self.design_desc_ctrl.SetName("Voice description")
        add_labeled(self.design_panel, desc_grid, "Voice description", self.design_desc_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        d_sizer.Add(desc_grid, 0, wx.EXPAND | wx.ALL, 4)
        d_tip = wx.StaticText(
            self.design_panel,
            label="Example: female, young adult, british accent. Attributes: "
                  "male/female, child/teenager/young adult/middle-aged/elderly, "
                  "pitch levels, whisper, accents (american, british, indian, ...).",
        )
        d_tip.Wrap(640)
        d_sizer.Add(d_tip, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        dname_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        dname_grid.AddGrowableCol(1)
        self.design_name_ctrl = wx.TextCtrl(self.design_panel)
        self.design_name_ctrl.SetName("Voice name")
        add_labeled(self.design_panel, dname_grid, "Voice name", self.design_name_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.design_create_btn = wx.Button(self.design_panel, label="Create voice")
        self.design_create_btn.SetName("Create design voice")
        dname_grid.Add(self.design_create_btn, 0, wx.ALL, 2)
        d_sizer.Add(dname_grid, 0, wx.EXPAND | wx.ALL, 4)
        self.design_status = wx.StaticText(self.design_panel, label="")
        self.design_status.SetName("Design voice status")
        d_sizer.Add(self.design_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        self.design_panel.SetSizer(d_sizer)
        sizer.Add(self.design_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)

        # -- voice library list -------------------------------------------
        lib_box = wx.StaticBox(self, label="My OmniVoice voices (shared by both engines)")
        lib = wx.StaticBoxSizer(lib_box, wx.VERTICAL)
        self.voices_list = wx.ListBox(lib_box, size=(-1, 130),
                                      name="My OmniVoice voices")
        lib.Add(self.voices_list, 1, wx.EXPAND | wx.ALL, 4)

        action_row = wx.BoxSizer(wx.HORIZONTAL)
        self.rename_btn = wx.Button(lib_box, label="Rename...")
        self.rename_btn.SetName("Rename selected voice")
        self.delete_btn = wx.Button(lib_box, label="Delete")
        self.delete_btn.SetName("Delete selected voice")
        action_row.Add(self.rename_btn, 0, wx.ALL, 2)
        action_row.Add(self.delete_btn, 0, wx.ALL, 2)
        lib.Add(action_row, 0, wx.LEFT, 2)

        pv_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        pv_grid.AddGrowableCol(1)
        self.preview_text = wx.TextCtrl(lib_box, value=self._PREVIEW_DEFAULT)
        self.preview_text.SetName("Preview text")
        add_labeled(lib_box, pv_grid, "Preview text", self.preview_text,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.preview_btn = wx.Button(lib_box, label="&Preview selected voice")
        self.preview_btn.SetName("Preview voice library voice")
        pv_grid.Add(self.preview_btn, 0, wx.ALL, 2)
        # Which back-end the preview runs on.  OmniVoice itself needs CUDA, so
        # GPU is the default here; the CPU entry is only offered for the sake
        # of consistency (the engine still runs on the GPU).
        from .compute_choice import make_compute_row  # noqa: PLC0415

        compute_label, self.compute_combo = make_compute_row(
            lib_box, self.settings, "omnivoice_engines", default="cuda"
        )
        lib.Add(compute_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)
        lib.Add(self.compute_combo, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        lib.Add(pv_grid, 0, wx.EXPAND | wx.ALL, 4)
        self.lib_status = wx.StaticText(lib_box, label="")
        self.lib_status.SetName("Voice library status")
        lib.Add(self.lib_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(lib, 0, wx.EXPAND | wx.ALL, 6)

        # -- events ---------------------------------------------------------
        self.engine_combo.Bind(wx.EVT_COMBOBOX, self._on_engine_change)
        self.mode_combo.Bind(wx.EVT_COMBOBOX, self._on_mode_change)
        browse_btn.Bind(wx.EVT_BUTTON, self._on_browse)
        self.clone_create_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_create("clone"))
        self.design_create_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_create("design"))
        self.voices_list.Bind(wx.EVT_LISTBOX, self._on_library_select)
        self.rename_btn.Bind(wx.EVT_BUTTON, self._on_rename)
        self.delete_btn.Bind(wx.EVT_BUTTON, self._on_delete)
        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)

        self.design_panel.Hide()
        self._on_engine_change(None)
        self._on_mode_change(None)
        self.rename_btn.Disable()
        self.delete_btn.Disable()
        self.preview_btn.Disable()

    # ------------------------------------------------------------- helpers
    def _selected_engine_id(self):
        sel = self.engine_combo.GetSelection()
        return self.engine_combo.GetClientData(sel) if sel >= 0 else "omnivoice_server"

    def _selected_library_voice(self):
        sel = self.voices_list.GetSelection()
        if sel < 0 or sel >= len(self._library):
            return None
        return self._library[sel]

    def _refresh_library(self):
        from ..omnivoice import voice_store  # noqa: PLC0415

        old = self._selected_library_voice()
        self._library = voice_store.omni_custom_voices(self.store)
        self._lib_names = [
            f"{v['name']} — {'clone' if v.get('mode') == 'clone' else 'design'}"
            + (f" — {language_choice.label(v.get('omni_language'))}"
               if v.get("omni_language") else "")
            for v in self._library
        ]
        self.voices_list.Clear()
        for label in self._lib_names:
            self.voices_list.Append(label)
        if old:
            for index, voice in enumerate(self._library):
                if voice["name"] == old["name"]:
                    self.voices_list.SetSelection(index)
                    break
        self._on_library_select(None)
        count = len(self._library)
        self.lib_status.SetLabel(
            f"{count} voice{'s' if count != 1 else ''} in your library - shared by "
            "both OmniVoice engines."
            if count
            else "No voices yet. Create one above."
        )

    def _on_library_select(self, _evt):
        has = self._selected_library_voice() is not None
        for btn in (self.rename_btn, self.delete_btn, self.preview_btn):
            if has:
                btn.Enable()
            else:
                btn.Disable()

    def access_key_hint(self, entry) -> str:
        """Why Alt+P cannot act here (the shared access_keys fallback)."""
        if entry.control is self.preview_btn and self._selected_library_voice() is None:
            return ("Alt+P: select a voice in \"My OmniVoice voices\" first - "
                    "Preview is unavailable until a voice is selected.")
        return super().access_key_hint(entry)

    def _on_engine_change(self, _evt):
        from ..omnivoice import voice_store  # noqa: PLC0415

        engine_id = self._selected_engine_id()
        label, pkg = voice_store.ENGINE_INFO.get(engine_id, (engine_id, ""))
        if engine_id in self._installed:
            tip = f"Preview will use {label} ({pkg} is installed)."
        else:
            tip = f"{label} is not installed yet (pip package {pkg}). " \
                  "Created voices still work with the other engine; install " \
                  "this one from the Compute tab to preview with it."
        self.mode_tip.SetLabel(self._mode_tip_text() + "\n" + tip)
        self.mode_tip.Wrap(640)
        self.Layout()

    def _mode_tip_text(self):
        mode = self.mode_combo.GetClientData(self.mode_combo.GetSelection())
        if mode == "clone":
            return ("Clone: pick a 3-15 second recording of the voice to copy "
                    "(the optional transcript improves cloning accuracy).")
        return ("Design: describe the voice with attributes such as "
                "'female, young adult, british accent'.")

    def _on_mode_change(self, _evt):
        mode = self.mode_combo.GetClientData(self.mode_combo.GetSelection())
        self.clone_panel.Show(mode == "clone")
        self.design_panel.Show(mode != "clone")
        self.mode_tip.SetLabel(self._mode_tip_text())
        self.mode_tip.Wrap(640)
        self.Layout()

    def _on_browse(self, _evt):
        dlg = wx.FileDialog(
            self,
            "Choose a voice sample",
            wildcard="Audio files (*.wav;*.mp3;*.flac;*.ogg)|*.wav;*.mp3;*.flac;*.ogg|"
                     "All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.sample_ctrl.SetValue(dlg.GetPath())
        finally:
            dlg.Destroy()

    # -------------------------------------------------------------- actions
    def _on_create(self, mode: str):
        from ..omnivoice import voice_store  # noqa: PLC0415

        if mode == "clone":
            name = self.clone_name_ctrl.GetValue().strip()
            ref_audio = self.sample_ctrl.GetValue().strip()
            ref_text = self.ref_text_ctrl.GetValue().strip()
            instruct = ""
            status = self.clone_status
        else:
            name = self.design_name_ctrl.GetValue().strip()
            ref_audio = ""
            ref_text = ""
            instruct = self.design_desc_ctrl.GetValue().strip()
            status = self.design_status
        if not name:
            wx.MessageBox("Type a voice name first.", "Create voice",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        if mode == "clone" and (not ref_audio or not os.path.isfile(ref_audio)):
            wx.MessageBox(
                "Choose a voice sample file first (Browse...).",
                "Create voice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        try:
            entry = voice_store.create_voice(
                self.store, name=name, mode=mode,
                ref_audio=ref_audio, ref_text=ref_text, instruct=instruct,
                language=language_choice.hint_of(self.omni_language_combo),
            )
        except ValueError as exc:
            status.SetLabel("")
            wx.MessageBox(str(exc), "Create voice",
                          style=wx.OK | wx.ICON_ERROR)
            return
        # Success: clear the form, refresh and select the new voice.
        status.SetLabel(
            f"Voice '{entry['name']}' created - it now works with both "
            "OmniVoice engines (see the list below)."
        )
        if mode == "clone":
            self.clone_name_ctrl.SetValue("")
            self.sample_ctrl.SetValue("")
            self.ref_text_ctrl.SetValue("")
        else:
            self.design_name_ctrl.SetValue("")
        self._refresh_library()
        for index, voice in enumerate(self._library):
            if voice["name"] == entry["name"]:
                self.voices_list.SetSelection(index)
                break
        self._on_library_select(None)
        self.Layout()

    def _on_rename(self, _evt):
        from ..omnivoice import voice_store  # noqa: PLC0415

        voice = self._selected_library_voice()
        if not voice:
            return
        dlg = wx.TextEntryDialog(
            self, f"New name for '{voice['name']}':", "Rename voice",
            value=voice["name"],
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            new_name = dlg.GetValue().strip()
        finally:
            dlg.Destroy()
        if not new_name or new_name == voice["name"]:
            return
        try:
            renamed = voice_store.rename_voice(self.store, voice["name"], new_name)
        except ValueError as exc:
            wx.MessageBox(str(exc), "Rename voice", style=wx.OK | wx.ICON_ERROR)
            return
        self._refresh_library()
        for index, entry in enumerate(self._library):
            if entry["name"] == renamed["name"]:
                self.voices_list.SetSelection(index)
                break
        self._on_library_select(None)
        self.lib_status.SetLabel(f"Renamed to '{renamed['name']}'.")

    def _on_delete(self, _evt):
        from ..omnivoice import voice_store  # noqa: PLC0415

        voice = self._selected_library_voice()
        if not voice:
            return
        answer = wx.MessageBox(
            f"Delete voice '{voice['name']}'? Its sample audio will also be "
            "removed from the voice library.",
            "Delete voice", style=wx.YES_NO | wx.ICON_QUESTION,
        )
        if answer != wx.YES:
            return
        if voice_store.delete_voice(self.store, voice["name"]):
            self._refresh_library()
            self.lib_status.SetLabel(f"Deleted '{voice['name']}'.")

    # -------------------------------------------------------------- preview
    def _on_preview(self, evt=None):
        # One Alt+P press reaches this handler more than once (see
        # access_keys.once): the guard keeps it to one preview.
        if not access_keys.once(evt if evt is not None else self.preview_btn):
            return
        from ..omnivoice import voice_store  # noqa: PLC0415

        voice = self._selected_library_voice()
        if not voice:
            return
        engine_id = self._selected_engine_id()
        label, _pkg = voice_store.ENGINE_INFO.get(engine_id, (engine_id, ""))
        entry = voice_store.preview_entry(voice, engine_id)
        text = self.preview_text.GetValue().strip() or self._PREVIEW_DEFAULT
        self.preview_btn.Disable()
        self.preview_status_label = label
        self.lib_status.SetLabel(
            f"Synthesizing a preview with {label}... (first use loads the "
            "model and can take a minute or two.)"
        )
        # Snapshot the compute choice on the UI thread; the job runs on a worker.
        from .compute_choice import _combo_value, provider_for_preview  # noqa: PLC0415

        choice = _combo_value(self.compute_combo)
        provider = provider_for_preview(choice, entry.get("engine"))
        threading.Thread(
            target=self._preview_job, args=(entry, text, label, provider),
            daemon=True,
        ).start()

    def _preview_job(self, entry, text, engine_label, provider="cuda"):
        from ..audio.output import write_wav  # noqa: PLC0415
        from ..tts.engine import (  # noqa: PLC0415
            EngineUnavailableError,
            get_engine,
            process_punctuation,
        )

        try:
            engine = get_engine(entry, provider=provider)
            punct = self.settings.get("recording.punctuation", "default")
            samples = engine.synthesize(
                process_punctuation(text, punct), sid=0, speed=1.0
            )
            import tempfile  # noqa: PLC0415

            fd, tmp = tempfile.mkstemp(prefix="aivs_voice_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None, engine_label)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc), engine_label)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}", engine_label)

    def _preview_done(self, tmp, error, engine_label):
        import wx.adv  # noqa: PLC0415

        self.preview_btn.Enable()
        if error:
            self.lib_status.SetLabel(error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()
        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            # Keep a reference: wxSound must outlive Play(SOUND_ASYNC) or the
            # preview is cut off before it is heard (garbage collection bug).
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            self.lib_status.SetLabel(f"Playing preview ({engine_label}).")
        else:
            self.lib_status.SetLabel("Preview file could not be played.")

    def _stop_preview_sound(self):
        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:  # noqa: BLE001
                pass
            self._preview_sound = None

    # -- dependency (OmniVoice package) -------------------------------------
    def _build_dependency_hint(self, sizer):
        """Where to get OmniVoice when its dependency is missing.

        Neither the direct engine (``omnivoice-triton``) nor the server
        (``omnivoice-server``) can be downloaded from this category - the
        Compute category installs them - so the missing case gets one button
        that moves the user to Compute.
        """
        box = wx.StaticBox(self, label="OmniVoice dependency")
        inner = wx.StaticBoxSizer(box, wx.VERTICAL)
        self.dependency_status = wx.StaticText(box, label="Checking...")
        self.dependency_status.SetName("OmniVoice dependency status")
        self.dependency_status.Wrap(640)
        inner.Add(self.dependency_status, 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)

        self.goto_compute_btn = wx.Button(
            box, label="Download OmniVoice — move to Compute"
        )
        self.goto_compute_btn.SetName(
            "Download OmniVoice dependency from the Compute category"
        )
        self.goto_compute_btn.SetToolTip(
            "OmniVoice is not installed. The Compute category downloads and "
            "installs the omnivoice-triton and omnivoice-server packages."
        )
        inner.Add(self.goto_compute_btn, 0, wx.ALL, 4)
        sizer.Add(inner, 0, wx.EXPAND | wx.ALL, 6)

        self.goto_compute_btn.Bind(wx.EVT_BUTTON, self._on_goto_compute)
        self._refresh_dependency_hint()

    def _refresh_dependency_hint(self):
        """Show/hide the Compute shortcut from the cached package status."""
        from ..omnivoice import voice_store  # noqa: PLC0415

        ready: list[str] = []
        missing: list[str] = []
        for engine_id, (label, package) in voice_store.ENGINE_INFO.items():
            # Each OmniVoice engine is probed in its own virtualenv first, then
            # in the shared addon environment (where pre-existing installs
            # live), exactly like the Voice Lab engines.
            version = venv_packages.version(package, engine=engine_id)
            if version is None:
                version = venv_packages.version(package)
            if version:
                ready.append(f"{label} (v{version})")
            else:
                missing.append(package)
                if not venv_packages.is_known(package, engine=engine_id):
                    venv_packages.request(
                        package, self._on_dependency_probe, engine=engine_id
                    )
        if ready and not missing:
            self.dependency_status.SetLabel("Installed: " + ", ".join(ready) + ".")
            self.goto_compute_btn.Hide()
        else:
            detail = ", ".join(missing) if missing else "the OmniVoice package"
            self.dependency_status.SetLabel(
                f"OmniVoice TTS is not installed (needs {detail}). It cannot be "
                "downloaded from this category - use the Compute category to "
                "download it."
            )
            self.goto_compute_btn.Show()
        self.Layout()

    def _refresh_engine_status(self):
        """Update every engine card's status from the shared package cache."""
        for eng in self._engines:
            package = eng["package"]
            env_id = eng["env"]
            version = venv_packages.version(package, engine=env_id)
            if version is None:
                version = venv_packages.version(package)
            if version:
                eng["status_label"].SetLabel(
                    f"Installed (v{version}) — ready to use"
                )
                eng["installed"] = True
            else:
                eng["status_label"].SetLabel(
                    "Not installed — install from Compute tab"
                )
                eng["installed"] = False
                if not venv_packages.is_known(package, engine=env_id):
                    venv_packages.request(
                        package, self._on_dependency_probe, engine=env_id
                    )
        self._installed = [eng["id"] for eng in self._engines if eng["installed"]]

    def _refresh_dependency_state(self):
        """Refresh the dependency hint, the engine cards and the library."""
        self._refresh_dependency_hint()
        self._refresh_engine_status()
        self._refresh_library()

    def _on_dependency_probe(self, _version=None):
        """A background package probe finished (worker thread)."""
        try:
            wx.CallAfter(self._refresh_dependency_state)
        except Exception:  # noqa: BLE001
            pass

    def _on_goto_compute(self, _evt=None):
        """Move to the Compute category, where OmniVoice is installed from."""
        dialog = wx.GetTopLevelParent(self)
        show = getattr(dialog, "_show_category", None)
        categories = getattr(dialog, "CATEGORIES", None)
        if not show or not categories or _ComputePanel not in categories:
            wx.MessageBox(
                "Open Settings > Compute to download and install the "
                "OmniVoice dependency (omnivoice-triton or omnivoice-server).",
                "Download OmniVoice", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        index = categories.index(_ComputePanel)
        try:
            if dialog.cat_list.GetItemCount() > index:
                dialog.cat_list.Select(index)
                dialog.cat_list.Focus(index)
            show(index, focus_panel=True)
        except Exception:  # noqa: BLE001
            pass

    def _build_engine_cards(self, sizer):
        """Build status cards for each OmniVoice engine variant."""
        engines = [
            {
                "id": "omnivoice_server",
                "name": "OmniVoice Server",
                "package": "omnivoice-server",
                "env": "omnivoice_server",
                "description": "OpenAI-compatible HTTP API. Other apps on the network can use it.",
                "speed": "Fastest (persistent server, model stays hot)",
                "quality": "High (32 inference steps default)",
                "features": "HTTP API, streaming, concurrent requests, network access",
            },
            {
                "id": "omnivoice_triton",
                "name": "OmniVoice Triton",
                "package": "omnivoice-triton",
                "env": "omnivoice",
                "description": "Stable GPU mode via Triton kernel fusion.",
                "speed": "Good (~1.5x faster than hybrid)",
                "quality": "High",
                "features": "Per-request inference, lowest VRAM usage",
            },
            {
                "id": "omnivoice_hybrid",
                "name": "OmniVoice Hybrid",
                "package": "omnivoice-triton",
                "env": "omnivoice",
                "description": "Fast GPU mode combining ONNX and CUDA kernels.",
                "speed": "Fastest per-request (~3.4x faster)",
                "quality": "High",
                "features": "Fastest single-request, may leak VRAM",
            },
        ]

        for eng in engines:
            card_sizer = wx.StaticBoxSizer(
                wx.StaticBox(self, label=eng["name"]), wx.VERTICAL
            )
            card_box = card_sizer.GetStaticBox()

            # Status
            status_grid = wx.FlexGridSizer(cols=2, vgap=4, hgap=8)
            status_grid.AddGrowableCol(1)
            status_label = wx.StaticText(card_box, label="Checking...")
            status_label.SetName(f"{eng['name']} status")
            add_labeled(card_box, status_grid, "Status", status_label,
                        flag=wx.LEFT | wx.RIGHT, border=2)

            # Install status is filled in by _refresh_engine_status() from the
            # shared package cache, so building this panel never waits for a
            # managed-venv probe (those waits made Settings open slowly).
            card_sizer.Add(status_grid, 0, wx.EXPAND | wx.ALL, 4)

            # Features
            feat_items = [
                ("Description", eng["description"]),
                ("Speed", eng["speed"]),
                ("Quality", eng["quality"]),
                ("Features", eng["features"]),
            ]
            for label, value in feat_items:
                feat_sizer = wx.BoxSizer(wx.HORIZONTAL)
                feat_label = wx.StaticText(card_box, label=f"{label}: ")
                feat_label.SetFont(feat_label.GetFont().Bold())
                feat_val = wx.StaticText(card_box, label=value)
                feat_sizer.Add(feat_label, 0, wx.RIGHT, 4)
                feat_sizer.Add(feat_val, 1, wx.EXPAND)
                card_sizer.Add(feat_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

            sizer.Add(card_sizer, 0, wx.EXPAND | wx.ALL, 6)
            self._engines.append({
                "id": eng["id"],
                "package": eng["package"],
                "env": eng["env"],
                "status_label": status_label,
                "installed": False,
            })

    def on_activated(self):
        super().on_activated()
        # Status, library and engine cards all answer from the shared caches
        # (venv package probe), so opening this category never waits.
        self._refresh_dependency_state()
        self._on_engine_change(None)

    def apply_to_settings(self):
        pass  # No settings to save for this panel


# ---------------------------------------------------------------------------
# OmniVoice Server category (network TTS server)
# ---------------------------------------------------------------------------
class _OmniVoiceServerPanel(_SettingsPanel):
    title = "OmniVoice Server"

    """OmniVoice Server settings panel.

    Configures the omnivoice-server HTTP server:
    - Enable/disable the server
    - Host/port binding (localhost vs network)
    - API key authentication
    - CORS origins for browser frontends
    - Inference quality settings
    - Auto-start on app launch
    """

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self._thread: threading.Thread | None = None
        self._status_thread: threading.Thread | None = None
        self._status_probe_running = False
        self._status_epoch = 0

        sizer = wx.BoxSizer(wx.VERTICAL)

        # -- Enable / Auto-start ----------------------------------------
        ov_cfg = settings.get("omnivoice_server", {})
        self.enable_cb = wx.CheckBox(self, label="Enable OmniVoice Server")
        self.enable_cb.SetName("Enable server")
        self.enable_cb.SetValue(ov_cfg.get("enabled", False))
        self.enable_cb.SetToolTip(
            "Start the OmniVoice HTTP server when the app launches. "
            "Requires omnivoice-server to be installed (see Compute tab)."
        )
        sizer.Add(self.enable_cb, 0, wx.ALL, 4)

        self.auto_start_cb = wx.CheckBox(self, label="Auto-start on app launch")
        self.auto_start_cb.SetName("Auto-start server")
        self.auto_start_cb.SetValue(ov_cfg.get("auto_start", False))
        self.auto_start_cb.SetToolTip(
            "Automatically start the server when AI Voice Studio starts."
        )
        sizer.Add(self.auto_start_cb, 0, wx.ALL, 4)

        # -- Server status -----------------------------------------------
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 6)
        self.status_label = wx.StaticText(self, label="Server status: Not running")
        self.status_label.SetName("Server status")
        sizer.Add(self.status_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        status_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self, label="Start server")
        self.start_btn.SetName("Start server")
        self.start_btn.SetToolTip("Start the OmniVoice HTTP server now")
        self.stop_btn = wx.Button(self, label="Stop server")
        self.stop_btn.SetName("Stop server")
        self.stop_btn.SetToolTip("Stop the running OmniVoice HTTP server")
        self.stop_btn.Disable()
        self.test_btn = wx.Button(self, label="Test connection")
        self.test_btn.SetName("Test server connection")
        self.test_btn.SetToolTip("Send a test request to the server")
        status_btns.Add(self.start_btn, 0, wx.ALL, 4)
        status_btns.Add(self.stop_btn, 0, wx.ALL, 4)
        status_btns.Add(self.test_btn, 0, wx.ALL, 4)
        sizer.Add(status_btns, 0, wx.LEFT, 2)

        sizer.AddSpacer(4)

        # -- Network settings --------------------------------------------
        sizer.Add(
            wx.StaticText(self, label="Network settings"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)

        self.host_ctrl = wx.TextCtrl(self, size=(-1, 28))
        self.host_ctrl.SetName("Server host")
        self.host_ctrl.SetValue(ov_cfg.get("host", "127.0.0.1"))
        self.host_ctrl.SetToolTip(
            "Bind address. Use 127.0.0.1 for local only, "
            "0.0.0.0 to allow network access."
        )
        add_labeled(self, grid, "Host", self.host_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.port_ctrl = wx.SpinCtrl(self, min=1024, max=65535,
                                      name="Server port")
        self.port_ctrl.SetValue(ov_cfg.get("port", 8881))
        self.port_ctrl.SetToolTip(
            "Port number (1024-65535). Default: 8880."
        )
        add_labeled(self, grid, "Port", self.port_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.network_cb = wx.CheckBox(
            self, label="Allow access from other devices on the network"
        )
        self.network_cb.SetName("Allow network access")
        self.network_cb.SetValue(ov_cfg.get("allow_network", False))
        self.network_cb.SetToolTip(
            "When checked, sets host to 0.0.0.0 so other devices "
            "on the same network can use the server. "
            "Uncheck for localhost-only."
        )
        grid.Add((1, 1))
        grid.Add(self.network_cb, 0, wx.ALL, 2)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        # -- Authentication ----------------------------------------------
        sizer.Add(
            wx.StaticText(self, label="Security"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        sec_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        sec_grid.AddGrowableCol(1)

        self.api_key_ctrl = wx.TextCtrl(self, size=(-1, 28))
        self.api_key_ctrl.SetName("API key")
        self.api_key_ctrl.SetValue(ov_cfg.get("api_key", ""))
        self.api_key_ctrl.SetToolTip(
            "Bearer token for API authentication. Leave empty for no auth."
        )
        add_labeled(self, sec_grid, "API key", self.api_key_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.cors_ctrl = wx.TextCtrl(self, size=(-1, 28))
        self.cors_ctrl.SetName("CORS origins")
        self.cors_ctrl.SetValue(ov_cfg.get("cors_origins", ""))
        self.cors_ctrl.SetToolTip(
            "Comma-separated allowed browser origins for CORS. "
            "Example: http://localhost:5173,http://127.0.0.1:5173"
        )
        add_labeled(self, sec_grid, "CORS origins", self.cors_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        sizer.Add(sec_grid, 0, wx.EXPAND | wx.ALL, 6)

        # -- Quality settings --------------------------------------------
        sizer.Add(
            wx.StaticText(self, label="Quality settings"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        q_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        q_grid.AddGrowableCol(1)

        self.steps_ctrl = wx.SpinCtrl(self, min=1, max=64,
                                       name="Inference steps")
        self.steps_ctrl.SetValue(ov_cfg.get("num_steps", 32))
        self.steps_ctrl.SetToolTip(
            "Inference steps (1-64). Higher = better quality but slower. "
            "Default: 32. Use 16 for faster output."
        )
        add_labeled(self, q_grid, "Inference steps", self.steps_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.concurrent_ctrl = wx.SpinCtrl(self, min=1, max=16,
                                            name="Max concurrent requests")
        self.concurrent_ctrl.SetValue(ov_cfg.get("max_concurrent", 2))
        self.concurrent_ctrl.SetToolTip(
            "Maximum concurrent synthesis requests. Default: 2. "
            "Increase for batch processing."
        )
        add_labeled(self, q_grid, "Max concurrent", self.concurrent_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.device_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Compute device")
        self.device_combo.Append("CUDA (GPU)", "cuda")
        self.device_combo.Append("CPU", "cpu")
        device_val = ov_cfg.get("device", "cuda")
        idx = 0 if device_val == "cuda" else 1
        self.device_combo.SetSelection(idx)
        self.device_combo.SetToolTip(
            "Compute device. CUDA requires NVIDIA GPU."
        )
        add_labeled(self, q_grid, "Device", self.device_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        sizer.Add(q_grid, 0, wx.EXPAND | wx.ALL, 6)

        # -- Voice profiles (server-stored clones) -------------------------
        self._build_profiles_ui(sizer)

        # -- Info --------------------------------------------------------
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="Server API: http://host:port/v1/audio/speech\n"
                      "Voice list: http://host:port/v1/voices\n"
                      "Health check: http://host:port/health\n"
                      "Metrics: http://host:port/metrics",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        self.SetSizer(sizer)

        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start)
        self.stop_btn.Bind(wx.EVT_BUTTON, self._on_stop)
        self.test_btn.Bind(wx.EVT_BUTTON, self._on_test)
        self.network_cb.Bind(wx.EVT_CHECKBOX, self._on_network_toggle)

    def on_activated(self):
        super().on_activated()
        self._refresh_status()

    def _on_network_toggle(self, _):
        """When network access is checked, suggest 0.0.0.0 as host."""
        if self.network_cb.GetValue():
            self.host_ctrl.SetValue("0.0.0.0")
        else:
            self.host_ctrl.SetValue("127.0.0.1")

    def _refresh_status(self):
        """Check if the server is running and update the status label and the
        voice-profile controls.

        The check is a network round trip (an unreachable host can take
        seconds to time out), so it runs on a worker thread: opening this
        category stays instant and the labels are filled in when the answer
        arrives.  A check already in flight is not restarted.
        """
        if self._status_probe_running:
            return
        host = self.host_ctrl.GetValue().strip() or "127.0.0.1"
        port = self.port_ctrl.GetValue()
        self.status_label.SetLabel("Server status: Checking...")
        self._status_probe_running = True
        self._status_epoch += 1
        self._status_thread = threading.Thread(
            target=self._probe_status,
            args=(host, port, self._status_epoch),
            daemon=True,
        )
        self._status_thread.start()

    def _probe_status(self, host: str, port: int, epoch: int):
        """Worker thread: is a server reachable on host:port?"""
        running = False
        try:
            from ..omnivoice_server import get_server_manager  # noqa: PLC0415
            mgr = get_server_manager(host=host, port=port)
            # A subprocess we started ourselves answers without a round
            # trip; anything else gets one short health probe.
            running = bool(mgr.process_alive) or mgr.health_check(
                retries=1, delay=0.0, timeout=2.0
            )
        except Exception:  # noqa: BLE001
            running = False
        try:
            wx.CallAfter(self._status_ready, running, host, port, epoch)
        except Exception:  # noqa: BLE001
            pass

    def _status_ready(self, running: bool, host: str, port: int, epoch: int):
        """Main thread: apply the result of a background status probe."""
        if epoch != self._status_epoch:
            # A start/stop happened while this probe was in flight: its
            # answer is about the old state, so drop it.
            return
        try:
            self._status_probe_running = False
            if running:
                self.status_label.SetLabel(
                    f"Server status: Running at http://{host}:{port}"
                )
                self.start_btn.Disable()
                self.stop_btn.Enable()
            else:
                self.status_label.SetLabel("Server status: Not running")
                self.start_btn.Enable()
                self.stop_btn.Disable()
            self._set_profiles_enabled(running)
            if running:
                self._on_refresh_profiles(None)
            self.Layout()
        except Exception:  # noqa: BLE001
            # The dialog (and this panel) may already be gone; nothing to do.
            pass

    def _on_start(self, _):
        """Start the OmniVoice server in a background thread."""
        self.start_btn.Disable()
        self.stop_btn.Disable()
        self.status_label.SetLabel("Starting server...")
        self._thread = threading.Thread(target=self._start_job, daemon=True)
        self._thread.start()

    def _start_job(self):
        from ..omnivoice_server import get_server_manager  # noqa: PLC0415
        host = self.host_ctrl.GetValue().strip() or "127.0.0.1"
        port = self.port_ctrl.GetValue()
        device_sel = self.device_combo.GetSelection()
        device = self.device_combo.GetClientData(device_sel) if device_sel >= 0 else "cuda"
        try:
            mgr = get_server_manager(
                host=host, port=port, device=device,
                num_steps=self.steps_ctrl.GetValue(),
                max_concurrent=self.concurrent_ctrl.GetValue(),
                api_key=self.api_key_ctrl.GetValue().strip(),
                cors_origins=self.cors_ctrl.GetValue().strip(),
            )
            mgr.start(timeout=180.0)
            wx.CallAfter(self._start_done, True, f"Server started at http://{host}:{port}")
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._start_done, False, f"Failed to start server: {exc}")

    def _start_done(self, success, message):
        self._thread = None
        # Starting/stopping changes the truth the status probe was asking
        # about, so any probe still in flight is discarded.
        self._status_epoch += 1
        self._status_probe_running = False
        self.status_label.SetLabel(f"Server status: {message}")
        self.start_btn.Enable(not success)
        self.stop_btn.Enable(success)
        if success:
            self._set_profiles_enabled(True)
            self._on_refresh_profiles(None)
        else:
            self._set_profiles_enabled(False)
            wx.MessageBox(message, "OmniVoice Server",
                          style=wx.OK | wx.ICON_ERROR)

    def _on_stop(self, _):
        """Stop the OmniVoice server."""
        try:
            from ..omnivoice_server import get_server_manager  # noqa: PLC0415
            mgr = get_server_manager()
            mgr.stop()
            self._status_epoch += 1
            self._status_probe_running = False
            self.status_label.SetLabel("Server status: Stopped")
            self.start_btn.Enable()
            self.stop_btn.Disable()
            self._set_profiles_enabled(False)
        except Exception as exc:  # noqa: BLE001
            self.status_label.SetLabel(f"Server status: Error stopping: {exc}")

    def _on_test(self, _):
        """Test the server connection."""
        try:
            from ..omnivoice_server import get_server_manager  # noqa: PLC0415
            host = self.host_ctrl.GetValue().strip() or "127.0.0.1"
            port = self.port_ctrl.GetValue()
            mgr = get_server_manager(host=host, port=port)
            if mgr.health_check():
                voices = mgr.get_voices()
                wx.MessageBox(
                    f"Server is running at http://{host}:{port}\n\n"
                    f"Available voices: {len(voices)}\n"
                    f"API: POST http://{host}:{port}/v1/audio/speech\n"
                    f"Health: GET http://{host}:{port}/health",
                    "Server test successful",
                    style=wx.OK | wx.ICON_INFORMATION,
                )
            else:
                wx.MessageBox(
                    f"Server at http://{host}:{port} is not responding.\n"
                    "Make sure the server is started.",
                    "Server test failed",
                    style=wx.OK | wx.ICON_ERROR,
                )
        except Exception as exc:  # noqa: BLE001
            wx.MessageBox(
                f"Connection test failed: {exc}",
                "Server test failed",
                style=wx.OK | wx.ICON_ERROR,
            )

    # -- Voice profiles (server-stored clones) ------------------------------

    def _build_profiles_ui(self, sizer):
        """Build the "Voice profiles" section (list / create / delete)."""
        self._profiles: list = []
        box = wx.StaticBoxSizer(
            wx.StaticBox(self, label="Voice profiles (clones stored on the server)"),
            wx.VERTICAL,
        )
        box.Add(
            wx.StaticText(
                box.GetStaticBox(),
                label="Profiles save a cloned voice (reference audio) on the server so any "
                      "app on the network can reuse it by profile id. For recordings in this "
                      "studio you can also clone per project from the Recording window's "
                      "OmniVoice voice options.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )

        # -- create row ----------------------------------------------------
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)

        self.profile_id_ctrl = wx.TextCtrl(box.GetStaticBox())
        self.profile_id_ctrl.SetName("Profile id")
        self.profile_id_ctrl.SetToolTip(
            "Unique profile id (letters, digits, dashes, underscores), e.g. "
            "my_narrator."
        )
        add_labeled(box.GetStaticBox(), grid, "Profile id", self.profile_id_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)

        self.profile_audio_ctrl = wx.TextCtrl(box.GetStaticBox())
        self.profile_audio_ctrl.SetName("Reference audio")
        browse_btn = wx.Button(box.GetStaticBox(), label="Browse...")
        browse_btn.SetName("Browse reference audio for profile")
        browse_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_profile_browse())
        audio_box = wx.BoxSizer(wx.HORIZONTAL)
        audio_box.Add(self.profile_audio_ctrl, 1, wx.EXPAND)
        audio_box.Add(browse_btn, 0, wx.LEFT, 4)
        grid.Add(
            wx.StaticText(box.GetStaticBox(), label="Reference audio:"),
            0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2,
        )
        grid.Add(audio_box, 1, wx.EXPAND)

        self.profile_ref_text_ctrl = wx.TextCtrl(box.GetStaticBox())
        self.profile_ref_text_ctrl.SetName("Reference text")
        self.profile_ref_text_ctrl.SetToolTip(
            "Transcript of the sample (optional; empty lets the engine "
            "auto-transcribe)."
        )
        add_labeled(box.GetStaticBox(), grid, "Reference text",
                    self.profile_ref_text_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        self.profile_overwrite_cb = wx.CheckBox(
            box.GetStaticBox(), label="Overwrite if the id already exists"
        )
        self.profile_overwrite_cb.SetName("Overwrite existing profile")
        grid.Add((1, 1))
        grid.Add(self.profile_overwrite_cb, 0, wx.ALL, 2)

        box.Add(grid, 0, wx.EXPAND | wx.ALL, 4)
        self.add_profile_btn = wx.Button(box.GetStaticBox(), label="Add profile")
        self.add_profile_btn.SetName("Add voice profile")
        self.add_profile_btn.SetToolTip(
            "Save the reference audio as a named profile on the server."
        )
        box.Add(self.add_profile_btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # -- list row -------------------------------------------------------
        self.profile_list = wx.ListCtrl(
            box.GetStaticBox(),
            style=wx.LC_REPORT | wx.LC_SINGLE_SEL,
            size=(-1, 120),
        )
        self.profile_list.SetName("Voice profiles on server")
        self.profile_list.InsertColumn(0, "Profile id", width=200)
        self.profile_list.InsertColumn(1, "Description", width=320)
        box.Add(self.profile_list, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.refresh_profiles_btn = wx.Button(box.GetStaticBox(), label="Refresh list")
        self.refresh_profiles_btn.SetName("Refresh voice profiles")
        self.delete_profile_btn = wx.Button(box.GetStaticBox(), label="Delete selected")
        self.delete_profile_btn.SetName("Delete selected voice profile")
        row.Add(self.refresh_profiles_btn, 0, wx.ALL, 4)
        row.Add(self.delete_profile_btn, 0, wx.ALL, 4)
        self.profiles_status = wx.StaticText(box.GetStaticBox(), label="")
        self.profiles_status.SetName("Voice profiles status")
        row.Add(self.profiles_status, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        box.Add(row, 0, wx.EXPAND | wx.LEFT, 2)

        sizer.Add(box, 0, wx.EXPAND | wx.ALL, 6)

        self.add_profile_btn.Bind(wx.EVT_BUTTON, self._on_add_profile)
        self.refresh_profiles_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_refresh_profiles(None))
        self.delete_profile_btn.Bind(wx.EVT_BUTTON, self._on_delete_profile)
        self.profile_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_profile_selected)
        self.profile_list.Bind(wx.EVT_LIST_ITEM_DESELECTED, self._on_profile_selected)

    def _server_manager(self):
        """Manager for the currently configured host/port (as other methods
        on this panel resolve it)."""
        from ..omnivoice_server import get_server_manager  # noqa: PLC0415
        host = self.host_ctrl.GetValue().strip() or "127.0.0.1"
        port = self.port_ctrl.GetValue()
        return get_server_manager(host=host, port=port)

    def _selected_profile_id(self):
        sel = self.profile_list.GetFirstSelected()
        if sel < 0 or sel >= len(self._profiles):
            return None
        return self._profiles[sel].get("profile_id")

    def _set_profiles_enabled(self, running: bool):
        """Enable/disable the profile controls depending on the server."""
        self.add_profile_btn.Enable(running)
        self.refresh_profiles_btn.Enable(running)
        self.delete_profile_btn.Enable(
            running and self._selected_profile_id() is not None
        )
        if not running:
            self.profile_list.DeleteAllItems()
            self._profiles = []
            self.profiles_status.SetLabel(
                "Start the server to manage voice profiles."
            )

    def _on_profile_selected(self, _):
        self.delete_profile_btn.Enable(
            self._selected_profile_id() is not None
        )

    def _on_profile_browse(self):
        with wx.FileDialog(
            self,
            "Choose the reference voice sample (3-15 seconds)",
            wildcard="Audio files (*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac)|"
                     "*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.profile_audio_ctrl.SetValue(dlg.GetPath())

    def _on_refresh_profiles(self, _):
        """Fetch the profile list from the server (background thread)."""
        self.refresh_profiles_btn.Disable()
        self.profiles_status.SetLabel("Refreshing profiles...")
        threading.Thread(target=self._refresh_profiles_job, daemon=True).start()

    def _refresh_profiles_job(self):
        try:
            mgr = self._server_manager()
            entries = mgr.list_profiles()
            wx.CallAfter(self._profiles_loaded, entries, None)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._profiles_loaded, None, str(exc))

    def _profiles_loaded(self, entries, error):
        self.refresh_profiles_btn.Enable(True)
        self._profiles = [e for e in (entries or []) if e.get("profile_id")]
        self.profile_list.DeleteAllItems()
        for i, entry in enumerate(self._profiles):
            self.profile_list.InsertItem(i, entry.get("profile_id", ""))
            self.profile_list.SetItem(i, 1, entry.get("description") or "")
        if error:
            self.profiles_status.SetLabel("Could not load profiles.")
            wx.MessageBox(
                f"Could not load voice profiles: {error}",
                "OmniVoice profiles", style=wx.OK | wx.ICON_ERROR,
            )
        else:
            self.profiles_status.SetLabel(
                f"{len(self._profiles)} profile(s) stored on the server."
            )
        self.delete_profile_btn.Enable(
            self._selected_profile_id() is not None
        )

    def _on_add_profile(self, _):
        """Save the reference audio as a named profile on the server."""
        import re  # noqa: PLC0415

        profile_id = self.profile_id_ctrl.GetValue().strip()
        if not profile_id:
            wx.MessageBox("Enter a profile id first.", "Add voice profile",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]+", profile_id):
            wx.MessageBox(
                "Profile ids may contain letters, digits, dashes and "
                "underscores only.",
                "Add voice profile", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        ref_audio = self.profile_audio_ctrl.GetValue().strip()
        if not ref_audio or not os.path.isfile(ref_audio):
            wx.MessageBox(
                "Choose a reference audio file for the profile first.",
                "Add voice profile", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        self.add_profile_btn.Disable()
        self.profiles_status.SetLabel(f"Saving profile '{profile_id}'...")
        threading.Thread(
            target=self._add_profile_job,
            args=(profile_id, ref_audio,
                  self.profile_ref_text_ctrl.GetValue().strip(),
                  self.profile_overwrite_cb.GetValue()),
            daemon=True,
        ).start()

    def _add_profile_job(self, profile_id, ref_audio, ref_text, overwrite):
        try:
            mgr = self._server_manager()
            mgr.save_profile(
                profile_id=profile_id,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                overwrite=bool(overwrite),
            )
            wx.CallAfter(self._profile_saved, profile_id, None)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._profile_saved, profile_id, str(exc))

    def _profile_saved(self, profile_id, error):
        self.add_profile_btn.Enable(True)
        if error:
            self.profiles_status.SetLabel("Could not save the profile.")
            wx.MessageBox(
                f"Could not save profile '{profile_id}': {error}",
                "OmniVoice profiles", style=wx.OK | wx.ICON_ERROR,
            )
            return
        self.profile_id_ctrl.SetValue("")
        self.profile_audio_ctrl.SetValue("")
        self.profile_ref_text_ctrl.SetValue("")
        self.profiles_status.SetLabel(f"Profile '{profile_id}' saved.")
        self._on_refresh_profiles(None)

    def _on_delete_profile(self, _):
        """Delete the selected profile from the server."""
        profile_id = self._selected_profile_id()
        if not profile_id:
            return
        if wx.MessageBox(
            f"Delete the voice profile '{profile_id}' from the server?",
            "Delete voice profile",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        self.delete_profile_btn.Disable()
        self.profiles_status.SetLabel(f"Deleting profile '{profile_id}'...")
        threading.Thread(
            target=self._delete_profile_job, args=(profile_id,), daemon=True
        ).start()

    def _delete_profile_job(self, profile_id):
        try:
            mgr = self._server_manager()
            ok = mgr.delete_profile(profile_id)
            wx.CallAfter(self._profile_deleted, profile_id, None if ok else "not found")
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._profile_deleted, profile_id, str(exc))

    def _profile_deleted(self, profile_id, error):
        if error:
            self.profiles_status.SetLabel("Could not delete the profile.")
            wx.MessageBox(
                f"Could not delete profile '{profile_id}': {error}",
                "OmniVoice profiles", style=wx.OK | wx.ICON_ERROR,
            )
            return
        self.profiles_status.SetLabel(f"Profile '{profile_id}' deleted.")
        self._on_refresh_profiles(None)

    def apply_to_settings(self):
        """Save server settings."""
        self.settings.set("omnivoice_server.enabled", self.enable_cb.GetValue())
        self.settings.set("omnivoice_server.auto_start", self.auto_start_cb.GetValue())
        self.settings.set("omnivoice_server.host", self.host_ctrl.GetValue().strip() or "127.0.0.1")
        self.settings.set("omnivoice_server.port", self.port_ctrl.GetValue())
        device_sel = self.device_combo.GetSelection()
        device = self.device_combo.GetClientData(device_sel) if device_sel >= 0 else "cuda"
        self.settings.set("omnivoice_server.device", device)
        self.settings.set("omnivoice_server.num_steps", self.steps_ctrl.GetValue())
        self.settings.set("omnivoice_server.max_concurrent", self.concurrent_ctrl.GetValue())
        self.settings.set("omnivoice_server.api_key", self.api_key_ctrl.GetValue().strip())
        self.settings.set("omnivoice_server.cors_origins", self.cors_ctrl.GetValue().strip())
        self.settings.set("omnivoice_server.allow_network", self.network_cb.GetValue())


# ---------------------------------------------------------------------------
# Compute category (optional runtimes / GPU dependency)
# ---------------------------------------------------------------------------
class _ComputePanel(_SettingsPanel):
    title = "Compute"

    """Manages the optional GPU (CUDA) runtime download.

    The CPU ONNX Runtime ships with the application; the GPU back-end needs
    the ~1.8 GB CUDA runtime, downloaded here into the user folder. It is
    activated on the next application start, so installing or removing it
    asks for a restart.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="CPU is always available. GPU (CUDA) requires "
                                      f"downloading the optional runtime (~{runtime.DOWNLOAD_SIZE_MB / 1024:.0f} GB)."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.gpu_status = wx.StaticText(self, label="")
        self.gpu_status.SetName("GPU runtime status")
        add_labeled(self, grid, "GPU (CUDA) runtime", self.gpu_status,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        self.gauge = wx.Gauge(self, range=100, size=(-1, 22))
        self.gauge.SetName("GPU runtime download progress")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 6)
        self.progress_label = wx.StaticText(self, label="No download in progress.")
        sizer.Add(self.progress_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.gpu_download_btn = wx.Button(self, label="Download GPU runtime")
        self.gpu_remove_btn = wx.Button(self, label="Remove GPU runtime")
        self.gpu_cancel_btn = wx.Button(self, label="Cancel download")
        self.gpu_download_btn.SetName("Download GPU runtime")
        self.gpu_download_btn.SetToolTip("Download the optional CUDA GPU runtime (~1.8 GB) for GPU-accelerated synthesis")
        self.gpu_remove_btn.SetName("Remove GPU runtime")
        self.gpu_remove_btn.SetToolTip("Delete the GPU runtime files and return to CPU-only mode")
        self.gpu_cancel_btn.SetName("Cancel download")
        self.gpu_cancel_btn.SetToolTip("Cancel the GPU runtime download")
        self.gpu_cancel_btn.Disable()
        btns.Add(self.gpu_download_btn, 0, wx.ALL, 4)
        btns.Add(self.gpu_remove_btn, 0, wx.ALL, 4)
        btns.Add(self.gpu_cancel_btn, 0, wx.ALL, 4)
        sizer.Add(btns, 0, wx.LEFT, 2)

        sizer.AddSpacer(6)
        sizer.Add(
            wx.StaticText(self, label="Restart the application after installing or removing "
                                      "the GPU runtime."),
            0, wx.ALL, 6,
        )

        # -- OmniVoice TTS (omnivoice-triton) ----------------------------
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(
            wx.StaticText(self, label="OmniVoice TTS (600+ languages, voice cloning)"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        sizer.Add(
            wx.StaticText(self, label="Optional GPU-accelerated TTS engine. Requires NVIDIA GPU. "
                                      "Package: omnivoice-triton (~2-3 GB)."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        ov_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        ov_grid.AddGrowableCol(1)
        self.omnivoice_status = wx.StaticText(self, label="Checking...")
        self.omnivoice_status.SetName("OmniVoice dependency status")
        add_labeled(self, ov_grid, "OmniVoice dependency", self.omnivoice_status,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(ov_grid, 0, wx.EXPAND | wx.ALL, 6)

        ov_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.omnivoice_install_btn = wx.Button(self, label="Install OmniVoice dependency")
        self.omnivoice_install_btn.SetName("Install OmniVoice dependency")
        self.omnivoice_install_btn.SetToolTip(
            "Opens a command window to install omnivoice-triton (pip install) "
            "into the OmniVoice environment (shared with OmniVoice Server). "
            "Requires NVIDIA GPU with CUDA."
        )
        self.omnivoice_remove_btn = wx.Button(self, label="Remove OmniVoice dependency")
        self.omnivoice_remove_btn.SetName("Remove OmniVoice dependency")
        self.omnivoice_remove_btn.SetToolTip(
            "Opens a command window to uninstall omnivoice-triton (pip uninstall)."
        )
        ov_btns.Add(self.omnivoice_install_btn, 0, wx.ALL, 4)
        ov_btns.Add(self.omnivoice_remove_btn, 0, wx.ALL, 4)
        sizer.Add(ov_btns, 0, wx.LEFT, 2)

        sizer.AddSpacer(6)

        self.SetSizer(sizer)

        self.gpu_download_btn.Bind(wx.EVT_BUTTON, self._on_download)
        self.gpu_remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.gpu_cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.omnivoice_install_btn.Bind(wx.EVT_BUTTON, self._on_omnivoice_install)
        self.omnivoice_remove_btn.Bind(wx.EVT_BUTTON, self._on_omnivoice_remove)
        self.Bind(EVT_DOWNLOAD_PROGRESS, self._on_progress)
        self.Bind(EVT_DOWNLOAD_FINISHED, self._on_finished)

        # OmniVoice Server section
        self._init_omnivoice_server_ui()

        self._refresh()

    def on_activated(self):
        super().on_activated()
        self._refresh_omnivoice()
        self._refresh_omnivoice_server()

    @staticmethod
    def _omnivoice_torch_ready() -> bool:
        """True when the shared OmniVoice environment already has PyTorch.

        The direct OmniVoice engine and the HTTP server install into the *same*
        environment (same model, same CUDA PyTorch), so whichever is installed
        second finds PyTorch already there.
        """
        try:
            return bool(venv_packages.installed("torch", engine="omnivoice"))
        except Exception:  # noqa: BLE001
            return False

    def _refresh_omnivoice(self):
        """Show the omnivoice-triton status (its own venv, then the shared one)."""
        package = "omnivoice-triton"
        env_id = "omnivoice"
        version = venv_packages.version(package, engine=env_id)
        if version is None:
            version = venv_packages.version(package)
        if version:
            self.omnivoice_status.SetLabel(
                f"Installed (v{version}). OmniVoice TTS is available."
            )
            self.omnivoice_install_btn.Disable()
            self.omnivoice_remove_btn.Enable()
        else:
            self.omnivoice_status.SetLabel(
                "Not installed. OmniVoice TTS is unavailable."
            )
            self.omnivoice_install_btn.Enable()
            self.omnivoice_remove_btn.Disable()
            if not venv_packages.is_known(package, engine=env_id):
                venv_packages.request(
                    package, lambda _v: wx.CallAfter(self._refresh_omnivoice),
                    engine=env_id,
                )

    def _on_omnivoice_install(self, _):
        """Install omnivoice-triton via pip with a progress dialog."""
        self.omnivoice_install_btn.Disable()
        self.omnivoice_remove_btn.Disable()

        # Create a simple progress dialog (like a copy dialog, no cancel)
        self._ov_install_dlg = wx.Dialog(
            self, title="Installing OmniVoice",
            style=wx.DEFAULT_DIALOG_STYLE,
            size=(420, 120),
        )
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        self._ov_install_label = wx.StaticText(
            self._ov_install_dlg,
            label="Installing omnivoice-triton package...",
        )
        dlg_sizer.Add(self._ov_install_label, 0, wx.ALL | wx.EXPAND, 10)
        self._ov_install_gauge = wx.Gauge(
            self._ov_install_dlg, range=0, size=(-1, 24),
            style=wx.GA_HORIZONTAL | wx.GA_SMOOTH,
        )
        dlg_sizer.Add(self._ov_install_gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self._ov_install_dlg.SetSizer(dlg_sizer)
        self._ov_install_dlg.Centre()
        self._ov_install_dlg.Show()

        self._ov_install_thread = threading.Thread(
            target=self._omnivoice_install_job, daemon=True,
        )
        self._ov_install_thread.start()

    @staticmethod
    def _pip_install_raw(pip_exe, args, env_extra=None):
        """Run pip directly (bypasses PythonRuntime cleaning).

        Returns ``{"ok": bool, "output": str, "error": str}``.
        """
        import subprocess as _sp  # noqa: PLC0415
        import tempfile as _tmp  # noqa: PLC0415
        cmd = [pip_exe, "install", "--no-warn-script-location"] + args
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        flags = 0
        if sys.platform == "win32":
            flags = getattr(_sp, "CREATE_NO_WINDOW", 0)
        try:
            proc = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                text=True, bufsize=1, cwd=_tmp.gettempdir(),
                creationflags=flags, env=env,
            )
        except OSError as exc:
            return {"ok": False, "output": "", "error": str(exc)}
        lines = []
        for line in proc.stdout:
            lines.append(line.rstrip())
        proc.wait()
        combined = "\n".join(lines)
        if proc.returncode != 0:
            return {"ok": False, "output": combined,
                    "error": f"pip install failed (exit {proc.returncode})"}
        return {"ok": True, "output": combined, "error": ""}

    def _omnivoice_install_job(self):
        """Background thread: install OmniVoice into the shared OmniVoice venv.

        On Windows the install order is:
        1. PyTorch with CUDA (from PyTorch index — ``--index-url``)
        2. ``triton-windows`` (Windows-compatible Triton fork)
        3. ``omnivoice`` (base package)
        4. ``sageattention``
        5. ``omnivoice-triton --no-deps`` (kernel fusion, skips broken
           ``triton`` dep since ``triton-windows`` provides it)

        Step 1 uses a raw pip call (not ``PythonRuntime.pip_install``)
        because we need ``--index-url`` to replace PyPI with the CUDA
        wheel index, and the runtime's arg-cleaning logic strips flags.
        """
        from ..python_runtime import get_runtime  # noqa: PLC0415
        from ..voicelab import engines as voice_lab_engines  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415
        # OmniVoice's *own* environment, so its CUDA PyTorch never shares
        # site-packages with another TTS engine.
        rt = get_runtime("omnivoice")
        try:
            def _progress(msg, _done, _total):
                wx.CallAfter(self._ov_update_label, msg)

            if _sys.platform == "win32":
                # Ensure the managed venv exists (creates it + pip
                # on first run — must happen before any pip calls).
                _progress("Preparing Python environment...", 0, 0)
                rt.ensure_pip()

                # -- Step 1: PyTorch with CUDA via PyTorch's own index ----
                # Both OmniVoice engines share one environment, so a second
                # install (Server first, say) must not fetch the CUDA wheels
                # all over again.
                if self._omnivoice_torch_ready():
                    _progress("Step 1/5: PyTorch with CUDA is already installed.",
                              0, 0)
                else:
                    _progress(
                        "Step 1/5: Installing PyTorch with CUDA (~2 GB)...",
                        0, 0,
                    )
                    r = self._pip_install_raw(
                        rt.pip_exe,
                        ["torch", "torchaudio",
                         "--index-url",
                         voice_lab_engines.PYTORCH_CUDA_INDEX],
                    )
                    if not r["ok"]:
                        wx.CallAfter(self._omnivoice_install_done, False,
                                     f"Step 1 failed: {r.get('error', '')}")
                        return

                # -- Steps 2-5: remaining packages via PyPI -----------------
                remaining = [
                    (
                        "Step 2/5: Installing triton-windows (GPU compiler)...",
                        ["triton-windows"],
                    ),
                    (
                        "Step 3/5: Installing omnivoice (base package)...",
                        ["omnivoice"],
                    ),
                    (
                        "Step 4/5: Installing sageattention...",
                        ["sageattention"],
                    ),
                    (
                        "Step 5/5: Installing omnivoice-triton (GPU kernels)...",
                        ["omnivoice-triton", "--no-deps"],
                    ),
                ]
                for label, pkgs in remaining:
                    _progress(label, 0, 0)
                    res = rt.pip_install(pkgs, progress=_progress)
                    if not res["ok"]:
                        wx.CallAfter(
                            self._omnivoice_install_done, False,
                            f"{label}\n{res.get('error', 'Install failed.')}",
                        )
                        return
            else:
                # Linux / macOS: plain install (triton works natively)
                _progress("Installing omnivoice-triton...", 0, 0)
                result = rt.pip_install(
                    "omnivoice-triton", progress=_progress,
                )
                if not result["ok"]:
                    wx.CallAfter(
                        self._omnivoice_install_done, False,
                        result.get("error") or "Installation failed.",
                    )
                    return

            wx.CallAfter(self._omnivoice_install_done, True,
                         "OmniVoice installed successfully.")
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._omnivoice_install_done, False, f"Installation failed: {exc}")

    def _ov_update_label(self, msg: str):
        """Update the progress dialog label from a background thread."""
        try:
            if self._ov_install_dlg and self._ov_install_label:
                self._ov_install_label.SetLabel(msg)
        except Exception:  # noqa: BLE001
            pass

    def _omnivoice_install_done(self, success, message):
        """Called on the GUI thread when pip install finishes."""
        # Close progress dialog
        try:
            if self._ov_install_dlg:
                self._ov_install_dlg.Hide()
                self._ov_install_dlg.Destroy()
                self._ov_install_dlg = None
        except Exception:  # noqa: BLE001
            pass
        venv_packages.invalidate("omnivoice-triton", engine="omnivoice")
        self._refresh_omnivoice()
        wx.MessageBox(
            message,
            "OmniVoice",
            style=(wx.OK | wx.ICON_INFORMATION) if success else (wx.OK | wx.ICON_ERROR),
        )

    def _on_omnivoice_remove(self, _):
        """Uninstall omnivoice-triton via pip with a progress dialog."""
        if wx.MessageBox(
            "Remove the OmniVoice dependency (omnivoice-triton)?\n\n"
            "This will uninstall the omnivoice-triton package and its "
            "dependencies.\n"
            "OmniVoice Server shares this environment, so it is removed too.\n"
            "The OmniVoice TTS voices will no longer be available.",
            "Remove OmniVoice",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        self.omnivoice_install_btn.Disable()
        self.omnivoice_remove_btn.Disable()

        # Create a simple progress dialog (no cancel)
        self._ov_install_dlg = wx.Dialog(
            self, title="Removing OmniVoice",
            style=wx.DEFAULT_DIALOG_STYLE,
            size=(420, 120),
        )
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        self._ov_install_label = wx.StaticText(
            self._ov_install_dlg,
            label="Uninstalling omnivoice-triton package...",
        )
        dlg_sizer.Add(self._ov_install_label, 0, wx.ALL | wx.EXPAND, 10)
        self._ov_install_gauge = wx.Gauge(
            self._ov_install_dlg, range=0, size=(-1, 24),
            style=wx.GA_HORIZONTAL | wx.GA_SMOOTH,
        )
        dlg_sizer.Add(self._ov_install_gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self._ov_install_dlg.SetSizer(dlg_sizer)
        self._ov_install_dlg.Centre()
        self._ov_install_dlg.Show()

        self._ov_install_thread = threading.Thread(
            target=self._omnivoice_remove_job, daemon=True,
        )
        self._ov_install_thread.start()

    def _omnivoice_remove_job(self):
        """Background thread: uninstall all OmniVoice packages from the managed venv."""
        from ..python_runtime import get_runtime  # noqa: PLC0415
        rt = get_runtime("omnivoice")
        try:
            # Remove all OmniVoice-related packages (not PyTorch — other
            # addons may still need it).  The HTTP server lives in the same
            # environment, so it goes with them.
            packages = [
                "omnivoice-triton",
                "omnivoice-server",
                "omnivoice",
                "sageattention",
                "triton-windows",
            ]
            result = rt.pip_uninstall(packages)
            if result["ok"]:
                wx.CallAfter(self._omnivoice_remove_done, True,
                             "OmniVoice removed. Restart the application.")
            else:
                error = result.get("error") or "Removal failed."
                wx.CallAfter(self._omnivoice_remove_done, False, error)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._omnivoice_remove_done, False, f"Removal failed: {exc}")

    def _omnivoice_remove_done(self, success, message):
        """Called on the GUI thread when pip uninstall finishes."""
        try:
            if self._ov_install_dlg:
                self._ov_install_dlg.Hide()
                self._ov_install_dlg.Destroy()
                self._ov_install_dlg = None
        except Exception:  # noqa: BLE001
            pass
        venv_packages.invalidate("omnivoice-triton", engine="omnivoice")
        self._refresh_omnivoice()
        wx.MessageBox(
            message,
            "OmniVoice",
            style=(wx.OK | wx.ICON_INFORMATION) if success else (wx.OK | wx.ICON_ERROR),
        )

    # -- OmniVoice Server (omnivoice-server) ------------------------------
    def _init_omnivoice_server_ui(self):
        """Build the OmniVoice Server section in the Compute panel."""
        sizer = self.GetSizer()
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(
            wx.StaticText(self, label="OmniVoice Server (OpenAI-compatible HTTP API)"),
            0, wx.LEFT | wx.RIGHT | wx.TOP, 6,
        )
        sizer.Add(
            wx.StaticText(self, label=
                "HTTP server for OmniVoice TTS. Other apps on the same "
                "network can also use it. Package: omnivoice-server."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        ov_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        ov_grid.AddGrowableCol(1)
        self.server_status = wx.StaticText(self, label="Checking...")
        self.server_status.SetName("OmniVoice Server status")
        add_labeled(self, ov_grid, "OmniVoice Server", self.server_status,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(ov_grid, 0, wx.EXPAND | wx.ALL, 6)

        server_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.server_install_btn = wx.Button(self, label="Install OmniVoice Server")
        self.server_install_btn.SetName("Install OmniVoice Server")
        self.server_install_btn.SetToolTip(
            "Install omnivoice-server via pip (includes PyTorch, omnivoice, etc.)"
        )
        self.server_remove_btn = wx.Button(self, label="Remove OmniVoice Server")
        self.server_remove_btn.SetName("Remove OmniVoice Server")
        self.server_remove_btn.SetToolTip(
            "Uninstall omnivoice-server and its dependencies"
        )
        server_btns.Add(self.server_install_btn, 0, wx.ALL, 4)
        server_btns.Add(self.server_remove_btn, 0, wx.ALL, 4)
        sizer.Add(server_btns, 0, wx.LEFT, 2)

        sizer.AddSpacer(6)

        self.server_install_btn.Bind(wx.EVT_BUTTON, self._on_server_install)
        self.server_remove_btn.Bind(wx.EVT_BUTTON, self._on_server_remove)

    def _refresh_omnivoice_server(self):
        """Show the omnivoice-server status.

        ``omnivoice_server`` resolves to the environment the direct OmniVoice
        engine uses (the two share one), so this reports that status.
        """
        package = "omnivoice-server"
        env_id = "omnivoice_server"
        version = venv_packages.version(package, engine=env_id)
        if version is None:
            version = venv_packages.version(package)
        if version:
            self.server_status.SetLabel(
                f"Installed (v{version}). OmniVoice Server is available."
            )
            self.server_install_btn.Disable()
            self.server_remove_btn.Enable()
        else:
            self.server_status.SetLabel(
                "Not installed. OmniVoice Server is unavailable."
            )
            self.server_install_btn.Enable()
            self.server_remove_btn.Disable()
            if not venv_packages.is_known(package, engine=env_id):
                venv_packages.request(
                    package,
                    lambda _v: wx.CallAfter(self._refresh_omnivoice_server),
                    engine=env_id,
                )

    def _on_server_install(self, _):
        """Install omnivoice-server via pip with a progress dialog."""
        self.server_install_btn.Disable()
        self.server_remove_btn.Disable()

        self._srv_install_dlg = wx.Dialog(
            self, title="Installing OmniVoice Server",
            style=wx.DEFAULT_DIALOG_STYLE,
            size=(420, 120),
        )
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        self._srv_install_label = wx.StaticText(
            self._srv_install_dlg,
            label="Installing omnivoice-server package...",
        )
        dlg_sizer.Add(self._srv_install_label, 0, wx.ALL | wx.EXPAND, 10)
        self._srv_install_gauge = wx.Gauge(
            self._srv_install_dlg, range=0, size=(-1, 24),
            style=wx.GA_HORIZONTAL | wx.GA_SMOOTH,
        )
        dlg_sizer.Add(self._srv_install_gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self._srv_install_dlg.SetSizer(dlg_sizer)
        self._srv_install_dlg.Centre()
        self._srv_install_dlg.Show()

        self._srv_install_thread = threading.Thread(
            target=self._server_install_job, daemon=True,
        )
        self._srv_install_thread.start()

    def _server_install_job(self):
        """Background thread: install omnivoice-server into the shared venv."""
        from ..python_runtime import get_runtime  # noqa: PLC0415
        from ..voicelab import engines as voice_lab_engines  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415
        # The HTTP server shares the OmniVoice environment.
        rt = get_runtime("omnivoice_server")
        try:
            def _progress(msg, _done, _total):
                wx.CallAfter(self._srv_update_label, msg)

            if _sys.platform == "win32":
                _progress("Preparing Python environment...", 0, 0)
                rt.ensure_pip()

                # Step 1: PyTorch with CUDA (shared with the direct engine:
                # both OmniVoice engines install into the same environment).
                if self._omnivoice_torch_ready():
                    _progress("Step 1/3: PyTorch with CUDA is already installed.",
                              0, 0)
                else:
                    _progress(
                        "Step 1/3: Installing PyTorch with CUDA (~2 GB)...",
                        0, 0,
                    )
                    r = self._pip_install_raw(
                        rt.pip_exe,
                        ["torch", "torchaudio",
                         "--index-url",
                         voice_lab_engines.PYTORCH_CUDA_INDEX],
                    )
                    if not r["ok"]:
                        wx.CallAfter(self._server_install_done, False,
                                     f"Step 1 failed: {r.get('error', '')}")
                        return

                # Step 2: omnivoice (base)
                _progress("Step 2/3: Installing omnivoice (base package)...", 0, 0)
                res = rt.pip_install(["omnivoice"], progress=_progress)
                if not res["ok"]:
                    wx.CallAfter(
                        self._server_install_done, False,
                        f"Step 2 failed: {res.get('error', 'Install failed.')}",
                    )
                    return

                # Step 3: omnivoice-server
                _progress("Step 3/3: Installing omnivoice-server...", 0, 0)
                res = rt.pip_install(["omnivoice-server"], progress=_progress)
                if not res["ok"]:
                    wx.CallAfter(
                        self._server_install_done, False,
                        f"Step 3 failed: {res.get('error', 'Install failed.')}",
                    )
                    return
            else:
                _progress("Installing omnivoice-server...", 0, 0)
                result = rt.pip_install(
                    "omnivoice-server", progress=_progress,
                )
                if not result["ok"]:
                    wx.CallAfter(
                        self._server_install_done, False,
                        result.get("error") or "Installation failed.",
                    )
                    return

            wx.CallAfter(self._server_install_done, True,
                         "OmniVoice Server installed successfully.")
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._server_install_done, False, f"Installation failed: {exc}")

    def _srv_update_label(self, msg: str):
        """Update the server progress dialog label from a background thread."""
        try:
            if self._srv_install_dlg and self._srv_install_label:
                self._srv_install_label.SetLabel(msg)
        except Exception:  # noqa: BLE001
            pass

    def _server_install_done(self, success, message):
        """Called on the GUI thread when pip install finishes."""
        try:
            if self._srv_install_dlg:
                self._srv_install_dlg.Hide()
                self._srv_install_dlg.Destroy()
                self._srv_install_dlg = None
        except Exception:  # noqa: BLE001
            pass
        venv_packages.invalidate("omnivoice-server", engine="omnivoice_server")
        self._refresh_omnivoice_server()
        wx.MessageBox(
            message,
            "OmniVoice Server",
            style=(wx.OK | wx.ICON_INFORMATION) if success else (wx.OK | wx.ICON_ERROR),
        )

    def _on_server_remove(self, _):
        """Uninstall omnivoice-server via pip with a progress dialog."""
        if wx.MessageBox(
            "Remove the OmniVoice Server (omnivoice-server)?\n\n"
            "This will uninstall omnivoice-server and its dependencies.\n"
            "The HTTP server and server-mode voices will no longer be available.",
            "Remove OmniVoice Server",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        self.server_install_btn.Disable()
        self.server_remove_btn.Disable()

        self._srv_install_dlg = wx.Dialog(
            self, title="Removing OmniVoice Server",
            style=wx.DEFAULT_DIALOG_STYLE,
            size=(420, 120),
        )
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        self._srv_install_label = wx.StaticText(
            self._srv_install_dlg,
            label="Uninstalling omnivoice-server package...",
        )
        dlg_sizer.Add(self._srv_install_label, 0, wx.ALL | wx.EXPAND, 10)
        self._srv_install_gauge = wx.Gauge(
            self._srv_install_dlg, range=0, size=(-1, 24),
            style=wx.GA_HORIZONTAL | wx.GA_SMOOTH,
        )
        dlg_sizer.Add(self._srv_install_gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self._srv_install_dlg.SetSizer(dlg_sizer)
        self._srv_install_dlg.Centre()
        self._srv_install_dlg.Show()

        self._srv_install_thread = threading.Thread(
            target=self._server_remove_job, daemon=True,
        )
        self._srv_install_thread.start()

    def _server_remove_job(self):
        """Background thread: uninstall omnivoice-server from the shared venv."""
        from ..python_runtime import get_runtime  # noqa: PLC0415
        rt = get_runtime("omnivoice_server")
        try:
            packages = ["omnivoice-server"]
            result = rt.pip_uninstall(packages)
            if result["ok"]:
                wx.CallAfter(self._server_remove_done, True,
                             "OmniVoice Server removed.")
            else:
                error = result.get("error") or "Removal failed."
                wx.CallAfter(self._server_remove_done, False, error)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._server_remove_done, False, f"Removal failed: {exc}")

    def _server_remove_done(self, success, message):
        """Called on the GUI thread when pip uninstall finishes."""
        try:
            if self._srv_install_dlg:
                self._srv_install_dlg.Hide()
                self._srv_install_dlg.Destroy()
                self._srv_install_dlg = None
        except Exception:  # noqa: BLE001
            pass
        venv_packages.invalidate("omnivoice-server", engine="omnivoice_server")
        self._refresh_omnivoice_server()
        wx.MessageBox(
            message,
            "OmniVoice Server",
            style=(wx.OK | wx.ICON_INFORMATION) if success else (wx.OK | wx.ICON_ERROR),
        )

    def _refresh(self):
        installed = runtime.is_installed()
        if installed:
            version = runtime.installed_version()
            self.gpu_status.SetLabel(
                f"Installed ({version or 'ready'}). The GPU back-end will be "
                "active after the application restarts."
            )
        else:
            self.gpu_status.SetLabel(
                "Not installed. CPU back-end only until the runtime is downloaded."
            )
        busy = self._thread is not None
        self.gpu_download_btn.Enable(not installed and not busy)
        self.gpu_remove_btn.Enable(installed and not busy)
        self.gpu_cancel_btn.Enable(busy)

    def _on_download(self, _):
        self._cancel_event = threading.Event()
        self.gpu_download_btn.Disable()
        self.gpu_remove_btn.Disable()
        self.gpu_cancel_btn.Enable()
        self.gauge.SetValue(0)
        self.progress_label.SetLabel("Downloading the GPU runtime...")
        self._thread = threading.Thread(target=self._download_job, daemon=True)
        self._thread.start()

    def _download_job(self):
        def progress(name, done, total):
            wx.PostEvent(self, DownloadProgressEvent(name, done, total))

        try:
            runtime.install(progress=progress, cancel_event=self._cancel_event)
            wx.PostEvent(
                self,
                DownloadFinishedEvent(
                    True, "GPU runtime downloaded. Restart the application to activate it."
                ),
            )
        except runtime.DownloadCancelled:
            wx.PostEvent(self, DownloadFinishedEvent(False, "Download cancelled."))
        except (runtime.DownloadError, OSError) as exc:
            wx.PostEvent(self, DownloadFinishedEvent(False, f"Download failed: {exc}"))

    def _on_cancel(self, _):
        self._cancel_event.set()
        self.gpu_cancel_btn.Disable()
        self.progress_label.SetLabel("Cancelling...")

    def _on_progress(self, evt: DownloadProgressEvent):
        done, total = evt.done, evt.total
        if total > 0:
            self.gauge.SetValue(int(min(100, done * 100 / total)))
        self.progress_label.SetLabel(
            f"Downloading {evt.name}: {done / (1024 * 1024):.1f} MB"
            + (f" of {total / (1024 * 1024):.1f} MB" if total else "")
        )

    def _on_finished(self, evt: DownloadFinishedEvent):
        self._thread = None
        self.gpu_cancel_btn.Disable()
        self.progress_label.SetLabel(evt.message)
        self.gauge.SetValue(100 if evt.success else 0)
        self._refresh()
        if evt.success:
            wx.MessageBox(
                evt.message,
                "GPU runtime",
                style=wx.OK | wx.ICON_INFORMATION,
            )

    def _on_remove(self, _):
        if wx.MessageBox(
            "Remove the GPU (CUDA) runtime? Its files will be deleted from "
            "your user folder. Restart the application afterwards.",
            "Remove GPU runtime",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            runtime.remove()
            self.progress_label.SetLabel(
                "GPU runtime removed. Restart the application to return to CPU-only."
            )
            self._refresh()


# ---------------------------------------------------------------------------
# Developer category (addon management, diagnostics)
# ---------------------------------------------------------------------------
class _DeveloperPanel(_SettingsPanel):
    title = "Developer"

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self._thread = None
        self._pip_thread: threading.Thread | None = None
        self._pip_probe_running = False
        sizer = wx.BoxSizer(wx.VERTICAL)

        self._dev_warning = wx.StaticText(
            self,
            label="Developer Mode is not enabled. Go to General and enable "
                  "Developer Mode to use these features.",
        )
        sizer.Add(self._dev_warning, 0, wx.ALL, 6)

        # -- Addon management ---------------------------------------------
        addon_box = wx.StaticBox(self, label="Installed Addons")
        addon_sizer = wx.StaticBoxSizer(addon_box, wx.VERTICAL)
        self.addon_list = wx.ListBox(addon_box, style=wx.LB_SINGLE)
        self.addon_list.SetName("Installed addons")
        self.addon_list.SetMinSize((-1, 120))
        addon_sizer.Add(self.addon_list, 1, wx.EXPAND | wx.ALL, 4)

        addon_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.addon_install_btn = wx.Button(addon_box, label="Install addon from ZIP...")
        self.addon_install_btn.SetName("Install addon from ZIP")
        self.addon_enable_btn = wx.Button(addon_box, label="Enable")
        self.addon_disable_btn = wx.Button(addon_box, label="Disable")
        self.addon_remove_btn = wx.Button(addon_box, label="Remove")
        self.addon_enable_btn.SetName("Enable selected addon")
        self.addon_enable_btn.SetToolTip("Enable the selected addon so it loads on the next restart")
        self.addon_disable_btn.SetName("Disable selected addon")
        self.addon_disable_btn.SetToolTip("Disable the selected addon so it no longer loads")
        self.addon_remove_btn.SetName("Remove selected addon permanently")
        self.addon_remove_btn.SetToolTip("Delete the selected addon and all its files")
        addon_btns.Add(self.addon_install_btn, 0, wx.ALL, 2)
        addon_btns.Add(self.addon_enable_btn, 0, wx.ALL, 2)
        addon_btns.Add(self.addon_disable_btn, 0, wx.ALL, 2)
        addon_btns.Add(self.addon_remove_btn, 0, wx.ALL, 2)
        addon_sizer.Add(addon_btns, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(addon_sizer, 1, wx.EXPAND | wx.ALL, 6)

        # -- Pip environment ----------------------------------------------
        pip_box = wx.StaticBox(self, label="Python Environment")
        pip_sizer = wx.StaticBoxSizer(pip_box, wx.VERTICAL)
        self.pip_status = wx.StaticText(pip_box, label="Checking...")
        self.pip_status.SetName("Python environment status")
        pip_sizer.Add(self.pip_status, 0, wx.ALL, 6)

        pip_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.pip_install_btn = wx.Button(pip_box, label="Install package...")
        self.pip_install_btn.SetName("Install pip package")
        self.pip_install_btn.SetToolTip("Install a Python package into the addon virtual environment")
        self.pip_list_btn = wx.Button(pip_box, label="List installed")
        self.pip_list_btn.SetName("List installed pip packages")
        self.pip_list_btn.SetToolTip("Show all packages installed in the addon virtual environment")
        pip_btns.Add(self.pip_install_btn, 0, wx.ALL, 2)
        pip_btns.Add(self.pip_list_btn, 0, wx.ALL, 2)
        pip_sizer.Add(pip_btns, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(pip_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # -- Diagnostics -------------------------------------------------
        diag_box = wx.StaticBox(self, label="Diagnostics")
        diag_sizer = wx.StaticBoxSizer(diag_box, wx.VERTICAL)
        self.log_path = wx.StaticText(diag_box, label="")
        self.log_path.SetName("Log file path")
        diag_sizer.Add(self.log_path, 0, wx.ALL, 6)
        diag_btns = wx.BoxSizer(wx.HORIZONTAL)
        self.open_log_btn = wx.Button(diag_box, label="Open log folder")
        self.open_log_btn.SetName("Open log folder")
        self.open_log_btn.SetToolTip("Open the logs folder in Windows Explorer")
        self.system_info_btn = wx.Button(diag_box, label="System info")
        self.system_info_btn.SetName("Show system information")
        self.system_info_btn.SetToolTip("Display detailed system information and write it to the log")
        diag_btns.Add(self.open_log_btn, 0, wx.ALL, 2)
        diag_btns.Add(self.system_info_btn, 0, wx.ALL, 2)
        diag_sizer.Add(diag_btns, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(diag_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # -- Log Viewer ---------------------------------------------------
        log_box = wx.StaticBox(self, label="Log Viewer")
        log_sizer = wx.StaticBoxSizer(log_box, wx.VERTICAL)
        log_row = wx.BoxSizer(wx.HORIZONTAL)
        log_row.Add(wx.StaticText(log_box, label="Log file to view:"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.log_file_combo = wx.ComboBox(log_box, style=wx.CB_READONLY,
                                             name="Log file to view")
        self.log_file_combo.SetToolTip("Choose which log file to display in the viewer below")
        self.log_file_combo.Append("app.log", "app.log")
        self.log_file_combo.Append("gui.log", "gui.log")
        self.log_file_combo.Append("tts.log", "tts.log")
        self.log_file_combo.Append("engine.log", "engine.log")
        self.log_file_combo.Append("recording.log", "recording.log")
        self.log_file_combo.Append("addons.log", "addons.log")
        self.log_file_combo.Append("download.log", "download.log")
        self.log_file_combo.Append("crash.log", "crash.log")
        self.log_file_combo.SetSelection(0)
        log_row.Add(self.log_file_combo, 1, wx.EXPAND)
        self.log_refresh_btn = wx.Button(log_box, label="Refresh")
        self.log_refresh_btn.SetName("Refresh log viewer")
        self.log_refresh_btn.SetToolTip("Reload the selected log file and display its latest content")
        log_row.Add(self.log_refresh_btn, 0, wx.LEFT, 6)
        log_sizer.Add(log_row, 0, wx.EXPAND | wx.ALL, 4)

        self.log_text = wx.TextCtrl(
            log_box, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
            size=(-1, 180),
        )
        self.log_text.SetName("Log file content")
        self.log_text.SetToolTip("Read-only view of the selected log file. Use arrow keys to scroll.")
        self.log_text.SetFont(
            wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        log_sizer.Add(self.log_text, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        log_hint = wx.StaticText(log_box, label="Showing last 200 lines. Click Refresh to update.")
        log_sizer.Add(log_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(log_sizer, 1, wx.EXPAND | wx.ALL, 6)

        self.SetSizer(sizer)

        self.log_file_combo.Bind(wx.EVT_COMBOBOX, self._on_log_file_change)
        self.log_refresh_btn.Bind(wx.EVT_BUTTON, self._on_log_refresh)
        # F5 keyboard shortcut for refresh
        self.log_refresh_btn.SetToolTip("Refresh log file (F5)")

        # Bind events
        self.addon_install_btn.Bind(wx.EVT_BUTTON, self._on_install_addon)
        self.addon_enable_btn.Bind(wx.EVT_BUTTON, self._on_enable_addon)
        self.addon_disable_btn.Bind(wx.EVT_BUTTON, self._on_disable_addon)
        self.addon_remove_btn.Bind(wx.EVT_BUTTON, self._on_remove_addon)
        self.pip_install_btn.Bind(wx.EVT_BUTTON, self._on_pip_install)
        self.pip_list_btn.Bind(wx.EVT_BUTTON, self._on_pip_list)
        self.open_log_btn.Bind(wx.EVT_BUTTON, self._on_open_log)
        self.system_info_btn.Bind(wx.EVT_BUTTON, self._on_system_info)
        self.addon_list.Bind(wx.EVT_LISTBOX, self._on_addon_select)

    def on_activated(self):
        super().on_activated()
        self._refresh_ui()

    def _refresh_ui(self):
        is_dev = self.settings.get("developer_mode", False)
        self._dev_warning.Show(not is_dev)
        for ctrl in (self.addon_install_btn, self.addon_enable_btn,
                     self.addon_disable_btn, self.addon_remove_btn,
                     self.pip_install_btn, self.pip_list_btn,
                     self.open_log_btn, self.system_info_btn,
                     self.log_file_combo, self.log_refresh_btn):
            ctrl.Enable(is_dev)
        if is_dev:
            self._load_log_file()
        self._refresh_addons()
        self._refresh_pip()
        self.log_path.SetLabel(f"Log directory: {paths.logs_dir()}")

    def _refresh_addons(self):
        from ..addons import get_addon_manager
        manager = get_addon_manager()
        manager.discover()
        self.addon_list.Clear()
        for addon in manager.list_addons():
            status = "[Enabled]" if addon.enabled else "[Disabled]"
            self.addon_list.Append(f"{status} {addon.name} v{addon.version}", addon.name)

    def _refresh_pip(self, force: bool = False):
        """Show the managed-environment status without blocking the UI.

        Counting the packages runs the environment's own Python (a
        subprocess), which is too slow to do while the category opens, so
        the answer is cached and the first probe runs on a worker thread.
        """
        from ..python_runtime import get_runtime
        rt = get_runtime()
        if not rt.is_created:
            self.pip_status.SetLabel(
                f"Environment not yet created. Will be created at: {rt.env_dir}"
            )
            return
        if not force:
            cached = _PIP_STATUS_CACHE.get(rt.env_dir)
            if cached is not None:
                self.pip_status.SetLabel(cached)
                return
        if self._pip_probe_running:
            return
        self._pip_probe_running = True
        self.pip_status.SetLabel("Checking the Python environment...")
        self._pip_thread = threading.Thread(
            target=self._probe_pip, args=(rt,), daemon=True
        )
        self._pip_thread.start()

    def _probe_pip(self, rt):
        """Worker thread: count the packages in the managed environment."""
        label = None
        try:
            pkgs = rt.pip_list()
            label = (
                f"Environment ready: {len(pkgs)} packages installed "
                f"at {rt.env_dir}"
            )
        except Exception:  # noqa: BLE001
            label = None
        try:
            wx.CallAfter(self._pip_ready, rt.env_dir, label)
        except Exception:  # noqa: BLE001
            pass

    def _pip_ready(self, env_dir: str, label):
        """Main thread: apply the result of a background environment probe."""
        self._pip_probe_running = False
        if label:
            _PIP_STATUS_CACHE[env_dir] = label
        try:
            self.pip_status.SetLabel(
                label if label else "Could not read the Python environment."
            )
        except Exception:  # noqa: BLE001
            pass

    def _on_addon_select(self, _):
        pass  # selection update

    def _on_install_addon(self, _):
        with wx.FileDialog(
            self,
            "Choose an addon archive",
            wildcard="ZIP files (*.zip)|*.zip|All files|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        from ..addons import get_addon_manager
        manager = get_addon_manager()
        if manager.install_addon(path):
            self._refresh_addons()
            wx.MessageBox(
                "Addon installed successfully. It will be loaded on the next restart.",
                "Addon installed", style=wx.OK | wx.ICON_INFORMATION,
            )
        else:
            wx.MessageBox(
                "Failed to install the addon. Check the manifest.json is valid.",
                "Addon install failed", style=wx.OK | wx.ICON_ERROR,
            )

    def _on_enable_addon(self, _):
        sel = self.addon_list.GetSelection()
        if sel < 0:
            return
        name = self.addon_list.GetClientData(sel)
        from ..addons import get_addon_manager
        get_addon_manager().enable_addon(name)
        self._refresh_addons()

    def _on_disable_addon(self, _):
        sel = self.addon_list.GetSelection()
        if sel < 0:
            return
        name = self.addon_list.GetClientData(sel)
        from ..addons import get_addon_manager
        get_addon_manager().disable_addon(name)
        self._refresh_addons()

    def _on_remove_addon(self, _):
        sel = self.addon_list.GetSelection()
        if sel < 0:
            return
        name = self.addon_list.GetClientData(sel)
        if wx.MessageBox(
            f"Remove addon '{name}'? This cannot be undone.",
            "Remove addon", style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            from ..addons import get_addon_manager
            get_addon_manager().uninstall_addon(name)
            self._refresh_addons()

    def _on_pip_install(self, _):
        dlg = wx.Dialog(self, title="Install pip package", size=(400, 120))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(dlg, label="Package name(s) (space-separated):"),
                  0, wx.ALL, 6)
        pkg_ctrl = wx.TextCtrl(dlg, size=(-1, 28))
        pkg_ctrl.SetName("Package names")
        sizer.Add(pkg_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(dlg, wx.ID_OK, "Install")
        cancel_btn = wx.Button(dlg, wx.ID_CANCEL)
        btn_sizer.Add(ok_btn, 0, wx.ALL, 4)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 4)
        sizer.Add(btn_sizer, 0, wx.ALL, 4)
        dlg.SetSizer(sizer)
        if dlg.ShowModal() == wx.ID_OK:
            packages = pkg_ctrl.GetValue().strip().split()
            if packages:
                from ..python_runtime import get_runtime
                rt = get_runtime()
                self.pip_status.SetLabel("Installing...")
                def _job():
                    result = rt.pip_install(packages)
                    wx.CallAfter(self._pip_install_done, result)
                self._thread = threading.Thread(target=_job, daemon=True)
                self._thread.start()
        dlg.Destroy()

    def _pip_install_done(self, result):
        self._thread = None
        self._refresh_pip(force=True)
        if result["ok"]:
            wx.MessageBox("Packages installed successfully.", "pip install",
                          style=wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox(f"Installation failed:\n{result['error']}", "pip install",
                          style=wx.OK | wx.ICON_ERROR)

    def _on_pip_list(self, _):
        from ..python_runtime import get_runtime
        rt = get_runtime()
        pkgs = rt.pip_list()
        if not pkgs:
            wx.MessageBox("No packages installed (or environment not ready).",
                          "pip list", style=wx.OK | wx.ICON_INFORMATION)
            return
        lines = [f"{p.get('name', '?')} {p.get('version', '?')}" for p in pkgs]
        wx.MessageBox("\n".join(lines), "Installed packages",
                      style=wx.OK | wx.ICON_INFORMATION)

    def _on_open_log(self, _):
        import webbrowser
        log_dir = paths.logs_dir()
        try:
            os.startfile(log_dir)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open("file:///" + log_dir.replace("\\", "/"))

    def _on_system_info(self, _):
        from ..heavy_logging import log_system_info
        import platform
        info = (
            f"AI Voice Studio v{__import__('ai_voice_studio', fromlist=['__version__']).__version__}\n"
            f"Python: {sys.version}\n"
            f"Platform: {platform.platform()}\n"
            f"Machine: {platform.machine()}\n"
            f"Executable: {sys.executable}\n"
            f"PID: {os.getpid()}\n"
        )
        log_system_info()
        wx.MessageBox(info, "System Information",
                      style=wx.OK | wx.ICON_INFORMATION)

    # -- Log Viewer --------------------------------------------------------
    _LOG_MAX_LINES = 200

    def _on_log_file_change(self, _):
        self._load_log_file()

    def _on_log_refresh(self, _):
        self._load_log_file()

    def _load_log_file(self):
        sel = self.log_file_combo.GetSelection()
        if sel < 0:
            return
        filename = self.log_file_combo.GetClientData(sel)
        if not filename:
            return
        log_dir = paths.logs_dir()
        log_path = os.path.join(log_dir, filename)
        if not os.path.isfile(log_path):
            msg = f"Log file not found: {log_path}. Heavy logging may not be enabled yet."
            self.log_text.SetValue(msg)
            # Announce to screen readers via accessible name update
            self.log_text.SetName(f"Log file {filename}: file not found")
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            tail = lines[-self._LOG_MAX_LINES:]
            self.log_text.SetValue("".join(tail))
            self.log_text.SetInsertionPointEnd()
            # Announce content loaded to screen readers
            total_lines = len(lines)
            shown = len(tail)
            self.log_text.SetName(
                f"Log file {filename}: showing last {shown} of {total_lines} lines"
            )
        except OSError as exc:
            error_msg = f"Could not read {log_path}: {exc}"
            self.log_text.SetValue(error_msg)
            self.log_text.SetName(f"Log file {filename}: read error")


# ---------------------------------------------------------------------------
# Reset category
# ---------------------------------------------------------------------------
class _ResetPanel(_SettingsPanel):
    title = "Reset"

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.reset_btn = wx.Button(self, label="Reset to default")
        self.reset_btn.SetName("Reset to default")
        sizer.Add(self.reset_btn, 0, wx.ALL, 4)
        self.SetSizer(sizer)
        self.reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)

    def _on_reset(self, _):
        if wx.MessageBox(
            "All settings will go back to their default values. Continue?",
            "Reset settings",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) == wx.YES:
            self.settings.reset()
            wx.MessageBox(
                "Settings have been reset to their default values. The other tabs "
                "now show the defaults; press OK or Apply to keep them.",
                "Reset complete",
                style=wx.OK | wx.ICON_INFORMATION,
            )
