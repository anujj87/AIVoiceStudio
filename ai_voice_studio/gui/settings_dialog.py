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
4. Voice clone            -- clone a voice from a short audio sample (XTTS v2)
5. Recording settings     -- speed, pitch, volume, preview
6. Punctuation            -- default punctuation mode (spoken-word expansion)
7. Audio file creation    -- 4 radio modes with descriptions
8. Compute                -- optional GPU (CUDA) runtime
9. Reset                  -- restore defaults
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import wx

from .. import paths, runtime
from ..constants import (
    AUDIO_MODE_CHOICES,
    AUDIO_MODE_DESCRIPTIONS,
    MODE_PAGE_WITH_H1,
    PITCH_MAX,
    PITCH_MIN,
    PROJECT_TYPE_DAISY_AUDIO,
    PROJECT_TYPE_DAISY_AUDIO_TEXT,
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
from ..settings import Settings
from ..tts import catalog
from ..tts.downloader import ModelDownloader
from ..tts.models import ModelStore
from .a11y import add_labeled
from .events import (
    DownloadFinishedEvent,
    DownloadProgressEvent,
    EVT_DOWNLOAD_FINISHED,
    EVT_DOWNLOAD_PROGRESS,
)
from .model_panels import AvailablePanel, DownloadPanel, VoiceClonePanel
from .theme import apply_theme

log = logging.getLogger(__name__)

_THEME_CHOICES = [
    (THEME_SYSTEM, "System default"),
    (THEME_LIGHT, "Light"),
    (THEME_DARK, "Dark"),
]


class _SettingsPanel(wx.Panel):
    """Base class for a settings category (NVDA ``SettingsPanel`` pattern).

    Each category is a panel with a ``title`` (shown in the category list), an
    optional spoken ``description``, and an ``apply_to_settings`` hook invoked
    when the user presses OK or Apply.

    Navigation pattern:
      1. Panel opens → focus lands on the first interactive control.
      2. Tab moves through controls top-to-bottom, left-to-right.
      3. The panel description is available via the accessible description
         so screen readers announce it when the panel is focused.
    """

    title = ""
    description = ""

    def on_activated(self):
        self.Show()
        # Focus the first enabled interactive control so the screen reader
        # announces the panel's purpose immediately.
        first = self._first_focusable()
        if first:
            wx.CallAfter(first.SetFocus)

    def on_deactivated(self):
        self.Hide()

    def apply_to_settings(self):
        pass

    def _first_focusable(self) -> wx.Window | None:
        """Find the first enabled, visible interactive child control."""
        _INTERACTIVE = (wx.TextCtrl, wx.ComboBox, wx.CheckBox,
                        wx.RadioButton, wx.ListBox, wx.Button, wx.Slider)
        stack = list(self.GetChildren())
        while stack:
            child = stack.pop(0)
            if (isinstance(child, _INTERACTIVE)
                    and child.IsShown() and child.IsEnabled()):
                return child
            stack.extend(child.GetChildren())
        return None


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
        # NVDA-style postInit: focus lands on the category list.
        wx.CallAfter(self.cat_list.SetFocus)

    # -- construction -------------------------------------------------------
    def _build_ui(self):
        main = wx.BoxSizer(wx.VERTICAL)

        body = wx.BoxSizer(wx.HORIZONTAL)
        # Categories label spans both columns (NVDA layout).
        body.Add(
            wx.StaticText(self, label="&Categories:"),
            0, wx.LEFT | wx.TOP | wx.RIGHT, 6,
        )
        # The list and the panel are placed on separate rows below.
        self.cat_list = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_NO_HEADER,
            size=(200, 380),
        )
        self.cat_list.InsertColumn(0, "Categories:")
        self.cat_list.SetName("Settings categories")

        self.container = wx.Panel(self)
        self.container.SetName("Settings panel")
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
            self.container_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 6)
            self._panels.append(panel)

        grid = wx.FlexGridSizer(cols=2, vgap=4, hgap=8)
        grid.AddGrowableRow(1)
        grid.AddGrowableCol(0, proportion=1)
        grid.AddGrowableCol(1, proportion=3)
        grid.Add(self.cat_list, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        grid.Add(self.container, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        main.Add(grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)

        buttons = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL | wx.APPLY)
        main.Add(buttons, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self.SetSizer(main)
        self.SetMinSize((700, 420))

    def _panel_args(self, cls):
        """Extra constructor arguments per panel class (NVDA panels take a
        parent plus category-specific dependencies)."""
        if cls is _ComputePanel:
            return ()
        if cls in (DownloadPanel,):
            return (self.store, self.downloader)
        if cls in (VoiceClonePanel,):
            return (self.store, self.downloader)
        if cls in (AvailablePanel,):
            return (self.store,)
        if cls in (_PunctuationPanel,):
            return (self.settings, self.store)
        return (self.settings,)

    # -- events -------------------------------------------------------------
    def _bind_events(self):
        self.cat_list.Bind(wx.EVT_LIST_ITEM_FOCUSED, self._on_category_focus)
        self.Bind(wx.EVT_BUTTON, self._on_apply, id=wx.ID_APPLY)
        self.Bind(wx.EVT_BUTTON, self._on_ok, id=wx.ID_OK)
        self.Bind(wx.EVT_BUTTON, self._on_cancel, id=wx.ID_CANCEL)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)

    def _on_category_focus(self, evt):
        self._show_category(evt.GetIndex())
        evt.Skip()

    def _show_category(self, index: int):
        """Show the panel for ``index`` and hide every other (NVDA pattern).

        After switching, focus lands on the first interactive control in the
        panel so the screen reader announces the panel's purpose immediately.
        """
        if not (0 <= index < len(self._panels)):
            return
        for i, panel in enumerate(self._panels):
            if i == index:
                panel.on_activated()
            else:
                panel.on_deactivated()
        self._current = index
        self.container.Layout()
        self.container.Refresh()
        # Move focus into the panel so NVDA/JAWS announce the first control.
        panel = self._panels[index]
        first = panel.FindWindowInDirection(wx.NavigationEnabled.NavigateDirection, wx.NavigationEnabled.NavigateDirection, False)
        if first and first.IsShown() and first.IsEnabled():
            wx.CallAfter(first.SetFocus)
        else:
            wx.CallAfter(panel.SetFocus)

    def _on_char_hook(self, evt):
        """NVDA-style keyboard: Ctrl+Tab switches category, Enter activates OK,
        Ctrl+S activates Apply. Enter is left alone inside multi-line text
        boxes (e.g. the Punctuation example) so it still inserts a new line."""
        key = evt.GetKeyCode()
        control = self.FindFocus()
        if evt.ControlDown() and key == wx.WXK_TAB:
            index = self.cat_list.GetFirstSelected()
            if index < 0:
                index = self._current
            step = -1 if evt.ShiftDown() else 1
            new_index = (index + step) % len(self._panels)
            self.cat_list.Select(new_index)
            self.cat_list.Focus(new_index)
            self._show_category(new_index)
            # _show_category moves focus into the panel; no need to
            # return focus to the category list.
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
            evt.Skip()

    # -- callbacks ----------------------------------------------------------
    def _refresh_available(self):
        self.available_panel.refresh()
        self.download_panel._refresh_buttons()

    def _save_from_ui(self):
        for panel in self._panels:
            try:
                panel.apply_to_settings()
            except Exception:  # noqa: BLE001
                log.exception("Panel %s failed to save", panel.title)
        self.settings.save()

    def _on_apply(self, _):
        self._save_from_ui()
        apply_theme(self, self.settings.theme)

    def _on_ok(self, _):
        self._save_from_ui()
        self.EndModal(wx.ID_OK)

    def _on_cancel(self, _):
        # Discard any in-flight changes by restoring the pre-open snapshot.
        self.settings.set_many({k: v for k, v in self._orig.items()})
        self.EndModal(wx.ID_CANCEL)

    # Convenience attributes kept for tests / external code.
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
    def clone_panel(self):
        return self._panels[3]

    @property
    def recording_panel(self):
        return self._panels[4]

    @property
    def punctuation_panel(self):
        return self._panels[5]

    @property
    def audio_mode_panel(self):
        return self._panels[6]

    @property
    def daisy_panel(self):
        return self._panels[7]

    @property
    def compute_panel(self):
        return self._panels[8]

    @property
    def developer_panel(self):
        return self._panels[9]

    @property
    def reset_panel(self):
        return self._panels[10]


# ---------------------------------------------------------------------------
# General category
# ---------------------------------------------------------------------------
class _GeneralPanel(_SettingsPanel):
    title = "General"
    description = (
        "Appearance, folders, and developer mode for AI Voice Studio."
    )

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


# ---------------------------------------------------------------------------
# Recording settings category
# ---------------------------------------------------------------------------
class _RecordingSettingsPanel(_SettingsPanel):
    title = "Recording settings"
    description = (
        "Default voice settings (speed, pitch, volume) used for new projects "
        "and a preview of the selected voice."
    )

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Default speed, pitch, and volume for new projects. "
                                      "Can be changed per project in the Recording window."),
            0, wx.ALL, 6,
        )

        self.rate = self._slider_row(sizer, "Speed", settings.get("recording.rate", 1.0),
                                     RATE_MIN, RATE_MAX)
        self.pitch = self._slider_row(sizer, "Pitch", settings.get("recording.pitch", 1.0),
                                      PITCH_MIN, PITCH_MAX)
        self.volume = self._slider_row(sizer, "Volume", settings.get("recording.volume", 1.0),
                                       VOLUME_MIN, VOLUME_MAX)

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

        self.preview_btn = wx.Button(self, label="Preview")
        self.preview_btn.SetName("Preview")
        sizer.Add(self.preview_btn, 0, wx.ALL, 4)
        self.preview_status = wx.StaticText(self, label="")
        sizer.Add(self.preview_status, 0, wx.ALL, 4)
        self.SetSizer(sizer)

        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)

    def _slider_row(self, sizer, name: str, value: float, lo: float, hi: float):
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        slider = wx.Slider(self, minValue=int(lo * 100), maxValue=int(hi * 100),
                           value=int(value * 100))
        slider.SetName(f"{name}: {value:.2f}")
        label = wx.StaticText(self, label=name + ":")
        label.SetName(name + " label")
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        value_label = wx.StaticText(self, label=f"{value:.2f}")
        slider.value_label = value_label  # type: ignore[attr-defined]
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(slider, 1, wx.EXPAND)
        box.Add(value_label, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6)
        grid.Add(box, 1, wx.EXPAND)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 4)
        slider.Bind(
            wx.EVT_SLIDER,
            lambda evt, s=slider, vl=value_label, n=name: (
                vl.SetLabel(f"{s.GetValue() / 100.0:.2f}"),
                s.SetName(f"{n}: {s.GetValue() / 100.0:.2f}"),
            ),
        )
        return slider

    def apply_to_settings(self):
        self.settings.set("recording.rate", self.rate.GetValue() / 100.0)
        self.settings.set("recording.pitch", self.pitch.GetValue() / 100.0)
        self.settings.set("recording.volume", self.volume.GetValue() / 100.0)

    def _on_preview(self, _):
        from ..tts.models import ModelStore

        store = ModelStore()
        voices = store.installed_voices()
        if not voices:
            wx.MessageBox(
                "No voices are downloaded yet. Use the 'Download and remove' tab "
                "to install a voice first.",
                "Preview unavailable",
                style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        voice = voices[0]
        # Snapshot the UI state on the UI thread; the job runs on a worker.
        text = self.sample_text.GetValue() or "Hello."
        punct = self.settings.get("recording.punctuation", "default")
        rate = self.rate.GetValue() / 100.0
        pitch = self.pitch.GetValue() / 100.0
        volume = self.volume.GetValue() / 100.0
        self.preview_btn.Disable()
        self.preview_status.SetLabel("Synthesizing preview...")
        threading.Thread(
            target=self._preview_job,
            args=(voice, text, punct, rate, pitch, volume),
            daemon=True,
        ).start()

    def _preview_job(self, voice, text, punct, rate, pitch, volume):
        from .. import compute as compute_mod
        from ..audio.output import write_wav
        from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation

        try:
            engine = get_engine(
                voice, provider=compute_mod.provider_for(compute_mod.resolve_compute("cpu"))
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

            tmp = os.path.join(tempfile.gettempdir(), "aivs_preview.wav")
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
            wx.MessageBox(error, "Preview failed", style=wx.OK | wx.ICON_ERROR)
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
    description = (
        "Default punctuation mode for new projects. 'All' reads every "
        "punctuation mark as a word so TTS voices that cannot pronounce "
        "punctuation still read text correctly."
    )

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
            wx.StaticText(self, label="Default punctuation mode for new projects."),
            0, wx.ALL, 6,
        )
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

        self.preview_btn = wx.Button(self, label="Preview")
        self.preview_btn.SetName("Preview")
        sizer.Add(self.preview_btn, 0, wx.ALL, 4)
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

    # -- voice cascade ------------------------------------------------------
    def _populate_voices(self):
        self._voices = self.store.installed_voices()
        for voice in self.store.custom_voices():
            self._voices.append({
                "tts": voice["tts"], "tts_name": voice["tts"],
                "language": "custom", "variant": "custom",
                "voice": voice["name"],
                "voice_name": f"Cloned voice: {voice['name']}",
                "sid": 0, "engine": voice.get("engine", "vits"),
                "dir": voice["dir"], "custom": True,
                "sample": voice.get("sample", ""),
                "reference": voice.get("reference", ""),
                "xtts_lang": voice.get("language", "en"),
            })
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
    def _on_preview(self, _):
        voice = self.selected_voice()
        if not voice:
            wx.MessageBox("Select a voice first.", "Preview",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        self.preview_btn.Disable()
        self.preview_status.SetLabel("Synthesizing preview...")
        mode = self.selected()
        text = self.example_text.GetValue() or "Hello."
        threading.Thread(
            target=self._preview_job, args=(voice, text, mode), daemon=True
        ).start()

    def _preview_job(self, voice, text, mode):
        from .. import compute as compute_mod
        from ..audio.output import write_wav
        from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation

        try:
            engine = get_engine(
                voice,
                provider=compute_mod.provider_for(compute_mod.resolve_compute("cpu")),
            )
            text = process_punctuation(text, mode)
            samples = engine.synthesize(text, sid=voice.get("sid", 0), speed=1.0)
            import tempfile  # noqa: PLC0415

            tmp = os.path.join(tempfile.gettempdir(), "aivs_preview.wav")
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
            wx.MessageBox(error, "Preview failed", style=wx.OK | wx.ICON_ERROR)
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
    description = "Choose how the document is split into audio files."

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Choose how the document is split into "
                                      "audio files."),
            0, wx.ALL, 6,
        )

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
            radio.Bind(wx.EVT_RADIOBUTTON, self._on_select)
            self.radios.append((radio, value))
            first = False
        self._update_description()
        self.SetSizer(sizer)

    def _on_select(self, _):
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
        self.settings.set("audio_mode", self.selected())


# ---------------------------------------------------------------------------
# DAISY book settings category
# ---------------------------------------------------------------------------
class _DaisySettingsPanel(_SettingsPanel):
    title = "DAISY settings"
    description = (
        "Default settings for DAISY 2.02 audio book creation: language, "
        "publisher, and whether to include text in audio+text books."
    )

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(
                self,
                label="Default DAISY 2.02 audio book settings for new projects.",
            ),
            0, wx.ALL, 6,
        )

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

        self.include_text_cb = wx.CheckBox(self, label="Include text in audio+text books")
        self.include_text_cb.SetName("Include text")
        self.include_text_cb.SetValue(settings.get("daisy.include_text", True))
        grid.Add((1, 1))  # spacer
        grid.Add(self.include_text_cb, 0, wx.ALL, 2)

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        sizer.Add(
            wx.StaticText(
                self,
                label="Language code (en, hi, fr, de, es) and optional publisher name.",
            ),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)

    def apply_to_settings(self):
        self.settings.set("daisy.language", self.lang_ctrl.GetValue().strip() or "en")
        self.settings.set("daisy.publisher", self.publisher_ctrl.GetValue().strip())
        self.settings.set("daisy.include_text", self.include_text_cb.GetValue())


# ---------------------------------------------------------------------------
# Compute category (optional runtimes / GPU dependency)
# ---------------------------------------------------------------------------
class _ComputePanel(_SettingsPanel):
    title = "Compute"
    description = (
        "Compute back-ends and the optional GPU (CUDA) runtime download."
    )

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
            wx.StaticText(self, label="Compute back-ends and the runtimes they need."),
            0, wx.ALL, 6,
        )
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

        sizer.Add(
            wx.StaticText(self, label="Restart the application after installing or removing "
                                      "the GPU runtime."),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)

        self.gpu_download_btn.Bind(wx.EVT_BUTTON, self._on_download)
        self.gpu_remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.gpu_cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.Bind(EVT_DOWNLOAD_PROGRESS, self._on_progress)
        self.Bind(EVT_DOWNLOAD_FINISHED, self._on_finished)
        self._refresh()

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
    description = (
        "Addon management, pip environment, and diagnostic tools. "
        "Enable Developer Mode in General first."
    )

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self._thread = None
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
        log_row.Add(wx.StaticText(log_box, label="Log file:"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.log_file_combo = wx.ComboBox(log_box, style=wx.CB_READONLY,
                                             name="Select log file to view")
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

    def _refresh_pip(self):
        from ..python_runtime import get_runtime
        rt = get_runtime()
        if rt.is_created:
            pkgs = rt.pip_list()
            self.pip_status.SetLabel(
                f"Environment ready: {len(pkgs)} packages installed "
                f"at {rt.env_dir}"
            )
        else:
            self.pip_status.SetLabel(
                f"Environment not yet created. Will be created at: {rt.env_dir}"
            )

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
        self._refresh_pip()
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
    description = "Restore every setting to its default value."

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Restore every setting in this window to "
                                      "its default value. Downloaded voices are "
                                      "not deleted."),
            0, wx.ALL, 6,
        )
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
