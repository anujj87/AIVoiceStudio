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
2. Download and remove ONNX TTS    -- model manager (DownloadPanel)
3. Available TTS          -- downloaded voices (AvailablePanel)
4. Voice Clone            -- the Voice Lab clone engines *and* the OmniVoice
                             engines (install, voice library, engine cards)
5. Recording settings     -- speed, pitch, volume, preview
6. Punctuation            -- default punctuation mode (spoken-word expansion)
7. Audio file creation    -- audio modes with descriptions + pages per file
8. DAISY settings         -- DAISY 2.02 audio book defaults
9. Compute               -- optional GPU (CUDA) runtime
10. Developer             -- addon management, pip, diagnostics
11. OmniVoice Server      -- network TTS server configuration
12. Reset                 -- restore defaults
13. Hugging Face account  -- the token a gated model (Gemma 3) needs

The 'OmniVoice engines' category that used to follow 'Voice Clone' is merged
into it: both list pip-installed cloning engines, and OmniVoice is installed
from the same page now, exactly like the Voice Lab engines.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import ClassVar

import wx
from wx.lib import scrolledpanel

from .. import gestures as gestures_model
from .. import hf_access, paths, runtime, venv_packages
from ..constants import (
    AUDIO_MODE_CHOICES,
    AUDIO_MODE_DESCRIPTIONS,
    DAISY_SPLIT_CHOICES,
    DAISY_SPLIT_DESCRIPTIONS,
    DAISY_SPLIT_H1,
    MODE_PAGE_ONLY,
    MODE_PAGE_WITH_H1,
    OMNIVOICE_BIND_ALL_INTERFACES,
    OMNIVOICE_BIND_LOOPBACK,
    PAGES_PER_FILE_MAX,
    PAGES_PER_FILE_MIN,
    PITCH_MAX,
    PITCH_MIN,
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
from ..describe import catalog as describe_catalog
from ..describe import engine as describe_engine
from ..describe import weights as describe_weights
from ..settings import Settings
from ..stt import catalog as stt_catalog
from ..stt import engine as stt_engine
from ..tts import catalog, windows_tts
from ..tts.downloader import ModelDownloader
from ..tts.models import ModelStore
from . import access_keys, announce, dialogs, dpi, language_choice
from .a11y import (
    add_labeled,
    finalize_accessibility,
    set_accessible_name,
    update_accessible_name,
)
from .clone_engines_panel import VoiceClonePanel
from .events import (
    EVT_DOWNLOAD_FINISHED,
    EVT_DOWNLOAD_PROGRESS,
    DownloadFinishedEvent,
    DownloadProgressEvent,
)
from .model_panels import AvailablePanel, DownloadPanel
from .progress import TaskProgressDialog
from .theme import (
    PATTERNS,
    all_theme_choices,
    apply_theme,
    get_wallpaper,
    set_wallpaper,
)

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

    def _persisted_now(self) -> None:
        """This panel wrote settings straight to disk (outside Apply/OK).

        The dialog's Cancel snapshot has to move with it, otherwise closing
        the settings panel would silently revert the save.
        """
        dialog = self.GetTopLevelParent()
        mark = getattr(dialog, "_mark_applied", None)
        if callable(mark):
            mark()

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
        except Exception:
            log.debug("Ignoring a failure in _SettingsPanel._on_voices_ready", exc_info=True)

    def _on_package_probe(self, package: str, engine: str | None = None):
        """A managed-venv probe finished; re-add voices when it is installed."""
        try:
            if venv_packages.version(package, engine=engine):
                self._on_voices_ready()
        except Exception:
            log.debug("Ignoring a failure in _SettingsPanel._on_package_probe", exc_info=True)


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
    # classes are defined further down this module).
    #
    # A panel is built the first time its category is shown.  Building all
    # seventeen up front cost 4.5-5.7 s and 392 window handles on the GUI
    # thread, and the heaviest category (Voice Clone, the Voice Lab engines
    # plus the OmniVoice studio) is the one a user opening Settings to change
    # the theme never looks at.  Nothing is lost by waiting: a category that is never
    # opened never writes anything either, so its stored values are already
    # correct - and one that *is* opened is built exactly as before, from the
    # same settings (see ``ensure_panel``).
    CATEGORIES: ClassVar[list] = []

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
        # Measured engine speeds decide which devices the panels list: load
        # them before any panel (the Voice Clone one especially) is built.
        try:
            from ..voicelab import benchmark

            benchmark.load_verdicts(settings)
        except Exception:
            log.debug("Ignoring a failure in SettingsDialog.__init__", exc_info=True)
        self.downloader = ModelDownloader(store)
        self._orig = settings.as_dict()
        #: One slot per category, in ``CATEGORIES`` order; ``None`` until that
        #: category is first shown (or asked for through the accessors below).
        self._panels: list[_SettingsPanel | None] = []
        self._current: int = 0
        #: Last category whose name was announced (see ``_show_category``).
        self._announced_category: int | None = None
        self.CATEGORIES = [
            _GeneralPanel,
            # Appearance sits right below General: it is where the look of the
            # window is chosen now that the theme list holds twenty themes.
            _AppearancePanel,
            DownloadPanel,
            AvailablePanel,
            _VoiceCloneCategoryPanel,
            _OmniVoiceServerPanel,
            _RecordingSettingsPanel,
            _PunctuationPanel,
            _AudioModePanel,
            _DaisySettingsPanel,
            _ManageSTTPanel,
            _ManageDescribePanel,
            _HuggingFacePanel,
            _InputGesturesPanel,
            _ComputePanel,
            _DeveloperPanel,
            _ResetPanel,
        ]

        self._panels = [None] * len(self.CATEGORIES)
        self._build_ui()
        self._bind_events()

        self._show_category(0)
        apply_theme(self, self.settings.theme, self.settings)
        # Real MSAA accNames for every labelled control (SetName alone is
        # ignored by MSAA on this wx build).
        finalize_accessibility(self)
        # Scale-derived geometry for the window that exists now; a category
        # built later scales itself in ``ensure_panel``.
        dpi.prepare(self)
        dpi.follow(self)
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

        # Only the category *names* are built here: the panels themselves wait
        # until a category is first shown (see ``ensure_panel``).
        for cls in self.CATEGORIES:
            self.cat_list.Append((cls.title,))

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
        if cls is _VoiceCloneCategoryPanel:
            return (self.settings, self.store)
        if cls in (_OmniVoiceServerPanel,):
            return (self.settings,)
        if cls in (_RecordingSettingsPanel, _PunctuationPanel):
            return (self.settings, self.store)
        return (self.settings,)

    # -- lazy panels ----------------------------------------------------------
    def panel_count(self) -> int:
        """How many categories the dialog has."""
        return len(self.CATEGORIES)

    def built_panels(self) -> list[_SettingsPanel]:
        """The panels that exist so far, in category order.

        Only these can be validated or saved: a category that was never shown
        has no controls to read and nothing to write, and the values it owns
        are already on disk untouched by this dialog.
        """
        return [panel for panel in self._panels if panel is not None]

    def ensure_panel(self, index: int) -> _SettingsPanel:
        """Return category ``index``'s panel, building it on first use.

        Building means: construct it with the same arguments it always had, add
        it to the scrolled container hidden, give it its MSAA property-page
        accessible, attach real accNames to its controls and paint it with the
        current theme.  Everything that used to happen to all eighteen panels
        at once now happens to one, once.
        """
        if not 0 <= index < len(self._panels):
            raise IndexError(f"no settings category at index {index}")
        existing = self._panels[index]
        if existing is not None:
            return existing
        cls = self.CATEGORIES[index]
        panel = cls(self.container, *self._panel_args(cls))
        panel.Hide()
        panel.description = cls.description
        try:
            panel.SetAccessible(_SettingsPanelAccessible(panel))
        except Exception:
            log.debug("Ignoring a failure in SettingsDialog.ensure_panel", exc_info=True)
        self.container_sizer.Add(
            panel, 1, wx.ALL | wx.EXPAND,
            border=10,  # NVDA BORDER_FOR_DIALOGS
        )
        # The item was added after the dialog was prepared, so its padding is
        # scaled here and joins the dialog's recorded geometry (a later display
        # change scales it too).
        dpi.scale_added_item(self.container_sizer, 10)
        self._panels[index] = panel
        # The dialog's own accessibility step ran before this panel existed, so
        # its controls are named here.  The access-key hook is *not* installed
        # a second time: the dialog already owns one and key events bubble up
        # to it, which is how the panels have always reached it.
        finalize_accessibility(panel, install_access_keys=False)
        # A panel built after the dialog was themed must be painted with it, or
        # it would appear in the system colours next to its dark neighbours.
        try:
            apply_theme(panel, self.settings.theme, self.settings)
        except Exception:
            log.debug("Ignoring a failure in SettingsDialog.ensure_panel", exc_info=True)
        # A category built after the dialog was prepared needs the same
        # scale-derived geometry as the one the dialog opened with.
        dpi.prepare(panel)
        self.container.Layout()
        return panel

    def panels(self) -> list[_SettingsPanel]:
        """Every panel, building the ones that do not exist yet.

        Used by the accessibility audit tools and by tests that sweep all
        eighteen categories; the dialog itself never needs this.
        """
        return [self.ensure_panel(index) for index in range(self.panel_count())]

    def _drop_built_panels(self, keep_index: int | None = None) -> None:
        """Throw the built panels away so they can be rebuilt from settings.

        Used after a reset: the panels that exist show the *old* values, and
        rebuilding them is what makes "the other tabs now show the defaults"
        true instead of merely reassuring.
        """
        for index, panel in enumerate(self._panels):
            if panel is None or index == keep_index:
                continue
            self._panels[index] = None
            self.container_sizer.Detach(panel)
            panel.Destroy()
        self.container.Layout()

    def _settings_reset(self) -> None:
        """Reset every setting and make the visible panels show the defaults."""
        self.settings.reset()
        # The panel that asked stays (it is in the middle of its own event);
        # every other built panel is thrown away and rebuilt on next visit.
        self._drop_built_panels(keep_index=self._current)
        self._orig = self.settings.as_dict()
        apply_theme(self, self.settings.theme, self.settings)
        self.container.Layout()

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
        # The category name is spoken first.  Several panels refresh their
        # status line while they are activated ("11 of 19 models installed",
        # "Server status: Checking...", "N voices in your library"), and those
        # live announcements used to arrive before the name of the category
        # they belong to.  Announcing the name here puts it in front of them:
        # announcements are delivered in order, one at a time.  It is skipped
        # while the user is arrow-keying the category list, where the screen
        # reader already reads the newly focused category.
        #
        # A panel refreshes its status line on every activation, so entering a
        # category a second time must speak its name again; otherwise the
        # panel's message would arrive on its own, without the category in
        # front of it.  Leaving a category therefore clears the record.
        if index != self._current:
            self._announced_category = None
        if index != self._announced_category:
            self._announced_category = index
            # Assertive on purpose: a category change is a discrete event, so
            # its name is delivered now, ahead of the panel's status lines,
            # and the previous category's still-waiting lines are dropped.
            # Announced politely, the name could queue behind - or be the
            # oldest message dropped from - the four-message status burst a
            # panel produces while it is activated, and the listener heard the
            # message with no category in front of it.
            announce.announce(self.CATEGORIES[index].title,
                              assertive=True, key="settings-category")
        # NVDA pattern: Freeze the container during the switch to prevent
        # controls from briefly appearing in wrong positions.
        self.container.Freeze()
        try:
            # The category being shown is built here, the first time it is
            # visited; the others are only *hidden* (a panel that was never
            # built has nothing to hide).
            shown = self.ensure_panel(index)
            for i, panel in enumerate(self._panels):
                if panel is None:
                    continue
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
            panel = shown
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
        _interactive = (wx.TextCtrl, wx.ComboBox, wx.Choice, wx.CheckBox,
                        wx.RadioButton, wx.ListBox, wx.Button, wx.Slider,
                        wx.SpinCtrl)
        stack = list(panel.GetChildren())
        while stack:
            child = stack.pop(0)
            if (isinstance(child, _interactive)
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
            # if the list didn't originally have focus.  Read the slot rather
            # than ``ensure_panel``: a category that was never shown is *not*
            # shown, so this must not build a panel just to ask.
            existing = self._panels[new_index]
            if not list_had_focus and existing is not None and existing.IsShown():
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
        elif self._is_apply_key(evt, key):
            # Ctrl+S runs the same path as the Apply button (and its Alt+A).
            self._apply_now()
        else:
            # NVDA pattern: let wx handle Escape, Alt+F4, and all other keys
            # via its default dialog behaviour.
            evt.Skip()

    # -- callbacks ----------------------------------------------------------

    def _validate_all(self) -> bool:
        """Check every panel that exists is valid (NVDA _validateAllPanels).

        A category the user never opened has no controls and cannot be
        invalid; its settings are the ones already stored, which were
        validated when they were written.  A panel that *was* opened is
        validated exactly as before.
        """
        for panel in self.built_panels():
            try:
                if not hasattr(panel, "isValid"):
                    log.warning("Panel %s has no isValid() method — skipping validation", panel.title)
                    continue
                if panel.isValid() is False:
                    log.warning("Panel %s failed validation", panel.title)
                    return False
            except Exception:
                log.exception("Panel %s raised during validation", panel.title)
                return False
        return True

    def _save_from_ui(self):
        """Validate the built panels, then save (NVDA _doSave pattern).

        Only a panel that was built can write anything, and an unbuilt one
        writes nothing - which is what keeps a value the user set in a
        category they did not open on this visit: it is already in the
        settings and nothing here touches it.
        """
        if not self._validate_all():
            return False
        for panel in self.built_panels():
            try:
                panel.apply_to_settings()
            except Exception:
                log.exception("Panel %s failed to save", panel.title)
        self.settings.save()
        return True

    @staticmethod
    def _is_apply_key(evt, key) -> bool:
        """True when a char-hook event is Ctrl+S, the Apply gesture.

        ``GetUnicodeKey()`` is not dependable here: on this wx/Windows build
        Ctrl+S is delivered with a Unicode key of 0 or of the control
        character 0x13 - not ``"S"`` - so a Unicode-only test made Ctrl+S
        silently do nothing.  The virtual key code is the reliable source,
        with the Unicode key kept as a fallback for builds that fill only
        that in.  Alt is excluded so AltGr (which arrives as Ctrl+Alt+letter)
        keeps typing its own character instead of applying settings.
        """
        if not evt.ControlDown() or evt.AltDown():
            return False
        if key in (ord("S"), ord("s")):
            return True
        try:
            unicode_key = evt.GetUnicodeKey()
        except Exception:
            return False
        return unicode_key in (ord("S"), ord("s"))

    def _apply_now(self) -> bool:
        """Validate, save and re-theme - the one path behind every apply.

        The Apply button, its Alt+A access key and the Ctrl+S accelerator all
        run this, so the three cannot drift apart.  The dialog and its owner
        are re-themed straight away so the change is visible immediately
        rather than only once the dialog closes.
        """
        if not self._save_from_ui():
            log.warning("Settings dialog: Apply failed (validation or save error)")
            return False
        log.info("Settings dialog: Apply succeeded")
        # Everything is on disk now, so Cancel / the window's X button must
        # fall back to *this* state rather than the one the dialog opened
        # with: closing the panel must not undo a change already applied.
        self._mark_applied()
        apply_theme(self, self.settings.theme, self.settings)
        owner = self.GetParent()
        if owner is not None:
            apply_theme(owner, self.settings.theme, self.settings)
            refresh_welcome = getattr(owner, "_apply_welcome_mode", None)
            if callable(refresh_welcome):
                refresh_welcome()
            # A rebound key takes effect straight away, not only after OK.
            reload_gestures = getattr(owner, "_reload_gestures", None)
            if callable(reload_gestures):
                reload_gestures()
        return True

    def _on_apply(self, _):
        log.info("Settings dialog: Apply button clicked")
        if self._apply_now():
            # Stay open — move focus back to the category list so the
            # user can continue configuring (matches NVDA pattern).
            wx.CallAfter(self.cat_list.SetFocus)

    def _on_ok(self, _):
        log.info("Settings dialog: OK button clicked")
        if self._save_from_ui():
            log.info("Settings dialog: OK succeeded — closing")
            self.EndModal(wx.ID_OK)
        else:
            log.warning("Settings dialog: OK failed (validation or save error)")

    def _mark_applied(self) -> None:
        """Move the Cancel snapshot forward to the state just written to disk.

        Apply (the button, its Alt+A access key, Ctrl+S) and every panel that
        saves straight away (a new custom theme, a wallpaper) have already
        persisted those values, so a later Cancel or window close restores
        *this* snapshot - closing the settings panel no longer reverts a
        change that was already applied.  Changes made after the snapshot
        and never applied are still discarded, exactly as before.
        """
        self._orig = self.settings.as_dict()

    def _on_cancel(self, _):
        log.info("Settings dialog: Cancel button clicked — discarding changes")
        # Discard the changes that were never applied: whatever was applied
        # or saved straight away already moved the snapshot forward.
        self.settings.set_many(dict(self._orig.items()))
        self.EndModal(wx.ID_CANCEL)

    # Convenience attributes kept for tests / external code.
    def _panel_by_title(self, title: str):
        """Look a category up by its title, building it if it is not built yet."""
        for index, cls in enumerate(self.CATEGORIES):
            if cls.title == title:
                return self.ensure_panel(index)
        raise KeyError(title)

    @property
    def general_panel(self):
        return self.ensure_panel(0)

    @property
    def appearance_panel(self):
        """The 'Appearance' category (theme, custom themes, wallpaper)."""
        return self._panel_by_title("Appearance")

    @property
    def download_panel(self):
        return self._panel_by_title("Download and remove ONNX TTS")

    @property
    def available_panel(self):
        return self._panel_by_title("Available TTS")

    @property
    def voice_clone_category_panel(self):
        """The 'Voice Clone' category as a whole: both halves on one page."""
        return self._panel_by_title("Voice Clone")

    @property
    def voice_clone_panel(self):
        """The Voice Lab half of the 'Voice Clone' category.

        The category itself is :attr:`voice_clone_category_panel`; this is the
        ``VoiceClonePanel`` inside it, which is what code that drives the
        Voice Lab engines means by "the Voice Clone panel".
        """
        return self.voice_clone_category_panel.voicelab

    @property
    def omnivoice_engines_panel(self):
        """The OmniVoice half of the 'Voice Clone' category.

        'OmniVoice engines' used to be a category of its own; it is now the
        second half of 'Voice Clone' (install buttons, voice library, engine
        cards), so this returns that half for tests and tools that drive it
        directly.
        """
        return self.voice_clone_category_panel.omnivoice

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
    def manage_stt_panel(self):
        return self._panel_by_title("Manage the STT")

    @property
    def manage_describe_panel(self):
        """The 'Manage the video describer' category (audio description)."""
        return self._panel_by_title("Manage the video describer")

    @property
    def hugging_face_panel(self):
        """The 'Hugging Face account' category (the token a gated model needs)."""
        return self._panel_by_title("Hugging Face account")

    @property
    def input_gestures_panel(self):
        """The 'Keyboard gestures' category (view/rebind keyboard gestures)."""
        return self._panel_by_title("Keyboard gestures")

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
# ---------------------------------------------------------------------------
# Appearance category
# ---------------------------------------------------------------------------
class _AppearancePanel(_SettingsPanel):
    """How the window looks: the theme list and the wallpaper.

    These controls used to share the General category with the file locations
    and the update settings.  With twenty predefined themes, the custom-theme
    studio and the wallpaper, appearance earned a category of its own.  Nothing
    was dropped: every control General kept had its own, and each one works
    exactly as before.  The category is built and saved like every other one,
    so a theme chosen here still writes ``theme`` and ``wallpapers``.
    """

    title = "Appearance"
    description = (
        "Choose how the application looks: the theme, the custom theme "
        "studio and the wallpaper of the welcome screen."
    )
    panelDescription = description

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, label="Theme"), 0, wx.ALL, 6)
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.theme_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Theme selector")
        self._theme_choices = all_theme_choices(settings)
        for value, label in self._theme_choices:
            self.theme_combo.Append(label, value)
        self.theme_combo.SetSelection(self._index_for_theme(settings.theme))
        add_labeled(self, grid, "Theme", self.theme_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        self.new_theme_btn = wx.Button(self, label="Create &custom theme...")
        self.new_theme_btn.SetName("Create custom theme")
        self.new_theme_btn.SetToolTip(
            "Design your own theme: pick colours and a style, and add a "
            "wallpaper. You can create as many themes as you want."
        )
        self.new_theme_btn.Bind(wx.EVT_BUTTON, self._on_create_theme)
        sizer.Add(self.new_theme_btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.preview = wx.StaticText(
            self,
            label="Note: the theme applies immediately, to the whole "
                  "application, when you press Apply or OK. The list holds "
                  "System default, the original Light and Dark themes, the "
                  "original ten predefined dark themes, ten vibrant dark "
                  "themes and every theme you create.",
        )
        self.preview.SetToolTip(
            "The vibrant themes (Neon Magenta, Inferno Red, Solar Flare, "
            "Voltage Yellow, Laser Lime, Emerald Surge, Cyber Cyan, Azure "
            "Rush, Electric Indigo, Ultraviolet) are the second group of ten."
        )
        sizer.Add(self.preview, 0, wx.ALL, 6)

        # -- wallpaper ------------------------------------------------------
        sizer.Add(wx.StaticText(self, label="Wallpaper"), 0, wx.ALL, 6)
        self.wallpaper_cb = wx.CheckBox(
            self,
            label="Show a wallpaper in the empty area of the welcome screen",
        )
        self.wallpaper_cb.SetName("Wallpaper")
        self.wallpaper_cb.SetToolTip(
            "Fills the empty area of the welcome screen - the region beside "
            "the project sidebar - with a painted background. The buttons "
            "and the recent projects list keep the theme colour. Works for "
            "the built-in, every predefined and any custom theme."
        )
        self.wallpaper_cb.Bind(wx.EVT_CHECKBOX, self._on_wallpaper)
        sizer.Add(self.wallpaper_cb, 0, wx.LEFT | wx.RIGHT, 6)

        wall_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        wall_grid.AddGrowableCol(1)
        self.wallpaper_image = wx.TextCtrl(self, style=wx.TE_READONLY)
        self.wallpaper_image.SetName("Wallpaper image")
        image_row = wx.BoxSizer(wx.HORIZONTAL)
        image_row.Add(self.wallpaper_image, 1, wx.EXPAND | wx.RIGHT, 6)
        self.wallpaper_browse_btn = wx.Button(self, label="Choose &image...")
        self.wallpaper_browse_btn.SetToolTip(
            "Choose a picture to use as the wallpaper; without one the "
            "theme's own painted style is used."
        )
        self.wallpaper_browse_btn.Bind(wx.EVT_BUTTON, self._on_choose_wallpaper)
        image_row.Add(self.wallpaper_browse_btn, 0, wx.RIGHT, 4)
        self.wallpaper_clear_btn = wx.Button(self, label="C&lear")
        self.wallpaper_clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_wallpaper)
        image_row.Add(self.wallpaper_clear_btn, 0)
        wall_grid.Add(wx.StaticText(self, label="Wallpaper image:"), 0,
                      wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        wall_grid.Add(image_row, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        self.wallpaper_style = wx.ComboBox(self, style=wx.CB_READONLY,
                                           name="Wallpaper background style")
        self.wallpaper_style.Append("(use the theme's own style)", "")
        for value, label in PATTERNS:
            self.wallpaper_style.Append(label, value)
        self.wallpaper_style.Bind(wx.EVT_COMBOBOX, self._on_wallpaper)
        add_labeled(self, wall_grid, "Background style when no image is chosen",
                    self.wallpaper_style, flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(wall_grid, 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="The wallpaper fills only the empty area of the "
                      "welcome screen - the region beside the projects "
                      "sidebar. The buttons and the recent projects list "
                      "keep the theme colour.",
            ),
            0, wx.ALL, 6,
        )
        self._load_wallpaper(settings.theme)

        self.SetSizer(sizer)
        self.theme_combo.Bind(wx.EVT_COMBOBOX, self._on_theme)

    # -- theme / wallpaper --------------------------------------------------
    def _index_for_theme(self, theme: str) -> int:
        for index, (value, _label) in enumerate(self._theme_choices):
            if value == theme:
                return index
        return 0

    def _on_theme(self, _):
        # Persist the wallpaper choices made for the previous theme before the
        # controls are reloaded for the newly selected one.
        self._commit_wallpaper()
        theme = self.selected()
        apply_theme(self, theme, self.settings)
        self._load_wallpaper(theme)

    def _on_create_theme(self, _evt=None):
        from .theme_studio import ThemeStudioDialog

        self._commit_wallpaper()
        dlg = ThemeStudioDialog(self, self.settings)
        try:
            dlg.ShowModal()
            saved = dlg.last_saved_theme_id
        finally:
            dlg.Destroy()
        if not saved:
            return
        self.settings.set("theme", saved)
        self.settings.save()
        # The theme is already on disk; closing the panel must keep it.
        self._persisted_now()
        self._theme_choices = all_theme_choices(self.settings)
        self.theme_combo.Clear()
        for value, label in self._theme_choices:
            self.theme_combo.Append(label, value)
        self.theme_combo.SetSelection(self._index_for_theme(saved))
        apply_theme(self, saved, self.settings)
        self._load_wallpaper(saved)

    def _load_wallpaper(self, theme: str) -> None:
        """Show the wallpaper record of ``theme`` in the controls."""
        self._wallpaper_theme = theme
        record = get_wallpaper(self.settings, theme)
        self.wallpaper_cb.SetValue(bool(record["enabled"]))
        self.wallpaper_image.SetValue(record["image"])
        wanted = record["pattern"]
        selection = 0
        for index in range(self.wallpaper_style.GetCount()):
            if self.wallpaper_style.GetClientData(index) == wanted:
                selection = index
                break
        self.wallpaper_style.SetSelection(selection)

    def _commit_wallpaper(self) -> None:
        theme = getattr(self, "_wallpaper_theme", "")
        if not theme:
            return
        set_wallpaper(
            self.settings, theme,
            enabled=self.wallpaper_cb.GetValue(),
            image=self.wallpaper_image.GetValue().strip(),
            pattern=self.wallpaper_style.GetClientData(
                self.wallpaper_style.GetSelection()) or "",
        )
        # set_wallpaper() persists immediately - keep it on Cancel too.
        self._persisted_now()

    def _repaint_theme(self) -> None:
        top = self.GetTopLevelParent() or self
        apply_theme(top, self.selected(), self.settings)

    def _on_wallpaper(self, _evt=None):
        self._commit_wallpaper()
        self._repaint_theme()

    def _on_choose_wallpaper(self, _evt=None):
        wildcard = (
            "Images (*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp)|"
            "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp|All files (*.*)|*.*"
        )
        with wx.FileDialog(
            self, "Choose a wallpaper image", wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            self.wallpaper_image.SetValue(dlg.GetPath())
        self.wallpaper_cb.SetValue(True)
        self._on_wallpaper()

    def _on_clear_wallpaper(self, _evt=None):
        self.wallpaper_image.SetValue("")
        self._on_wallpaper()

    def selected(self) -> str:
        sel = self.theme_combo.GetSelection()
        return self.theme_combo.GetClientData(sel) if sel >= 0 else THEME_SYSTEM

    def apply_to_settings(self):
        self.settings.set("theme", self.selected())
        self._commit_wallpaper()


# ---------------------------------------------------------------------------
# General category
# ---------------------------------------------------------------------------
class _GeneralPanel(_SettingsPanel):
    """Everything that is not the look of the window.

    The theme and wallpaper controls moved to the Appearance category; what
    stays here is the developer switch, the file locations the application
    writes to and the update behaviour.
    """

    title = "General"
    description = (
        "Developer Mode, the folders the application saves to, and updates."
    )
    panelDescription = description

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Developer Mode checkbox
        self.dev_mode_cb = wx.CheckBox(self, label="Enable Developer Mode")
        self.dev_mode_cb.SetName("Developer Mode")
        self.dev_mode_cb.SetValue(settings.get("developer_mode", False))
        self.dev_mode_cb.SetToolTip(
            "Enables advanced features: heavy logging to diagnostic files, "
            "addon management, and additional developer tools. "
            "Requires restarting the application for full effect."
        )
        sizer.Add(self.dev_mode_cb, 0, wx.ALL, 6)

        sizer.Add(
            wx.StaticText(self, label="File locations"),
            0, wx.ALL, 6,
        )
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.recordings_ctrl = self._path_row(
            grid, "Project save location",
            settings.get("paths.recordings_dir") or paths.recordings_dir(),
            browse_label="Browse project location...",
            browse_prompt="Choose the folder that holds your projects",
        )
        self.models_ctrl = self._path_row(
            grid, "Model files location",
            settings.get("paths.models_dir") or paths.models_dir(),
            browse_label="Browse model location...",
            browse_prompt="Choose the folder that holds downloaded models",
        )
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="New projects, their recordings and downloads are saved "
                      "to these folders. Existing files are not moved.",
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

    def _path_row(self, grid, name: str, current: str,
                  browse_label: str = "Browse...",
                  browse_prompt: str = "") -> wx.TextCtrl:
        """Add ``name`` + read-only path field + its own Browse button.

        Each row names what it browses for ("Browse project location...",
        "Browse model location...") so a screen-reader user hears which folder
        a button changes instead of a list of identical "Browse..." buttons.
        """
        text = wx.TextCtrl(self, style=wx.TE_READONLY)
        text.SetName(name)
        text.SetValue(current)
        browse = wx.Button(self, label=browse_label)
        browse.SetName(browse_label.rstrip("."))
        if browse_prompt:
            browse.SetToolTip(browse_prompt)
        browse.Bind(
            wx.EVT_BUTTON,
            lambda _evt, c=text, n=name, m=browse_prompt: self._on_browse(c, n, m),
        )
        label = wx.StaticText(self, label=name + ":")
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(text, 1, wx.EXPAND)
        row.Add(browse, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6)
        grid.Add(row, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        return text

    def _on_browse(self, ctrl: wx.TextCtrl, name: str, message: str = ""):
        with wx.DirDialog(
            self,
            message=message or ("Choose folder for " + name.lower()),
            defaultPath=ctrl.GetValue() or paths.user_data_dir(),
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                ctrl.SetValue(dlg.GetPath())

    def _on_check_updates_now(self, _evt=None):
        """Run a check straight from Settings (Help has the same action)."""
        from .update_dialog import check_for_updates_interactive

        check_for_updates_interactive(self, self.settings)

    def apply_to_settings(self):
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
        from .compute_choice import make_compute_row

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
            from ..omnivoice import voice_store
            if voice_store.omni_custom_voices(self.store):
                installed = voice_store.engine_ids_installed()
                self._voices.extend(
                    voice_store.consumer_entries(self.store, installed)
                )
        except Exception:
            log.debug("Ignoring a failure in _RecordingSettingsPanel._populate_voices", exc_info=True)
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
            announce.status_error(self.preview_status,
                "No voices downloaded. Use the 'Download and remove ONNX TTS' tab first."
            )
            self._load_defaults_for_tts(None)
        self._refresh_voice_list()

    def _inject_pip_installed_voices(self):
        """Inject voices for TTS engines installed via pip (e.g. OmniVoice)."""
        try:
            # Voice Lab engines (Pocket TTS, F5-TTS): their pre-made
            # voices are provided by the installed package rather than by a
            # catalog variant.
            from ..voicelab import builtin_voice_entries

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
        except Exception:
            log.debug("Ignoring a failure in "
                      "_RecordingSettingsPanel._inject_pip_installed_voices",
                      exc_info=True)

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
        announce.status_text(self.preview_status,
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
        slider.SetValue(round(float(value) * 100))
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
        announce.status_text(self.preview_status, "Synthesizing preview...")
        # Snapshot the compute choice on the UI thread; the job runs on a worker.
        from .compute_choice import _combo_value

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
            text = process_punctuation(text, punct, voice.get("engine"))
            samples = engine.synthesize(
                text,
                sid=voice.get("sid", 0),
                speed=rate,
                pitch=pitch,
                volume=volume,
            )
            import tempfile

            fd, tmp = tempfile.mkstemp(prefix="aivs_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except Exception as exc:
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}")

    def _preview_done(self, tmp, error):
        import wx.adv

        self.preview_btn.Enable()
        if error:
            announce.status_error(self.preview_status, error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()
        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            # Keep a reference: wxSound must outlive Play(SOUND_ASYNC) or the
            # preview is cut off before it is heard (garbage collection bug).
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            announce.status_text(self.preview_status, "Playing preview.")
        else:
            announce.status_error(self.preview_status, "Preview file could not be played.")

    def _stop_preview_sound(self):
        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:
                log.debug("Ignoring a failure in _RecordingSettingsPanel._stop_preview_sound", exc_info=True)
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
        # An engine can read expression markup of its own (Kitten TTS 2.0:
        # [emotion], <event>, (((emphasis)))).  Those spans are director's notes
        # to it rather than punctuation, so no mode touches them - which the
        # example below shows as soon as such a voice is selected.
        sizer.Add(
            wx.StaticText(
                self,
                label="Expression tags (Kitten TTS 2.0's [emotion], <event> and "
                      "(((emphasis)))) are kept exactly as typed, whichever mode "
                      "is chosen.",
            ),
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
        from .compute_choice import make_compute_row

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
        # The example's fate depends on the engine that would speak it - one
        # with expression markup keeps its tags - so it follows the whole voice
        # cascade, not only the mode and the text.
        self.variant_combo.Bind(wx.EVT_COMBOBOX, self._on_variant_changed)
        self.voice_combo.Bind(wx.EVT_COMBOBOX, self._update_example)
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
            from ..omnivoice import voice_store
            if voice_store.omni_custom_voices(self.store):
                installed = voice_store.engine_ids_installed()
                self._voices.extend(
                    voice_store.consumer_entries(self.store, installed)
                )
        except Exception:
            log.debug("Ignoring a failure in _PunctuationPanel._populate_voices", exc_info=True)
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
            announce.status_error(self.preview_status,
                "No voices downloaded. Use the 'Download and remove ONNX TTS' tab first."
            )

    def _inject_pip_installed_voices(self):
        """Inject voices for TTS engines installed via pip (e.g. OmniVoice).

        These engines don't use the artifact download system, so their
        voices never appear in ``store.installed_voices()``.  We check if
        the package is installed in the *managed venv* (not the main
        process) and, if so, create voice entries from the catalog.
        """
        try:
            # Voice Lab engines (Pocket TTS, F5-TTS): their pre-made
            # voices are provided by the installed package rather than by a
            # catalog variant.
            from ..voicelab import builtin_voice_entries

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
        except Exception:
            log.debug("Ignoring a failure in _PunctuationPanel._inject_pip_installed_voices", exc_info=True)

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
        # A different engine may treat the example's markup differently.
        self._update_example()

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
    def _on_variant_changed(self, event=None):
        """Re-run the voice cascade, then re-read the example for that engine."""
        self._on_variant(event)
        self._update_example()

    def _preview_engine(self):
        """The engine of the voice the Preview would speak with (may be None)."""
        voice = self.selected_voice() or {}
        return voice.get("engine")

    def _update_example(self, _evt=None):
        """Show what the *selected voice's* engine is about to be given.

        The engine matters: one that reads expression markup keeps it, so the
        line updates when the voice cascade changes as well as when the mode or
        the text does.
        """
        from ..tts.engine import process_punctuation

        mode = self.selected()
        text = self.example_text.GetValue() or "Hello."
        self.example_out.SetLabel(
            process_punctuation(text, mode, self._preview_engine())
        )

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
        announce.status_text(self.preview_status, "Synthesizing preview...")
        mode = self.selected()
        text = self.example_text.GetValue() or "Hello."
        # Snapshot the compute choice on the UI thread; the job runs on a worker.
        from .compute_choice import _combo_value

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
            text = process_punctuation(text, mode, voice.get("engine"))
            samples = engine.synthesize(text, sid=voice.get("sid", 0), speed=1.0)
            import tempfile

            fd, tmp = tempfile.mkstemp(prefix="aivs_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except Exception as exc:
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}")

    def _preview_done(self, tmp, error):
        import wx.adv

        self.preview_btn.Enable()
        if error:
            announce.status_error(self.preview_status, error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()

        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            announce.status_text(self.preview_status, "Playing preview.")
        else:
            announce.status_error(self.preview_status, "Preview file could not be played.")

    def _stop_preview_sound(self):
        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:
                log.debug("Ignoring a failure in _PunctuationPanel._stop_preview_sound", exc_info=True)
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
# OmniVoice engines (the second half of the Voice Clone category)
# ---------------------------------------------------------------------------
class _OmniVoiceEnginesPanel(_SettingsPanel):
    title = "OmniVoice engines"

    """OmniVoice Engines panel - part of the 'Voice Clone' category.

    Three parts:

    0. **Packages** - **Install / Remove OmniVoice dependency**
       (``omnivoice-triton``) and **Install / Remove OmniVoice Server**
       (``omnivoice-server``), with their install status.  Both are pip
       packages in the shared ``tts_envs\\omnivoice`` environment, so they
       are installed here like the Voice Lab engines rather than in Compute.

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
        #: Announcements start when the half is shown: the page shows either
        #: this half or the Voice Lab one, and building the hidden half must
        #: not make the screen reader read controls nobody can see.
        self._activated = False

        sizer = wx.BoxSizer(wx.VERTICAL)
        self._engines: list = []  # list of dicts for each engine variant

        # -- Dependency shortcut -------------------------------------------
        # OmniVoice is two pip packages, so it is installed from right here
        # (like the Voice Lab engines) instead of from the Compute category.
        self._build_install_section(sizer)

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
        self.engine_label = add_labeled(
            self, grid, "TTS version", self.engine_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )
        self.mode_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Voice creation mode")
        self.mode_combo.Append("Clone a voice from a sample", "clone")
        self.mode_combo.Append("Describe a new voice", "design")
        self.mode_combo.SetSelection(0)
        add_labeled(self, grid, "Creation mode", self.mode_combo,
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
        add_labeled(self, grid, "Language", self.omni_language_combo,
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
        from .compute_choice import make_compute_row

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
        from ..omnivoice import voice_store

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
        self._say(self.lib_status,
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
        from ..omnivoice import voice_store

        engine_id = self._selected_engine_id()
        label, pkg = voice_store.ENGINE_INFO.get(engine_id, (engine_id, ""))
        if engine_id in self._installed:
            state = f"Installed ({pkg}). Preview will use it."
        else:
            state = (f"Not installed yet (pip package {pkg}). Created voices "
                     "still work with the other engine; install this one "
                     "with the Install buttons above to preview with it.")
        self.mode_tip.SetLabel(self._mode_tip_text() + "\n" + f"{label}. {state}")
        self.mode_tip.Wrap(640)
        # The engine's name and install state come first, then the details -
        # the listener has just picked the engine and needs to know which one
        # it is and whether it can speak before hearing anything else.
        if self._activated:
            announce.announce(f"{label}. {state}")
        self.Layout()

    def select_engine(self, engine_id: str) -> bool:
        """Select an OmniVoice engine by id; False when it is not listed."""
        for index in range(self.engine_combo.GetCount()):
            if self.engine_combo.GetClientData(index) == engine_id:
                self.engine_combo.SetSelection(index)
                self._on_engine_change(None)
                return True
        return False

    def collapse_engine_row(self) -> None:
        """Hide this panel's own engine combo (the category page has one)."""
        for control in (self.engine_label, self.engine_combo):
            control.Hide()
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
        from ..omnivoice import voice_store

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
        announce.status_text(
            status,
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
        from ..omnivoice import voice_store

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
        announce.status_text(self.lib_status, f"Renamed to '{renamed['name']}'.")

    def _on_delete(self, _evt):
        from ..omnivoice import voice_store

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
            announce.status_text(self.lib_status, f"Deleted '{voice['name']}'.")

    # -------------------------------------------------------------- preview
    def _on_preview(self, evt=None):
        # One Alt+P press reaches this handler more than once (see
        # access_keys.once): the guard keeps it to one preview.
        if not access_keys.once(evt if evt is not None else self.preview_btn):
            return
        from ..omnivoice import voice_store

        voice = self._selected_library_voice()
        if not voice:
            return
        engine_id = self._selected_engine_id()
        label, _pkg = voice_store.ENGINE_INFO.get(engine_id, (engine_id, ""))
        entry = voice_store.preview_entry(voice, engine_id)
        text = self.preview_text.GetValue().strip() or self._PREVIEW_DEFAULT
        self.preview_btn.Disable()
        self.preview_status_label = label
        announce.status_text(self.lib_status,
            f"Synthesizing a preview with {label}... (first use loads the "
            "model and can take a minute or two.)"
        )
        # Snapshot the compute choice on the UI thread; the job runs on a worker.
        from .compute_choice import _combo_value, provider_for_preview

        choice = _combo_value(self.compute_combo)
        provider = provider_for_preview(choice, entry.get("engine"))
        threading.Thread(
            target=self._preview_job, args=(entry, text, label, provider),
            daemon=True,
        ).start()

    def _preview_job(self, entry, text, engine_label, provider="cuda"):
        from ..audio.output import write_wav
        from ..tts.engine import (
            EngineUnavailableError,
            get_engine,
            process_punctuation,
        )

        try:
            engine = get_engine(entry, provider=provider)
            punct = self.settings.get("recording.punctuation", "default")
            samples = engine.synthesize(
                process_punctuation(text, punct, entry.get("engine")),
                sid=0, speed=1.0,
            )
            import tempfile

            fd, tmp = tempfile.mkstemp(prefix="aivs_voice_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None, engine_label)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc), engine_label)
        except Exception as exc:
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}", engine_label)

    def _preview_done(self, tmp, error, engine_label):
        import wx.adv

        self.preview_btn.Enable()
        if error:
            announce.status_error(self.lib_status, error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()
        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            # Keep a reference: wxSound must outlive Play(SOUND_ASYNC) or the
            # preview is cut off before it is heard (garbage collection bug).
            self._preview_sound = sound
            sound.Play(wx.adv.SOUND_ASYNC)
            announce.status_text(self.lib_status, f"Playing preview ({engine_label}).")
        else:
            announce.status_error(self.lib_status, "Preview file could not be played.")

    def _stop_preview_sound(self):
        sound = getattr(self, "_preview_sound", None)
        if sound is not None:
            try:
                sound.Stop()
            except Exception:
                log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._stop_preview_sound", exc_info=True)
            self._preview_sound = None

    # -- packages (OmniVoice dependency + server) ---------------------------
    def _build_install_section(self, sizer):
        """Install / remove both OmniVoice packages from this page.

        OmniVoice is two pip packages (``omnivoice-triton`` for the direct
        engine and ``omnivoice-server`` for the HTTP server), so they are
        installed here exactly like the Voice Lab engines - the Compute
        category now only carries the ONNX GPU (CUDA) runtime.
        """
        box = wx.StaticBox(self, label="OmniVoice packages")
        inner = wx.StaticBoxSizer(box, wx.VERTICAL)
        tip = wx.StaticText(
            box,
            label="Both engines install into one shared environment "
                  "(tts_envs\\omnivoice) and need an NVIDIA GPU with CUDA. "
                  "The first install downloads PyTorch with CUDA (~2 GB).",
        )
        tip.Wrap(640)
        inner.Add(tip, 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.dependency_status = wx.StaticText(box, label="Checking...")
        self.dependency_status.SetName("OmniVoice dependency status")
        add_labeled(box, grid, "OmniVoice dependency", self.dependency_status,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.server_status = wx.StaticText(box, label="Checking...")
        self.server_status.SetName("OmniVoice Server status")
        add_labeled(box, grid, "OmniVoice Server", self.server_status,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        inner.Add(grid, 0, wx.EXPAND | wx.ALL, 4)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.omnivoice_install_btn = wx.Button(box, label="Install OmniVoice dependency")
        self.omnivoice_install_btn.SetName("Install OmniVoice dependency")
        self.omnivoice_install_btn.SetToolTip(
            "Install omnivoice-triton (pip install) into the OmniVoice "
            "environment (shared with OmniVoice Server). Requires an NVIDIA "
            "GPU with CUDA."
        )
        self.omnivoice_remove_btn = wx.Button(box, label="Remove OmniVoice dependency")
        self.omnivoice_remove_btn.SetName("Remove OmniVoice dependency")
        self.omnivoice_remove_btn.SetToolTip(
            "Uninstall omnivoice-triton (pip uninstall)."
        )
        self.server_install_btn = wx.Button(box, label="Install OmniVoice Server")
        self.server_install_btn.SetName("Install OmniVoice Server")
        self.server_install_btn.SetToolTip(
            "Install omnivoice-server via pip (includes PyTorch, omnivoice, etc.)"
        )
        self.server_remove_btn = wx.Button(box, label="Remove OmniVoice Server")
        self.server_remove_btn.SetName("Remove OmniVoice Server")
        self.server_remove_btn.SetToolTip(
            "Uninstall omnivoice-server and its dependencies"
        )
        for btn in (self.omnivoice_install_btn, self.omnivoice_remove_btn,
                    self.server_install_btn, self.server_remove_btn):
            btns.Add(btn, 0, wx.ALL, 4)
        inner.Add(btns, 0, wx.ALL, 4)
        sizer.Add(inner, 0, wx.EXPAND | wx.ALL, 6)

        self.omnivoice_install_btn.Bind(wx.EVT_BUTTON, self._on_omnivoice_install)
        self.omnivoice_remove_btn.Bind(wx.EVT_BUTTON, self._on_omnivoice_remove)
        self.server_install_btn.Bind(wx.EVT_BUTTON, self._on_server_install)
        self.server_remove_btn.Bind(wx.EVT_BUTTON, self._on_server_remove)
        self._refresh_packages()

    def _refresh_packages(self):
        """Refresh both package rows from the cached probe results."""
        self._refresh_omnivoice()
        self._refresh_omnivoice_server()
        self.Layout()

    @staticmethod
    def _omnivoice_torch_ready() -> bool:
        """True when the shared OmniVoice environment already has PyTorch.

        The direct OmniVoice engine and the HTTP server install into the *same*
        environment (same model, same CUDA PyTorch), so whichever is installed
        second finds PyTorch already there.
        """
        try:
            return bool(venv_packages.installed("torch", engine="omnivoice"))
        except Exception:
            return False

    def _refresh_omnivoice(self):
        """Show the omnivoice-triton status (its own venv, then the shared one)."""
        package = "omnivoice-triton"
        env_id = "omnivoice"
        version = venv_packages.version(package, engine=env_id)
        if version is None:
            version = venv_packages.version(package)
        if version:
            self._say(self.dependency_status,
                f"Installed (v{version}). OmniVoice TTS is available."
            )
            self.omnivoice_install_btn.Disable()
            self.omnivoice_remove_btn.Enable()
        else:
            self._say(self.dependency_status,
                "Not installed. OmniVoice TTS is unavailable."
            )
            self.omnivoice_install_btn.Enable()
            self.omnivoice_remove_btn.Disable()
            if not venv_packages.is_known(package, engine=env_id):
                venv_packages.request(
                    package, self._on_dependency_probe, engine=env_id,
                )

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
            self._say(self.server_status,
                f"Installed (v{version}). OmniVoice Server is available."
            )
            self.server_install_btn.Disable()
            self.server_remove_btn.Enable()
        else:
            self._say(self.server_status,
                "Not installed. OmniVoice Server is unavailable."
            )
            self.server_install_btn.Enable()
            self.server_remove_btn.Disable()
            if not venv_packages.is_known(package, engine=env_id):
                venv_packages.request(
                    package, self._on_dependency_probe, engine=env_id,
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
        import subprocess as _sp
        import tempfile as _tmp
        cmd = [pip_exe, "install", "--no-warn-script-location", *args]
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
        import sys as _sys

        from ..python_runtime import get_runtime
        from ..voicelab import engines as voice_lab_engines
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
        except Exception as exc:
            wx.CallAfter(self._omnivoice_install_done, False, f"Installation failed: {exc}")

    def _ov_update_label(self, msg: str):
        """Update the progress dialog label from a background thread."""
        try:
            if self._ov_install_dlg and self._ov_install_label:
                self._ov_install_label.SetLabel(msg)
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._ov_update_label", exc_info=True)

    def _omnivoice_install_done(self, success, message):
        """Called on the GUI thread when pip install finishes."""
        # Close progress dialog
        try:
            if self._ov_install_dlg:
                self._ov_install_dlg.Hide()
                self._ov_install_dlg.Destroy()
                self._ov_install_dlg = None
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._omnivoice_install_done", exc_info=True)
        venv_packages.invalidate("omnivoice-triton", engine="omnivoice")
        self._refresh_dependency_state()
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
        from ..python_runtime import get_runtime
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
        except Exception as exc:
            wx.CallAfter(self._omnivoice_remove_done, False, f"Removal failed: {exc}")

    def _omnivoice_remove_done(self, success, message):
        """Called on the GUI thread when pip uninstall finishes."""
        try:
            if self._ov_install_dlg:
                self._ov_install_dlg.Hide()
                self._ov_install_dlg.Destroy()
                self._ov_install_dlg = None
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._omnivoice_remove_done", exc_info=True)
        venv_packages.invalidate("omnivoice-triton", engine="omnivoice")
        self._refresh_dependency_state()
        wx.MessageBox(
            message,
            "OmniVoice",
            style=(wx.OK | wx.ICON_INFORMATION) if success else (wx.OK | wx.ICON_ERROR),
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
        import sys as _sys

        from ..python_runtime import get_runtime
        from ..voicelab import engines as voice_lab_engines
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
        except Exception as exc:
            wx.CallAfter(self._server_install_done, False, f"Installation failed: {exc}")

    def _srv_update_label(self, msg: str):
        """Update the server progress dialog label from a background thread."""
        try:
            if self._srv_install_dlg and self._srv_install_label:
                self._srv_install_label.SetLabel(msg)
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._srv_update_label", exc_info=True)

    def _server_install_done(self, success, message):
        """Called on the GUI thread when pip install finishes."""
        try:
            if self._srv_install_dlg:
                self._srv_install_dlg.Hide()
                self._srv_install_dlg.Destroy()
                self._srv_install_dlg = None
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._server_install_done", exc_info=True)
        venv_packages.invalidate("omnivoice-server", engine="omnivoice_server")
        self._refresh_dependency_state()
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
        from ..python_runtime import get_runtime
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
        except Exception as exc:
            wx.CallAfter(self._server_remove_done, False, f"Removal failed: {exc}")

    def _server_remove_done(self, success, message):
        """Called on the GUI thread when pip uninstall finishes."""
        try:
            if self._srv_install_dlg:
                self._srv_install_dlg.Hide()
                self._srv_install_dlg.Destroy()
                self._srv_install_dlg = None
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._server_remove_done", exc_info=True)
        venv_packages.invalidate("omnivoice-server", engine="omnivoice_server")
        self._refresh_dependency_state()
        wx.MessageBox(
            message,
            "OmniVoice Server",
            style=(wx.OK | wx.ICON_INFORMATION) if success else (wx.OK | wx.ICON_ERROR),
        )

    def _refresh_engine_status(self):
        """Update every engine card's status from the shared package cache."""
        for eng in self._engines:
            package = eng["package"]
            env_id = eng["env"]
            version = venv_packages.version(package, engine=env_id)
            if version is None:
                version = venv_packages.version(package)
            if version:
                self._say(eng["status_label"],
                    f"Installed (v{version}) — ready to use"
                )
                eng["installed"] = True
            else:
                self._say(eng["status_label"],
                    "Not installed — use the Install buttons above"
                )
                eng["installed"] = False
                if not venv_packages.is_known(package, engine=env_id):
                    venv_packages.request(
                        package, self._on_dependency_probe, engine=env_id
                    )
        self._installed = [eng["id"] for eng in self._engines if eng["installed"]]

    def _refresh_dependency_state(self):
        """Refresh the package rows, the engine cards and the library."""
        self._refresh_packages()
        self._refresh_engine_status()
        self._refresh_library()

    def _on_dependency_probe(self, _version=None):
        """A background package probe finished (worker thread)."""
        try:
            wx.CallAfter(self._refresh_dependency_state)
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceEnginesPanel._on_dependency_probe", exc_info=True)

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
        self.mark_activated()
        self._refresh_dependency_state()
        self._on_engine_change(None)
        self.Layout()

    def mark_activated(self) -> None:
        """Announce status from now on (this is the half being shown)."""
        self._activated = True

    def _say(self, label, message: str) -> None:
        """Put ``message`` in ``label``; announce it once the half is open."""
        label.SetLabel(message)
        if self._activated:
            announce.announce(message)

    def apply_to_settings(self):
        pass  # No settings to save for this panel


# ---------------------------------------------------------------------------
# Voice Clone category (Voice Lab engines + OmniVoice studio, one page)
# ---------------------------------------------------------------------------
#: The OmniVoice engines in the Voice Clone page's one engine list.
_OMNIVOICE_ENGINES: tuple[tuple[str, str], ...] = (
    ("OmniVoice Server", "omnivoice_server"),
    ("OmniVoice (direct, Triton)", "omnivoice"),
)
_OMNIVOICE_ENGINE_IDS = frozenset(
    engine_id for _label, engine_id in _OMNIVOICE_ENGINES
)


class _VoiceCloneCategoryPanel(_SettingsPanel):
    """The 'Voice Clone' category: every pip-installed cloning engine.

    One engine list, one page.  The list carries both families - the Voice Lab
    engines (:class:`VoiceClonePanel`) first, then the OmniVoice engines - and
    the controls of the chosen family follow below it:

    1. the **Voice Lab** half (:class:`VoiceClonePanel`) with its install,
       device and voice controls, and
    2. the **OmniVoice** half (:class:`_OmniVoiceEnginesPanel`): the Install /
       Remove buttons for ``omnivoice-triton`` and ``omnivoice-server``, the
       engine cards and the shared voice library.

    The two halves used to carry a combo box each.  They are one list now, so
    an engine is picked once rather than twice; everything else in both halves
    - install and remove buttons, device list, sample box, voice library and
    engine cards - is unchanged and follows the choice.

    'OmniVoice engines' used to be a category of its own; it was merged into
    this one so the install buttons sit beside the other cloning engines'
    install buttons instead of in the Compute category, which now only holds
    the ONNX GPU (CUDA) runtime.
    """

    title = "Voice Clone"
    #: Deliberately empty: the page is driven by its engine list, so a long
    #: paragraph read on entry only delays the control the user came for.  The
    #: intro text below says the same thing, and the engine status says what
    #: applies to the engine that is selected.
    description = ""

    def __init__(self, parent, settings: Settings, store: ModelStore):
        from ..voicelab import engines as voice_lab

        super().__init__(parent)
        self.settings = settings
        self.store = store

        sizer = wx.BoxSizer(wx.VERTICAL)

        # -- one engine list for both families ------------------------------
        box = wx.StaticBox(self, label="Engine")
        engine_box = wx.StaticBoxSizer(box, wx.VERTICAL)
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.engine_combo = wx.ComboBox(box, style=wx.CB_READONLY,
                                        name="TTS engine")
        add_labeled(box, grid, "TTS engine", self.engine_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        engine_box.Add(grid, 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(engine_box, 0, wx.EXPAND | wx.ALL, 6)

        for info in voice_lab.ENGINES:
            self.engine_combo.Append(info["name"], info["id"])
        for label, engine_id in _OMNIVOICE_ENGINES:
            self.engine_combo.Append(label, engine_id)

        # -- the Voice Lab engines -----------------------------------------
        self.voicelab = VoiceClonePanel(self, settings, store)
        sizer.Add(self.voicelab, 0, wx.EXPAND)

        # -- the OmniVoice half --------------------------------------------
        self.omni_line = wx.StaticLine(self)
        sizer.Add(self.omni_line, 0, wx.EXPAND | wx.ALL, 6)
        self.omni_title = wx.StaticText(
            self, label="OmniVoice engines (Server, Triton, Hybrid)"
        )
        self.omni_title.SetFont(self.omni_title.GetFont().Bold())
        sizer.Add(self.omni_title, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.omni_tip = wx.StaticText(
            self,
            label="OmniVoice is the second family of cloning engines: 600+ "
                  "languages, voice design from a description and zero-shot "
                  "cloning. Install it here, then create a voice below - the "
                  "voice works with both engines and everywhere in the app.",
        )
        self.omni_tip.Wrap(660)
        sizer.Add(self.omni_tip, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self.omnivoice = _OmniVoiceEnginesPanel(self, settings, store)
        sizer.Add(self.omnivoice, 0, wx.EXPAND)

        self.SetSizer(sizer)

        # The halves keep their own selectors for their own internals (and the
        # tests and tools that drive them directly); only this page's list is
        # shown, so there is one "which TTS engine" control on the page.
        self.voicelab.collapse_engine_row()
        self.omnivoice.collapse_engine_row()

        saved = (self.settings.get("voice_clone.engine", "")
                 or self.settings.get("clone_engines.engine", ""))
        self._show_engine(saved or self._engine_value(0), refresh=False)
        self.engine_combo.Bind(wx.EVT_COMBOBOX, self._on_engine_select)

    # -- the one engine list -----------------------------------------------
    def _engine_value(self, index: int) -> str:
        return str(self.engine_combo.GetClientData(index) or "")

    def selected_engine_id(self) -> str:
        """The engine chosen on this page (Voice Lab or OmniVoice)."""
        index = self.engine_combo.GetSelection()
        if index < 0:
            return self._engine_value(0) if self.engine_combo.GetCount() else ""
        return self._engine_value(index)

    def _on_engine_select(self, _evt):
        self._show_engine(self.selected_engine_id())

    def _show_engine(self, engine_id: str, *, refresh: bool = True) -> None:
        """Show the half that owns ``engine_id`` and select it there.

        ``refresh`` is off while the page is being built: both halves have
        already selected their own engine and announced its status, so doing
        it again would only repeat those announcements.
        """
        is_omni = engine_id in _OMNIVOICE_ENGINE_IDS
        for index in range(self.engine_combo.GetCount()):
            if self._engine_value(index) == engine_id:
                self.engine_combo.SetSelection(index)
                break
        self.voicelab.Show(not is_omni)
        for control in (self.omni_line, self.omni_title, self.omni_tip,
                        self.omnivoice):
            control.Show(is_omni)
        if refresh and is_omni:
            self.omnivoice.mark_activated()
            self.omnivoice.select_engine(engine_id)
        elif refresh:
            self.voicelab.mark_activated()
            self.voicelab.select_engine(engine_id)
        self.Layout()
        parent = self.GetParent()
        if parent is not None:
            parent.Layout()

    # -- the two halves behave as one category -----------------------------
    #
    # The halves keep the attribute names they always had (``engine_combo`` on
    # the Voice Lab half, ``compute_combo`` on the OmniVoice half) and are read
    # through the two attributes above, rather than by forwarding attribute
    # lookups: a forwarding ``__getattr__`` would make ``mock.patch.object`` and
    # ``hasattr`` answer about the wrapper while the real control lives in a
    # half, which is a trap rather than a convenience.
    def on_activated(self):
        super().on_activated()
        # Only the visible half refreshes: the hidden one would announce its own
        # status lines ("Not installed...", "N voices in your library") for a
        # family the user is not looking at.
        if self.voicelab.IsShown():
            self.voicelab.on_activated()
        else:
            self.omnivoice.on_activated()
        self.Layout()

    def apply_to_settings(self):
        self.voicelab.apply_to_settings()
        self.omnivoice.apply_to_settings()
        # Remember the engine picked here, whichever family it belongs to, so
        # the page opens on it next time.
        self.settings.set("voice_clone.engine", self.selected_engine_id())

    def isValid(self) -> bool:
        return self.voicelab.isValid()


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
            "Requires omnivoice-server to be installed (Settings > Voice Clone)."
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
        self.host_ctrl.SetValue(ov_cfg.get("host", OMNIVOICE_BIND_LOOPBACK))
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
        """When network access is checked, suggest the all-interfaces bind address."""
        if self.network_cb.GetValue():
            self.host_ctrl.SetValue(OMNIVOICE_BIND_ALL_INTERFACES)
        else:
            self.host_ctrl.SetValue(OMNIVOICE_BIND_LOOPBACK)

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
        host = self.host_ctrl.GetValue().strip() or OMNIVOICE_BIND_LOOPBACK
        port = self.port_ctrl.GetValue()
        announce.status_text(self.status_label, "Server status: Checking...")
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
            from ..omnivoice_server import get_server_manager
            mgr = get_server_manager(host=host, port=port)
            # A subprocess we started ourselves answers without a round
            # trip; anything else gets one short health probe.
            running = bool(mgr.process_alive) or mgr.health_check(
                retries=1, delay=0.0, timeout=2.0
            )
        except Exception:
            running = False
        try:
            wx.CallAfter(self._status_ready, running, host, port, epoch)
        except Exception:
            log.debug("Ignoring a failure in _OmniVoiceServerPanel._probe_status", exc_info=True)

    def _status_ready(self, running: bool, host: str, port: int, epoch: int):
        """Main thread: apply the result of a background status probe."""
        if epoch != self._status_epoch:
            # A start/stop happened while this probe was in flight: its
            # answer is about the old state, so drop it.
            return
        try:
            self._status_probe_running = False
            if running:
                announce.status_text(self.status_label,
                    f"Server status: Running at http://{host}:{port}"
                )
                self.start_btn.Disable()
                self.stop_btn.Enable()
            else:
                announce.status_text(self.status_label, "Server status: Not running")
                self.start_btn.Enable()
                self.stop_btn.Disable()
            self._set_profiles_enabled(running)
            if running:
                self._on_refresh_profiles(None)
            self.Layout()
        except Exception:
            # The dialog (and this panel) may already be gone; nothing to do.
            log.debug("Ignoring a failure in _OmniVoiceServerPanel._status_ready", exc_info=True)

    def _on_start(self, _):
        """Start the OmniVoice server in a background thread."""
        self.start_btn.Disable()
        self.stop_btn.Disable()
        announce.status_text(self.status_label, "Starting server...")
        self._thread = threading.Thread(target=self._start_job, daemon=True)
        self._thread.start()

    def _start_job(self):
        from ..omnivoice_server import get_server_manager
        host = self.host_ctrl.GetValue().strip() or OMNIVOICE_BIND_LOOPBACK
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
        except Exception as exc:
            wx.CallAfter(self._start_done, False, f"Failed to start server: {exc}")

    def _start_done(self, success, message):
        self._thread = None
        # Starting/stopping changes the truth the status probe was asking
        # about, so any probe still in flight is discarded.
        self._status_epoch += 1
        self._status_probe_running = False
        announce.status_text(self.status_label, f"Server status: {message}")
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
            from ..omnivoice_server import get_server_manager
            mgr = get_server_manager()
            mgr.stop()
            self._status_epoch += 1
            self._status_probe_running = False
            announce.status_text(self.status_label, "Server status: Stopped")
            self.start_btn.Enable()
            self.stop_btn.Disable()
            self._set_profiles_enabled(False)
        except Exception as exc:
            announce.status_error(self.status_label, f"Server status: Error stopping: {exc}")

    def _on_test(self, _):
        """Test the server connection."""
        try:
            from ..omnivoice_server import get_server_manager
            host = self.host_ctrl.GetValue().strip() or OMNIVOICE_BIND_LOOPBACK
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
        except Exception as exc:
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
        from ..omnivoice_server import get_server_manager
        host = self.host_ctrl.GetValue().strip() or OMNIVOICE_BIND_LOOPBACK
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
            announce.status_text(self.profiles_status,
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
        announce.status_text(self.profiles_status, "Refreshing profiles...")
        threading.Thread(target=self._refresh_profiles_job, daemon=True).start()

    def _refresh_profiles_job(self):
        try:
            mgr = self._server_manager()
            entries = mgr.list_profiles()
            wx.CallAfter(self._profiles_loaded, entries, None)
        except Exception as exc:
            wx.CallAfter(self._profiles_loaded, None, str(exc))

    def _profiles_loaded(self, entries, error):
        self.refresh_profiles_btn.Enable(True)
        self._profiles = [e for e in (entries or []) if e.get("profile_id")]
        self.profile_list.DeleteAllItems()
        for i, entry in enumerate(self._profiles):
            self.profile_list.InsertItem(i, entry.get("profile_id", ""))
            self.profile_list.SetItem(i, 1, entry.get("description") or "")
        if error:
            announce.status_error(self.profiles_status, "Could not load profiles.")
            wx.MessageBox(
                f"Could not load voice profiles: {error}",
                "OmniVoice profiles", style=wx.OK | wx.ICON_ERROR,
            )
        else:
            announce.status_text(self.profiles_status,
                f"{len(self._profiles)} profile(s) stored on the server."
            )
        self.delete_profile_btn.Enable(
            self._selected_profile_id() is not None
        )

    def _on_add_profile(self, _):
        """Save the reference audio as a named profile on the server."""
        import re

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
        announce.status_text(self.profiles_status, f"Saving profile '{profile_id}'...")
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
        except Exception as exc:
            wx.CallAfter(self._profile_saved, profile_id, str(exc))

    def _profile_saved(self, profile_id, error):
        self.add_profile_btn.Enable(True)
        if error:
            announce.status_error(self.profiles_status, "Could not save the profile.")
            wx.MessageBox(
                f"Could not save profile '{profile_id}': {error}",
                "OmniVoice profiles", style=wx.OK | wx.ICON_ERROR,
            )
            return
        self.profile_id_ctrl.SetValue("")
        self.profile_audio_ctrl.SetValue("")
        self.profile_ref_text_ctrl.SetValue("")
        announce.status_text(self.profiles_status, f"Profile '{profile_id}' saved.")
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
        announce.status_text(self.profiles_status, f"Deleting profile '{profile_id}'...")
        threading.Thread(
            target=self._delete_profile_job, args=(profile_id,), daemon=True
        ).start()

    def _delete_profile_job(self, profile_id):
        try:
            mgr = self._server_manager()
            ok = mgr.delete_profile(profile_id)
            wx.CallAfter(self._profile_deleted, profile_id, None if ok else "not found")
        except Exception as exc:
            wx.CallAfter(self._profile_deleted, profile_id, str(exc))

    def _profile_deleted(self, profile_id, error):
        if error:
            announce.status_error(self.profiles_status, "Could not delete the profile.")
            wx.MessageBox(
                f"Could not delete profile '{profile_id}': {error}",
                "OmniVoice profiles", style=wx.OK | wx.ICON_ERROR,
            )
            return
        announce.status_text(self.profiles_status, f"Profile '{profile_id}' deleted.")
        self._on_refresh_profiles(None)

    def apply_to_settings(self):
        """Save server settings."""
        self.settings.set("omnivoice_server.enabled", self.enable_cb.GetValue())
        self.settings.set("omnivoice_server.auto_start", self.auto_start_cb.GetValue())
        self.settings.set("omnivoice_server.host",
                          self.host_ctrl.GetValue().strip() or OMNIVOICE_BIND_LOOPBACK)
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

    The OmniVoice packages are *not* installed here: they are pip packages,
    so they install from Settings -> Voice Clone beside the other cloning
    engines (see ``_VoiceCloneCategoryPanel``).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

        sizer = wx.BoxSizer(wx.VERTICAL)
        runtime_gb = runtime.DOWNLOAD_SIZE_MB / 1024
        sizer.Add(
            wx.StaticText(
                self,
                label="CPU is always available. GPU (CUDA) requires "
                      f"downloading the optional runtime (~{runtime_gb:.0f} GB).",
            ),
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
        self.gpu_download_btn.SetToolTip(
            "Download the optional CUDA GPU runtime (~1.8 GB) for "
            "GPU-accelerated synthesis"
        )
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

        self.SetSizer(sizer)

        self.gpu_download_btn.Bind(wx.EVT_BUTTON, self._on_download)
        self.gpu_remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.gpu_cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.Bind(EVT_DOWNLOAD_PROGRESS, self._on_progress)
        self.Bind(EVT_DOWNLOAD_FINISHED, self._on_finished)

        self._refresh()

    def on_activated(self):
        super().on_activated()
        self._refresh()

    def _refresh(self):
        installed = runtime.is_installed()
        if installed:
            version = runtime.installed_version()
            announce.status_text(self.gpu_status,
                f"Installed ({version or 'ready'}). The GPU back-end will be "
                "active after the application restarts."
            )
        else:
            announce.status_text(self.gpu_status,
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
        announce.status_text(self.progress_label, "Downloading the GPU runtime...")
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
        announce.status_text(self.progress_label, "Cancelling...")

    def _on_progress(self, evt: DownloadProgressEvent):
        done, total = evt.done, evt.total
        if total > 0:
            self.gauge.SetValue(int(min(100, done * 100 / total)))
        announce.status_text(self.progress_label,
            f"Downloading {evt.name}: {done / (1024 * 1024):.1f} MB"
            + (f" of {total / (1024 * 1024):.1f} MB" if total else "")
        )

    def _on_finished(self, evt: DownloadFinishedEvent):
        self._thread = None
        self.gpu_cancel_btn.Disable()
        announce.status_text(self.progress_label, evt.message)
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
            announce.status_text(self.progress_label,
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
            announce.status_text(self.pip_status,
                f"Environment not yet created. Will be created at: {rt.env_dir}"
            )
            return
        if not force:
            cached = _PIP_STATUS_CACHE.get(rt.env_dir)
            if cached is not None:
                announce.status_text(self.pip_status, cached)
                return
        if self._pip_probe_running:
            return
        self._pip_probe_running = True
        announce.status_text(self.pip_status, "Checking the Python environment...")
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
        except Exception:
            label = None
        try:
            wx.CallAfter(self._pip_ready, rt.env_dir, label)
        except Exception:
            log.debug("Ignoring a failure in _DeveloperPanel._probe_pip", exc_info=True)

    def _pip_ready(self, env_dir: str, label):
        """Main thread: apply the result of a background environment probe."""
        self._pip_probe_running = False
        if label:
            _PIP_STATUS_CACHE[env_dir] = label
        try:
            announce.status_text(self.pip_status,
                label if label else "Could not read the Python environment."
            )
        except Exception:
            log.debug("Ignoring a failure in _DeveloperPanel._pip_ready", exc_info=True)

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
                announce.status_text(self.pip_status, "Installing...")
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
        import platform

        from ..heavy_logging import log_system_info
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
            with open(log_path, encoding="utf-8", errors="replace") as fh:
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
class _ManageSTTPanel(_SettingsPanel):
    """Settings > Manage the STT: the speech-to-text model table.

    Lists every speech-to-text model the application knows about with its
    licence, language coverage, whether it fits a CPU or a GPU of up to 12 GB,
    and a short note.  Selecting a row reports whether a back-end for it is
    installed in the managed Python environment and, when it is not, the
    package that has to be installed.
    """

    title = "Manage the STT"
    description = (
        "Every speech-to-text model with its licence, languages, CPU or GPU "
        "fit and notes.  Select a model to see whether it can run now."
    )
    panelDescription = description

    _COLUMNS = (
        ("Status", 110),
        ("Model", 170),
        ("License", 110),
        ("Languages", 120),
        ("GPU/CPU Fit", 140),
        ("Notes", 340),
    )

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, label="Speech-to-text models"),
                  0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="Choose a Speech to text project base in the New "
                      "Project wizard to transcribe with these models.  Each "
                      "model runs from the managed Python environment; the "
                      "status line under the table says what has to be "
                      "installed for the selected model.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        self.table = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL,
            size=wx.Size(-1, 280),
        )
        for index, (label, width) in enumerate(self._COLUMNS):
            self.table.InsertColumn(index, label, width=width)
        for model_id, row in zip(stt_catalog.model_ids(),
                                 stt_catalog.summary_rows(), strict=False):
            item = self.table.InsertItem(
                self.table.GetItemCount(), self._status_text(model_id),
            )
            for column, value in enumerate(row, start=1):
                self.table.SetItem(item, column, value)
        set_accessible_name(self.table, "Speech-to-text models")
        self.table.Bind(wx.EVT_LIST_ITEM_SELECTED, lambda _e: self._show_status())
        self.table.Bind(wx.EVT_LIST_ITEM_FOCUSED, lambda _e: self._show_status())
        sizer.Add(self.table, 1, wx.EXPAND | wx.ALL, 6)

        # One button pair per model - each model is installed into its own
        # environment, so installing one never touches another's packages.
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.install_btn = wx.Button(self, label="&Install selected model")
        self.install_btn.SetName("Install selected model")
        self.install_btn.SetToolTip(
            "Install this model's back-end into its own Python environment"
        )
        self.remove_btn = wx.Button(self, label="&Remove selected model")
        self.remove_btn.SetName("Remove selected model")
        self.remove_btn.SetToolTip(
            "Delete this model's own Python environment and its packages"
        )
        buttons.Add(self.install_btn, 0, wx.ALL, 4)
        buttons.Add(self.remove_btn, 0, wx.ALL, 4)
        sizer.Add(buttons, 0, wx.ALL, 4)

        self.status = wx.StaticText(self, label="")
        set_accessible_name(self.status, "Selected model status")
        self.status.Wrap(700)
        sizer.Add(self.status, 0, wx.ALL, 6)

        self.env_label = wx.StaticText(self, label="")
        set_accessible_name(self.env_label, "Selected model environment")
        self.env_label.Wrap(700)
        sizer.Add(self.env_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self._install_dialog = None
        self._install_cancel = None
        self.install_btn.Bind(wx.EVT_BUTTON, self._on_install)
        self.remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)

        if self.table.GetItemCount():
            self.table.Select(0)
            self.table.Focus(0)
        self._show_status()
        self.SetSizer(sizer)

    # -- selection ---------------------------------------------------------
    def _selected_model(self):
        index = self.table.GetFirstSelected()
        if index < 0:
            index = self.table.GetFocusedItem()
        ids = stt_catalog.model_ids()
        if 0 <= index < len(ids):
            return ids[index]
        return None

    def _show_status(self) -> None:
        model_id = self._selected_model()
        if not model_id:
            announce.status_text(self.status, "Select a model.")
            self.env_label.SetLabel("")
            self._refresh_buttons()
            return
        entry = stt_catalog.model(model_id)
        note = f"{entry['name']}: {stt_engine.availability_note(model_id)}"
        missing = stt_engine.gpu_runtime_missing(model_id)
        if missing:
            note += (
                "\nThe GPU runtime this model needs is missing ("
                + ", ".join(missing)
                + "); without it a GPU run falls back to the CPU. Use "
                "Install selected model to repair it."
            )
        announce.status_text(self.status, note)
        self.env_label.SetLabel(
            f"Own environment: {stt_engine.env_dir(model_id)}"
        )
        self._refresh_buttons()

    @staticmethod
    def _status_text(model_id: str) -> str:
        """'Installed' / 'Not installed' for this model's own environment."""
        return "Installed" if stt_engine.is_installed(model_id) else "Not installed"

    def _refresh_table(self) -> None:
        """Re-read the Status column from the environment probes on disk."""
        count = self.table.GetItemCount()
        for index, model_id in enumerate(stt_catalog.model_ids()):
            if index < count:
                self.table.SetItem(index, 0, self._status_text(model_id))

    def _refresh_buttons(self) -> None:
        model_id = self._selected_model()
        installed = bool(model_id) and stt_engine.is_installed(model_id)
        created = bool(model_id) and stt_engine.env_exists(model_id)
        # A model whose environment exists but is missing the CUDA runtime its
        # GPU path needs (an environment installed before that wheel joined the
        # catalog) can be repaired by installing again.
        repair = bool(model_id) and installed and bool(
            stt_engine.gpu_runtime_missing(model_id)
        )
        self.install_btn.SetLabel(
            "&Repair selected model" if repair else "&Install selected model"
        )
        self.install_btn.Enable(bool(model_id) and (not installed or repair))
        self.remove_btn.Enable(bool(model_id) and (created or installed))

    def on_activated(self):
        super().on_activated()
        self._refresh_table()
        self._show_status()

    # -- install / remove --------------------------------------------------
    def _on_install(self, _):
        model_id = self._selected_model()
        if not model_id:
            return
        name = stt_catalog.model(model_id)["name"]
        packages = ", ".join(stt_engine.missing_packages(model_id))
        missing = stt_engine.gpu_runtime_missing(model_id)
        if missing:
            packages += ((", " if packages else "") + ", ".join(missing))
        answer = wx.MessageBox(
            f"Install {name} into its own Python environment?\n\n"
            f"Environment: {stt_engine.env_dir(model_id)}\n"
            f"Packages: {packages}\n\n"
            "This may download several gigabytes.",
            "Install model",
            style=wx.YES_NO | wx.ICON_QUESTION,
        )
        if answer != wx.YES:
            return
        name = stt_catalog.model(model_id)["name"]
        self._install_cancel = threading.Event()
        self._install_dialog = TaskProgressDialog(
            self, "Install model", f"Preparing {name}...",
            cancel_label="&Cancel", on_cancel=self._on_install_cancel,
        )
        self._install_dialog.start_pulse()
        self._install_dialog.Show()
        self.install_btn.Disable()
        self.remove_btn.Disable()
        threading.Thread(
            target=self._install_job, args=(model_id,), daemon=True,
            name=f"aivs-stt-install-{model_id}",
        ).start()

    def _on_install_cancel(self) -> None:
        if self._install_cancel is not None:
            self._install_cancel.set()

    def _install_job(self, model_id: str) -> None:
        def progress(message: str) -> None:
            wx.CallAfter(self._install_progress, message)

        result = stt_engine.install(
            model_id, progress=progress, cancel_event=self._install_cancel,
        )
        wx.CallAfter(self._install_done, model_id, result)

    def _install_progress(self, message: str) -> None:
        dialog = self._install_dialog
        if dialog is not None:
            try:
                dialog.set_message(message)
            except Exception:
                log.debug("Ignoring a failure in _ManageSTTPanel._install_progress", exc_info=True)

    def _install_done(self, model_id: str, result) -> None:
        dialog = self._install_dialog
        self._install_dialog = None
        self._install_cancel = None
        if dialog is not None:
            dialog.finish()
        name = stt_catalog.model(model_id)["name"]
        self._refresh_table()
        self._show_status()
        if result.get("ok"):
            wx.MessageBox(
                f"{name} is installed in its own environment and ready to "
                "transcribe.",
                "Install model", style=wx.OK | wx.ICON_INFORMATION,
            )
        else:
            error = str(result.get("error") or "Install failed.")
            wx.MessageBox(
                f"{name} could not be installed:\n\n{error}",
                "Install model", style=wx.OK | wx.ICON_ERROR,
            )

    def _on_remove(self, _):
        model_id = self._selected_model()
        if not model_id:
            return
        name = stt_catalog.model(model_id)["name"]
        if wx.MessageBox(
            f"Delete the Python environment of {name}?\n\n"
            f"{stt_engine.env_dir(model_id)}",
            "Remove model", style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        note = stt_engine.remove(model_id)
        self._refresh_table()
        self._show_status()
        if note:
            wx.MessageBox(note, "Remove model", style=wx.OK | wx.ICON_INFORMATION)


# ---------------------------------------------------------------------------
# Audio description models
# ---------------------------------------------------------------------------
class _ManageDescribePanel(_SettingsPanel):
    """Settings > Manage the video describer: the description model table.

    A describing model is a *vision-language* model - it answers with a short,
    objective description of what it sees - so this category lists the model,
    the licence, and then, for the chosen model, every **variant** with the two
    numbers that decide whether it can run on the user's machine: how big the
    download is and how much video memory it needs.

    Two different things are managed here, and keeping them apart is the point:

    * the **environment** - the packages the model runs on (``transformers``,
      PyTorch, a multimodal processor).  It belongs to the *model*: every
      variant of it shares one, and *Install model environment* builds it; and
    * the **checkpoint** - the weights of one variant alone.  *Download
      selected variant* fetches those now; without it they arrive during the
      first description of that variant.  *Remove variant download* deletes
      them again, leaving the model's environment and its other variants' files
      alone.

    A machine should only carry the checkpoints it can run, so there is
    deliberately no "download every variant" button.
    """

    title = "Manage the video describer"
    description = (
        "Every video-description model with its licence, download size and the "
        "video memory each variant needs, so you can pick by your hardware.  "
        "Each model runs in its own Python environment."
    )
    panelDescription = description

    _COLUMNS = (
        ("Variant", 220),
        ("Size", 90),
        ("VRAM", 90),
        ("Fit", 90),
        ("Status", 150),
        ("Note", 260),
    )

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self._syncing = False
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, label="Video description models"),
                  0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="Choose an Audio description project base in the New "
                      "Project wizard to describe videos with these models.  "
                      "Pick a model, then the variant whose download size and "
                      "video memory suit your machine; the Status column says "
                      "whether that variant's files are already on disk, and "
                      "the environment line under the table says what still "
                      "has to be installed.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        # Model + variant combos: the variant is the actual choice, because two
        # variants of one model differ by gigabytes of weights.
        model_row = wx.BoxSizer(wx.HORIZONTAL)
        model_row.Add(wx.StaticText(self, label="Video description model"), 0,
                      wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.model_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Video description model")
        for model_id in describe_catalog.model_ids():
            self.model_combo.Append(self._model_label(model_id), model_id)
        model_row.Add(self.model_combo, 1, wx.EXPAND)
        sizer.Add(model_row, 0, wx.EXPAND | wx.ALL, 6)

        variant_row = wx.BoxSizer(wx.HORIZONTAL)
        variant_row.Add(wx.StaticText(self, label="Model variant"), 0,
                        wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Model variant")
        variant_row.Add(self.variant_combo, 1, wx.EXPAND)
        sizer.Add(variant_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # Per-variant numbers: the whole point of this table.
        self.table = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL,
            size=wx.Size(-1, 180),
        )
        for index, (label, width) in enumerate(self._COLUMNS):
            self.table.InsertColumn(index, label, width=width)
        set_accessible_name(self.table, "Model variants with size and video memory")
        sizer.Add(self.table, 1, wx.EXPAND | wx.ALL, 6)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.install_btn = wx.Button(self, label="&Install model environment")
        self.install_btn.SetName("Install model environment")
        self.install_btn.SetToolTip(
            "Build this model's own Python environment (transformers, PyTorch, "
            "a multimodal processor).  Every variant of this model shares it."
        )
        self.remove_btn = wx.Button(self, label="Remove model &environment")
        self.remove_btn.SetName("Remove model environment")
        self.remove_btn.SetToolTip(
            "Delete this model's own Python environment and its packages.  "
            "Downloaded checkpoint files are kept; remove those with Remove "
            "variant download."
        )
        # A variant's checkpoint is a separate thing from the model's
        # environment, and it is the half that used to have no button at all: a
        # model whose environment was installed could neither fetch nor drop the
        # weights of another of its variants.
        self.download_btn = wx.Button(self, label="&Download selected variant")
        self.download_btn.SetName("Download selected variant")
        self.remove_variant_btn = wx.Button(self, label="Remove variant download")
        self.remove_variant_btn.SetName("Remove variant download")
        self.remove_variant_btn.SetToolTip(
            "Delete the downloaded checkpoint files of the selected variant"
        )
        buttons.Add(self.install_btn, 0, wx.ALL, 4)
        buttons.Add(self.remove_btn, 0, wx.ALL, 4)
        buttons.Add(self.download_btn, 0, wx.ALL, 4)
        buttons.Add(self.remove_variant_btn, 0, wx.ALL, 4)
        sizer.Add(buttons, 0, wx.ALL, 4)

        self.status = wx.StaticText(self, label="")
        set_accessible_name(self.status, "Selected model status")
        self.status.Wrap(700)
        sizer.Add(self.status, 0, wx.ALL, 6)

        self.env_label = wx.StaticText(self, label="")
        set_accessible_name(self.env_label, "Selected model environment")
        self.env_label.Wrap(700)
        sizer.Add(self.env_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self._install_dialog = None
        self._install_cancel = None
        self._download_dialog = None
        self._download_cancel = None
        self.install_btn.Bind(wx.EVT_BUTTON, self._on_install)
        self.remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.download_btn.Bind(wx.EVT_BUTTON, self._on_download)
        self.remove_variant_btn.Bind(wx.EVT_BUTTON, self._on_remove_variant)
        self.model_combo.Bind(wx.EVT_COMBOBOX, lambda _e: self._on_model())
        self.variant_combo.Bind(wx.EVT_COMBOBOX, lambda _e: self._on_variant())
        self.table.Bind(wx.EVT_LIST_ITEM_SELECTED,
                        lambda _e: self._on_variant_row())
        self.table.Bind(wx.EVT_LIST_ITEM_FOCUSED,
                        lambda _e: self._on_variant_row())

        if self.model_combo.GetCount():
            self.model_combo.SetSelection(0)
        self._on_model()
        self.SetSizer(sizer)

    # -- selection ---------------------------------------------------------
    @staticmethod
    def _model_label(model_id: str) -> str:
        """A model's combo label, marked when it still has to be installed."""
        label = describe_catalog.model(model_id)["name"]
        if not describe_engine.is_installed(model_id):
            label += " (not installed)"
        return label

    def _selected_model(self):
        sel = self.model_combo.GetSelection()
        ids = describe_catalog.model_ids()
        if 0 <= sel < len(ids):
            return ids[sel]
        return ids[0] if ids else None

    def _selected_variant(self):
        sel = self.variant_combo.GetSelection()
        if sel < 0:
            return ""
        return self.variant_combo.GetClientData(sel) or ""

    @staticmethod
    def _combo_index(combo: wx.ComboBox, value) -> int:
        for index in range(combo.GetCount()):
            if combo.GetClientData(index) == value:
                return index
        return -1

    def _on_model(self) -> None:
        """Refill the variant combo and the per-variant table."""
        model_id = self._selected_model()
        self.variant_combo.Clear()
        self.table.DeleteAllItems()
        if model_id:
            for value, label in describe_catalog.variants(model_id):
                self.variant_combo.Append(label, value)
            # Re-select the variant the wizard last used for this model.
            remembered = str(self.settings.get("describe.variant", "") or "")
            wanted = remembered if self._combo_index(
                self.variant_combo, remembered) >= 0 else describe_catalog.default_variant(
                model_id)
            index = self._combo_index(self.variant_combo, wanted)
            if index < 0 and self.variant_combo.GetCount():
                index = 0
            if index >= 0:
                self._syncing = True
                try:
                    self.variant_combo.SetSelection(index)
                finally:
                    self._syncing = False
            for value, label, size, vram, fit_label, note in (
                    describe_catalog.variant_rows(model_id)):
                item = self.table.InsertItem(self.table.GetItemCount(), label)
                self.table.SetItem(item, 1, size)
                self.table.SetItem(item, 2, vram)
                self.table.SetItem(item, 3, fit_label)
                # The environment is shared by every variant of a model; the
                # checkpoint is not, so each row says which of the two it is
                # waiting for.
                self.table.SetItem(
                    item, 4,
                    describe_engine.variant_state_label(model_id, value),
                )
                self.table.SetItem(item, 5, note)
                self.table.SetItemData(item, index)
        self._select_variant_row(self._selected_variant())
        self._show_status()

    def _on_variant(self) -> None:
        if self._syncing:
            return
        self._select_variant_row(self._selected_variant())
        self._show_status()

    def _on_variant_row(self) -> None:
        """Selecting a table row selects that variant in the combo."""
        if self._syncing:
            return
        row = self.table.GetFirstSelected()
        if row < 0:
            row = self.table.GetFocusedItem()
        if row < 0:
            return
        index = self._combo_index(self.variant_combo, self._row_variant(row))
        if index < 0 or index == self.variant_combo.GetSelection():
            return
        self._syncing = True
        try:
            self.variant_combo.SetSelection(index)
        finally:
            self._syncing = False
        self._show_status()

    def _row_variant(self, row: int):
        """The variant value of a table row (its index in the combo)."""
        index = self.table.GetItemData(row)
        if index is None or index < 0 or index >= self.variant_combo.GetCount():
            return None
        return self.variant_combo.GetClientData(index)

    def _select_variant_row(self, value) -> None:
        """Select the table row whose variant is ``value``."""
        target = -1
        for row in range(self.table.GetItemCount()):
            if self._row_variant(row) == value:
                target = row
                break
        if target < 0:
            return
        self._syncing = True
        try:
            if self.table.GetItemCount():
                self.table.Select(target)
                self.table.Focus(target)
        finally:
            self._syncing = False

    # -- status ------------------------------------------------------------
    def _show_status(self) -> None:
        model_id = self._selected_model()
        if not model_id:
            announce.status_text(self.status, "Select a model.")
            self.env_label.SetLabel("")
            self._refresh_buttons()
            return
        entry = describe_catalog.model(model_id)
        variant = self._selected_variant()
        lines = [f"{entry['name']}: {describe_engine.availability_note(model_id)}"]
        if variant:
            lines.append(
                f"Selected variant: {variant} - download about "
                f"{describe_catalog.variant_size_gb(model_id, variant):.1f} GB, "
                "video memory needed about "
                f"{describe_catalog.variant_vram_gb(model_id, variant):.1f} GB."
            )
            # The two layers in words, because they are what the four buttons
            # act on: the environment belongs to the model, the checkpoint to
            # this variant alone.
            lines.append(
                "This variant's checkpoint files: "
                + describe_engine.variant_state_label(model_id, variant)
                + ".  The environment is shared by every variant of this "
                "model; the checkpoint is not."
            )
        note = describe_catalog.hallucination_note(model_id)
        if note:
            lines.append("Hallucination: " + note)
        gpu_missing = describe_engine.gpu_runtime_missing(model_id)
        if gpu_missing:
            lines.append(
                "The extra packages a GPU run needs are missing ("
                + ", ".join(gpu_missing)
                + ").  A checkpoint larger than this machine's card is loaded "
                "in 4-bit, which needs them, so a GPU run of such a variant "
                "fails until they are installed.  Use Repair model environment "
                "to install them."
            )
        # The packages the model's own files import are a separate matter from
        # the GPU extras, and the one failure that made this necessary is
        # MiniCPM-o 2.6: an environment that predates its ``torchaudio`` and
        # ``librosa`` requirements reports itself installed and then fails on the
        # checkpoint's first import.  Naming them here is what makes Repair
        # reachable instead of a Remove-and-rebuild.
        declared_missing = [
            name for name in describe_engine.missing_packages(model_id)
            if name not in gpu_missing
        ]
        if declared_missing:
            lines.append(
                "Its environment is missing "
                + ", ".join(declared_missing)
                + ", which this model needs to load.  Use Repair model "
                "environment to install what is missing."
            )
        announce.status_text(self.status, "\n".join(lines))
        self.env_label.SetLabel(
            f"Own environment: {describe_engine.env_dir(model_id)}"
        )
        self._refresh_buttons()

    def _refresh_statuses(self) -> None:
        """Re-read the '(not installed)' suffixes from the environment probes."""
        current = self._selected_model()
        self._syncing = True
        try:
            for index, model_id in enumerate(describe_catalog.model_ids()):
                if index < self.model_combo.GetCount():
                    self.model_combo.SetString(index, self._model_label(model_id))
        finally:
            self._syncing = False
        if current:
            self._on_model()

    def _refresh_buttons(self) -> None:
        """Drive all four buttons from the two layers they act on.

        The environment pair (install / remove) is about the model; the variant
        pair (download / remove download) is about the selected checkpoint.  A
        model whose environment exists but whose selected variant's files do not
        therefore still offers a download - which is exactly the state this
        panel used to leave the user stuck in.
        """
        model_id = self._selected_model()
        variant = self._selected_variant()
        installed = bool(model_id) and describe_engine.is_installed(model_id)
        created = bool(model_id) and describe_engine.env_exists(model_id)
        repair = bool(model_id) and installed and bool(
            describe_engine.missing_packages(model_id)
            or describe_engine.gpu_runtime_missing(model_id)
        )
        busy = self._busy()
        downloaded = self._variant_downloaded(model_id, variant)
        self.install_btn.SetLabel(
            "&Repair model environment" if repair
            else "&Install model environment"
        )
        self.install_btn.Enable(bool(model_id) and (not installed or repair)
                                and not busy)
        self.remove_btn.Enable(bool(model_id) and (created or installed)
                               and not busy)
        self.download_btn.Enable(bool(model_id) and bool(variant) and installed
                                 and not downloaded and not busy)
        self.remove_variant_btn.Enable(downloaded and not busy)
        self._explain_variant_buttons(model_id, variant, installed, downloaded)

    def _busy(self) -> bool:
        """True while an install or a download is running (one at a time)."""
        return (self._install_dialog is not None
                or self._download_dialog is not None)

    @staticmethod
    def _variant_downloaded(model_id, variant) -> bool:
        if not model_id or not variant:
            return False
        try:
            return describe_engine.variant_downloaded(model_id, variant)
        except Exception:
            return False

    def _explain_variant_buttons(self, model_id, variant, installed: bool,
                                 downloaded: bool) -> None:
        """Put the reason a variant button cannot act into its tooltip.

        A disabled button and a key that answers to it look identical from the
        keyboard (nothing happens), so the reason is kept where a mouse user
        sees it and a screen reader can read it.
        """
        if not model_id or not variant:
            self.download_btn.SetToolTip(
                "Download selected variant - choose a model and a variant "
                "first."
            )
        elif not installed:
            self.download_btn.SetToolTip(
                "Download selected variant - unavailable: install this model's "
                "environment first, because that is the environment that "
                "carries the downloader."
            )
        elif downloaded:
            self.download_btn.SetToolTip(
                f"Download selected variant - {variant} is already on disk."
            )
        else:
            self.download_btn.SetToolTip(
                "Fetch this variant's checkpoint files now, instead of at the "
                "first description.  The environment is shared by every "
                "variant; the checkpoint is not."
            )
        self.remove_variant_btn.SetToolTip(
            "Delete the downloaded checkpoint files of the selected variant"
            if downloaded else
            "Remove variant download - unavailable: this variant's files have "
            "not been downloaded."
        )

    def on_activated(self):
        super().on_activated()
        self._refresh_statuses()
        self._show_status()

    # -- install / remove --------------------------------------------------
    def _on_install(self, _):
        model_id = self._selected_model()
        if not model_id:
            return
        name = describe_catalog.model(model_id)["name"]
        variant = self._selected_variant()
        size = describe_catalog.variant_size_gb(model_id, variant)
        vram = describe_catalog.variant_vram_gb(model_id, variant)
        packages = ", ".join(describe_engine.missing_packages(model_id))
        missing = describe_engine.gpu_runtime_missing(model_id)
        if missing:
            packages += ((", " if packages else "") + ", ".join(missing))
        answer = wx.MessageBox(
            f"Install {name} into its own Python environment?\n\n"
            f"Environment: {describe_engine.env_dir(model_id)}\n"
            f"Download size: {size:.1f} GB\n"
            f"Video memory needed: {vram:.1f} GB\n"
            f"Packages: {packages}\n\n"
            "This may download several gigabytes, and it is per model: every "
            "variant of this model shares this environment.  A variant's own "
            "checkpoint files are separate - use Download selected variant to "
            "fetch them now, or they arrive during that variant's first "
            "description.",
            "Install model",
            style=wx.YES_NO | wx.ICON_QUESTION,
        )
        if answer != wx.YES:
            return
        self._install_cancel = threading.Event()
        self._install_dialog = TaskProgressDialog(
            self, "Install model", f"Preparing {name}...",
            cancel_label="&Cancel", on_cancel=self._on_install_cancel,
        )
        self._install_dialog.start_pulse()
        self._install_dialog.Show()
        self.install_btn.Disable()
        self.remove_btn.Disable()
        threading.Thread(
            target=self._install_job, args=(model_id,), daemon=True,
            name=f"aivs-describe-install-{model_id}",
        ).start()

    def _on_install_cancel(self) -> None:
        if self._install_cancel is not None:
            self._install_cancel.set()

    def _install_job(self, model_id: str) -> None:
        def progress(message: str) -> None:
            wx.CallAfter(self._install_progress, message)

        result = describe_engine.install(
            model_id, progress=progress, cancel_event=self._install_cancel,
        )
        wx.CallAfter(self._install_done, model_id, result)

    def _install_progress(self, message: str) -> None:
        dialog = self._install_dialog
        if dialog is not None:
            try:
                dialog.set_message(message)
            except Exception:
                log.debug("Ignoring a failure in _ManageDescribePanel._install_progress", exc_info=True)

    def _install_done(self, model_id: str, result) -> None:
        dialog = self._install_dialog
        self._install_dialog = None
        self._install_cancel = None
        if dialog is not None:
            dialog.finish()
        name = describe_catalog.model(model_id)["name"]
        self._refresh_statuses()
        self._show_status()
        if result.get("ok"):
            wx.MessageBox(
                f"{name} is installed in its own environment and ready to "
                "describe video.",
                "Install model", style=wx.OK | wx.ICON_INFORMATION,
            )
        else:
            error = str(result.get("error") or "Install failed.")
            wx.MessageBox(
                f"{name} could not be installed:\n\n{error}",
                "Install model", style=wx.OK | wx.ICON_ERROR,
            )

    def _on_remove(self, _):
        model_id = self._selected_model()
        if not model_id:
            return
        name = describe_catalog.model(model_id)["name"]
        if wx.MessageBox(
            f"Delete the Python environment of {name}?\n\n"
            f"{describe_engine.env_dir(model_id)}",
            "Remove model", style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        note = describe_engine.remove(model_id)
        self._refresh_statuses()
        self._show_status()
        if note:
            wx.MessageBox(note, "Remove model environment",
                          style=wx.OK | wx.ICON_INFORMATION)

    # -- one variant's checkpoint files ------------------------------------
    def _on_download(self, _):
        """Fetch the selected variant's checkpoint files, without describing."""
        model_id = self._selected_model()
        variant = self._selected_variant()
        if not model_id or not variant:
            return
        name = describe_catalog.model(model_id)["name"]
        repo = describe_engine.variant_repo(model_id, variant)
        size = describe_catalog.variant_size_gb(model_id, variant)
        vram = describe_catalog.variant_vram_gb(model_id, variant)
        answer = wx.MessageBox(
            f"Download the checkpoint files of {name}, variant {variant}?\n\n"
            f"Repository: {repo}\n"
            f"Download size: {size:.1f} GB\n"
            f"Video memory needed: {vram:.1f} GB\n"
            f"Kept in: {describe_weights.path_for(repo)}\n\n"
            "These are the files the first description of this variant would "
            "fetch by itself; downloading them here means that run starts "
            "describing instead of downloading.  Only these files belong to "
            "this variant - the environment above is shared with every other "
            "variant of the model.",
            "Download variant",
            style=wx.YES_NO | wx.ICON_QUESTION,
        )
        if answer != wx.YES:
            return
        self._download_cancel = threading.Event()
        self._download_dialog = TaskProgressDialog(
            self, "Download variant", f"Preparing {repo}...",
            cancel_label="&Cancel", on_cancel=self._on_download_cancel,
        )
        self._download_dialog.start_pulse()
        self._download_dialog.Show()
        self._refresh_buttons()
        threading.Thread(
            target=self._download_job, args=(model_id, variant), daemon=True,
            name=f"aivs-describe-download-{model_id}",
        ).start()

    def _on_download_cancel(self) -> None:
        if self._download_cancel is not None:
            self._download_cancel.set()

    def _download_job(self, model_id: str, variant: str) -> None:
        def progress(message: str) -> None:
            wx.CallAfter(self._download_progress, message)

        result = describe_engine.download_variant(
            model_id, variant, progress=progress,
            cancel_event=self._download_cancel,
        )
        wx.CallAfter(self._download_done, model_id, variant, result)

    def _download_progress(self, message: str) -> None:
        dialog = self._download_dialog
        if dialog is not None:
            try:
                dialog.set_message(message)
            except Exception:
                log.debug("Ignoring a failure in _ManageDescribePanel._download_progress", exc_info=True)

    def _download_done(self, model_id: str, variant: str, result) -> None:
        dialog = self._download_dialog
        self._download_dialog = None
        self._download_cancel = None
        if dialog is not None:
            dialog.finish()
        name = describe_catalog.model(model_id)["name"]
        self._refresh_statuses()
        self._show_status()
        if result.get("ok"):
            wx.MessageBox(
                f"{name} ({variant}) is downloaded and ready to describe "
                "with.",
                "Download variant", style=wx.OK | wx.ICON_INFORMATION,
            )
        else:
            error = str(result.get("error") or "The download failed.")
            wx.MessageBox(
                f"{name} ({variant}) could not be downloaded:\n\n{error}",
                "Download variant", style=wx.OK | wx.ICON_ERROR,
            )

    def _on_remove_variant(self, _):
        """Delete one variant's checkpoint files; the model is untouched."""
        model_id = self._selected_model()
        variant = self._selected_variant()
        if not model_id or not variant:
            return
        name = describe_catalog.model(model_id)["name"]
        repo = describe_engine.variant_repo(model_id, variant)
        if wx.MessageBox(
            f"Delete the downloaded checkpoint files of {name}, variant "
            f"{variant}?\n\n{describe_weights.path_for(repo)}\n\n"
            "The model's environment and its other variants' downloads are "
            "kept.  Describing with this variant again downloads these files.",
            "Remove variant download", style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        note = describe_engine.remove_variant(model_id, variant)
        self._refresh_statuses()
        self._show_status()
        if note:
            wx.MessageBox(note, "Remove variant download",
                          style=wx.OK | wx.ICON_INFORMATION)


# ---------------------------------------------------------------------------
# Keyboard gestures (after NVDA's Input Gestures dialog)
# ---------------------------------------------------------------------------
#: Named keys that are not a printable character, by wx key code.
_GESTURE_KEY_NAMES = {
    wx.WXK_ESCAPE: "escape",
    wx.WXK_SPACE: "space",
    wx.WXK_TAB: "tab",
    wx.WXK_RETURN: "enter",
    wx.WXK_BACK: "back",
    wx.WXK_DELETE: "delete",
    wx.WXK_INSERT: "insert",
    wx.WXK_HOME: "home",
    wx.WXK_END: "end",
    wx.WXK_PAGEUP: "pageup",
    wx.WXK_PAGEDOWN: "pagedown",
    wx.WXK_UP: "up",
    wx.WXK_DOWN: "down",
    wx.WXK_LEFT: "left",
    wx.WXK_RIGHT: "right",
}
for _index in range(1, 25):  # F1 .. F24
    _GESTURE_KEY_NAMES[wx.WXK_F1 + _index - 1] = f"f{_index}"

#: Modifier-only presses (they cannot be a gesture on their own).
_GESTURE_MODIFIER_CODES = {
    wx.WXK_CONTROL, wx.WXK_SHIFT, wx.WXK_ALT, wx.WXK_WINDOWS_LEFT,
    wx.WXK_WINDOWS_RIGHT,
}


def gesture_key_name(code: int) -> str:
    """The gesture chunk for one wx key code (``None``-safe, ``""`` if none)."""
    if code in _GESTURE_KEY_NAMES:
        return _GESTURE_KEY_NAMES[code]
    if 32 < code < 127:
        return chr(code).lower()
    return ""


def gesture_from_event(evt: wx.KeyEvent) -> str:
    """The normalized gesture a key event represents (``""`` if none yet)."""
    code = evt.GetKeyCode()
    if code in _GESTURE_MODIFIER_CODES:
        return ""
    key = gesture_key_name(code)
    if not key:
        return ""
    modifiers = []
    if evt.ControlDown():
        modifiers.append("control")
    if evt.AltDown():
        modifiers.append("alt")
    if evt.ShiftDown():
        modifiers.append("shift")
    if getattr(evt, "MetaDown", lambda: False)():
        modifiers.append("windows")
    return gestures_model.normalize("+".join([*modifiers, key]))


def gesture_problem(identifier: str) -> str:
    """Why ``identifier`` cannot be used as a global gesture (``""`` if it can).

    A bare letter, digit or space would shadow ordinary typing, so a gesture
    needs a Control, Alt or Windows modifier unless it is a function key.
    """
    if not gestures_model.is_valid(identifier):
        return "Press a key combination with Ctrl, Alt or Windows (or a function key)."
    body = identifier.split(":", 1)[1]
    chunks = body.split("+")
    key = chunks[-1]
    named = key in _GESTURE_KEY_NAMES.values() or (
        key.startswith("f") and key[1:].isdigit()
    )
    held = any(chunk in ("control", "alt", "windows") for chunk in chunks)
    if not held and not named:
        return ("A gesture with no Ctrl, Alt or Windows key would take over "
                "ordinary typing. Add one of those modifiers.")
    return ""


class _GestureCaptureDialog(wx.Dialog):
    """Ask the user to press the key combination they want to assign."""

    def __init__(self, parent: wx.Window, command_label: str):
        super().__init__(parent, title="Add input gesture")
        self.gesture = ""
        sizer = wx.BoxSizer(wx.VERTICAL)
        intro = wx.StaticText(
            self,
            label=f"Press the key combination to assign to \"{command_label}\".",
        )
        intro.Wrap(420)
        sizer.Add(intro, 0, wx.ALL, 10)

        self.capture = wx.TextCtrl(self, style=wx.TE_READONLY)
        set_accessible_name(self.capture, "Captured gesture")
        self.capture.SetName("Captured gesture")
        sizer.Add(self.capture, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        self.hint = wx.StaticText(self, label="Waiting for a key...")
        set_accessible_name(self.hint, "Capture hint")
        self.hint.Wrap(420)
        sizer.Add(self.hint, 0, wx.ALL, 10)

        sizer.Add(
            self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL),
            0, wx.EXPAND | wx.ALL, 10,
        )
        self.SetSizer(sizer)
        self.SetMinSize((460, 220))
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        finalize_accessibility(self)
        wx.CallAfter(self.capture.SetFocus)

    def _on_char_hook(self, evt: wx.KeyEvent) -> None:
        if evt.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
            return
        if evt.GetKeyCode() in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER):
            if self.gesture:
                self.EndModal(wx.ID_OK)
            return
        candidate = gesture_from_event(evt)
        if not candidate:
            evt.Skip()
            return
        problem = gesture_problem(candidate)
        self.gesture = "" if problem else candidate
        self.capture.SetValue(gestures_model.display_name(candidate))
        self.hint.SetLabel(problem or "Press Enter to accept, Escape to cancel.")
        self.hint.Wrap(420)
        self.Layout()


class _HuggingFacePanel(_SettingsPanel):
    """Settings > Hugging Face account: the token a gated model needs.

    Nothing here is needed by most of the catalog: only the entries whose
    repository is *gated* (Google's Gemma 3, and Kyutai's voice-cloning Pocket
    TTS weights) download for an account that has accepted their licence.
    Without this panel such a user could only be told to run
    ``huggingface-cli login``, which a person who installed the setup file does
    not have - so the token is asked for here, saved in this user's settings,
    and handed to the model's worker as ``HF_TOKEN``.

    A gated model needs **two** things, and this panel does both: the licence
    accepted on the model's own page (there is no per-model token - one read
    token serves them all), and that token given to the application.  So the
    page of each gated repository the application ships is one button away, and
    *Check access* asks Hugging Face the question the failure itself leaves
    unanswered - whether this token's account may download *this* model, which
    separates a mistyped token from a licence that was never accepted.
    """

    title = "Hugging Face account"
    description = (
        "Access token for models whose download is gated on Hugging Face - "
        "Google's Gemma 3 and Pocket TTS's voice-cloning weights. Open models "
        "work with no token at all."
    )

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        #: The exact text that was verified last, so a token edited after a
        #: successful check never keeps the old account beside it.
        self._checked_token = ""
        self._account = ""
        #: The last answer *Check access* gave (kept so the line survives a
        #: status refresh of the token beside it).
        self._access_status = ""
        sizer = wx.BoxSizer(wx.VERTICAL)

        intro = wx.StaticText(
            self,
            label="Most models download with no account. A few - Google's "
                  "Gemma 3 among them - live in a *gated* repository: their "
                  "licence has to be accepted on the model's own Hugging Face "
                  "page, and the download then has to say who is asking. "
                  "This panel is that account.",
        )
        intro.Wrap(520)
        sizer.Add(intro, 0, wx.ALL, 6)

        box = wx.StaticBox(self, label="Access token")
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(box, label="&Token:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.token_ctrl = wx.TextCtrl(box, style=wx.TE_PASSWORD | wx.TE_PROCESS_ENTER)
        self.token_ctrl.SetName("Hugging Face access token")
        self.token_ctrl.SetToolTip(
            "Paste a read token from huggingface.co/settings/tokens, from the "
            "account that accepted the model's licence. It is stored in this "
            "user's settings file only."
        )
        self.token_ctrl.SetValue(
            str(settings.get(hf_access.TOKEN_SETTING, "") or "")
        )
        self.token_ctrl.Bind(wx.EVT_TEXT, self._on_token_edited)
        row.Add(self.token_ctrl, 1, wx.EXPAND)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 6)

        self.show_cb = wx.CheckBox(box, label="Show the token")
        self.show_cb.SetName("Show the token")
        self.show_cb.SetToolTip(
            "Show the token as plain text so you can check what was pasted"
        )
        self.show_cb.Bind(wx.EVT_CHECKBOX, self._on_show_token)
        box_sizer.Add(self.show_cb, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.check_btn = wx.Button(box, label="&Check token")
        self.check_btn.SetName("Check token")
        self.check_btn.SetToolTip(
            "Ask huggingface.co whose token this is, and report the account"
        )
        self.check_btn.Bind(wx.EVT_BUTTON, self._on_check_token)
        self.clear_btn = wx.Button(box, label="C&lear token")
        self.clear_btn.SetName("Clear token")
        self.clear_btn.SetToolTip(
            "Remove the token from this application (saved when you press "
            "Apply or OK)"
        )
        self.clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_token)
        self.get_btn = wx.Button(box, label="&Get a token...")
        self.get_btn.SetName("Get a token")
        self.get_btn.SetToolTip(
            "Open huggingface.co in your browser, where a read token is created"
        )
        self.get_btn.Bind(wx.EVT_BUTTON, self._on_open_token_page)
        for button in (self.check_btn, self.clear_btn, self.get_btn):
            buttons.Add(button, 0, wx.ALL, 4)
        box_sizer.Add(buttons, 0, wx.ALL, 2)

        self.status = wx.StaticText(box, label="")
        self.status.SetName("Token status")
        self.status.Wrap(520)
        box_sizer.Add(self.status, 0, wx.ALL, 6)
        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 6)

        # The other half of the job: the licence, on the model's own page, and
        # the proof that it is in place.  One token serves every gated model, so
        # "which token for Gemma 3?" is answered here rather than in prose.
        gated = wx.StaticBox(self, label="Gated models (accept the licence "
                                         "first)")
        gated_sizer = wx.StaticBoxSizer(gated, wx.VERTICAL)
        self._gated_rows = hf_access.gated_repos()
        pick = wx.BoxSizer(wx.HORIZONTAL)
        pick.Add(wx.StaticText(gated, label="&Model:"), 0,
                 wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.repo_choice = wx.Choice(
            gated,
            choices=[f"{row['label']}  -  {row['repo']}"
                     for row in self._gated_rows],
        )
        self.repo_choice.SetName("Gated model")
        self.repo_choice.SetToolTip(
            "The gated model to set up. Its licence is accepted on its own "
            "page, with the account the token belongs to."
        )
        self.repo_choice.SetSelection(self._default_gated_index())
        self.repo_choice.Bind(wx.EVT_CHOICE, self._on_repo_changed)
        pick.Add(self.repo_choice, 1, wx.EXPAND)
        gated_sizer.Add(pick, 0, wx.EXPAND | wx.ALL, 6)

        gated_buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.open_btn = wx.Button(gated, label="&Open the model page...")
        self.open_btn.SetName("Open the model page")
        self.open_btn.SetToolTip(
            "Open this model's page on huggingface.co, where its licence "
            "(terms of use) is accepted with your account"
        )
        self.open_btn.Bind(wx.EVT_BUTTON, self._on_open_model_page)
        self.access_btn = wx.Button(gated, label="Check &access")
        self.access_btn.SetName("Check access")
        self.access_btn.SetToolTip(
            "Ask huggingface.co whether this token's account may download "
            "this model - the answer that tells a wrong token from a licence "
            "that was never accepted"
        )
        self.access_btn.Bind(wx.EVT_BUTTON, self._on_check_access)
        for button in (self.open_btn, self.access_btn):
            gated_buttons.Add(button, 0, wx.ALL, 4)
        gated_sizer.Add(gated_buttons, 0, wx.ALL, 2)

        self.access_status = wx.StaticText(gated, label="")
        self.access_status.SetName("Model access status")
        self.access_status.Wrap(520)
        gated_sizer.Add(self.access_status, 0, wx.ALL, 6)
        sizer.Add(gated_sizer, 0, wx.EXPAND | wx.ALL, 6)
        self._set_access_status(self._idle_access_line())

        steps = wx.StaticText(
            self,
            label="Three steps, once per gated model. There is no per-model "
                  "token: one read token serves every gated model, and what "
                  "grants access to Gemma 3 is accepting *its* licence.\n"
                  "1. Choose the model above, press Open the model page and "
                  "press the page's own accept button (Gemma 3 asks Google "
                  "for that click; Pocket TTS's voice cloning approves "
                  "itself immediately).\n"
                  "2. On huggingface.co/settings/tokens press Create new "
                  "token, choose the Read role, give it any name and copy "
                  "it.\n"
                  "3. Paste it above, press Check token (it prints your "
                  "account name), then press Check access (it says whether "
                  "the licence is in place for that model).\n"
                  "The token stays on this computer, in your own settings "
                  "file, and is sent nowhere except huggingface.co when a "
                  "model downloads.",
        )
        steps.Wrap(520)
        sizer.Add(steps, 0, wx.ALL, 6)
        self.SetSizer(sizer)
        self._refresh_status()

    # -- events -------------------------------------------------------------
    def on_activated(self) -> None:
        super().on_activated()
        self._refresh_status()

    def _on_token_edited(self, _evt=None) -> None:
        """A hand-edited token invalidates the verified account beside it."""
        if self.token_ctrl.GetValue().strip() != self._checked_token:
            self._account = ""
        self._refresh_status()

    def _on_show_token(self, _evt=None) -> None:
        style = self.token_ctrl.GetWindowStyle()
        if self.show_cb.GetValue():
            self.token_ctrl.SetWindowStyle(style & ~wx.TE_PASSWORD)
        else:
            self.token_ctrl.SetWindowStyle(style | wx.TE_PASSWORD)
        self.token_ctrl.Refresh()

    def _on_clear_token(self, _evt=None) -> None:
        self.token_ctrl.SetValue("")
        self._checked_token = ""
        self._account = ""
        self._refresh_status("The token was cleared. Press Apply or OK to save.")

    def _on_open_token_page(self, _evt=None) -> None:
        import webbrowser

        webbrowser.open(hf_access.TOKEN_PAGE)

    def _on_check_token(self, _evt=None) -> None:
        """Ask huggingface.co whose token this is (never the GUI thread)."""
        value = self.token_ctrl.GetValue().strip()
        if not value:
            self._refresh_status("Enter a token first.")
            return
        self.check_btn.Enable(False)
        announce.status_text(self.status, "Checking the token with huggingface.co...")
        self.status.Wrap(520)
        self.Layout()

        def _check() -> None:
            ok, detail = hf_access.verify(value)
            wx.CallAfter(self._checked, value, ok, detail)

        threading.Thread(target=_check, name="aivs-hf-token-check",
                         daemon=True).start()

    def _checked(self, value: str, ok: bool, detail: str) -> None:
        try:
            self.check_btn.Enable(True)
        except Exception:
            return
        self._checked_token = value if ok else ""
        self._account = detail if ok else ""
        if ok:
            self._refresh_status(f"The token works: signed in as {detail}.")
        else:
            self._refresh_status(detail)

    def _refresh_status(self, prefix: str = "") -> None:
        """One line saying whether a gated model can download right now.

        The three places a token can come from are named, because a token in
        ``HF_TOKEN`` or one left by ``huggingface-cli login`` is in use even
        though this field is empty.
        """
        value = self.token_ctrl.GetValue().strip()
        source = ("this panel" if value
                  else hf_access.token_source(self.settings))
        account = self._account or hf_access.account(self.settings)
        if value or source:
            where = "this panel" if value else source
            text = f"A token is configured in {where}."
            if account:
                text += f" Last verified account: {account}."
            text += (" Gated models (Gemma 3, Pocket TTS voice cloning) can "
                     "download; open models do not need it.")
        else:
            text = ("No token is configured. Open models download without one; "
                    "a gated one (Gemma 3, Pocket TTS voice cloning) will say "
                    "so before it starts.")
        announce.status_text(self.status, f"{prefix} {text}".strip())
        self.status.Wrap(520)
        self.Layout()

    def apply_to_settings(self):
        previous = str(self.settings.get(hf_access.TOKEN_SETTING, "") or "").strip()
        value = self.token_ctrl.GetValue().strip()
        self.settings.set(hf_access.TOKEN_SETTING, value)
        # Only a token that was actually checked carries an account name; a
        # token typed and never verified leaves the old one out rather than
        # claiming an account that was never confirmed for it.
        if value and self._checked_token == value and self._account:
            self.settings.set(hf_access.ACCOUNT_SETTING, self._account)
        elif not value or self._checked_token != value:
            self.settings.set(hf_access.ACCOUNT_SETTING, "")
        if value != previous:
            self._restart_engine_workers()

    @staticmethod
    def _restart_engine_workers() -> None:
        """Drop the Voice Lab workers so the next one starts with the new token.

        A worker is handed ``HF_TOKEN`` once, when it is launched, and its engine
        keeps its (possibly cloning-free) weights loaded between requests - so a
        token pasted now would otherwise only take effect after the application
        was restarted, and a user who did exactly what the message asked would
        see the same failure again.  Closing them costs one model load on the
        next request.
        """
        try:
            from .. import voicelab

            voicelab.close_workers()
        except Exception as exc:
            log.info("Could not restart the Voice Lab workers: %s", exc)


    # -- the gated models --------------------------------------------------
    def _default_gated_index(self) -> int:
        """Gemma 3 12B when it is offered, else the first gated repository.

        It is the one this whole panel exists for - the model whose 401 started
        the feature - so it is what the picker opens showing.
        """
        for index, row in enumerate(self._gated_rows):
            if row.get("repo") == "google/gemma-3-12b-it":
                return index
        return 0 if self._gated_rows else -1

    def _selected_repo(self) -> str:
        """The repository the picker is showing (``""`` when there is none)."""
        index = self.repo_choice.GetSelection()
        if index < 0 or index >= len(self._gated_rows):
            return ""
        return str(self._gated_rows[index].get("repo") or "")

    def _idle_access_line(self) -> str:
        repo = self._selected_repo()
        if not repo:
            return ""
        return (f"Press Check access to ask huggingface.co whether "
                f"{hf_access.gated_label(repo)} can be downloaded with this "
                "token.")

    def _set_access_status(self, text: str) -> None:
        self._access_status = text
        try:
            announce.status_text(self.access_status, text)
            self.access_status.Wrap(520)
            self.Layout()
        except Exception:
            log.debug("Ignoring a failure in _HuggingFacePanel._set_access_status", exc_info=True)

    def _on_repo_changed(self, _evt=None) -> None:
        """A different model is a different licence: say so, do not carry over."""
        self._set_access_status(self._idle_access_line())

    def _on_open_model_page(self, _evt=None) -> None:
        """Open the chosen model's page - where its licence is accepted.

        This is the half of the job the token page cannot do, and the reason a
        user can follow every instruction and still be refused: access is
        granted by accepting *that* repository's terms with the account the
        token belongs to.
        """
        import webbrowser

        repo = self._selected_repo()
        if not repo:
            return
        webbrowser.open(hf_access.model_page(repo))
        self._set_access_status(
            f"Opened {hf_access.model_page(repo)} in your browser. Accept its "
            "licence there (the button is called \"Agree and access "
            "repository\"), then press Check access."
        )

    def _on_check_access(self, _evt=None) -> None:
        """Ask Hugging Face whether this token may download this model."""
        repo = self._selected_repo()
        if not repo:
            return
        value = self.token_ctrl.GetValue().strip()
        self.access_btn.Enable(False)
        self._set_access_status(
            f"Asking huggingface.co whether {repo} can be downloaded..."
        )

        def _check() -> None:
            state, detail = hf_access.check_access(repo, value)
            wx.CallAfter(self._access_checked, repo, state, detail)

        threading.Thread(target=_check, name="aivs-hf-access-check",
                         daemon=True).start()

    def _access_checked(self, repo: str, state: str, detail: str) -> None:
        try:
            self.access_btn.Enable(True)
        except Exception:
            return
        self._set_access_status(detail)
        if state == hf_access.ACCESS_OK:
            log.info("Hugging Face access to %s is in place for this token", repo)


class _InputGesturesPanel(_SettingsPanel):
    """Settings > Keyboard gestures: view and rebind the app's commands.

    Mirrors NVDA's Input Gestures dialog: commands are grouped by category in
    a tree, the selected command's gestures are listed beside it, and the
    user can add a key combination, remove one, or restore the defaults.  A
    key already taken by another command is reported and can be reassigned.
    """

    title = "Keyboard gestures"
    description = (
        "Every command with the keyboard keys that trigger it. Select a "
        "command to see its gestures, then add or remove keys. A key that "
        "another command already uses is reported before it is reassigned."
    )
    panelDescription = description

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self.gesture_map = gestures_model.GestureMap(
            settings.get("gestures.bindings", {}) or {}
        )
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, label="Commands"), 0, wx.ALL, 6)
        self.filter = wx.TextCtrl(self)
        set_accessible_name(self.filter, "Filter commands")
        self.filter.SetName("Filter commands")
        self.filter.SetHint("Type to filter the commands")
        self.filter.Bind(wx.EVT_TEXT, lambda _e: self._rebuild_tree())
        sizer.Add(self.filter, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.tree = wx.TreeCtrl(
            self, style=wx.TR_HIDE_ROOT | wx.TR_SINGLE | wx.TR_LINES_AT_ROOT,
            size=wx.Size(-1, 220),
        )
        set_accessible_name(self.tree, "Commands")
        self.tree.SetName("Commands")
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self._on_command_selected)
        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 6)

        sizer.Add(wx.StaticText(self, label="Gestures for the selected command"),
                  0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.gesture_list = wx.ListBox(self, size=wx.Size(-1, 120))
        set_accessible_name(self.gesture_list, "Gestures for the selected command")
        self.gesture_list.SetName("Gestures for the selected command")
        self.gesture_list.Bind(wx.EVT_LISTBOX, lambda _e: self._refresh_buttons())
        sizer.Add(self.gesture_list, 0, wx.EXPAND | wx.ALL, 6)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.add_btn = wx.Button(self, label="&Add gesture...")
        self.add_btn.SetName("Add gesture")
        self.add_btn.SetToolTip("Assign a key combination to the selected command")
        self.remove_btn = wx.Button(self, label="&Remove gesture")
        self.remove_btn.SetName("Remove gesture")
        self.remove_btn.SetToolTip("Take the selected key away from this command")
        self.reset_btn = wx.Button(self, label="Restore &defaults")
        self.reset_btn.SetName("Restore defaults")
        self.reset_btn.SetToolTip("Forget your changes to the selected command")
        for button in (self.add_btn, self.remove_btn, self.reset_btn):
            buttons.Add(button, 0, wx.ALL, 4)
        sizer.Add(buttons, 0, wx.ALL, 4)

        self.status = wx.StaticText(self, label="")
        set_accessible_name(self.status, "Gesture status")
        self.status.Wrap(700)
        sizer.Add(self.status, 0, wx.ALL, 6)

        self.add_btn.Bind(wx.EVT_BUTTON, self._on_add)
        self.remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.reset_btn.Bind(wx.EVT_BUTTON, self._on_reset)

        self.SetSizer(sizer)
        self._rebuild_tree()
        self._refresh_gestures()

    # -- tree ---------------------------------------------------------------
    def _rebuild_tree(self) -> None:
        """Rebuild the category/command tree for the current filter text."""
        needle = (self.filter.GetValue() or "").strip().lower()
        selected = self.selected_command()
        self.tree.DeleteAllItems()
        root = self.tree.AddRoot("Commands")
        first = None
        for category in gestures_model.categories_present():
            entries = [
                command for command in gestures_model.commands_in(category)
                if not needle or needle in command.label.lower()
                or needle in command.id.lower()
            ]
            if not entries:
                continue
            branch = self.tree.AppendItem(root, category)
            for command in entries:
                node = self.tree.AppendItem(branch, command.label)
                self.tree.SetItemData(node, command.id)
                if first is None:
                    first = node
        self.tree.ExpandAll()
        if selected:
            node = self._find_node(selected)
            if node:
                self.tree.SelectItem(node)
                return
        if first is not None:
            self.tree.SelectItem(first)
        self._refresh_gestures()

    def _find_node(self, command_id: str):
        root = self.tree.GetRootItem()
        if not root.IsOk():
            return None
        branch, cookie = self.tree.GetFirstChild(root)
        while branch.IsOk():
            child, cookie2 = self.tree.GetFirstChild(branch)
            while child.IsOk():
                if self.tree.GetItemData(child) == command_id:
                    return child
                child, cookie2 = self.tree.GetNextChild(branch, cookie2)
            branch, cookie = self.tree.GetNextChild(root, cookie)
        return None

    def selected_command(self) -> str:
        try:
            item = self.tree.GetSelection()
        except RuntimeError:
            # The tree was destroyed (the dialog is closing); no selection.
            return ""
        if not item.IsOk():
            return ""
        return self.tree.GetItemData(item) or ""

    def _on_command_selected(self, _evt) -> None:
        # This can fire while the dialog is being destroyed, when the C++
        # TreeCtrl is already gone.
        try:
            self._refresh_gestures()
        except RuntimeError:
            log.debug("Ignoring a failure in _InputGesturesPanel._on_command_selected", exc_info=True)

    # -- gestures -----------------------------------------------------------
    def _refresh_gestures(self) -> None:
        command_id = self.selected_command()
        self.gesture_list.Clear()
        for gesture in self.gesture_map.bindings_for(command_id) if command_id else []:
            self.gesture_list.Append(
                f"{gestures_model.source_display(gesture)}  "
                f"{gestures_model.display_name(gesture)}",
                gesture,
            )
        entry = gestures_model.command(command_id) if command_id else None
        if entry is not None and self.gesture_map.is_customized(command_id):
            state = "changed from the default"
        else:
            state = "the default"
        conflicts = self.gesture_map.conflicts()
        conflict_text = ""
        if conflicts:
            parts = [
                f"{gestures_model.display_name(gesture)} "
                f"({', '.join(ids)})"
                for gesture, ids in sorted(conflicts.items())
            ]
            conflict_text = "  Conflicts: " + "; ".join(parts)
        if entry is None:
            announce.status_text(self.status, "Select a command." + conflict_text)
        else:
            announce.status_text(self.status,
                f"{entry.label}: {self.gesture_list.GetCount()} "
                f"gesture(s), {state}.{conflict_text}"
            )
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        command_id = self.selected_command()
        self.add_btn.Enable(bool(command_id))
        self.remove_btn.Enable(self.gesture_list.GetSelection() >= 0)
        self.reset_btn.Enable(bool(command_id)
                              and self.gesture_map.is_customized(command_id))

    # -- editing ------------------------------------------------------------
    def _on_add(self, _) -> None:
        command_id = self.selected_command()
        entry = gestures_model.command(command_id) if command_id else None
        if entry is None:
            return
        dlg = _GestureCaptureDialog(self, entry.label)
        try:
            if dlg.ShowModal() != wx.ID_OK or not dlg.gesture:
                return
            gesture = dlg.gesture
        finally:
            dlg.Destroy()
        owner = self.gesture_map.owner(gesture)
        if owner and owner != command_id:
            other = gestures_model.command(owner)
            answer = wx.MessageBox(
                f"{gestures_model.display_name(gesture)} is already assigned "
                f"to \"{other.label if other else owner}\".\n\n"
                "Take it away from that command and give it to this one?",
                "Input gesture already in use",
                style=wx.YES_NO | wx.ICON_QUESTION,
            )
            if answer != wx.YES:
                return
            self.gesture_map.add(command_id, gesture, replace_owner=True)
        else:
            self.gesture_map.add(command_id, gesture)
        self._rebuild_tree()

    def _on_remove(self, _) -> None:
        command_id = self.selected_command()
        index = self.gesture_list.GetSelection()
        if not command_id or index < 0:
            return
        gesture = self.gesture_list.GetClientData(index)
        self.gesture_map.remove(command_id, gesture)
        self._refresh_gestures()

    def _on_reset(self, _) -> None:
        command_id = self.selected_command()
        if command_id:
            self.gesture_map.reset(command_id)
            self._rebuild_tree()

    def on_activated(self):
        super().on_activated()
        self._refresh_gestures()

    def apply_to_settings(self):
        self.settings.set("gestures.bindings", self.gesture_map.export())


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
            # Reset through the dialog when it is there, so the categories
            # that are already built drop their old widgets and show the
            # defaults when they are next opened (and the ones that were
            # never built read the defaults anyway).
            reset_all = getattr(self.GetTopLevelParent(), "_settings_reset", None)
            if callable(reset_all):
                reset_all()
            else:
                self.settings.reset()
            wx.MessageBox(
                "Settings have been reset to their default values. The other tabs "
                "now show the defaults; press OK or Apply to keep them.",
                "Reset complete",
                style=wx.OK | wx.ICON_INFORMATION,
            )
