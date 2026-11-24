"""Main window (SPEC 3.2, 3.5, 3.6).

Menu bar with mnemonics and accelerators:
* File    -> New Project, Open Project, Recent Projects, Exit
* Edit    -> Resume Recording, Restart Project, Restart All Recording,
             Start Selected Recording, Remove Project, Show project folder
* Tools   -> Record (Ctrl+Shift+R), Settings (Ctrl+,)
* Help    -> Read Me, User Guide, About

Ctrl+Shift+N belongs to the welcome panel's *Create New Project* button (not
to the File menu): pressing it sends that button its own click event, so the
gesture and the mouse click run one and the same code path.  Ctrl+Shift+R
(Record) and Ctrl+, (Settings) stay with the Tools-menu items.  The frame's
own key handler routes all three, because the menu bar's accelerator parsing
proves unreliable for Shift+letter combinations on this wx build.

The welcome screen offers the same actions with large labelled buttons and a
recent-projects list that screen readers can navigate with arrow keys.  It is
laid out as three regions: a full-width action bar of those buttons, a
project sidebar on the left holding the recent-projects list and separated by
a one-pixel rule, and the large empty area that remains - the only region a
wallpaper is ever painted into, so it never covers a button or the list.  The
Edit-menu actions act on the project selected in the Recent projects list
(right-click the list for the same actions).
"""

from __future__ import annotations

import logging
import os
import sys

import wx

from .. import __version__, project
from .. import gestures as gestures_model
from ..constants import (
    DEFAULT_PROJECT_BASE,
    PROJECT_BASE_AD,
    PROJECT_BASE_STT,
    PROJECT_TYPE_AD,
    PROJECT_TYPE_STT,
)
from ..settings import Settings
from ..tts.models import ModelStore
from . import announce, dpi
from .a11y import finalize_accessibility, update_accessible_name
from .describe_dialog import DescribeDialog
from .dialogs import open_folder
from .new_project_wizard import NewProjectWizard
from .recording_dialog import RecordingDialog
from .settings_dialog import SettingsDialog, gesture_from_event
from .stt_dialog import STTTranscribeDialog
from .theme import (
    apply_theme,
    current_theme,
    get_wallpaper,
    mix,
    paint_background,
    resolve_palette,
    theme_pattern,
)

log = logging.getLogger(__name__)

#: Smallest width of the welcome screen's project sidebar.  It is wide enough
#: for the heading to sit on one line without clipping ("Welcome to AI Voice
#: Studio" at 16pt bold measures ~280px) plus its margins.
_SIDEBAR_MIN_WIDTH = 320
#: The action bar is taller than the buttons it carries by this many pixels on
#: each side - comfortably more than one millimetre at every supported DPI, so
#: the row reads as a bar instead of buttons glued to the window edge.
_ACTION_BAR_PAD = 10


def _wrap_label(text: wx.StaticText, label: str, width: int) -> None:
    """Word-wrap ``label`` into ``text`` so no line is wider than ``width``.

    ``wx.StaticText.Wrap`` is not dependable here: on the first call for a
    freshly laid-out control it silently does nothing, and it would leave the
    problem invisible until the text hung off the panel edge.  Building the
    lines against the control's own font metrics always works and always
    produces the same result.
    """
    if width <= 0:
        return
    dc = wx.ClientDC(text)
    dc.SetFont(text.GetFont())
    lines: list[str] = []
    current = ""
    for word in label.split():
        candidate = word if not current else f"{current} {word}"
        if current and dc.GetTextExtent(candidate)[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    wrapped = "\n".join(lines)
    if text.GetLabel() != wrapped:
        text.SetLabel(wrapped)
        text.InvalidateBestSize()


class _WallpaperPanel(wx.Panel):
    """The *empty* region of the welcome screen.

    That is the whole point of this panel: it holds no controls of its own,
    so the theme's wallpaper (the user's image, aspect-filled, or one of the
    built-in patterns) can fill it without ever painting behind a button or
    the recent projects list - those live in sibling panels.  With no
    wallpaper enabled it is painted in the theme's plain background colour,
    the very colour it displayed before themes existed.
    """

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda _evt: None)

    def _on_paint(self, _evt) -> None:
        dc = wx.AutoBufferedPaintDC(self)
        width, height = self.GetClientSize()
        theme = current_theme()
        colors = resolve_palette(theme, self.settings)
        record = get_wallpaper(self.settings, theme)
        if record["enabled"]:
            paint_background(
                dc, (width, height), colors,
                theme_pattern(theme, self.settings), record["image"],
            )
        else:
            paint_background(dc, (width, height), colors, "solid", "")


class _DividerPanel(wx.Panel):
    """The one-pixel rule a sidebar draws along its inside edge.

    ``wx.StaticLine`` keeps its own system colour, which disappears against a
    dark theme.  This paints the theme background and then a line blended
    between it and the theme's heading colour, so the separation stays visible
    in every theme - including the light and dark built-ins.
    """

    def __init__(self, parent, settings):
        super().__init__(parent, size=(1, -1))
        self.settings = settings
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda _evt: None)

    def _on_paint(self, _evt) -> None:
        dc = wx.AutoBufferedPaintDC(self)
        width, height = self.GetClientSize()
        colors = resolve_palette(current_theme(), self.settings)
        background = wx.Colour(colors.get("bg", "#000000"))
        heading = (colors.get("heading") or colors.get("accent")
                   or colors.get("fg") or "#ffffff")
        dc.SetPen(wx.Pen(background))
        dc.SetBrush(wx.Brush(background))
        dc.DrawRectangle(0, 0, width, height)
        dc.SetPen(wx.Pen(mix(colors.get("bg", "#000000"), heading, 0.5)))
        x = max(0, width // 2)
        dc.DrawLine(x, 0, x, height)

#: Windows virtual keys and scan codes the frame's key handler needs.
_VK_CONTROL, _VK_SHIFT, _VK_N, _VK_R = 0x11, 0x10, 0x4E, 0x52
_SCAN_N, _SCAN_R = 0x31, 0x13


def _physical_key_down(vk: int) -> bool:
    """True while the key ``vk`` is physically held (Windows only).

    wx can deliver a Shift+letter combination with the modifier bits missing
    from the event, which would make the shortcut unrecognisable; the real
    keyboard is the only authority left.  False everywhere else, where the
    event's own modifiers are trusted.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetKeyState(vk) & 0x8000)
    except Exception:
        return False


ID_NEW_PROJECT = wx.NewIdRef()
ID_OPEN_PROJECT = wx.NewIdRef()
ID_RECORD = wx.NewIdRef()
ID_SETTINGS = wx.NewIdRef()
ID_README = wx.NewIdRef()
ID_USER_GUIDE = wx.NewIdRef()
ID_ADDON_GUIDE = wx.NewIdRef()
ID_ACCESSIBILITY_GUIDE = wx.NewIdRef()
ID_PYTHON_BOOK = wx.NewIdRef()
ID_THIRD_PARTY_LICENSES = wx.NewIdRef()
ID_CHECK_UPDATES = wx.NewIdRef()
ID_ABOUT = wx.NewIdRef()
ID_RESUME_RECORDING = wx.NewIdRef()
ID_RESTART_PROJECT = wx.NewIdRef()
ID_RESTART_ALL = wx.NewIdRef()
ID_RESTART_SELECTED = wx.NewIdRef()
ID_REMOVE_PROJECT = wx.NewIdRef()
ID_SHOW_PROJECT_FOLDER = wx.NewIdRef()
ID_WHATS_NEW = wx.NewIdRef()

#: Edit-menu labels for an audio description project.  Every project has the
#: same actions - resume, restart, start a chosen file - but an audio
#: description project speaks about *descriptions*, not "recordings" or
#: "files", so the same item ids answer with these labels instead (see
#: :meth:`MainFrame._edit_label`).  The access keys stay unique inside the menu
#: (R, P, A, S, M, F).
_AD_EDIT_LABELS = {
    ID_RESUME_RECORDING: "&Resume the project",
    ID_RESTART_PROJECT: "Restart &Project...",
    ID_RESTART_ALL: "Restart &All Descriptions",
    ID_RESTART_SELECTED: "Start &Selected Descriptions...",
    ID_REMOVE_PROJECT: "Re&move Project...",
    ID_SHOW_PROJECT_FOLDER: "Show project &folder",
}


class MainFrame(wx.Frame):
    def __init__(self, parent=None, settings: Settings = None, store: ModelStore = None):
        super().__init__(parent, title="AI Voice Studio", size=(820, 600))
        if settings is None:
            settings = Settings()
        if store is None:
            store = ModelStore()
        self.settings = settings
        self.store = store
        # The Hugging Face token is read by the model engines, which never see
        # this window; handing them the live settings store means a token pasted
        # into Settings > Hugging Face account is in force for the very next run.
        from .. import hf_access

        hf_access.use_settings(settings)
        # Measured engine speeds (Settings > Voice Clone > 'Measure speed...')
        # decide which devices are offered; load them once, before any dialog
        # that lists them is built.
        try:
            from ..voicelab import benchmark

            benchmark.load_verdicts(settings)
        except Exception:
            log.debug("Ignoring a failure in MainFrame.__init__", exc_info=True)
        self._recent_menu: wx.Menu | None = None
        # The user's gesture overrides, loaded once and reloaded when the
        # settings dialog closes (see _reload_gestures).
        self._gestures = gestures_model.GestureMap(
            self.settings.get("gestures.bindings", {}) or {}
        )
        # Command id -> the name of the method that runs it in this window
        # (see gestures.COMMANDS).  Names rather than bound methods, so a
        # later override of the method is what actually runs.
        self._gesture_actions = {
            "project.new": "_press_new_project_button",
            "project.open": "_open_project",
            "project.resume": "_resume_recording",
            "project.restart": "_restart_project",
            "project.restart_all": "_restart_all",
            "project.start_selected": "_restart_selected",
            "project.remove": "_remove_project",
            "project.show_folder": "_show_project_folder",
            "record.start": "_record",
            "app.settings": "_settings",
            "help.readme": "_readme",
            "help.user_guide": "_user_guide",
            "help.whats_new": "_whats_new",
            "help.check_updates": "_check_updates",
            "help.about": "_about",
        }

        self._build_menu()
        self._build_welcome()
        self._build_statusbar()
        self.Centre()
        # Startup focus lands on the Recent projects list (all projects),
        # with its accessible name already set by finalize_accessibility.
        wx.CallAfter(self.recent_list.SetFocus)

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menubar = wx.MenuBar()

        file_menu = wx.Menu()
        # No \t accelerator here on purpose: Ctrl+Shift+N is bound to the
        # welcome panel's Create New Project button (see _build_welcome and
        # _on_global_char_hook), so the menu item must not claim the same
        # gesture a second time.
        file_menu.Append(ID_NEW_PROJECT, "&New Project")
        file_menu.Append(ID_OPEN_PROJECT, "&Open Project...")
        self._recent_menu = wx.Menu()
        file_menu.AppendSubMenu(self._recent_menu, "&Recent Projects")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "E&xit")
        menubar.Append(file_menu, "&File")

        # One access key per item inside a menu: Windows fires only the first
        # item that matches a letter, so "&Resume Recording" and a second
        # "&Restart/R&emove..." would leave the later ones unreachable by
        # keyboard.  The markers below are unique within this menu (R, P, A,
        # S, M, F).
        #
        # Each entry is ``(id, text-to-speech label, speech-to-text label)``.
        # A recording project has "recordings"; a transcription project has
        # "files" and an audio description project has "descriptions", so the
        # menu's nature follows the selected project's base (see
        # :meth:`_refresh_edit_menu`; the description labels live in
        # ``_AD_EDIT_LABELS``).  ``None`` is a separator.
        self._edit_items = [
            (ID_RESUME_RECORDING, "&Resume the project", "&Resume the project"),
            (ID_RESTART_PROJECT, "Restart &Project...",
             "Restart &Project..."),
            (ID_RESTART_ALL, "Restart &All Recording", "Restart &All Files"),
            (ID_RESTART_SELECTED, "Start &Selected Recording...",
             "Start &Selected Files..."),
            (None, None, None),
            (ID_REMOVE_PROJECT, "Re&move Project...", "Re&move Project..."),
            (ID_SHOW_PROJECT_FOLDER, "Show project &folder",
             "Show project &folder"),
        ]
        self._edit_handlers = {
            ID_RESUME_RECORDING: self._resume_recording,
            ID_RESTART_PROJECT: self._restart_project,
            ID_RESTART_ALL: self._restart_all,
            ID_RESTART_SELECTED: self._restart_selected,
            ID_REMOVE_PROJECT: self._remove_project,
            ID_SHOW_PROJECT_FOLDER: self._show_project_folder,
        }
        self._edit_menu = wx.Menu()
        for item_id, tts_label, _stt_label in self._edit_items:
            if item_id is None:
                self._edit_menu.AppendSeparator()
            else:
                self._edit_menu.Append(item_id, tts_label)
        menubar.Append(self._edit_menu, "&Edit")

        tools_menu = wx.Menu()
        tools_menu.Append(ID_RECORD, "&Record...\tCtrl+Shift+R")
        tools_menu.Append(ID_SETTINGS, "&Settings...\tCtrl+,")
        menubar.Append(tools_menu, "&Tools")

        # Unique access keys inside each menu (R, U, A, C, B, T, S): three
        # items in this menu used to answer to Alt+A, so opening Help and
        # pressing A always chose the first one and the others were
        # unreachable from the keyboard.
        help_menu = wx.Menu()
        help_menu.Append(ID_README, "&Read Me")
        help_menu.Append(ID_USER_GUIDE, "&User Guide")
        help_menu.Append(ID_ADDON_GUIDE, "&Addon Development Guide")
        help_menu.Append(ID_ACCESSIBILITY_GUIDE, "Ac&cessibility Guidelines")
        help_menu.AppendSeparator()
        help_menu.Append(ID_PYTHON_BOOK, "Python & wxPython Book	F1")
        help_menu.Append(ID_THIRD_PARTY_LICENSES, "&Third-Party Licences")
        help_menu.Append(ID_WHATS_NEW, "What's &New")
        help_menu.AppendSeparator()
        help_menu.Append(ID_CHECK_UPDATES, "Check for Up&dates...")
        help_menu.Append(ID_ABOUT, "About AI Voice &Studio")
        menubar.Append(help_menu, "&Help")

        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, lambda _: self._new_project(), id=ID_NEW_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._open_project(), id=ID_OPEN_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._record(), id=ID_RECORD)
        self.Bind(wx.EVT_MENU, lambda _: self._settings(), id=ID_SETTINGS)
        self.Bind(wx.EVT_MENU, lambda _: self._readme(), id=ID_README)
        self.Bind(wx.EVT_MENU, lambda _: self._user_guide(), id=ID_USER_GUIDE)
        self.Bind(wx.EVT_MENU, lambda _: self._addon_guide(), id=ID_ADDON_GUIDE)
        self.Bind(wx.EVT_MENU, lambda _: self._accessibility_guide(), id=ID_ACCESSIBILITY_GUIDE)
        self.Bind(wx.EVT_MENU, lambda _: self._python_book(), id=ID_PYTHON_BOOK)
        self.Bind(wx.EVT_MENU, lambda _: self._third_party_licences(), id=ID_THIRD_PARTY_LICENSES)
        self.Bind(wx.EVT_MENU, lambda _: self._whats_new(), id=ID_WHATS_NEW)
        self.Bind(wx.EVT_MENU, lambda _: self._check_updates(), id=ID_CHECK_UPDATES)
        self.Bind(wx.EVT_MENU, lambda _: self._about(), id=ID_ABOUT)
        self.Bind(wx.EVT_MENU, lambda _: self._resume_recording(), id=ID_RESUME_RECORDING)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_project(), id=ID_RESTART_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_all(), id=ID_RESTART_ALL)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_selected(), id=ID_RESTART_SELECTED)
        self.Bind(wx.EVT_MENU, lambda _: self._remove_project(), id=ID_REMOVE_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._show_project_folder(),
                  id=ID_SHOW_PROJECT_FOLDER)
        self.Bind(wx.EVT_MENU, lambda _: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU_OPEN, lambda _: self._on_menu_open())

        # Frame-level accelerator table for the two shortcuts that belong to
        # a menu item.  These fire the same menu IDs, but do not depend on
        # the menu bar's own accelerator parsing, which can be unreliable for
        # Shift+letter combos on some wx builds.  Ctrl+Shift+N is *not* here:
        # it is the Create New Project button's gesture, routed by the char
        # hook below.
        self.SetAcceleratorTable(wx.AcceleratorTable([
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("R"), ID_RECORD),
            (wx.ACCEL_CTRL, ord(","), ID_SETTINGS),
        ]))

        # Belt-and-braces: a char hook catches the shortcuts even when the
        # accelerator table / menu accelerator parsing misses them (observed
        # on wx 3.3.3 msw for Ctrl+Shift+letter) - and it is the *only* route
        # Ctrl+Shift+N has, since it belongs to the welcome button.  Only
        # fires when no modal dialog is open, so it never duplicates the
        # dialog's own handling and never swallows ordinary typing.
        self.Bind(wx.EVT_CHAR_HOOK, self._on_global_char_hook)

    def _on_global_char_hook(self, evt: wx.KeyEvent) -> None:
        """Global keyboard hook for the three main shortcuts.

        wx's menu accelerators proved unreliable for Ctrl+Shift+letter on
        this wx build, so the hook routes the keys itself.  Ctrl+Shift+N is
        handed to the welcome panel's Create New Project button - that button
        owns the gesture, and there is no other binding for it anywhere.

        Measured with a real keystroke on wx 3.3.3 msw: a Shift+letter
        combination can arrive as key code 0 *and* with the modifier bits
        missing from the event, so the modifiers are confirmed against the
        physical keyboard and the held letter is identified by the event's
        scan code or, failing that, by the key state itself.  Every other key
        is passed straight on.
        """
        key = evt.GetKeyCode()
        mods = evt.GetModifiers()
        ctrl = bool(mods & wx.MOD_CONTROL)
        shift = bool(mods & wx.MOD_SHIFT)
        if wx.IsBusy():
            evt.Skip()
            return
        ctrl = ctrl or _physical_key_down(_VK_CONTROL)
        shift = shift or _physical_key_down(_VK_SHIFT)
        # A user-configured gesture (Settings > Keyboard gestures) runs before
        # built-in shortcuts, so a rebinding can take over any command - and a
        # gesture that matches is consumed here rather than falling through.
        if self._dispatch_gesture(evt):
            return
        if ctrl and shift and key in (0, ord("N"), ord("n"), ord("R"), ord("r")):
            scan = 0
            try:
                scan = (evt.GetRawKeyFlags() >> 16) & 0xFF
            except Exception:
                log.debug("Ignoring a failure in MainFrame._on_global_char_hook", exc_info=True)
            is_n = (key in (ord("N"), ord("n")) or scan == _SCAN_N
                    or _physical_key_down(_VK_N))
            is_r = (key in (ord("R"), ord("r")) or scan == _SCAN_R
                    or _physical_key_down(_VK_R))
            if is_n:
                self._press_new_project_button()
                return
            if is_r:
                self._record()
                return
        if ctrl and key in (ord(","),):
            self._settings()
            return
        evt.Skip()

    def _dispatch_gesture(self, evt: wx.KeyEvent) -> bool:
        """Run the command a configured gesture names; True when one matched."""
        gesture = gesture_from_event(evt)
        if not gesture:
            return False
        command_id = self._gestures.owner(gesture)
        if not command_id:
            return False
        name = self._gesture_actions.get(command_id)
        handler = getattr(self, name, None) if name else None
        if handler is None:
            return False
        log.info("Input gesture %s -> %s", gesture, command_id)
        handler()
        return True

    def _reload_gestures(self) -> None:
        """Pick up gesture changes the settings dialog has just saved."""
        self._gestures = gestures_model.GestureMap(
            self.settings.get("gestures.bindings", {}) or {}
        )

    def _press_new_project_button(self) -> None:
        """Do exactly what clicking the welcome panel's button does.

        Ctrl+Shift+N is that button's gesture: the key sends the button its
        own click event rather than calling the action directly, so the two
        ways into the New Project wizard cannot drift apart.
        """
        button = getattr(self, "new_project_btn", None)
        if button is None:  # pragma: no cover - the panel is always built
            self._new_project()
            return
        event = wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId())
        event.SetEventObject(button)
        button.GetEventHandler().ProcessEvent(event)

    def _build_welcome(self):
        # The welcome screen is three regions, each with its own background:
        #
        #   * an action bar spanning the full width, carrying the four main
        #     buttons and only as tall as they are plus a little padding;
        #   * a project sidebar on the left, holding the recent-projects list
        #     and separated from the rest by a one-pixel rule;
        #   * everything that is left over - the large empty area of the
        #     window - which is where the wallpaper goes.
        #
        # The wallpaper deliberately has a panel of its own rather than being
        # painted behind the whole screen: that is what keeps it off the
        # buttons and off the recent projects list, whichever theme is in use.
        panel = wx.Panel(self)
        self.welcome_panel = panel
        outer = wx.BoxSizer(wx.VERTICAL)

        bar = wx.Panel(panel)
        bar.SetName("Project actions")
        self.action_bar = bar
        bar_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for label, handler, tip in (
            ("Create New Project", self._new_project,
             "Open the New Project wizard to create a project from a document"
             " (Ctrl+Shift+N)"),
            ("Record (open a project)", self._record,
             "Open the most recent project for recording"),
            ("Settings (download voices)", self._settings,
             "Open Settings to download voices and configure the application"),
        ):
            btn = wx.Button(bar, label=label)
            btn.SetName(label)
            btn.SetToolTip(tip)
            btn.Bind(wx.EVT_BUTTON, lambda _, h=handler: h())
            if label == "Create New Project":
                # Ctrl+Shift+N activates this very button; keeping the
                # reference lets the frame's key handler press it instead of
                # calling the action behind it (see _press_new_project_button).
                self.new_project_btn = btn
            bar_sizer.Add(btn, 0, wx.LEFT | wx.TOP | wx.BOTTOM, _ACTION_BAR_PAD)

        # Alt+J removes the project selected in the list beside it.  It must
        # not be Alt+R: the menu bar answers Alt+R first in a frame (&Resume
        # Recording, and &Recent Projects in the File menu), so an R here
        # would be a key that looks available and never reaches this button.
        self.remove_btn = wx.Button(bar, label="Remove selected pro&ject")
        self.remove_btn.SetName("Remove selected project")
        self.remove_btn.SetToolTip(
            "Delete the selected project and all its recordings permanently "
            "(Alt+J)"
        )
        self.remove_btn.Bind(wx.EVT_BUTTON, lambda _: self._remove_project())
        bar_sizer.Add(self.remove_btn, 0, wx.ALL, _ACTION_BAR_PAD)
        bar.SetSizer(bar_sizer)
        outer.Add(bar, 0, wx.EXPAND)

        lower = wx.BoxSizer(wx.HORIZONTAL)

        sidebar = wx.Panel(panel)
        sidebar.SetName("Projects")
        sidebar.SetMinSize((_SIDEBAR_MIN_WIDTH, -1))
        self.sidebar = sidebar
        side_sizer = wx.BoxSizer(wx.VERTICAL)
        self._welcome_title = wx.StaticText(sidebar, label="Welcome to AI Voice Studio")
        self._welcome_title.SetFont(
            wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        side_sizer.Add(self._welcome_title, 0, wx.ALL, 12)
        self._welcome_recent_label = wx.StaticText(sidebar, label="Recent projects:")
        side_sizer.Add(self._welcome_recent_label, 0,
                       wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self.recent_list = wx.ListBox(sidebar, style=wx.LB_SINGLE)
        self.recent_list.SetName("Recent projects list")
        self.recent_list.SetToolTip("Select a project and press Enter or double-click to open it")
        side_sizer.Add(self.recent_list, 1,
                       wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self.recent_list.Bind(wx.EVT_LISTBOX_DCLICK, lambda _: self._open_recent_selected())
        self.recent_list.Bind(wx.EVT_KEY_DOWN, self._on_recent_key)
        self.recent_list.Bind(wx.EVT_CONTEXT_MENU, self._on_recent_context_menu)

        self._welcome_note_label = (
            "Right-click a project (or press the application-menu key) for "
            "more actions (resume, restart, remove, show the project folder) "
            "- also in the Edit menu."
        )
        self._welcome_note = wx.StaticText(sidebar, label=self._welcome_note_label)
        side_sizer.Add(self._welcome_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        sidebar.SetSizer(side_sizer)
        # The sidebar is narrow, so its long hint has to re-wrap to its width.
        sidebar.Bind(wx.EVT_SIZE, self._on_sidebar_size)
        lower.Add(sidebar, 0, wx.EXPAND)
        lower.Add(_DividerPanel(panel, self.settings), 0, wx.EXPAND)

        wallpaper = _WallpaperPanel(panel, self.settings)
        self.wallpaper_panel = wallpaper
        self._welcome_text_labels = [
            "Convert documents into spoken audio files with free, offline "
            "neural voices.",
            "Start with a new project: choose a document, choose how it is "
            "split, and record it. Voices are downloaded once from Settings "
            "and reused for all projects.",
        ]
        self._welcome_texts = [
            wx.StaticText(wallpaper, label=text)
            for text in self._welcome_text_labels
        ]
        wall_sizer = wx.BoxSizer(wx.VERTICAL)
        wall_sizer.Add(self._welcome_texts[0], 0, wx.ALL, 16)
        wall_sizer.Add(self._welcome_texts[1], 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 16)
        wallpaper.SetSizer(wall_sizer)
        # The descriptive text shares the empty area, so it has to re-wrap
        # whenever the wallpaper panel is resized.
        wallpaper.Bind(wx.EVT_SIZE, self._on_wallpaper_size)
        lower.Add(wallpaper, 1, wx.EXPAND)

        outer.Add(lower, 1, wx.EXPAND)
        panel.SetSizer(outer)
        self._refresh_recent_list()
        # Real MSAA accNames (e.g. the Recent projects list name).
        finalize_accessibility(self)
        self._apply_welcome_mode()
        # Scale-derived geometry and a live eye on the display: this is the
        # window a user moves between monitors with different scaling.  Last,
        # so the size that is scaled is the size the window settled on.
        dpi.prepare(self)
        dpi.follow(self)

    def _on_wallpaper_size(self, event) -> None:
        event.Skip()
        self._wrap_welcome_texts()

    def _wrap_welcome_texts(self) -> None:
        """Re-wrap the descriptive text to the empty area's current width."""
        panel = getattr(self, "wallpaper_panel", None)
        if panel is None:
            return
        width = panel.GetClientSize().width - 32
        if width < 80:
            return
        for text, label in zip(getattr(self, "_welcome_texts", []),
                               getattr(self, "_welcome_text_labels", []), strict=False):
            _wrap_label(text, label, width)

    def _on_sidebar_size(self, event) -> None:
        event.Skip()
        self._wrap_sidebar_texts()

    def _wrap_sidebar_texts(self) -> None:
        """Re-wrap the sidebar's right-click hint to the sidebar width."""
        sidebar = getattr(self, "sidebar", None)
        note = getattr(self, "_welcome_note", None)
        if sidebar is None or note is None:
            return
        width = sidebar.GetClientSize().width - 16
        if width < 80:
            return
        _wrap_label(note, self._welcome_note_label, width)

    def _apply_welcome_mode(self) -> None:
        """Show or hide the descriptive text depending on the wallpaper.

        The wallpaper only ever fills the empty region of the welcome screen
        (the panel beside the project sidebar); the action bar, the sidebar and
        the recent projects list keep the theme's plain colour.  That region is
        where the two descriptive paragraphs live too, so a wallpaper steps
        them aside; the title and the button labels stay put.
        """
        panel = getattr(self, "welcome_panel", None)
        if panel is None:
            return
        record = get_wallpaper(self.settings, current_theme())
        wallpaper = bool(record["enabled"])
        for text in getattr(self, "_welcome_texts", []):
            text.Show(not wallpaper)
        note = getattr(self, "_welcome_note", None)
        if note is not None:
            note.Show(not wallpaper)
        colors = resolve_palette(current_theme(), self.settings)
        try:
            self._welcome_title.SetForegroundColour(
                wx.Colour(colors.get("heading", colors.get("fg", "#ffffff")))
            )
        except Exception:
            log.debug("Ignoring a failure in MainFrame._apply_welcome_mode", exc_info=True)
        panel.Layout()
        # Wrapping changes a StaticText's height, so the layout runs again once
        # the text has been re-flowed to its new width.
        self._wrap_welcome_texts()
        self._wrap_sidebar_texts()
        panel.Layout()
        panel.Refresh()
        empty = getattr(self, "wallpaper_panel", None)
        if empty is not None:
            empty.Refresh()

    def _build_statusbar(self):
        self.CreateStatusBar(1)
        self.SetStatusText("Ready.")

    def _set_status(self, message: str) -> None:
        """Put ``message`` on the status bar and say it to the screen reader.

        The status bar is announced on demand only, so a message that is
        written here and never spoken is a message a blind user never hears
        (SPEC 3.2: "all status messages are written to a status bar").
        """
        self.SetStatusText(message)
        announce.announce(message, key="status")

    # ------------------------------------------------------------- actions
    def _new_project(self):
        wizard = NewProjectWizard(self, self.settings, self.store)
        wizard.RunWizard(wizard.page_details)
        wizard.Destroy()
        self._refresh_recent_list()

    def _open_project(self):
        with wx.FileDialog(
            self, "Open project", wildcard="AI Voice Studio project (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        project_folder = os.path.dirname(path)
        self._open_recording(project_folder)

    def _open_recent_selected(self):
        sel = self.recent_list.GetSelection()
        if sel < 0:
            return
        recents = self.settings.get("recent_projects", [])
        if sel < len(recents):
            self._open_recording(recents[sel]["path"])

    def _on_recent_key(self, evt):
        if evt.GetKeyCode() == wx.WXK_RETURN and self.recent_list.GetSelection() >= 0:
            self._open_recent_selected()
        else:
            evt.Skip()

    def _record(self):
        recents = self.settings.get("recent_projects", [])
        if not recents:
            wx.MessageBox(
                "No projects yet. Create a new project first (File, New Project "
                "or press Ctrl+Shift+N).",
                "Record", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        self._open_recording(recents[0]["path"])

    def _open_recording(self, project_folder: str, start_index: int | None = None,
                        single_segment: bool = False):
        # A Speech-to-Text project is reopened in its own transcribe window and
        # an Audio description project in its own describe window, never in the
        # TTS Recording window (their segments carry no text to record).
        if start_index is None and not single_segment:
            try:
                data = project.load_project(project_folder)
            except (OSError, ValueError, KeyError):
                data = {}
            if data.get("project_type") == PROJECT_TYPE_STT:
                self._open_transcribe(project_folder)
                return
            if data.get("project_type") == PROJECT_TYPE_AD:
                self._open_describe(project_folder)
                return
        try:
            dlg = RecordingDialog(
                self, project_folder, self.settings, self.store,
                start_index=start_index, single_segment=single_segment,
            )
        except (OSError, ValueError, KeyError) as exc:
            wx.MessageBox(f"Could not open the project: {exc}", "Open project",
                          style=wx.OK | wx.ICON_ERROR)
            return
        dlg.ShowModal()
        dlg.Destroy()
        self._refresh_recent_list()

    def _open_transcribe(self, project_folder: str, sources: list | None = None):
        """Open the transcribe window, optionally for a subset of sources.

        ``sources`` is the file list the Edit > Start Selected Files picker
        chose; ``None`` transcribes every source of the project.
        """
        try:
            dlg = STTTranscribeDialog(
                self, project_folder, self.settings, self.store,
                sources=sources,
            )
        except (OSError, ValueError, KeyError) as exc:
            wx.MessageBox(f"Could not open the project: {exc}", "Open project",
                          style=wx.OK | wx.ICON_ERROR)
            return
        dlg.ShowModal()
        dlg.Destroy()
        self._refresh_recent_list()

    def _open_describe(self, project_folder: str, sources: list | None = None):
        """Open the audio description window, optionally for a subset of videos.

        ``sources`` is the file list the Edit > Start Selected Descriptions
        picker chose; ``None`` describes every source of the project.
        """
        try:
            dlg = DescribeDialog(
                self, project_folder, self.settings, self.store,
                sources=sources,
            )
        except (OSError, ValueError, KeyError) as exc:
            wx.MessageBox(f"Could not open the project: {exc}", "Open project",
                          style=wx.OK | wx.ICON_ERROR)
            return
        dlg.ShowModal()
        dlg.Destroy()
        self._refresh_recent_list()

    def _settings(self):
        dlg = SettingsDialog(self, self.settings, self.store)
        dlg.ShowModal()
        dlg.Destroy()
        apply_theme(self, self.settings.theme, self.settings)
        self._apply_welcome_mode()
        # Settings > Keyboard gestures may have rebound anything.
        self._reload_gestures()
        self._refresh_recent_list()

    def _refresh_recent_list(self):
        recents = self.settings.get("recent_projects", [])
        self.recent_list.Clear()
        for recent in recents:
            self.recent_list.Append(recent["name"])

    def _selected_recent(self):
        """The recent project the user selected (or the only one, if single)."""
        recents = self.settings.get("recent_projects", [])
        if not recents:
            return None
        sel = self.recent_list.GetSelection()
        if 0 <= sel < len(recents):
            return recents[sel]
        if len(recents) == 1:
            return recents[0]
        return None

    def _require_selected(self) -> dict | None:
        recent = self._selected_recent()
        if recent is None:
            wx.MessageBox(
                "Select a project in the Recent projects list first.",
                "AI Voice Studio", style=wx.OK | wx.ICON_INFORMATION,
            )
        return recent

    def _on_menu_open(self) -> None:
        """Refresh the menus that depend on the current selection/state."""
        self._rebuild_recent()
        self._refresh_edit_menu()

    def _selected_project_base(self) -> str:
        """The base of the selected recent project (``tts``, ``stt`` or ``ad``)."""
        recent = self._selected_recent()
        if not recent:
            return DEFAULT_PROJECT_BASE
        try:
            data = project.load_project(recent["path"])
        except (OSError, ValueError, KeyError):
            return DEFAULT_PROJECT_BASE
        if data.get("project_type") == PROJECT_TYPE_STT:
            return PROJECT_BASE_STT
        if data.get("project_type") == PROJECT_TYPE_AD:
            return PROJECT_BASE_AD
        return DEFAULT_PROJECT_BASE

    @staticmethod
    def _edit_label(entry, base: str) -> str:
        """The Edit-menu text for one entry, given the selected project base.

        The TTS and STT wording comes straight from the entry; an audio
        description project has its own wording, keyed by the same item id, so
        the menu keeps one entry list while speaking three ways.
        """
        item_id, tts_label, stt_label = entry
        if base == PROJECT_BASE_AD:
            return _AD_EDIT_LABELS.get(item_id, tts_label)
        return stt_label if base == PROJECT_BASE_STT else tts_label

    def _refresh_edit_menu(self) -> None:
        """Re-label the Edit menu for the selected project's nature."""
        menu = getattr(self, "_edit_menu", None)
        if menu is None:
            return
        base = self._selected_project_base()
        for entry in getattr(self, "_edit_items", []):
            item_id = entry[0]
            if item_id is None:
                continue
            item = menu.FindItemById(item_id)
            if item is not None:
                item.SetItemLabel(self._edit_label(entry, base))

    def _on_recent_context_menu(self, evt):
        # Right-click carries a position; the keyboard application-menu key
        # (and Shift+F10) does not, and then the already-selected project is
        # the one the menu acts on.
        pos = evt.GetPosition()
        if pos != wx.DefaultPosition:
            item = self.recent_list.HitTest(self.recent_list.ScreenToClient(pos))
            if item != wx.NOT_FOUND:
                self.recent_list.SetSelection(item)
        base = self._selected_project_base()
        menu = wx.Menu()
        for entry in getattr(self, "_edit_items", []):
            item_id = entry[0]
            if item_id is None:
                menu.AppendSeparator()
                continue
            new_id = wx.NewIdRef()
            menu.Append(new_id, self._edit_label(entry, base))
            handler = self._edit_handlers[item_id]
            menu.Bind(wx.EVT_MENU, lambda _, h=handler: h(), id=new_id.GetId())
        self.recent_list.PopupMenu(menu)
        menu.Destroy()

    def _show_project_folder(self):
        """Open the selected project's folder in the OS file manager."""
        recent = self._require_selected()
        if not recent:
            return
        folder = recent["path"]
        if not os.path.isdir(folder):
            wx.MessageBox(
                f"The project folder no longer exists:\n{folder}",
                "Show project folder", style=wx.OK | wx.ICON_WARNING,
            )
            return
        open_folder(folder)
        self._set_status(f"Opened {folder}")

    # --------------------------------------------------- project lifecycle
    def _resume_recording(self):
        recent = self._require_selected()
        if recent:
            self._open_recording(recent["path"])

    def _restart_project(self):
        recent = self._require_selected()
        if not recent:
            return
        if wx.MessageBox(
            f"Restart project '{recent['name']}'?\n\n"
            "All recordings or transcribe files and project settings will be "
            "deleted; only the project name is kept. You will choose a "
            "document and the audio creation or transcribe options again.",
            "Restart project", style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        pdir = recent["path"]
        name = recent["name"]
        project.remove_project(pdir)
        wizard = NewProjectWizard(self, self.settings, self.store, initial_name=name)
        finished = wizard.RunWizard(wizard.page_details)
        wizard.Destroy()
        if not finished:
            # The folder is gone; drop the stale recent entry.
            self.settings.remove_recent_project(pdir)
        self._refresh_recent_list()

    def _restart_all(self):
        recent = self._require_selected()
        if not recent:
            return
        if wx.MessageBox(
            f"Restart all recording or transcribe files for "
            f"'{recent['name']}'?\n\n"
            "All recorded or transcribe files will be deleted and every "
            "segment will be recorded or transcribed again from the "
            "beginning.",
            "Restart all files", style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        removed = project.reset_recordings(recent["path"])
        removed += self._clear_describe_outputs(recent["path"])
        self._set_status(
            f"Deleted {removed} file(s); the project starts from the beginning."
        )
        self._open_recording(recent["path"])

    def _clear_describe_outputs(self, project_dir: str) -> int:
        """Delete an audio description project's documents; returns how many.

        *Restart all* means "throw away everything this project produced", so an
        audio description project has to lose its scripts and subtitle tracks
        too - otherwise the menu entry would claim a restart and leave the old
        descriptions behind.  Any other project type is left untouched.
        """
        try:
            data = project.load_project(project_dir)
        except (OSError, ValueError, KeyError):
            return 0
        if data.get("project_type") != PROJECT_TYPE_AD:
            return 0
        from ..describe import output as describe_output

        return describe_output.remove_outputs(project_dir, data)

    def _restart_selected(self):
        recent = self._require_selected()
        if not recent:
            return
        try:
            data = project.load_project(recent["path"])
        except (OSError, ValueError, KeyError) as exc:
            wx.MessageBox(f"Could not open the project: {exc}",
                          "Start selected recording", style=wx.OK | wx.ICON_ERROR)
            return
        # A transcription project holds transcribe files and an audio
        # description project holds videos, not recordings, so each gets its
        # own picker (see _StartTranscribeDialog and _StartDescribeDialog).
        if data.get("project_type") == PROJECT_TYPE_STT:
            self._restart_selected_stt(recent, data)
            return
        if data.get("project_type") == PROJECT_TYPE_AD:
            self._restart_selected_ad(recent, data)
            return
        # Use the audio files actually on disk (not just segment metadata) so
        # "select an audio file" shows everything that was recorded.
        saved = project.recorded_files(recent["path"])
        # Deliberately no "has anything been recorded?" guard: a long book may
        # be recorded out of order (chapter 75 of 100 today, the rest later),
        # so the picker always opens.  With nothing recorded yet it starts in
        # "select by file break" mode (see _StartRecordingDialog).  The only
        # real requirement is a project whose document was split into breaks.
        if not data.get("segments"):
            wx.MessageBox(
                "This project has no file breaks yet, so there is nothing to "
                "record.\n\nOpen the project once (double-click it in the "
                "list) so its document is split into file breaks, then try "
                "again.",
                "Start selected recording", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        dlg = _StartRecordingDialog(self, _picker_entries(data, saved))
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            start_index = dlg.start_index()
            only_selected = dlg.only_selected()
            chosen = dlg.chosen_file()
        finally:
            dlg.Destroy()
        if start_index is None:
            return
        # Delete the chosen recording so it is recorded again.  In "select by
        # file break" mode the chosen break may not have a file yet, and in
        # "record all files from here" mode the breaks after it are simply
        # overwritten as the run goes on.
        if chosen:
            project.remove_recording(recent["path"], chosen)
        self._open_recording(
            recent["path"], start_index=start_index, single_segment=only_selected,
        )

    def _restart_selected_stt(self, recent: dict, data: dict) -> None:
        """Start Selected Files for a speech-to-text project.

        The picker names the project's base (always *Speech to text* here),
        then offers the transcribe files already in the project or every file
        break, and finally whether to transcribe only the chosen file or the
        chosen file and every file after it.  The chosen files are handed to
        the transcribe window, which runs the job for them alone.
        """
        entries = _transcribe_entries(data)
        if not entries:
            wx.MessageBox(
                "This project has no transcribe files yet, so there is "
                "nothing to transcribe.\n\nCreate the project again with at "
                "least one source file, then try again.",
                "Start selected files", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        dlg = _StartTranscribeDialog(self, entries)
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            start_index = dlg.start_index()
            only_selected = dlg.only_selected()
        finally:
            dlg.Destroy()
        if start_index is None:
            return
        chosen = (entries[start_index:start_index + 1] if only_selected
                  else entries[start_index:])
        chosen_sources = [entry["source"] for entry in chosen
                          if entry.get("source")]
        if not chosen_sources:
            return
        self._open_transcribe(recent["path"], chosen_sources)

    def _restart_selected_ad(self, recent: dict, data: dict) -> None:
        """Start Selected Descriptions for an audio description project.

        The picker names the project's base (always *Audio description* here),
        then offers the videos already in the project or every file break, and
        finally whether to describe only the chosen file or the chosen file and
        every file after it.  The chosen videos are handed to the describe
        window, which runs the job for them alone.
        """
        entries = _describe_entries(data)
        if not entries:
            wx.MessageBox(
                "This project has no video files yet, so there is nothing to "
                "describe.\n\nCreate the project again with at least one video "
                "file, then try again.",
                "Start selected descriptions", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        dlg = _StartDescribeDialog(self, entries)
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            start_index = dlg.start_index()
            only_selected = dlg.only_selected()
        finally:
            dlg.Destroy()
        if start_index is None:
            return
        chosen = (entries[start_index:start_index + 1] if only_selected
                  else entries[start_index:])
        chosen_sources = [entry["source"] for entry in chosen
                          if entry.get("source")]
        if not chosen_sources:
            return
        self._open_describe(recent["path"], chosen_sources)

    def _remove_project(self):
        recent = self._require_selected()
        if not recent:
            return
        if wx.MessageBox(
            f"Remove project '{recent['name']}' and ALL its recordings from "
            "your hard disk?\n\nThis cannot be undone.",
            "Remove project", style=wx.YES_NO | wx.ICON_WARNING,
        ) != wx.YES:
            return
        project.remove_project(recent["path"])
        self.settings.remove_recent_project(recent["path"])
        self._refresh_recent_list()
        self._set_status(f"Removed project '{recent['name']}'.")

    def _rebuild_recent(self):
        if not self._recent_menu:
            return
        # wx.Menu has no Clear(); delete items individually (crash fix:
        # "AttributeError: 'Menu' object has no attribute 'Clear'").
        for item in list(self._recent_menu.GetMenuItems()):
            self._recent_menu.Delete(item)
        recents = self.settings.get("recent_projects", [])
        if not recents:
            item = self._recent_menu.Append(wx.ID_ANY, "No recent projects")
            item.Enable(False)
            return
        for recent in recents:
            # Unique ID per item (wx.NewIdRef()): manual arithmetic on
            # ID_RECENT_BASE used to collide with the sequentially-generated
            # IDs of Record/Settings/Help menu items, so Ctrl+, (Settings)
            # opened the Recording window of a recent project instead.
            item_id = wx.NewIdRef()
            item = self._recent_menu.Append(
                item_id,
                f"{recent['name']}  ({os.path.dirname(recent['path'])})",
            )
            self.Bind(
                wx.EVT_MENU,
                lambda _, r=recent: self._open_recording(r["path"]),
                id=item.GetId(),
            )

    # -------------------------------------------------------------- help
    def _doc_path(self, name: str) -> str:
        """The packaged HTML doc, or the repository copy when running from
        source (e.g. the PyInstaller bundle keeps docs next to the exe)."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(here, "docs", name),
            os.path.join(os.getcwd(), "docs", name),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _open_doc(self, name: str, fallback_title: str):
        import webbrowser

        path = self._doc_path(name)
        if not path:
            wx.MessageBox(
                f"The documentation file ({name}) was not found next to the "
                "application.",
                fallback_title, style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        try:
            webbrowser.open("file:///" + path.replace("\\", "/"))
        except Exception:
            os.startfile(path)  # type: ignore[attr-defined]

    def _readme(self):
        self._open_doc("README.html", "Read Me - AI Voice Studio")

    def _user_guide(self):
        self._open_doc("UserGuide.html", "User Guide - AI Voice Studio")

    def _addon_guide(self):
        self._open_doc("AddonDevelopmentGuide.html", "Addon Development Guide - AI Voice Studio")

    def _accessibility_guide(self):
        self._open_doc("AccessibilityGuide.html", "Accessibility Guidelines - AI Voice Studio")

    def _third_party_licences(self):
        self._open_doc("THIRD-PARTY-LICENSES.html", "Third-Party Licences - AI Voice Studio")

    def _whats_new(self):
        self._open_doc("WHATS-NEW.html", "What's New - AI Voice Studio")

    def _check_updates(self):
        """Help ▸ Check for updates: ask GitHub for the latest release."""
        from .update_dialog import check_for_updates_interactive

        check_for_updates_interactive(self, self.settings)

    def _python_book(self):
        """Open the Python & wxPython book in the default browser."""
        import webbrowser

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # In PyInstaller bundles, docs live next to the package folder
        # (e.g. _internal/docs/), so also check one level up from the package.
        parent = os.path.dirname(here)
        candidates = [
            os.path.join(here, "docs", "book", "index.html"),
            os.path.join(parent, "docs", "book", "index.html"),
            os.path.join(os.getcwd(), "docs", "book", "index.html"),
        ]
        path = ""
        for candidate in candidates:
            if os.path.isfile(candidate):
                path = candidate
                break
        if not path:
            wx.MessageBox(
                "The Python & wxPython book (docs/book/index.html) was not found.\n"
                "Please reinstall AI Voice Studio to get the book.",
                "Python & wxPython Book", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        try:
            webbrowser.open("file:///" + path.replace("\\", "/"))
        except Exception:
            os.startfile(path)  # type: ignore[attr-defined]

    def _about(self):
        import wx.adv

        info = wx.adv.AboutDialogInfo()
        info.SetName("AI Voice Studio")
        info.SetVersion(__version__)
        info.SetDescription(
            "Convert PDF, TXT, DOC/DOCX, HTML, Markdown and clipboard text into\n"
            "spoken WAV/MP3/FLAC audio using free, offline ONNX neural voices.\n\n"
            "Built with wxPython and sherpa-onnx. Screen-reader friendly."
        )
        info.SetCopyright("(c) Anuj Sharma - GPL-3.0")
        wx.adv.AboutBox(info)


def _picker_entries(data: dict, saved: list[str]) -> list[dict]:
    """One entry per project segment for the picker dialog.

    Each entry is ``{"position", "title", "file"}``: the 0-based segment
    position the synthesis worker starts from, the file break's name, and the
    recorded audio file that break already has (``None`` when it is still to
    be recorded).  The recorded files on disk are matched back to their
    break through the project's ``saved`` field first and by file name
    second, so a project whose metadata is out of date still lines up.
    """
    from ..util import sanitize_filename

    segments = data.get("segments", []) or []
    entries = []
    by_stem: dict[str, int] = {}
    by_name: dict[str, int] = {}
    for position, seg in enumerate(segments):
        title = str(seg.get("title") or f"segment {position + 1}")
        entries.append({"position": position, "title": title, "file": None})
        by_stem.setdefault(sanitize_filename(title, 80), position)
        stored = os.path.basename(str(seg.get("saved") or ""))
        if stored:
            by_name.setdefault(stored, position)
    for name in saved:
        position = by_name.get(name)
        if position is None:
            position = by_stem.get(os.path.splitext(name)[0])
        if position is None:
            continue  # A stray file that no file break refers to.
        if entries[position]["file"] is None:
            entries[position]["file"] = name
    return entries


def _transcribe_entries(data: dict) -> list[dict]:
    """One entry per source file of a speech-to-text project.

    Each entry is ``{"position", "title", "source", "file"}``: the 0-based
    position the transcribe window starts from, the file break's name, the
    full path of the source file, and that file's name (``None`` when the
    break has no source).  A transcription project stores one segment per
    source file (see the New Project wizard), so the break's ``title`` is the
    source file's name and ``file`` is what the picker lists under "Select a
    transcribe file".
    """
    segments = data.get("segments", []) or []
    entries = []
    for position, seg in enumerate(segments):
        source = str(seg.get("source") or "")
        title = str(
            seg.get("title") or os.path.basename(source)
            or f"file {position + 1}"
        )
        entries.append({
            "position": position,
            "title": title,
            "source": source,
            "file": os.path.basename(source) if source else None,
        })
    return entries


def _describe_entries(data: dict) -> list[dict]:
    """One entry per source video of an audio description project.

    Each entry is ``{"position", "title", "source", "file"}``: the 0-based
    position the describe window starts from, the file break's name, the full
    path of the source video, and that file's name (``None`` when the break has
    no source).  An audio description project stores one segment per source file
    (see the New Project wizard), so the break's ``title`` is the video's name
    and ``file`` is what the picker lists under "Select a video file".
    """
    segments = data.get("segments", []) or []
    entries = []
    for position, seg in enumerate(segments):
        source = str(seg.get("source") or "")
        title = str(
            seg.get("title") or os.path.basename(source)
            or f"file {position + 1}"
        )
        entries.append({
            "position": position,
            "title": title,
            "source": source,
            "file": os.path.basename(source) if source else None,
        })
    return entries


class _StartRecordingDialog(wx.Dialog):
    """Choose which file to record and how far to record (Edit menu).

    The sidebar radio buttons choose how the combo box is filled:

    * **Select an audio file** (default) - the combo lists the audio files
      already recorded in the project folder, so one of them is recorded
      again.
    * **Select by file break** - the combo lists every file break of the
      project, recorded or not, so recording starts at any break.

    The second radio group chooses the scope: only the chosen file, or the
    chosen file and every file after it.
    """

    def __init__(self, parent, entries, base_label="Text to speech"):
        super().__init__(parent, title="Start selected recording", size=(600, 360))
        self._entries = [dict(entry) for entry in entries]
        self._focus_done = False

        outer = wx.BoxSizer(wx.VERTICAL)

        # Project base: exactly one item - the selected project's base - so
        # the dialog always says which kind of project it acts on.  A read-only
        # combo is used rather than a label so a screen reader announces it in
        # the same "Project base" voice as the rest of the picker.
        base_row = wx.BoxSizer(wx.HORIZONTAL)
        base_row.Add(wx.StaticText(self, label="Project base"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.base_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Project base")
        self.base_combo.Append(base_label, "")
        self.base_combo.SetSelection(0)
        base_row.Add(self.base_combo, 1, wx.EXPAND)
        outer.Add(base_row, 0, wx.EXPAND | wx.ALL, 8)

        row = wx.BoxSizer(wx.HORIZONTAL)

        # -- sidebar: how the combo is filled -------------------------------
        side_panel = wx.Panel(self)
        sidebar = wx.BoxSizer(wx.VERTICAL)
        self.file_radio = wx.RadioButton(
            side_panel, label="Select an audio file", style=wx.RB_GROUP)
        self.file_radio.SetToolTip(
            "Choose one audio file that was already recorded, and record it again")
        self.break_radio = wx.RadioButton(side_panel, label="Select by file break")
        self.break_radio.SetToolTip(
            "Choose any file break of the project - recorded or not - and "
            "start recording at it")
        for radio in (self.file_radio, self.break_radio):
            radio.SetName(radio.GetLabel())
            sidebar.Add(radio, 0, wx.ALL, 6)
        side_panel.SetSizerAndFit(sidebar)
        side_panel.SetMinSize((240, -1))
        side_panel.SetName("Recording selection mode")
        if not any(entry.get("file") for entry in self._entries):
            # Nothing recorded yet: the only useful mode is by file break.
            self.break_radio.SetValue(True)
        row.Add(side_panel, 0, wx.EXPAND | wx.ALL, 8)

        # -- combo + scope ---------------------------------------------------
        content = wx.BoxSizer(wx.VERTICAL)
        self.choice_label = wx.StaticText(self, label="")
        self.choice_label.SetName("Recording selection")
        content.Add(self.choice_label, 0, wx.BOTTOM, 4)
        self.combo = wx.ComboBox(self, style=wx.CB_READONLY)
        content.Add(self.combo, 0, wx.EXPAND | wx.BOTTOM, 12)

        self.only_radio = wx.RadioButton(
            self, label="Only record selected file", style=wx.RB_GROUP)
        self.only_radio.SetToolTip(
            "Record just the chosen file; the files after it are left alone")
        self.all_radio = wx.RadioButton(
            self, label="Record all files from here")
        self.all_radio.SetToolTip(
            "Record the chosen file and then every file after it to the end "
            "of the project")
        for radio in (self.only_radio, self.all_radio):
            radio.SetName(radio.GetLabel())
            content.Add(radio, 0, wx.ALL, 6)

        content.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0,
                    wx.EXPAND | wx.TOP, 12)
        row.Add(content, 1, wx.EXPAND | wx.ALL, 8)
        outer.Add(row, 1, wx.EXPAND)
        self.SetSizer(outer)

        self.file_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self.break_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self._fill_combo()
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)
        self.SetMinSize(self.GetSize())
        if parent is not None and parent.IsShown():
            self.CentreOnParent()
        else:
            self.Centre()
        # Focus the sidebar mode radio button - the first control of the
        # dialog and the checked one - so a screen reader starts on "Select an
        # audio file" instead of jumping straight into the combo box.  It has
        # to wait for the show event: a dialog hands the focus to its default
        # (OK) button while it is being created, and that would win.
        self.Bind(wx.EVT_SHOW, self._on_show)

    # ---------------------------------------------------------------- focus
    def _on_show(self, event):
        """Move the cursor to the mode button once the dialog is on screen."""
        event.Skip()
        if event.IsShown() and not self._focus_done:
            self._focus_done = True
            wx.CallAfter(self._focus_first_control)

    def initial_focus_control(self):
        """The control the dialog focuses when it opens.

        The sidebar mode radio button comes first: it says how the combo box
        below is filled, so it is what the user (and a screen reader) has to
        meet first.  The checked one is chosen, which is always "Select an
        audio file" unless nothing has been recorded yet.
        """
        return self.break_radio if self.break_radio.GetValue() else self.file_radio

    def _focus_first_control(self):
        """Put the keyboard cursor on :meth:`initial_focus_control`."""
        self.initial_focus_control().SetFocus()

    # ------------------------------------------------------------------ state
    def _fill_combo(self):
        """(Re)fill the combo box for the selected sidebar mode."""
        self.combo.Clear()
        # The combo's accessible name is the row label itself (minus its
        # colon), so what a screen reader announces is exactly the sentence
        # on screen - speech input can then use the visible wording, and the
        # two can never drift apart.
        if self.file_radio.GetValue():
            self.choice_label.SetLabel(
                "Choose which recorded file to record again:")
            update_accessible_name(
                self.combo, "Choose which recorded file to record again")
            self.combo.SetToolTip(
                "Audio files that are already recorded in this project")
            items = [(entry["file"], entry["position"])
                     for entry in self._entries if entry.get("file")]
        else:
            self.choice_label.SetLabel(
                "Choose which file to record by file break:")
            update_accessible_name(
                self.combo, "Choose which file to record by file break")
            self.combo.SetToolTip(
                "Every file break of the project, recorded or not")
            items = [(entry["title"], entry["position"])
                     for entry in self._entries]
        for label, position in items:
            self.combo.Append(str(label), position)
        if self.combo.GetCount():
            self.combo.SetSelection(0)
        self.combo.Enable(self.combo.GetCount() > 0)
        ok_btn = self.FindWindowById(wx.ID_OK)
        if ok_btn is not None:
            ok_btn.Enable(self.combo.GetCount() > 0)
        self.Layout()

    def _on_mode(self, _evt):
        self._fill_combo()

    def _selected_entry(self) -> dict | None:
        position = self.start_index()
        if position is None:
            return None
        for entry in self._entries:
            if entry["position"] == position:
                return entry
        return None

    # ---------------------------------------------------------------- answers
    def start_index(self) -> int | None:
        """0-based segment position to start recording at (``None`` = none)."""
        sel = self.combo.GetSelection()
        if sel < 0:
            return None
        value = self.combo.GetClientData(sel)
        return None if value is None else int(value)

    def only_selected(self) -> bool:
        """True = record only the chosen file; False = record from it on."""
        return bool(self.only_radio.GetValue())

    def chosen_file(self) -> str | None:
        """The recorded audio file of the chosen break, when it has one."""
        entry = self._selected_entry()
        if entry is None:
            return None
        if entry.get("file"):
            return entry["file"]
        if self.file_radio.GetValue():
            return self.combo.GetStringSelection() or None
        return None


class _StartTranscribeDialog(wx.Dialog):
    """Choose which transcribe file to transcribe and how far (Edit menu).

    The Edit menu's *Start Selected Files...* action opens this dialog for a
    speech-to-text project.  It mirrors :class:`_StartRecordingDialog`: a
    read-only **Project base** combo names the project's base, the sidebar
    radio buttons choose how the combo box is filled, and the second radio
    group chooses the scope.

    * **Select a transcribe file** (default) - the combo lists the source
      files the project already holds, so one of them is transcribed again.
    * **Select by file break** - the combo lists every file break of the
      project, so transcription starts at any of them.

    The second radio group chooses the scope: only the chosen file, or the
    chosen file and every file after it.
    """

    def __init__(self, parent, entries, base_label="Speech to text"):
        super().__init__(parent, title="Start selected files", size=(620, 400))
        self._entries = [dict(entry) for entry in entries]
        self._focus_done = False

        outer = wx.BoxSizer(wx.VERTICAL)

        # Project base: exactly one item - the selected project's base.
        base_row = wx.BoxSizer(wx.HORIZONTAL)
        base_row.Add(wx.StaticText(self, label="Project base"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.base_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Project base")
        self.base_combo.Append(base_label, "")
        self.base_combo.SetSelection(0)
        base_row.Add(self.base_combo, 1, wx.EXPAND)
        outer.Add(base_row, 0, wx.EXPAND | wx.ALL, 8)

        row = wx.BoxSizer(wx.HORIZONTAL)

        # -- sidebar: how the combo is filled -------------------------------
        side_panel = wx.Panel(self)
        sidebar = wx.BoxSizer(wx.VERTICAL)
        self.file_radio = wx.RadioButton(
            side_panel, label="Select a transcribe file", style=wx.RB_GROUP)
        self.file_radio.SetToolTip(
            "Choose one transcribe file of this project and transcribe it "
            "again")
        self.break_radio = wx.RadioButton(
            side_panel, label="Select by file break")
        self.break_radio.SetToolTip(
            "Choose any file break of the project and start transcribing "
            "at it")
        for radio in (self.file_radio, self.break_radio):
            radio.SetName(radio.GetLabel())
            sidebar.Add(radio, 0, wx.ALL, 6)
        side_panel.SetSizerAndFit(sidebar)
        side_panel.SetMinSize((240, -1))
        side_panel.SetName("Transcribe selection mode")
        if not any(entry.get("file") for entry in self._entries):
            # No source files: the only useful mode is by file break.
            self.break_radio.SetValue(True)
        row.Add(side_panel, 0, wx.EXPAND | wx.ALL, 8)

        # -- combo + scope ---------------------------------------------------
        content = wx.BoxSizer(wx.VERTICAL)
        self.choice_label = wx.StaticText(self, label="")
        self.choice_label.SetName("Transcribe selection")
        content.Add(self.choice_label, 0, wx.BOTTOM, 4)
        self.combo = wx.ComboBox(self, style=wx.CB_READONLY)
        content.Add(self.combo, 0, wx.EXPAND | wx.BOTTOM, 12)

        self.only_radio = wx.RadioButton(
            self, label="Only transcribe selected file", style=wx.RB_GROUP)
        self.only_radio.SetToolTip(
            "Transcribe just the chosen file; the files after it are left "
            "alone")
        self.all_radio = wx.RadioButton(
            self, label="Transcribe all files from here")
        self.all_radio.SetToolTip(
            "Transcribe the chosen file and then every file after it to the "
            "end of the project")
        for radio in (self.only_radio, self.all_radio):
            radio.SetName(radio.GetLabel())
            content.Add(radio, 0, wx.ALL, 6)

        content.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0,
                    wx.EXPAND | wx.TOP, 12)
        row.Add(content, 1, wx.EXPAND | wx.ALL, 8)
        outer.Add(row, 1, wx.EXPAND)
        self.SetSizer(outer)

        self.file_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self.break_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self._fill_combo()
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)
        self.SetMinSize(self.GetSize())
        if parent is not None and parent.IsShown():
            self.CentreOnParent()
        else:
            self.Centre()
        # Focus the checked sidebar mode button, not the combo, exactly as
        # the recording picker does; the dialog's own focus has to be undone
        # after it is shown.
        self.Bind(wx.EVT_SHOW, self._on_show)

    # ---------------------------------------------------------------- focus
    def _on_show(self, event):
        """Move the cursor to the mode button once the dialog is on screen."""
        event.Skip()
        if event.IsShown() and not self._focus_done:
            self._focus_done = True
            wx.CallAfter(self._focus_first_control)

    def initial_focus_control(self):
        """The control the dialog focuses when it opens."""
        return self.break_radio if self.break_radio.GetValue() else self.file_radio

    def _focus_first_control(self):
        """Put the keyboard cursor on :meth:`initial_focus_control`."""
        self.initial_focus_control().SetFocus()

    # ------------------------------------------------------------------ state
    def _fill_combo(self):
        """(Re)fill the combo box for the selected sidebar mode."""
        self.combo.Clear()
        if self.file_radio.GetValue():
            self.choice_label.SetLabel(
                "Choose which transcribed file to transcribe again:")
            update_accessible_name(
                self.combo,
                "Choose which transcribed file to transcribe again")
            self.combo.SetToolTip(
                "Source files that are already part of this project")
            items = [(entry["file"], entry["position"])
                     for entry in self._entries if entry.get("file")]
        else:
            self.choice_label.SetLabel(
                "Choose which file to transcribe by file break:")
            update_accessible_name(
                self.combo, "Choose which file to transcribe by file break")
            self.combo.SetToolTip(
                "Every file break of the project")
            items = [(entry["title"], entry["position"])
                     for entry in self._entries]
        for label, position in items:
            self.combo.Append(str(label), position)
        if self.combo.GetCount():
            self.combo.SetSelection(0)
        self.combo.Enable(self.combo.GetCount() > 0)
        ok_btn = self.FindWindowById(wx.ID_OK)
        if ok_btn is not None:
            ok_btn.Enable(self.combo.GetCount() > 0)
        self.Layout()

    def _on_mode(self, _evt):
        self._fill_combo()

    def _selected_entry(self) -> dict | None:
        position = self.start_index()
        if position is None:
            return None
        for entry in self._entries:
            if entry["position"] == position:
                return entry
        return None

    # ---------------------------------------------------------------- answers
    def start_index(self) -> int | None:
        """0-based segment position to start transcribing at (``None`` = none)."""
        sel = self.combo.GetSelection()
        if sel < 0:
            return None
        value = self.combo.GetClientData(sel)
        return None if value is None else int(value)

    def only_selected(self) -> bool:
        """True = transcribe only the chosen file; False = from it on."""
        return bool(self.only_radio.GetValue())

    def chosen_file(self) -> str | None:
        """The transcribe file of the chosen break, when it has one."""
        entry = self._selected_entry()
        if entry is None:
            return None
        if entry.get("file"):
            return entry["file"]
        if self.file_radio.GetValue():
            return self.combo.GetStringSelection() or None
        return None


class _StartDescribeDialog(wx.Dialog):
    """Choose which video to describe and how far (Edit menu).

    The Edit menu's *Start Selected Descriptions...* action opens this dialog for
    an audio description project.  It mirrors :class:`_StartTranscribeDialog`: a
    read-only **Project base** combo names the project's base, the sidebar radio
    buttons choose how the combo box is filled, and the second radio group
    chooses the scope.

    * **Select a video file** (default) - the combo lists the videos the project
      already holds, so one of them is described again.
    * **Select by file break** - the combo lists every file break of the project,
      so describing starts at any of them.

    The second radio group chooses the scope: only the chosen file, or the chosen
    file and every file after it.
    """

    def __init__(self, parent, entries, base_label="Audio description"):
        super().__init__(parent, title="Start selected descriptions",
                         size=(620, 400))
        self._entries = [dict(entry) for entry in entries]
        self._focus_done = False

        outer = wx.BoxSizer(wx.VERTICAL)

        # Project base: exactly one item - the selected project's base.
        base_row = wx.BoxSizer(wx.HORIZONTAL)
        base_row.Add(wx.StaticText(self, label="Project base"), 0,
                     wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.base_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Project base")
        self.base_combo.Append(base_label, "")
        self.base_combo.SetSelection(0)
        base_row.Add(self.base_combo, 1, wx.EXPAND)
        outer.Add(base_row, 0, wx.EXPAND | wx.ALL, 8)

        row = wx.BoxSizer(wx.HORIZONTAL)

        # -- sidebar: how the combo is filled -------------------------------
        side_panel = wx.Panel(self)
        sidebar = wx.BoxSizer(wx.VERTICAL)
        self.file_radio = wx.RadioButton(
            side_panel, label="Select a video file", style=wx.RB_GROUP)
        self.file_radio.SetToolTip(
            "Choose one video of this project and describe it again")
        self.break_radio = wx.RadioButton(
            side_panel, label="Select by file break")
        self.break_radio.SetToolTip(
            "Choose any file break of the project and start describing at it")
        for radio in (self.file_radio, self.break_radio):
            radio.SetName(radio.GetLabel())
            sidebar.Add(radio, 0, wx.ALL, 6)
        side_panel.SetSizerAndFit(sidebar)
        side_panel.SetMinSize((240, -1))
        side_panel.SetName("Description selection mode")
        if not any(entry.get("file") for entry in self._entries):
            # No video files: the only useful mode is by file break.
            self.break_radio.SetValue(True)
        row.Add(side_panel, 0, wx.EXPAND | wx.ALL, 8)

        # -- combo + scope ---------------------------------------------------
        content = wx.BoxSizer(wx.VERTICAL)
        self.choice_label = wx.StaticText(self, label="")
        self.choice_label.SetName("Description selection")
        content.Add(self.choice_label, 0, wx.BOTTOM, 4)
        self.combo = wx.ComboBox(self, style=wx.CB_READONLY)
        content.Add(self.combo, 0, wx.EXPAND | wx.BOTTOM, 12)

        self.only_radio = wx.RadioButton(
            self, label="Only describe selected file", style=wx.RB_GROUP)
        self.only_radio.SetToolTip(
            "Describe just the chosen file; the files after it are left alone")
        self.all_radio = wx.RadioButton(
            self, label="Describe all files from here")
        self.all_radio.SetToolTip(
            "Describe the chosen file and then every file after it to the end "
            "of the project")
        for radio in (self.only_radio, self.all_radio):
            radio.SetName(radio.GetLabel())
            content.Add(radio, 0, wx.ALL, 6)

        content.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0,
                    wx.EXPAND | wx.TOP, 12)
        row.Add(content, 1, wx.EXPAND | wx.ALL, 8)
        outer.Add(row, 1, wx.EXPAND)
        self.SetSizer(outer)

        self.file_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self.break_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self._fill_combo()
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)
        self.SetMinSize(self.GetSize())
        if parent is not None and parent.IsShown():
            self.CentreOnParent()
        else:
            self.Centre()
        # Focus the checked sidebar mode button, not the combo, exactly as the
        # other pickers do; the dialog's own focus has to be undone after it is
        # shown.
        self.Bind(wx.EVT_SHOW, self._on_show)

    # ---------------------------------------------------------------- focus
    def _on_show(self, event):
        """Move the cursor to the mode button once the dialog is on screen."""
        event.Skip()
        if event.IsShown() and not self._focus_done:
            self._focus_done = True
            wx.CallAfter(self._focus_first_control)

    def initial_focus_control(self):
        """The control the dialog focuses when it opens."""
        return self.break_radio if self.break_radio.GetValue() else self.file_radio

    def _focus_first_control(self):
        """Put the keyboard cursor on :meth:`initial_focus_control`."""
        self.initial_focus_control().SetFocus()

    # ------------------------------------------------------------------ state
    def _fill_combo(self):
        """(Re)fill the combo box for the selected sidebar mode."""
        self.combo.Clear()
        if self.file_radio.GetValue():
            self.choice_label.SetLabel(
                "Choose which described file to describe again:")
            update_accessible_name(
                self.combo,
                "Choose which described file to describe again")
            self.combo.SetToolTip(
                "Video files that are already part of this project")
            items = [(entry["file"], entry["position"])
                     for entry in self._entries if entry.get("file")]
        else:
            self.choice_label.SetLabel(
                "Choose which file to describe by file break:")
            update_accessible_name(
                self.combo, "Choose which file to describe by file break")
            self.combo.SetToolTip(
                "Every file break of the project")
            items = [(entry["title"], entry["position"])
                     for entry in self._entries]
        for label, position in items:
            self.combo.Append(str(label), position)
        if self.combo.GetCount():
            self.combo.SetSelection(0)
        self.combo.Enable(self.combo.GetCount() > 0)
        ok_btn = self.FindWindowById(wx.ID_OK)
        if ok_btn is not None:
            ok_btn.Enable(self.combo.GetCount() > 0)
        self.Layout()

    def _on_mode(self, _evt):
        self._fill_combo()

    def _selected_entry(self) -> dict | None:
        position = self.start_index()
        if position is None:
            return None
        for entry in self._entries:
            if entry["position"] == position:
                return entry
        return None

    # ---------------------------------------------------------------- answers
    def start_index(self) -> int | None:
        """0-based segment position to start describing at (``None`` = none)."""
        sel = self.combo.GetSelection()
        if sel < 0:
            return None
        value = self.combo.GetClientData(sel)
        return None if value is None else int(value)

    def only_selected(self) -> bool:
        """True = describe only the chosen file; False = from it on."""
        return bool(self.only_radio.GetValue())

    def chosen_file(self) -> str | None:
        """The video of the chosen break, when it has one."""
        entry = self._selected_entry()
        if entry is None:
            return None
        if entry.get("file"):
            return entry["file"]
        if self.file_radio.GetValue():
            return self.combo.GetStringSelection() or None
        return None
