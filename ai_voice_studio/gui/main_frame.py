"""Main window (SPEC 3.2, 3.5, 3.6).

Menu bar with mnemonics and accelerators:
* File    -> New Project (Ctrl+Shift+N), Open Project, Recent Projects, Exit
* Edit    -> Resume Recording, Restart Project, Restart All Recording,
             Start Selected Recording, Remove Project, Show project folder
* Tools   -> Record (Ctrl+Shift+R), Settings (Ctrl+,)
* Help    -> Read Me, User Guide, About

The welcome panel offers the same actions with large labelled buttons and a
recent-projects list that screen readers can navigate with arrow keys. The
Edit-menu actions act on the project selected in the Recent projects list
(right-click the list for the same actions).
"""

from __future__ import annotations

import logging
import os
import sys
import wx

from .. import __version__, project
from ..settings import Settings
from ..tts.models import ModelStore
from .a11y import finalize_accessibility, update_accessible_name
from .dialogs import open_folder
from .new_project_wizard import NewProjectWizard
from .recording_dialog import RecordingDialog
from .settings_dialog import SettingsDialog
from .theme import apply_theme

log = logging.getLogger(__name__)

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
ID_ABOUT = wx.NewIdRef()
ID_RESUME_RECORDING = wx.NewIdRef()
ID_RESTART_PROJECT = wx.NewIdRef()
ID_RESTART_ALL = wx.NewIdRef()
ID_RESTART_SELECTED = wx.NewIdRef()
ID_REMOVE_PROJECT = wx.NewIdRef()
ID_SHOW_PROJECT_FOLDER = wx.NewIdRef()


class MainFrame(wx.Frame):
    def __init__(self, parent=None, settings: Settings = None, store: ModelStore = None):
        super().__init__(parent, title="AI Voice Studio", size=(820, 600))
        if settings is None:
            settings = Settings()
        if store is None:
            store = ModelStore()
        self.settings = settings
        self.store = store
        self._recent_menu: wx.Menu | None = None

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
        file_menu.Append(ID_NEW_PROJECT, "&New Project\tCtrl+Shift+N")
        file_menu.Append(ID_OPEN_PROJECT, "&Open Project...")
        self._recent_menu = wx.Menu()
        file_menu.AppendSubMenu(self._recent_menu, "&Recent Projects")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "E&xit")
        menubar.Append(file_menu, "&File")

        edit_menu = wx.Menu()
        edit_menu.Append(ID_RESUME_RECORDING, "&Resume Recording")
        edit_menu.Append(ID_RESTART_PROJECT, "&Restart Project...")
        edit_menu.Append(ID_RESTART_ALL, "Restart &All Recording")
        edit_menu.Append(ID_RESTART_SELECTED, "Start &Selected Recording...")
        edit_menu.AppendSeparator()
        edit_menu.Append(ID_REMOVE_PROJECT, "&Remove Project...")
        edit_menu.Append(ID_SHOW_PROJECT_FOLDER, "Show project &folder")
        menubar.Append(edit_menu, "&Edit")

        tools_menu = wx.Menu()
        tools_menu.Append(ID_RECORD, "&Record...\tCtrl+Shift+R")
        tools_menu.Append(ID_SETTINGS, "&Settings...\tCtrl+,")
        menubar.Append(tools_menu, "&Tools")

        help_menu = wx.Menu()
        help_menu.Append(ID_README, "&Read Me")
        help_menu.Append(ID_USER_GUIDE, "&User Guide")
        help_menu.Append(ID_ADDON_GUIDE, "&Addon Development Guide")
        help_menu.Append(ID_ACCESSIBILITY_GUIDE, "&Accessibility Guidelines")
        help_menu.AppendSeparator()
        help_menu.Append(ID_PYTHON_BOOK, "Python & wxPython Book	F1")
        help_menu.Append(ID_THIRD_PARTY_LICENSES, "&Third-Party Licences")
        help_menu.Append(ID_ABOUT, "&About AI Voice Studio")
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
        self.Bind(wx.EVT_MENU, lambda _: self._about(), id=ID_ABOUT)
        self.Bind(wx.EVT_MENU, lambda _: self._resume_recording(), id=ID_RESUME_RECORDING)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_project(), id=ID_RESTART_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_all(), id=ID_RESTART_ALL)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_selected(), id=ID_RESTART_SELECTED)
        self.Bind(wx.EVT_MENU, lambda _: self._remove_project(), id=ID_REMOVE_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._show_project_folder(),
                  id=ID_SHOW_PROJECT_FOLDER)
        self.Bind(wx.EVT_MENU, lambda _: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU_OPEN, lambda _: self._rebuild_recent())

        # Frame-level accelerator table for the three global shortcuts.
        # These fire the same menu IDs, but do not depend on the menu
        # bar's own accelerator parsing, which can be unreliable for
        # Shift+letter combos on some wx builds.
        self.SetAcceleratorTable(wx.AcceleratorTable([
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("N"), ID_NEW_PROJECT),
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("R"), ID_RECORD),
            (wx.ACCEL_CTRL, ord(","), ID_SETTINGS),
        ]))

        # Belt-and-braces: a char hook catches the same shortcuts even when
        # the accelerator table / menu accelerator parsing misses them
        # (observed on wx 3.3.3 msw for Ctrl+Shift+letter).  Only fires when
        # no modal dialog is open, so it never duplicates the dialog's own
        # handling and never swallows ordinary typing.
        self.Bind(wx.EVT_CHAR_HOOK, self._on_global_char_hook)

    def _on_global_char_hook(self, evt: wx.KeyEvent) -> None:
        """Global keyboard hook for the three main shortcuts.

        wx's menu accelerators proved unreliable for Ctrl+Shift+letter on
        this wx build, so the hook routes the keys itself.  It only acts
        when the frame itself is active (no modal dialog owns the keys) and
        passes every other key through untouched.
        """
        key = evt.GetKeyCode()
        mods = evt.GetModifiers()
        ctrl = bool(mods & wx.MOD_CONTROL)
        shift = bool(mods & wx.MOD_SHIFT)
        if wx.IsBusy():
            evt.Skip()
            return
        # wx 3.3.3 msw translates letter+Ctrl+Shift to keycode 0 (raw 255),
        # which is why neither the menu accelerators nor the accelerator
        # table can match Ctrl+Shift+N / Ctrl+Shift+R on this build.  For
        # those untranslated events, identify the held letter via the event
        # scan code first, then Win32 GetKeyState as a fallback.
        if sys.platform == "win32" and ctrl and shift and key in (0, ord("N"), ord("n"), ord("R"), ord("r")):
            scan = 0
            try:
                scan = (evt.GetRawKeyFlags() >> 16) & 0xFF
            except Exception:  # noqa: BLE001
                pass
            import ctypes  # noqa: PLC0415
            user32 = ctypes.windll.user32
            def _held(vk: int) -> bool:
                return bool(user32.GetKeyState(vk) & 0x8000)
            is_n = key in (ord("N"), ord("n")) or scan == 0x31 or _held(0x4E)
            is_r = key in (ord("R"), ord("r")) or scan == 0x13 or _held(0x52)
            if is_n:
                self._new_project()
                return
            if is_r:
                self._record()
                return
        if ctrl and shift and key in (ord("N"), ord("n")):
            self._new_project()
            return
        if ctrl and shift and key in (ord("R"), ord("r")):
            self._record()
            return
        if ctrl and key in (ord(","),):
            self._settings()
            return
        evt.Skip()

    def _build_welcome(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(panel, label="Welcome to AI Voice Studio")
        title.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        sizer.Add(title, 0, wx.ALL, 12)
        sizer.Add(
            wx.StaticText(panel, label="Convert documents into spoken audio "
                                       "files with free, offline neural voices."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12,
        )

        sizer.Add(
            wx.StaticText(panel, label="Start with a new project: choose a "
                                       "document, choose how it is split, and "
                                       "record it. Voices are downloaded once "
                                       "from Settings and reused for all "
                                       "projects."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12,
        )

        for label, handler, tip in (
            ("Create New Project", self._new_project,
             "Open the New Project wizard to create a project from a document"),
            ("Record (open a project)", self._record,
             "Open the most recent project for recording"),
            ("Settings (download voices)", self._settings,
             "Open Settings to download voices and configure the application"),
        ):
            btn = wx.Button(panel, label=label)
            btn.SetName(label)
            btn.SetToolTip(tip)
            btn.Bind(wx.EVT_BUTTON, lambda _, h=handler: h())
            sizer.Add(btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        sizer.Add(wx.StaticText(panel, label="Recent projects:"), 0, wx.ALL, 8)
        self.recent_list = wx.ListBox(panel, style=wx.LB_SINGLE)
        self.recent_list.SetName("Recent projects list")
        self.recent_list.SetToolTip("Select a project and press Enter or double-click to open it")
        sizer.Add(self.recent_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self.recent_list.Bind(wx.EVT_LISTBOX_DCLICK, lambda _: self._open_recent_selected())
        self.recent_list.Bind(wx.EVT_KEY_DOWN, self._on_recent_key)
        self.recent_list.Bind(wx.EVT_CONTEXT_MENU, self._on_recent_context_menu)

        # Alt+R removes the project selected in the list below.
        self.remove_btn = wx.Button(panel, label="&Remove selected project")
        self.remove_btn.SetName("Remove selected project")
        self.remove_btn.SetToolTip("Delete the selected project and all its recordings permanently")
        self.remove_btn.Bind(wx.EVT_BUTTON, lambda _: self._remove_project())
        sizer.Add(self.remove_btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(
            wx.StaticText(panel, label="Right-click a project (or press the "
                                       "application-menu key) for more actions "
                                       "(resume, restart, remove, show the "
                                       "project folder) - also in the Edit "
                                       "menu."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8,
        )

        panel.SetSizer(sizer)
        self._refresh_recent_list()
        # Real MSAA accNames (e.g. the Recent projects list name).
        finalize_accessibility(self)

    def _build_statusbar(self):
        self.CreateStatusBar(1)
        self.SetStatusText("Ready.")

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

    def _settings(self):
        dlg = SettingsDialog(self, self.settings, self.store)
        dlg.ShowModal()
        dlg.Destroy()
        apply_theme(self, self.settings.theme)
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

    def _on_recent_context_menu(self, evt):
        # Right-click carries a position; the keyboard application-menu key
        # (and Shift+F10) does not, and then the already-selected project is
        # the one the menu acts on.
        pos = evt.GetPosition()
        if pos != wx.DefaultPosition:
            item = self.recent_list.HitTest(self.recent_list.ScreenToClient(pos))
            if item != wx.NOT_FOUND:
                self.recent_list.SetSelection(item)
        menu = wx.Menu()
        for label, handler in (
            ("Resume Recording", self._resume_recording),
            ("Restart Project...", self._restart_project),
            ("Restart All Recording", self._restart_all),
            ("Start Selected Recording...", self._restart_selected),
            (None, None),
            ("Remove Project...", self._remove_project),
            ("Show project folder", self._show_project_folder),
        ):
            if label is None:
                menu.AppendSeparator()
                continue
            item_id = wx.NewIdRef()
            menu.Append(item_id, label)
            menu.Bind(wx.EVT_MENU, lambda _, h=handler: h(), id=item_id.GetId())
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
        self.SetStatusText(f"Opened {folder}")

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
            "All recordings and project settings will be deleted; only the "
            "project name is kept. You will choose a document and the audio "
            "creation options again.",
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
            f"Restart all recording for '{recent['name']}'?\n\n"
            "All recorded audio files will be deleted and every segment will "
            "be recorded again from the beginning.",
            "Restart all recording", style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        removed = project.reset_recordings(recent["path"])
        self.SetStatusText(
            f"Deleted {removed} audio file(s); recording starts from the beginning."
        )
        self._open_recording(recent["path"])

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
        self.SetStatusText(f"Removed project '{recent['name']}'.")

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
        for idx, recent in enumerate(recents):
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
        import webbrowser  # noqa: PLC0415

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
        except Exception:  # noqa: BLE001
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: SIM115

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

    def _python_book(self):
        """Open the Python & wxPython book in the default browser."""
        import webbrowser  # noqa: PLC0415

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
        except Exception:  # noqa: BLE001
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: SIM115

    def _about(self):
        import wx.adv  # noqa: PLC0415

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
    from ..util import sanitize_filename  # noqa: PLC0415

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

    def __init__(self, parent, entries):
        super().__init__(parent, title="Start selected recording", size=(600, 320))
        self._entries = [dict(entry) for entry in entries]
        self._focus_done = False

        outer = wx.BoxSizer(wx.HORIZONTAL)

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
        outer.Add(side_panel, 0, wx.EXPAND | wx.ALL, 8)

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
        outer.Add(content, 1, wx.EXPAND | wx.ALL, 8)
        self.SetSizer(outer)

        self.file_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self.break_radio.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
        self._fill_combo()
        finalize_accessibility(self)
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
