"""Main window (SPEC 3.2, 3.5, 3.6).

Menu bar with mnemonics and accelerators:
* File    -> New Project (Ctrl+Shift+N), Open Project, Recent Projects, Exit
* Edit    -> Resume Recording, Restart Project, Restart All Recording,
             Restart Selected Recording, Remove Project
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
import wx

from .. import __version__, project
from ..settings import Settings
from ..tts.models import ModelStore
from .new_project_wizard import NewProjectWizard
from .recording_dialog import RecordingDialog
from .settings_dialog import SettingsDialog
from .theme import apply_theme

log = logging.getLogger(__name__)

ID_NEW_PROJECT = wx.NewIdRef()
ID_OPEN_PROJECT = wx.NewIdRef()
ID_RECENT_BASE = wx.NewIdRef()
ID_RECORD = wx.NewIdRef()
ID_SETTINGS = wx.NewIdRef()
ID_README = wx.NewIdRef()
ID_USER_GUIDE = wx.NewIdRef()
ID_ADDON_GUIDE = wx.NewIdRef()
ID_ACCESSIBILITY_GUIDE = wx.NewIdRef()
ID_ABOUT = wx.NewIdRef()
ID_RESUME_RECORDING = wx.NewIdRef()
ID_RESTART_PROJECT = wx.NewIdRef()
ID_RESTART_ALL = wx.NewIdRef()
ID_RESTART_SELECTED = wx.NewIdRef()
ID_REMOVE_PROJECT = wx.NewIdRef()


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
        edit_menu.Append(ID_RESTART_SELECTED, "Restart &Selected Recording...")
        edit_menu.AppendSeparator()
        edit_menu.Append(ID_REMOVE_PROJECT, "&Remove Project...")
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
        self.Bind(wx.EVT_MENU, lambda _: self._about(), id=ID_ABOUT)
        self.Bind(wx.EVT_MENU, lambda _: self._resume_recording(), id=ID_RESUME_RECORDING)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_project(), id=ID_RESTART_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_all(), id=ID_RESTART_ALL)
        self.Bind(wx.EVT_MENU, lambda _: self._restart_selected(), id=ID_RESTART_SELECTED)
        self.Bind(wx.EVT_MENU, lambda _: self._remove_project(), id=ID_REMOVE_PROJECT)
        self.Bind(wx.EVT_MENU, lambda _: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU_OPEN, lambda _: self._rebuild_recent())

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

        self.remove_btn = wx.Button(panel, label="Remove selected project")
        self.remove_btn.SetName("Remove selected project")
        self.remove_btn.SetToolTip("Delete the selected project and all its recordings permanently")
        self.remove_btn.Bind(wx.EVT_BUTTON, lambda _: self._remove_project())
        sizer.Add(self.remove_btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(
            wx.StaticText(panel, label="Right-click a project for more actions "
                                       "(resume, restart, remove) - also in the "
                                       "Edit menu."),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8,
        )

        panel.SetSizer(sizer)
        self._refresh_recent_list()

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

    def _open_recording(self, project_folder: str):
        try:
            dlg = RecordingDialog(self, project_folder, self.settings, self.store)
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
        pos = evt.GetPosition()
        if pos == wx.DefaultPosition:
            pos = wx.GetMousePosition()
        item = self.recent_list.HitTest(self.recent_list.ScreenToClient(pos))
        if item != wx.NOT_FOUND:
            self.recent_list.SetSelection(item)
        menu = wx.Menu()
        for label, handler in (
            ("Resume Recording", self._resume_recording),
            ("Restart Project...", self._restart_project),
            ("Restart All Recording", self._restart_all),
            ("Restart Selected Recording...", self._restart_selected),
            (None, None),
            ("Remove Project...", self._remove_project),
        ):
            if label is None:
                menu.AppendSeparator()
                continue
            item_id = wx.NewIdRef()
            menu.Append(item_id, label)
            menu.Bind(wx.EVT_MENU, lambda _, h=handler: h(), id=item_id.GetId())
        self.recent_list.PopupMenu(menu)
        menu.Destroy()

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
            project.load_project(recent["path"])
        except (OSError, ValueError, KeyError) as exc:
            wx.MessageBox(f"Could not open the project: {exc}",
                          "Restart selected recording", style=wx.OK | wx.ICON_ERROR)
            return
        # Use the audio files actually on disk (not just segment metadata) so
        # the picker shows everything that was recorded.
        saved = project.recorded_files(recent["path"])
        if not saved:
            wx.MessageBox(
                "This project has no recorded files yet. Record it first.",
                "Restart selected recording", style=wx.OK | wx.ICON_INFORMATION,
            )
            return
        dlg = _RecordingPickerDialog(self, saved)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                chosen = dlg.combo.GetStringSelection()
                project.remove_recording(recent["path"], chosen)
                self._open_recording(recent["path"])
        finally:
            dlg.Destroy()

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
            item = self._recent_menu.Append(
                int(ID_RECENT_BASE) + idx,
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


class _RecordingPickerDialog(wx.Dialog):
    """Combo box to choose one recorded file to record again (Edit menu)."""

    def __init__(self, parent, files):
        super().__init__(parent, title="Restart selected recording", size=(480, 170))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(
            wx.StaticText(self, label="Choose which recorded file to record again:"),
            0, wx.ALL, 8,
        )
        self.combo = wx.ComboBox(self, style=wx.CB_READONLY)
        self.combo.SetName("Recorded files")
        for name in files:
            self.combo.Append(name)
        self.combo.SetSelection(0)
        sizer.Add(self.combo, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        sizer.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 6)
        self.SetSizer(sizer)
