"""New Project wizard (SPEC 3.5).

Page 1: project name + Open document button.
Page 2: audio file creation mode (the same radio group as in Settings).

Finishing parses the document, splits it per the chosen mode, writes
``project.json`` and opens the Recording window.
"""

from __future__ import annotations

import logging
import os
import wx
from wx.adv import EVT_WIZARD_FINISHED, Wizard, WizardPage

from .. import project
from ..constants import (
    AUDIO_MODE_CHOICES,
    AUDIO_MODE_DESCRIPTIONS,
    MODE_PAGE_WITH_H1,
    PROJECT_TYPE_AUDIO_PLAYLIST,
    PROJECT_TYPE_CLIPBOARD,
    PROJECT_TYPE_DAISY_AUDIO,
    PROJECT_TYPE_DAISY_AUDIO_TEXT,
    PROJECT_TYPES,
    PROJECT_TYPE_DESCRIPTIONS,
    PUNCTUATION_CHOICES,
    PUNCTUATION_DEFAULT,
    SUPPORTED_EXTENSIONS,
)
from ..documents.parsers import ParseError, parse_document
from ..documents.splitter import split_document
from ..paths import project_dir
from ..settings import Settings
from ..tts.models import ModelStore
from .a11y import add_labeled
from .recording_dialog import RecordingDialog

log = logging.getLogger(__name__)


class NewProjectWizard(Wizard):
    def __init__(self, parent, settings: Settings, store: ModelStore,
                 initial_name: str = ""):
        super().__init__(parent, title="New Project - AI Voice Studio")
        self.settings = settings
        self.store = store
        self.source_path: str = ""

        self.page_details = _DetailsPage(self)
        self.page_mode = _ModePage(self, settings)
        self.GetPageAreaSizer().Add(self.page_details)
        if initial_name:
            self.page_details.name_ctrl.SetValue(initial_name)

        self.Bind(EVT_WIZARD_FINISHED, self._on_finish)
        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGED, self._on_page_changed)

    def _on_page_changed(self, evt):
        # NVDA-friendly: focus lands on the first control of the new page.
        page = evt.GetPage()
        if evt.GetDirection():
            if page is self.page_details:
                self.page_details.name_ctrl.SetFocus()
            elif page is self.page_mode and self.page_mode.radios:
                wx.CallAfter(self.page_mode.radios[0][0].SetFocus)
        evt.Skip()

    def _on_finish(self, _):
        from ..documents.splitter import split_daisy_chapters  # noqa: PLC0415

        name = self.page_details.name_ctrl.GetValue().strip() or "Untitled project"
        ptype = self.page_details.selected_project_type()
        mode = self.page_mode.selected()

        # Clipboard mode: no document needed
        if ptype == PROJECT_TYPE_CLIPBOARD:
            pdir = project_dir(name)
            last = self.settings.get("last_model", {})
            project.create_project(
                pdir, name, "<clipboard>", "clipboard",
                [{"index": 1, "title": "clipboard text", "text": ""}],
                tts_settings={
                    "tts": last.get("tts"),
                    "language": last.get("language"),
                    "variant": last.get("variant"),
                    "voice": last.get("voice"),
                    "rate": self.settings.get("recording.rate", 1.0),
                    "pitch": self.settings.get("recording.pitch", 1.0),
                    "volume": self.settings.get("recording.volume", 1.0),
                    "punctuation": self.page_details.selected_punctuation(),
                    "output_format": self.settings.get("recording.output_format", "wav"),
                    "compute": self.settings.compute,
                },
                project_type=ptype,
            )
            self.settings.add_recent_project(name, pdir)
            self.EndModal(wx.ID_OK)
            dlg = RecordingDialog(self.GetParent(), pdir, self.settings, self.store)
            dlg.ShowModal()
            dlg.Destroy()
            return

        # All other modes need a document
        if not self.source_path:
            wx.MessageBox("Choose a document to read first.", "New project",
                          style=wx.OK | wx.ICON_INFORMATION)
            return

        try:
            wx.BeginBusyCursor()
            doc = parse_document(self.source_path)
            if ptype in (PROJECT_TYPE_DAISY_AUDIO, PROJECT_TYPE_DAISY_AUDIO_TEXT):
                segments = split_daisy_chapters(doc.blocks)
            else:
                segments = split_document(doc, mode)
            wx.EndBusyCursor()
        except ParseError as exc:
            wx.EndBusyCursor()
            wx.MessageBox(str(exc), "Could not read the document",
                          style=wx.OK | wx.ICON_ERROR)
            return
        except Exception as exc:  # noqa: BLE001
            wx.EndBusyCursor()
            wx.MessageBox(f"Preparing the document failed: {exc}",
                          "New project", style=wx.OK | wx.ICON_ERROR)
            return

        if not segments:
            wx.MessageBox("The document contains no readable text.",
                          "New project", style=wx.OK | wx.ICON_WARNING)
            return

        pdir = project_dir(name)
        last = self.settings.get("last_model", {})
        project.create_project(
            pdir,
            name,
            self.source_path,
            mode,
            [
                {"index": s.index, "title": s.title, "text": s.text}
                for s in segments
            ],
            tts_settings={
                "tts": last.get("tts"),
                "language": last.get("language"),
                "variant": last.get("variant"),
                "voice": last.get("voice"),
                "rate": self.settings.get("recording.rate", 1.0),
                "pitch": self.settings.get("recording.pitch", 1.0),
                "volume": self.settings.get("recording.volume", 1.0),
                "punctuation": self.page_details.selected_punctuation(),
                "output_format": self.settings.get("recording.output_format", "wav"),
                "compute": self.settings.compute,
            },
            project_type=ptype,
        )
        self.settings.add_recent_project(name, pdir)
        self.EndModal(wx.ID_OK)

        dlg = RecordingDialog(self.GetParent(), pdir, self.settings, self.store)
        dlg.ShowModal()
        dlg.Destroy()


class _DetailsPage(WizardPage):
    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self._build_ui()

    def GetNext(self):
        return self.wizard.page_mode

    def GetPrev(self):
        return None

    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Project details"),
                  0, wx.ALL, 6)

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.name_ctrl = wx.TextCtrl(self)
        self.name_ctrl.SetName("Project name")
        add_labeled(self, grid, "Project name", self.name_ctrl, flag=wx.LEFT | wx.RIGHT, border=2)

        # Project type combo
        self.project_type_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                               name="Project type")
        for value, label in PROJECT_TYPES:
            self.project_type_combo.Append(label, value)
        # Default to audio playlist (old behavior)
        default_idx = next(
            (i for i, (v, _) in enumerate(PROJECT_TYPES)
             if v == PROJECT_TYPE_AUDIO_PLAYLIST), 0
        )
        self.project_type_combo.SetSelection(default_idx)
        add_labeled(self, grid, "Project type", self.project_type_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        self.type_desc = wx.StaticText(self, label="")
        self.type_desc.Wrap(680)
        sizer.Add(self.type_desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self._update_type_desc()

        doc_row = wx.BoxSizer(wx.HORIZONTAL)
        self.open_btn = wx.Button(self, label="Open document")
        self.open_btn.SetName("Open document file")
        self.open_btn.SetToolTip("Browse for a PDF, TXT, DOC, DOCX, HTML, Markdown, or EPUB file")
        doc_row.Add(self.open_btn, 0, wx.ALL, 4)
        self.file_label = wx.StaticText(self, label="No document selected.")
        doc_row.Add(self.file_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        sizer.Add(doc_row, 0, wx.EXPAND)

        # Punctuation is chosen here, right after the document, so it becomes
        # part of the project. The Recording window no longer shows it.
        punct_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        punct_grid.AddGrowableCol(1)
        self.punct_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                        name="Punctuation mode")
        for value, label in PUNCTUATION_CHOICES:
            self.punct_combo.Append(label, value)
        current = self.wizard.settings.get(
            "recording.punctuation", PUNCTUATION_DEFAULT
        )
        idx = next(
            (i for i, (v, _) in enumerate(PUNCTUATION_CHOICES) if v == current), 0
        )
        self.punct_combo.SetSelection(idx)
        add_labeled(self, punct_grid, "Punctuation", self.punct_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(punct_grid, 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="Choose how punctuation marks are spoken. 'All' reads "
                      "every mark as a word, for example quote, dot, left "
                      "paren, tic - useful for TTS voices that cannot "
                      "pronounce punctuation.",
            ),
            0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6,
        )

        sizer.Add(
            wx.StaticText(
                self,
                label="Supported files: PDF, TXT, DOC, DOCX, HTML, Markdown, "
                      "EPUB. You can also copy text to the clipboard and "
                      "paste it here.",
            ),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)

        self.open_btn.Bind(wx.EVT_BUTTON, self._on_open)

    def selected_punctuation(self) -> str:
        sel = self.punct_combo.GetSelection()
        return self.punct_combo.GetClientData(sel) if sel >= 0 else PUNCTUATION_DEFAULT

    def selected_project_type(self) -> str:
        sel = self.project_type_combo.GetSelection()
        return self.project_type_combo.GetClientData(sel) if sel >= 0 else PROJECT_TYPE_AUDIO_PLAYLIST

    def _update_type_desc(self):
        ptype = self.selected_project_type()
        self.type_desc.SetLabel(PROJECT_TYPE_DESCRIPTIONS.get(ptype, ""))
        self.type_desc.Wrap(680)

    def _on_open(self, _):
        wildcard = (
            "Documents (*.pdf;*.txt;*.md;*.doc;*.docx;*.html;*.htm;*.epub)|"
            "*.pdf;*.txt;*.md;*.markdown;*.doc;*.docx;*.html;*.htm;*.epub|"
            "All files|*.*"
        )
        with wx.FileDialog(self, "Open document", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            wx.MessageBox(
                f"Unsupported file type '{ext or '(none)'}'.",
                "Open document", style=wx.OK | wx.ICON_WARNING,
            )
            return
        self.wizard.source_path = path
        self.file_label.SetLabel(f"Document: {path}")
        self.file_label.SetName("Selected document")


class _ModePage(WizardPage):
    def __init__(self, wizard, settings: Settings):
        super().__init__(wizard)
        self.wizard = wizard
        self._build_ui(settings)

    def GetNext(self):
        return None

    def GetPrev(self):
        return self.wizard.page_details

    def _build_ui(self, settings: Settings):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Audio file creation"),
                  0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(self, label="Choose how the document is split into "
                                      "audio files, then press Finish to "
                                      "prepare the document and open the "
                                      "Recording window."),
            0, wx.ALL, 6,
        )
        self.radios = []
        self.description = wx.StaticText(self, label="")
        sizer.Add(self.description, 0, wx.ALL, 6)
        current = settings.get("audio_mode", MODE_PAGE_WITH_H1)
        first = True
        for value, label in AUDIO_MODE_CHOICES:
            radio = wx.RadioButton(
                self, label=label, style=wx.RB_GROUP if first else 0
            )
            radio.SetName(label)
            radio.SetValue(value == current)
            sizer.Add(radio, 0, wx.ALL, 4)
            radio.Bind(wx.EVT_RADIOBUTTON, lambda _: self._update_description())
            self.radios.append((radio, value))
            first = False
        self._update_description()
        self.SetSizer(sizer)

    def _update_description(self):
        self.description.SetLabel(AUDIO_MODE_DESCRIPTIONS.get(self.selected(), ""))
        self.description.Wrap(680)

    def selected(self) -> str:
        for radio, value in self.radios:
            if radio.GetValue():
                return value
        return MODE_PAGE_WITH_H1
