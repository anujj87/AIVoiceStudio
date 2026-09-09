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
    DAISY_SPLIT_ALL_HEADINGS,
    DAISY_SPLIT_CHOICES,
    DAISY_SPLIT_DESCRIPTIONS,
    DAISY_SPLIT_H1,
    MODE_PAGE_ONLY,
    MODE_PAGE_WITH_H1,
    PAGES_PER_FILE_MAX,
    PAGES_PER_FILE_MIN,
    PROJECT_TYPE_AUDIO_PLAYLIST,
    PROJECT_TYPE_CLIPBOARD,
    PROJECT_TYPE_DAISY_AUDIO,
    PROJECT_TYPE_DAISY3_AUDIO_TEXT,
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
from .a11y import add_labeled, finalize_accessibility
from .progress import TaskProgressDialog
from .recording_dialog import RecordingDialog

log = logging.getLogger(__name__)

# Project types routed to the DAISY-specific second wizard page.
_DAISY_TYPES = (
    PROJECT_TYPE_DAISY_AUDIO,
    PROJECT_TYPE_DAISY3_AUDIO_TEXT,
)


class NewProjectWizard(Wizard):
    def __init__(self, parent, settings: Settings, store: ModelStore,
                 initial_name: str = ""):
        super().__init__(parent, title="New Project - AI Voice Studio")
        self.settings = settings
        self.store = store
        self.source_path: str = ""

        self.page_details = _DetailsPage(self)
        self.page_mode = _ModePage(self, settings)
        self.page_daisy = _DaisyPage(self, settings)
        self.GetPageAreaSizer().Add(self.page_details)
        if initial_name:
            self.page_details.name_ctrl.SetValue(initial_name)
        # Real MSAA accNames for the name box / type combo (SetName alone is
        # ignored by MSAA on this wx build).
        finalize_accessibility(self)

        self.Bind(EVT_WIZARD_FINISHED, self._on_finish)
        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGED, self._on_page_changed)
        # Announce the first field (project name) when the wizard opens.
        wx.CallAfter(self.page_details.name_ctrl.SetFocus)

    def _on_page_changed(self, evt):
        # NVDA-friendly: focus lands on the first control of the new page.
        page = evt.GetPage()
        if evt.GetDirection():
            if page is self.page_details:
                self.page_details.name_ctrl.SetFocus()
            elif page is self.page_mode and self.page_mode.radios:
                wx.CallAfter(self.page_mode.radios[0][0].SetFocus)
            elif page is self.page_daisy and self.page_daisy.radios:
                wx.CallAfter(self.page_daisy.radios[0][0].SetFocus)
        evt.Skip()

    def _recording_defaults(self) -> dict:
        """Rate/pitch/volume defaults for the new project.

        When the last used TTS engine has its own Speed/Pitch/Volume saved in
        Settings > Recording settings, those are used; otherwise the global
        defaults apply.
        """
        per_tts = self.settings.get("recording.per_tts", {}) or {}
        last_tts = self.settings.get("last_model", {}).get("tts")
        if last_tts and per_tts.get(last_tts):
            entry = per_tts[last_tts]
            return {
                "rate": float(entry.get("rate", 1.0)),
                "pitch": float(entry.get("pitch", 1.0)),
                "volume": float(entry.get("volume", 1.0)),
            }
        return {
            "rate": self.settings.get("recording.rate", 1.0),
            "pitch": self.settings.get("recording.pitch", 1.0),
            "volume": self.settings.get("recording.volume", 1.0),
        }

    def _on_finish(self, _):
        from ..documents.splitter import split_daisy_chapters  # noqa: PLC0415

        name = self.page_details.name_ctrl.GetValue().strip() or "Untitled project"
        ptype = self.page_details.selected_project_type()
        mode = self.page_mode.selected()
        is_daisy = ptype in _DAISY_TYPES

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
                    **self._recording_defaults(),
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

        # Progress dialog while the document is read and split into segments
        # (item-level feedback between the wizard and the Recording window).
        progress = TaskProgressDialog(self, "New Project",
                                      "Reading the document...")
        progress.Show()
        try:

            def _report(message, fraction):
                progress.update(int(max(0.0, min(1.0, fraction)) * 100), message)

            doc = parse_document(self.source_path, on_progress=_report)
            if is_daisy:
                progress.update(90, "Splitting the text into DAISY chapters...")
                segments = split_daisy_chapters(
                    doc.blocks, break_level=self.page_daisy.selected_splitting()
                )
            else:
                pages = self.page_mode.pages_per_file() if mode == MODE_PAGE_ONLY else 1
                progress.update(90, "Splitting the text into audio segments...")
                segments = split_document(doc, mode, pages_per_file=pages)
            progress.update(100, "Opening the Recording window...")
        except ParseError as exc:
            progress.finish()
            wx.MessageBox(str(exc), "Could not read the document",
                          style=wx.OK | wx.ICON_ERROR)
            return
        except Exception as exc:  # noqa: BLE001
            progress.finish()
            wx.MessageBox(f"Preparing the document failed: {exc}",
                          "New project", style=wx.OK | wx.ICON_ERROR)
            return

        if not segments:
            progress.finish()
            wx.MessageBox("The document contains no readable text.",
                          "New project", style=wx.OK | wx.ICON_WARNING)
            return
        progress.finish()

        pdir = project_dir(name)
        last = self.settings.get("last_model", {})
        daisy_settings = None
        if is_daisy:
            daisy_settings = {
                "splitting": self.page_daisy.selected_splitting(),
                "language": self.page_daisy.language(),
                "publisher": self.page_daisy.publisher(),
            }
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
                **self._recording_defaults(),
                "punctuation": self.page_details.selected_punctuation(),
                "output_format": self.settings.get("recording.output_format", "wav"),
                "compute": self.settings.compute,
            },
            project_type=ptype,
            daisy_settings=daisy_settings,
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
        """DAISY project types get the DAISY-specific second page; every other
        type gets the regular audio file creation page."""
        if self.selected_project_type() in _DAISY_TYPES:
            return self.wizard.page_daisy
        return self.wizard.page_mode

    def GetPrev(self):
        return None

    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Project details"),
                  0, wx.ALL, 6)

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)

        # Project name edit box.  The visible label and the accessible name
        # must agree exactly so a screen reader never confuses this field
        # with the Project type combo below it.
        self.name_ctrl = wx.TextCtrl(self)
        self.name_ctrl.SetName("Project name")
        name_lbl = add_labeled(self, grid, "Project name", self.name_ctrl,
                               flag=wx.LEFT | wx.RIGHT, border=2)
        name_lbl.SetName("Project name")
        self.name_ctrl.SetName("Project name")

        # Project type combo: "Audio files with playlist" first, "Clipboard"
        # second, then the DAISY types (order comes from PROJECT_TYPES).
        self.project_type_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                               name="Project type")
        for value, label in PROJECT_TYPES:
            self.project_type_combo.Append(label, value)
        # Default to audio playlist (first choice).
        default_idx = next(
            (i for i, (v, _) in enumerate(PROJECT_TYPES)
             if v == PROJECT_TYPE_AUDIO_PLAYLIST), 0
        )
        self.project_type_combo.SetSelection(default_idx)
        type_lbl = add_labeled(self, grid, "Project type", self.project_type_combo,
                               flag=wx.LEFT | wx.RIGHT, border=2)
        type_lbl.SetName("Project type")
        self.project_type_combo.SetName("Project type")
        self.project_type_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._update_type_desc())
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
            radio.Bind(
                wx.EVT_RADIOBUTTON,
                lambda _evt, v=value: self._on_select(v),
            )
            self.radios.append((radio, value))
            first = False

        # Pages per audio file: only meaningful for "Page by page only".
        # It is disabled for every other mode (and skipped by screen readers
        # in that state) so its label never gets confused with another field.
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
            "How many pages are recorded into one audio file (1 to 50). "
            "Each finished group can be resumed later."
        )
        pages_row.Add(self.pages_spin, 0, wx.RIGHT, 4)
        self.pages_row = pages_row
        sizer.Add(pages_row, 0, wx.ALL, 4)

        self._on_select(self.selected())
        self._update_description()
        self.SetSizer(sizer)

    def _on_select(self, value: str):
        """Radio changed: enable the pages-per-file control only for
        "Page by page only", and refresh the mode description."""
        self.pages_spin.Enable(value == MODE_PAGE_ONLY)
        self._update_description()

    def _update_description(self):
        self.description.SetLabel(AUDIO_MODE_DESCRIPTIONS.get(self.selected(), ""))
        self.description.Wrap(680)

    def pages_per_file(self) -> int:
        """Selected pages-per-file value (clamped to the valid range)."""
        value = self.pages_spin.GetValue()
        return max(PAGES_PER_FILE_MIN, min(PAGES_PER_FILE_MAX, int(value)))

    def selected(self) -> str:
        for radio, value in self.radios:
            if radio.GetValue():
                return value
        return MODE_PAGE_WITH_H1


class _DaisyPage(WizardPage):
    """Second wizard page for DAISY project types.

    Replaces the generic audio file creation page when the user chooses a
    DAISY book type: here the chapter splitting, text inclusion and the DAISY
    metadata (language, publisher) are chosen. The choices are stored in the
    project and in Settings > DAISY settings.
    """

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
        sizer.Add(wx.StaticText(self, label="DAISY book setup"),
                  0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(self, label="Choose how the document is split into "
                                      "DAISY chapters, then press Finish to "
                                      "prepare the document and open the "
                                      "Recording window."),
            0, wx.ALL, 6,
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
                label="After recording, the DAISY book is created in the "
                      "project folder (DAISY for DAISY 2.02 audio books, "
                      "DAISY3 for DAISY 3 audio and text with images books) "
                      "and can be exported as a ZIP for DAISY readers.",
            ),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)

    def selected_splitting(self) -> str:
        for radio, value in self.radios:
            if radio.GetValue():
                return value
        return DAISY_SPLIT_H1

    def language(self) -> str:
        return (self.lang_ctrl.GetValue().strip() or "en").lower()

    def publisher(self) -> str:
        return self.publisher_ctrl.GetValue().strip()

    def _update_description(self):
        self.description.SetLabel(
            DAISY_SPLIT_DESCRIPTIONS.get(self.selected_splitting(), "")
        )
        self.description.Wrap(680)
