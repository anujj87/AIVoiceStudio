"""New Project wizard (SPEC 3.5).

Page 1: project name + Open document button.
Page 2: audio file creation mode (the same radio group as in Settings).

Finishing parses the document, splits it per the chosen mode, writes
``project.json`` and opens the Recording window.
"""

from __future__ import annotations

import datetime
import logging
import os

import wx
from wx.adv import EVT_WIZARD_FINISHED, Wizard, WizardPage

from .. import project
from ..constants import (
    AD_CHUNK_CHOICES,
    AD_CHUNK_SECONDS,
    AD_DEFAULT_FPS,
    AD_DEFAULT_OUTPUT,
    AD_DEFAULT_PLACEMENT,
    AD_DEFAULT_TRACK_PRIORITY,
    AD_FPS_CHOICES,
    AD_MAX_DESCRIPTION_SECONDS,
    AD_MIN_GAP_SECONDS,
    AD_NARRATION_VOLUME_DEFAULT,
    AD_OUTPUT_CHOICES,
    AD_WARN_SECONDS,
    AD_WINDOW_SECONDS,
    AUDIO_MODE_CHOICES,
    AUDIO_MODE_DESCRIPTIONS,
    DAISY_SPLIT_CHOICES,
    DAISY_SPLIT_DESCRIPTIONS,
    DAISY_SPLIT_H1,
    DEFAULT_PROJECT_BASE,
    MODE_PAGE_ONLY,
    MODE_PAGE_WITH_H1,
    PAGES_PER_FILE_MAX,
    PAGES_PER_FILE_MIN,
    PROJECT_BASE_AD,
    PROJECT_BASE_CHOICES,
    PROJECT_BASE_STT,
    PROJECT_TYPE_AD,
    PROJECT_TYPE_AUDIO_PLAYLIST,
    PROJECT_TYPE_CLIPBOARD,
    PROJECT_TYPE_DAISY3_AUDIO_TEXT,
    PROJECT_TYPE_DAISY_AUDIO,
    PROJECT_TYPE_DESCRIPTIONS,
    PROJECT_TYPE_STT,
    PROJECT_TYPES,
    PUNCTUATION_CHOICES,
    PUNCTUATION_DEFAULT,
    STT_DEFAULT_MINUTES,
    STT_DEFAULT_OUTPUT,
    STT_DEFAULT_SOURCE,
    STT_DEFAULT_SPLIT,
    STT_OUTPUT_CHOICES,
    STT_SOURCE_CHOICES,
    STT_SOURCE_FOLDER,
    STT_SOURCE_MULTIPLE,
    STT_SOURCE_SINGLE,
    STT_SPLIT_BY_TIME,
    STT_SPLIT_CHOICES,
    STT_TIME_CHOICES,
    SUPPORTED_EXTENSIONS,
)
from ..describe import catalog as describe_catalog
from ..describe import engine as describe_engine
from ..describe import media as describe_media
from ..describe import pipeline as describe_pipeline
from ..documents.parsers import ParseError, parse_document
from ..documents.splitter import split_document
from ..paths import project_dir
from ..settings import Settings
from ..stt.output import scan_source
from ..tts.models import ModelStore
from . import announce, dpi
from .a11y import add_check, add_labeled, finalize_accessibility
from .describe_dialog import DescribeDialog
from .progress import TaskProgressDialog
from .recording_dialog import RecordingDialog
from .stt_dialog import STTTranscribeDialog

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
        # Speech-to-Text flow: the media files the user added as sources.
        self.stt_sources: list[str] = []
        # Audio description flow: the video files the user added as sources.
        self.describe_sources: list[str] = []

        self.page_details = _DetailsPage(self)
        self.page_mode = _ModePage(self, settings)
        self.page_daisy = _DaisyPage(self, settings)
        self.page_info = _DaisyInfoPage(self, settings)
        self.page_stt_mode = _SttModePage(self, settings)
        self.GetPageAreaSizer().Add(self.page_details)
        if initial_name:
            self.page_details.name_ctrl.SetValue(initial_name)
        # Real MSAA accNames for the name box / type combo (SetName alone is
        # ignored by MSAA on this wx build).
        finalize_accessibility(self)
        dpi.prepare(self)
        dpi.follow(self)

        self.Bind(EVT_WIZARD_FINISHED, self._on_finish)
        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGED, self._on_page_changed)
        # Announce the first field (the project base) when the wizard opens: the
        # base decides which other fields this page shows, so it is the one the
        # user has to choose first - not the project name.
        wx.CallAfter(self.page_details.base_combo.SetFocus)

    def _on_page_changed(self, evt):
        # NVDA-friendly: focus lands on the first control of the new page.
        page = evt.GetPage()
        if evt.GetDirection():
            if page is self.page_details:
                wx.CallAfter(self.page_details.base_combo.SetFocus)
            elif page is self.page_mode and self.page_mode.radios:
                wx.CallAfter(self.page_mode.radios[0][0].SetFocus)
            elif page is self.page_daisy and self.page_daisy.radios:
                wx.CallAfter(self.page_daisy.radios[0][0].SetFocus)
            elif page is self.page_stt_mode and self.page_stt_mode.radios:
                wx.CallAfter(self.page_stt_mode.radios[0][0].SetFocus)
            elif page is self.page_info:
                wx.CallAfter(self.page_info.focus_first_field)
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
        from ..documents.splitter import split_daisy_chapters

        name = self.page_details.name_ctrl.GetValue().strip() or "Untitled project"
        # Remember the project base so the next New Project starts where this
        # one did.
        self.settings.set("project_base", self.page_details.selected_base())
        if self.page_details.selected_base() == PROJECT_BASE_STT:
            self._finish_stt(name)
            return
        if self.page_details.selected_base() == PROJECT_BASE_AD:
            self._finish_ad(name)
            return
        ptype = self.page_details.selected_project_type()
        mode = self.page_mode.selected()
        is_daisy = ptype in _DAISY_TYPES

        # Clipboard mode: no document needed; the pasted text is the content.
        if ptype == PROJECT_TYPE_CLIPBOARD:
            text = self.page_details.clipboard_content()
            if not text:
                wx.MessageBox(
                    "Paste or type the text to record in the Clipboard text "
                    "box first.",
                    "New project",
                    style=wx.OK | wx.ICON_INFORMATION,
                )
                return
            pdir = project_dir(name)
            last = self.settings.get("last_model", {})
            project.create_project(
                pdir, name, "<clipboard>", "clipboard",
                [{"index": 1, "title": name, "text": text}],
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
        except Exception as exc:
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
                **self.page_info.metadata(),
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

    # -- speech to text -----------------------------------------------------
    def _finish_stt(self, name: str) -> None:
        """Create the transcription project and open the transcribe widget."""
        sources = self.stt_sources
        if not sources:
            wx.MessageBox(
                "Add at least one source file or folder first.",
                "New project", style=wx.OK | wx.ICON_INFORMATION,
            )
            return

        # Expand folders into the media files they hold, so the job receives
        # plain file paths.  An entry that yields nothing is kept as-is so the
        # job can report it instead of silently dropping the user's choice.
        expanded: list[str] = []
        for entry in sources:
            found = scan_source(entry)
            expanded.extend(found or [entry])
        # De-duplicate while preserving order.
        seen: set[str] = set()
        files: list[str] = []
        for path in expanded:
            if path not in seen:
                seen.add(path)
                files.append(path)

        details = self.page_details
        stt_settings = {
            "model": self.settings.get("stt.model", "whisper_large_v3"),
            "variant": "",
            "language": "auto",
            "accuracy": self.settings.get("stt.accuracy", "balanced"),
            "translate_to": "",
            "compute": self.settings.compute,
            "output_format": details.selected_output(),
            "include_timestamp": details.include_timestamp(),
            "source_type": details.selected_source_type(),
            "split_mode": self.page_stt_mode.selected(),
            "minutes": self.page_stt_mode.minutes(),
            "sources": files,
        }
        # Remember the choices for the next project.
        self.settings.set("stt.output_format", stt_settings["output_format"])
        self.settings.set("stt.include_timestamp", stt_settings["include_timestamp"])
        self.settings.set("stt.source_type", stt_settings["source_type"])
        self.settings.set("stt.split_mode", stt_settings["split_mode"])
        self.settings.set("stt.minutes", stt_settings["minutes"])
        self.settings.save()

        pdir = project_dir(name)
        segments = [
            {"index": i + 1, "title": os.path.basename(src), "text": "",
             "source": src}
            for i, src in enumerate(files)
        ]
        project.create_project(
            pdir,
            name,
            files[0] if files else "",
            stt_settings["split_mode"],
            segments,
            project_type=PROJECT_TYPE_STT,
            stt_settings=stt_settings,
        )
        self.settings.add_recent_project(name, pdir)
        self.EndModal(wx.ID_OK)

        dlg = STTTranscribeDialog(
            self.GetParent(), pdir, self.settings, self.store,
        )
        dlg.ShowModal()
        dlg.Destroy()

    # -- audio description -------------------------------------------------
    def _finish_ad(self, name: str) -> None:
        """Create the audio description project and open the describe dialog.

        The wizard only collects the video files and the describing model; how
        the video is chunked and what documents are written is asked in the
        describe window, exactly as the transcribe window asks it for a
        transcription project.
        """
        sources = self.describe_sources
        if not sources:
            wx.MessageBox(
                "Add at least one video file first.",
                "New project", style=wx.OK | wx.ICON_INFORMATION,
            )
            return

        # A folder (however it got here) stands for the video files inside it.
        # An entry that yields nothing is kept as-is, so the run can report it
        # instead of silently dropping the user's choice.
        expanded: list[str] = []
        for entry in sources:
            found = describe_media.scan_videos(entry)
            expanded.extend(found or [entry])
        # De-duplicate while preserving order, so a file added twice is
        # described once.
        seen: set[str] = set()
        files: list[str] = []
        for path in expanded:
            if path not in seen:
                seen.add(path)
                files.append(path)

        details = self.page_details
        describe_settings = {
            "model": details.selected_ad_model(),
            "variant": details.selected_ad_variant(),
            "compute": details.selected_ad_compute(),
            "fps": details.selected_ad_fps(),
            "chunk_seconds": details.selected_ad_chunk(),
            # How much video one description covers.  A project records it so the
            # describe window can say what the project was created with; the
            # value itself is the catalogue's cadence unless a run has changed
            # it, because it is what decides how much a listener is told.
            "window_seconds": float(
                self.settings.get("describe.window_seconds", AD_WINDOW_SECONDS)
            ),
            "output_format": details.selected_ad_output(),
            "gap_fitting": bool(self.settings.get("describe.gap_fitting", False)),
            "min_gap": float(
                self.settings.get("describe.min_gap", AD_MIN_GAP_SECONDS)
            ),
            "max_description": float(
                self.settings.get(
                    "describe.max_description", AD_MAX_DESCRIPTION_SECONDS
                )
            ),
            "language": "en",
            "sources": files,
            # Speak the descriptions as well? The next project starts where the
            # last one left off, like the model choices above; the voice ids are
            # resolved against the installed voices by the describe window.
            "placement": str(
                self.settings.get("describe.placement", AD_DEFAULT_PLACEMENT)
            ),
            # Which of the two tracks gives way when the description is heard
            # over the film's own soundtrack (standard placement only).  The
            # describe window is where it is chosen; the project records the
            # value it was created with, so the next run starts there.
            "track_priority": str(
                self.settings.get("describe.track_priority",
                                  AD_DEFAULT_TRACK_PRIORITY)
            ),
            "narration": bool(self.settings.get("describe.narration", False)),
            # How loud the description is spoken.  The describe window is where
            # it is chosen; the project records the value it was created with,
            # so the record and the mix agree about it.
            "narration_volume": float(
                self.settings.get("describe.narration_volume",
                                  AD_NARRATION_VOLUME_DEFAULT)
            ),
            "narration_voice": {
                key: value
                for key, value in (
                    self.settings.get("describe.narration_voice", {}) or {}
                ).items()
                if key in ("tts", "language", "variant", "voice") and value
            },
        }
        # Remember the choices for the next project.
        self.settings.set("describe.model", describe_settings["model"])
        self.settings.set("describe.variant", describe_settings["variant"])
        self.settings.set("describe.compute", describe_settings["compute"])
        self.settings.set("describe.fps", describe_settings["fps"])
        self.settings.set(
            "describe.chunk_seconds", describe_settings["chunk_seconds"]
        )
        self.settings.set(
            "describe.window_seconds", describe_settings["window_seconds"]
        )
        self.settings.set(
            "describe.output_format", describe_settings["output_format"]
        )
        self.settings.set(
            "describe.narration_volume", describe_settings["narration_volume"]
        )
        self.settings.save()

        pdir = project_dir(name)
        segments = [
            {"index": i + 1, "title": os.path.basename(src), "text": "",
             "source": src}
            for i, src in enumerate(files)
        ]
        project.create_project(
            pdir,
            name,
            files[0] if files else "",
            "chunked",
            segments,
            project_type=PROJECT_TYPE_AD,
            describe_settings=describe_settings,
        )
        self.settings.add_recent_project(name, pdir)
        self.EndModal(wx.ID_OK)

        dlg = DescribeDialog(
            self.GetParent(), pdir, self.settings, self.store,
        )
        dlg.ShowModal()
        dlg.Destroy()


class _DetailsPage(WizardPage):
    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self._build_ui()

    def GetNext(self):
        """Speech-to-Text projects go to the file-creation page; Audio
        description and Clipboard finish straight from this page; DAISY project
        types get the DAISY-specific page; every other type gets the regular
        audio creation page."""
        if self.selected_base() == PROJECT_BASE_STT:
            return self.wizard.page_stt_mode
        if self.selected_base() == PROJECT_BASE_AD:
            # Only the video files and the describing model are chosen here;
            # the describe window asks for the rest, so there is nothing more
            # for this wizard to collect.
            return None
        ptype = self.selected_project_type()
        if ptype in _DAISY_TYPES:
            return self.wizard.page_daisy
        if ptype == PROJECT_TYPE_CLIPBOARD:
            return None  # Finish button instead of Next
        return self.wizard.page_mode

    def GetPrev(self):
        return None

    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Project details"),
                  0, wx.ALL, 6)

        # Project base: "Text to speech" (the long-standing behaviour) or the
        # newer "Speech to text".  The rest of this page swaps between the two.
        base_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        base_grid.AddGrowableCol(1)
        self.base_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Project base")
        for value, label in PROJECT_BASE_CHOICES:
            self.base_combo.Append(label, value)
        current_base = self.wizard.settings.get(
            "project_base", DEFAULT_PROJECT_BASE
        )
        base_idx = next(
            (i for i, (v, _) in enumerate(PROJECT_BASE_CHOICES)
             if v == current_base), 0
        )
        self.base_combo.SetSelection(base_idx)
        base_lbl = add_labeled(self, base_grid, "Project base", self.base_combo,
                               flag=wx.LEFT | wx.RIGHT, border=2)
        base_lbl.SetName("Project base")
        sizer.Add(base_grid, 0, wx.EXPAND | wx.ALL, 6)
        self.base_combo.Bind(
            wx.EVT_COMBOBOX, lambda _: self._update_visibility()
        )

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
        self.type_lbl = add_labeled(self, grid, "Project type",
                                    self.project_type_combo,
                                    flag=wx.LEFT | wx.RIGHT, border=2)
        self.type_lbl.SetName("Project type")
        self.project_type_combo.SetName("Project type")
        self.project_type_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._update_type_desc())
        # Clipboard projects: show a multi-line edit box for pasting text.
        self.project_type_combo.Bind(
            wx.EVT_COMBOBOX, lambda _: self._update_visibility()
        )

        # ------------------------------------------------------------------
        # Speech-to-Text controls (shown only when the base is Speech to text)
        # ------------------------------------------------------------------
        self.stt_output_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                            name="Project output type")
        for value, label in STT_OUTPUT_CHOICES:
            self.stt_output_combo.Append(label, value)
        current_out = self.wizard.settings.get("stt.output_format", STT_DEFAULT_OUTPUT)
        out_idx = next(
            (i for i, (v, _) in enumerate(STT_OUTPUT_CHOICES) if v == current_out), 0
        )
        self.stt_output_combo.SetSelection(out_idx)
        self.stt_output_lbl = add_labeled(
            self, grid, "Project output type", self.stt_output_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        self.stt_source_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                            name="Source type")
        for value, label in STT_SOURCE_CHOICES:
            self.stt_source_combo.Append(label, value)
        current_src = self.wizard.settings.get("stt.source_type", STT_DEFAULT_SOURCE)
        src_idx = next(
            (i for i, (v, _) in enumerate(STT_SOURCE_CHOICES) if v == current_src), 0
        )
        self.stt_source_combo.SetSelection(src_idx)
        self.stt_source_lbl = add_labeled(
            self, grid, "Select source type", self.stt_source_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        self.timestamp_check = add_check(
            self, sizer, "Include timestamp in output",
            checked=bool(self.wizard.settings.get("stt.include_timestamp", False)),
        )

        source_row = wx.BoxSizer(wx.HORIZONTAL)
        self.add_source_btn = wx.Button(self, label="&Add source")
        self.add_source_btn.SetName("Add source")
        self.add_source_btn.SetToolTip(
            "Pick a media file, several files, or a whole folder of them"
        )
        source_row.Add(self.add_source_btn, 0, wx.ALL, 4)
        self.sources_count = wx.StaticText(self, label="No sources added yet.")
        source_row.Add(self.sources_count, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        sizer.Add(source_row, 0, wx.EXPAND)

        self.sources_list = wx.ListBox(self, name="Selected sources",
                                       size=wx.Size(-1, 90))
        sizer.Add(self.sources_list, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.stt_note = wx.StaticText(
            self,
            label="Audio and video files are accepted.  A folder is searched "
                  "for media files (sub-folders included).",
        )
        sizer.Add(self.stt_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # ------------------------------------------------------------------
        # Audio description controls (shown only when the base is Audio
        # description).  A describing model is a plain Python package with a
        # virtualenv of its own, exactly like a speech-to-text model, so the
        # model and variant combos below mirror the STT ones - but a variant
        # here also has to say what it costs, because download size and video
        # memory are what decide whether it can run on this machine at all.
        # ------------------------------------------------------------------
        # Known total length of the chosen videos, probed once whenever the
        # source list changes (each probe is a subprocess, and the estimate is
        # refreshed on every combo change).
        self._ad_total_duration = 0.0

        self.ad_model_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                          name="Video description model")
        for model_id in describe_catalog.model_ids():
            # Say which models can describe a video right now and which still
            # have to be installed: a list of names alone leaves the user to
            # find that out by starting a run and reading a failure.
            self.ad_model_combo.Append(self._ad_model_label(model_id), model_id)
        # Start on the model last used, or on the smallest CPU-capable one:
        # opening on the first table row would offer a 37 GB model that cannot
        # run on most machines.
        self._select_ad_data(
            self.ad_model_combo,
            self.wizard.settings.get("describe.model")
            or describe_catalog.default_model_id(),
            fallback=0,
        )
        self.ad_model_combo.SetToolTip(
            "The model that describes the picture of the video. It is "
            "downloaded from Settings, Manage the video describer."
        )
        self.ad_model_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_ad_model())
        self.ad_model_lbl = add_labeled(
            self, grid, "Video description model", self.ad_model_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        self.ad_variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                            name="Model variant")
        self.ad_variant_combo.SetToolTip(
            "Which checkpoint to download and run; its size and video memory "
            "need are shown below"
        )
        self.ad_variant_combo.Bind(
            wx.EVT_COMBOBOX, lambda _: self._on_ad_change())
        self.ad_variant_lbl = add_labeled(
            self, grid, "Model variant", self.ad_variant_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        self.ad_compute_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                            name="Compute")
        self.ad_compute_combo.SetToolTip(
            "Run the describer on the CPU or on an NVIDIA GPU; a GPU is much "
            "faster and needs the video memory named below"
        )
        self.ad_compute_combo.Bind(
            wx.EVT_COMBOBOX, lambda _: self._on_ad_change())
        self.ad_compute_lbl = add_labeled(
            self, grid, "Compute", self.ad_compute_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        self.ad_fps_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                        name="Frames per second")
        for value, label in AD_FPS_CHOICES:
            self.ad_fps_combo.Append(label, value)
        # A read-only combo shows its first entry before anything is selected,
        # and its selection is still -1 - so the default is selected explicitly
        # and the shown value can never disagree with the value that is read.
        self._select_ad_data(
            self.ad_fps_combo,
            float(self.wizard.settings.get("describe.fps", AD_DEFAULT_FPS)),
            fallback=next(
                (i for i, (v, _) in enumerate(AD_FPS_CHOICES)
                 if v == AD_DEFAULT_FPS), 0
            ),
        )
        self.ad_fps_combo.SetToolTip(
            "How much of the video the describer sees; fewer frames is faster "
            "and coarser"
        )
        self.ad_fps_combo.Bind(wx.EVT_COMBOBOX, lambda _: self._on_ad_change())
        self.ad_fps_lbl = add_labeled(
            self, grid, "Frames per second", self.ad_fps_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        self.ad_chunk_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                          name="Describe in chunks of")
        for value, label in AD_CHUNK_CHOICES:
            self.ad_chunk_combo.Append(label, value)
        self._select_ad_data(
            self.ad_chunk_combo,
            float(self.wizard.settings.get("describe.chunk_seconds",
                                           AD_CHUNK_SECONDS)),
            fallback=next(
                (i for i, (v, _) in enumerate(AD_CHUNK_CHOICES)
                 if v == AD_CHUNK_SECONDS), 0
            ),
        )
        self.ad_chunk_combo.SetToolTip(
            "A long video is described one chunk at a time; shorter chunks use "
            "less memory and can be stopped and resumed"
        )
        self.ad_chunk_combo.Bind(
            wx.EVT_COMBOBOX, lambda _: self._on_ad_change())
        self.ad_chunk_lbl = add_labeled(
            self, grid, "Describe in chunks of", self.ad_chunk_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        self.ad_output_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                           name="Description output type")
        for value, label in AD_OUTPUT_CHOICES:
            self.ad_output_combo.Append(label, value)
        self._select_ad_data(
            self.ad_output_combo,
            self.wizard.settings.get("describe.output_format",
                                     AD_DEFAULT_OUTPUT),
            fallback=0,
        )
        self.ad_output_combo.SetToolTip(
            "The description script is the reviewable draft; the subtitle "
            "formats are what a player reads out at the right moment"
        )
        self.ad_output_lbl = add_labeled(
            self, grid, "Description output type", self.ad_output_combo,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )

        ad_source_row = wx.BoxSizer(wx.HORIZONTAL)
        self.ad_add_btn = wx.Button(self, label="&Add video")
        self.ad_add_btn.SetName("Add video")
        self.ad_add_btn.SetToolTip(
            "Pick one or several video files to describe"
        )
        ad_source_row.Add(self.ad_add_btn, 0, wx.ALL, 4)
        self.ad_sources_count = wx.StaticText(self, label="No videos added yet.")
        ad_source_row.Add(self.ad_sources_count, 1,
                          wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        sizer.Add(ad_source_row, 0, wx.EXPAND)

        self.ad_sources_list = wx.ListBox(self, name="Selected videos",
                                          size=wx.Size(-1, 90))
        sizer.Add(self.ad_sources_list, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # Which model/variant states have to be re-read when the page is shown
        # again: installing a model in Settings changes what these combos say.
        self.ad_state_note = wx.StaticText(self, label="")
        self.ad_state_note.SetName("Video description model state")
        self.ad_state_note.Wrap(680)
        sizer.Add(self.ad_state_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.ad_note = wx.StaticText(self, label="")
        self.ad_note.SetName("Audio description time estimate")
        self.ad_note.Wrap(680)
        sizer.Add(self.ad_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # What the chosen variant costs, and how much it tends to invent - the
        # second is the failure mode a blind listener cannot check for
        # themselves, so it is stated rather than hidden behind benchmarks.
        self.ad_model_note = wx.StaticText(self, label="")
        self.ad_model_note.SetName("Video description model note")
        self.ad_model_note.Wrap(680)
        sizer.Add(self.ad_model_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # ------------------------------------------------------------------
        # TTS controls (shown only when the base is Text to speech)
        # ------------------------------------------------------------------
        self.type_desc = wx.StaticText(self, label="")
        self.type_desc.Wrap(680)
        sizer.Add(self.type_desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self._update_type_desc()

        self.doc_row = wx.BoxSizer(wx.HORIZONTAL)
        self.open_btn = wx.Button(self, label="Open document")
        self.open_btn.SetName("Open document file")
        self.open_btn.SetToolTip("Browse for a PDF, TXT, DOC, DOCX, HTML, Markdown, or EPUB file")
        self.doc_row.Add(self.open_btn, 0, wx.ALL, 4)
        self.file_label = wx.StaticText(self, label="No document selected.")
        self.doc_row.Add(self.file_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        sizer.Add(self.doc_row, 0, wx.EXPAND)

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
        self.punct_lbl = add_labeled(self, punct_grid, "Punctuation",
                                     self.punct_combo,
                                     flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(punct_grid, 0, wx.EXPAND | wx.ALL, 6)
        self.punct_hint = wx.StaticText(
            self,
            label="Choose how punctuation marks are spoken. 'All' reads "
                  "every mark as a word, for example quote, dot, left "
                  "paren, tic - useful for TTS voices that cannot "
                  "pronounce punctuation.",
        )
        sizer.Add(self.punct_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Clipboard text: only shown when the Clipboard project type is
        # selected. Paste or type the text to record here.
        self.clipboard_label = wx.StaticText(self, label="Clipboard text:")
        self.clipboard_label.SetName("Clipboard text")
        sizer.Add(self.clipboard_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.clipboard_text = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_DONTWRAP, name="Clipboard text",
            size=wx.Size(-1, 140),
        )
        self.clipboard_text.SetName("Clipboard text")
        sizer.Add(self.clipboard_text, 1, wx.EXPAND | wx.ALL, 6)
        self.clipboard_hint = wx.StaticText(
            self,
            label="Paste (Ctrl+V) or type the text to record, then press "
                  "Finish to open the Recording window.",
        )
        sizer.Add(self.clipboard_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.files_note = wx.StaticText(
            self,
            label="Supported files: PDF, TXT, DOC, DOCX, HTML, Markdown, "
                  "EPUB. You can also copy text to the clipboard and "
                  "paste it here.",
        )
        sizer.Add(self.files_note, 0, wx.ALL, 6)
        self.SetSizer(sizer)

        self.open_btn.Bind(wx.EVT_BUTTON, self._on_open)
        self.add_source_btn.Bind(wx.EVT_BUTTON, self._on_add_source)
        self.ad_add_btn.Bind(wx.EVT_BUTTON, self._on_ad_add)
        # Fill the variant/compute combos and the estimate once, before the
        # page decides which base's controls to show.
        self._on_ad_model()
        self._update_visibility()

    # -- project base ------------------------------------------------------
    def selected_base(self) -> str:
        sel = self.base_combo.GetSelection()
        return (self.base_combo.GetClientData(sel) if sel >= 0
                else DEFAULT_PROJECT_BASE)

    def _is_stt(self) -> bool:
        return self.selected_base() == PROJECT_BASE_STT

    def _is_ad(self) -> bool:
        return self.selected_base() == PROJECT_BASE_AD

    def _tts_controls(self):
        return (
            self.type_lbl, self.project_type_combo, self.type_desc,
            self.open_btn, self.file_label, self.punct_lbl, self.punct_combo,
            self.punct_hint, self.files_note,
        )

    def _stt_controls(self):
        return (
            self.stt_output_lbl, self.stt_output_combo,
            self.stt_source_lbl, self.stt_source_combo,
            self.timestamp_check, self.add_source_btn, self.sources_count,
            self.sources_list, self.stt_note,
        )

    def _ad_controls(self):
        return (
            self.ad_model_lbl, self.ad_model_combo,
            self.ad_variant_lbl, self.ad_variant_combo,
            self.ad_compute_lbl, self.ad_compute_combo,
            self.ad_fps_lbl, self.ad_fps_combo,
            self.ad_chunk_lbl, self.ad_chunk_combo,
            self.ad_output_lbl, self.ad_output_combo,
            self.ad_add_btn, self.ad_sources_count, self.ad_sources_list,
            self.ad_state_note, self.ad_note, self.ad_model_note,
        )

    def _update_visibility(self):
        """Show the controls of the selected project base.

        Text to speech keeps its long-standing page (project type, document,
        punctuation, clipboard); Speech to text swaps those for the output,
        source and timestamp controls; Audio description swaps in the video
        source, describing model and the time estimate instead.  Exactly one
        base's controls are ever visible.
        """
        is_stt = self._is_stt()
        is_ad = self._is_ad()
        for ctrl in self._ad_controls():
            ctrl.Show(is_ad)
        for ctrl in self._stt_controls():
            ctrl.Show(is_stt and not is_ad)
        for ctrl in self._tts_controls():
            ctrl.Show(not is_stt and not is_ad)
        if is_ad:
            self._refresh_ad_states()
        if is_stt or is_ad:
            # Neither of the newer bases shows the clipboard paste box.
            self.clipboard_label.Hide()
            self.clipboard_text.Hide()
            self.clipboard_hint.Hide()
        else:
            # The TTS side still hides the document controls for Clipboard.
            self._update_clipboard_visibility()
        self.Layout()

    def _update_clipboard_visibility(self):
        """Clipboard projects: hide the document controls and show the paste
        box instead.  Every other type: the reverse."""
        is_clipboard = (
            not self._is_stt()
            and self.selected_project_type() == PROJECT_TYPE_CLIPBOARD
        )
        # Clipboard: name + type + punctuation + paste box, no document.
        self.open_btn.Show(not is_clipboard)
        self.file_label.Show(not is_clipboard)
        self.files_note.Show(not is_clipboard)
        for ctrl in (self.clipboard_label, self.clipboard_text,
                     self.clipboard_hint):
            ctrl.Show(is_clipboard)
        self.Layout()

    def clipboard_content(self) -> str:
        return self.clipboard_text.GetValue().strip()

    def selected_punctuation(self) -> str:
        sel = self.punct_combo.GetSelection()
        return self.punct_combo.GetClientData(sel) if sel >= 0 else PUNCTUATION_DEFAULT

    def selected_project_type(self) -> str:
        sel = self.project_type_combo.GetSelection()
        return self.project_type_combo.GetClientData(sel) if sel >= 0 else PROJECT_TYPE_AUDIO_PLAYLIST

    # -- audio description ------------------------------------------------
    def selected_ad_model(self) -> str:
        sel = self.ad_model_combo.GetSelection()
        if sel < 0:
            return describe_catalog.default_model_id()
        return (self.ad_model_combo.GetClientData(sel)
                or describe_catalog.default_model_id())

    def selected_ad_variant(self) -> str:
        sel = self.ad_variant_combo.GetSelection()
        if sel < 0:
            return describe_catalog.default_variant(self.selected_ad_model())
        return self.ad_variant_combo.GetClientData(sel) or ""

    def selected_ad_compute(self) -> str:
        sel = self.ad_compute_combo.GetSelection()
        if sel < 0:
            return "cpu"
        return self.ad_compute_combo.GetClientData(sel) or "cpu"

    def selected_ad_fps(self) -> float:
        sel = self.ad_fps_combo.GetSelection()
        if sel < 0:
            return AD_DEFAULT_FPS
        return float(self.ad_fps_combo.GetClientData(sel) or AD_DEFAULT_FPS)

    def selected_ad_chunk(self) -> float:
        sel = self.ad_chunk_combo.GetSelection()
        if sel < 0:
            return AD_CHUNK_SECONDS
        return float(self.ad_chunk_combo.GetClientData(sel) or AD_CHUNK_SECONDS)

    def selected_ad_output(self) -> str:
        sel = self.ad_output_combo.GetSelection()
        if sel < 0:
            return AD_DEFAULT_OUTPUT
        return self.ad_output_combo.GetClientData(sel) or AD_DEFAULT_OUTPUT

    @staticmethod
    def _select_ad_data(combo, value, fallback: int = 0) -> None:
        """Select the entry carrying ``value`` (or ``fallback`` when absent)."""
        for index in range(combo.GetCount()):
            if combo.GetClientData(index) == value:
                combo.SetSelection(index)
                return
        if 0 <= fallback < combo.GetCount():
            combo.SetSelection(fallback)

    # -- readiness ---------------------------------------------------------
    @staticmethod
    def _ad_model_label(model_id: str) -> str:
        """A model's combo label, saying whether it can be used yet."""
        name = describe_catalog.model(model_id)["name"]
        return f"{name} - {describe_engine.model_state_label(model_id)}"

    def _ad_variant_label(self, model_id: str, value: str, label: str) -> str:
        """A variant's combo label, saying whether its weights are here yet."""
        if not describe_catalog.hf_model(model_id, value):
            return f"{label} - no download for this choice"
        if describe_engine.variant_downloaded(model_id, value):
            return f"{label} - downloaded"
        # A download that stopped part-way leaves files but no usable
        # checkpoint, and the download that would finish it is still offered:
        # saying "not downloaded yet" would send the user looking for the
        # gigabytes that are already on the disk.
        if describe_engine.variant_partial(model_id, value):
            return f"{label} - download stopped part-way"
        return f"{label} - not downloaded yet"

    def _on_ad_model(self) -> None:
        """Refill the variant and compute combos for the chosen model."""
        model_id = self.selected_ad_model()
        self.ad_variant_combo.Clear()
        for value, label in describe_catalog.variants(model_id):
            self.ad_variant_combo.Append(
                self._ad_variant_label(model_id, value, label), value)
        self._select_ad_data(
            self.ad_variant_combo,
            self.wizard.settings.get("describe.variant")
            or describe_catalog.default_variant(model_id),
            fallback=0,
        )
        self._refresh_ad_compute()
        self._on_ad_change()

    def _refresh_ad_compute(self) -> None:
        """Fill the compute combo with the devices this model/variant can use.

        Only a model that genuinely cannot run without a GPU is offered the GPU
        alone; every other one offers both the CPU and the GPU (the preferred
        device first) whenever the machine has an NVIDIA card, and the CPU alone
        when it has not.
        """
        remembered = self.wizard.settings.get("describe.compute", "")
        self.ad_compute_combo.Clear()
        for option in describe_engine.compute_options(
            self.selected_ad_model(), self.selected_ad_variant()
        ):
            self.ad_compute_combo.Append(option["label"], option["value"])
        if not self.ad_compute_combo.GetCount():
            self.ad_compute_combo.Append("CPU", "cpu")
        self._select_ad_data(self.ad_compute_combo, remembered, fallback=0)

    def _on_ad_change(self) -> None:
        """A choice changed: refresh the model note and the time estimate."""
        self._refresh_ad_notes()

    def _refresh_ad_states(self) -> None:
        """Re-read the readiness suffixes from the environment and the cache.

        Called when the page becomes visible again, because installing a model
        or downloading a variant in Settings is exactly what happens between
        two visits to this page.
        """
        for index, model_id in enumerate(describe_catalog.model_ids()):
            if index < self.ad_model_combo.GetCount():
                self.ad_model_combo.SetString(index, self._ad_model_label(model_id))
        # Rebuilds the variant labels for the model that is still selected, and
        # refreshes the notes from it.
        self._on_ad_model()

    def _refresh_ad_notes(self) -> None:
        """Show what the chosen variant costs and how long the run will take."""
        model_id = self.selected_ad_model()
        variant = self.selected_ad_variant()
        entry = describe_catalog.model(model_id)
        # The two things that decide whether this choice can run right now: the
        # model's environment, and this variant's checkpoint.  They are separate
        # on purpose and are stated separately.
        if not describe_engine.model_available(model_id):
            self.ad_state_note.SetLabel(
                f"{entry['name']} is not installed yet. Open Settings > Manage "
                "the video describer and install it there; each model builds "
                "its own Python environment."
            )
        elif describe_engine.variant_downloaded(model_id, variant):
            self.ad_state_note.SetLabel(
                f"{entry['name']} is ready to use and this variant is already "
                "downloaded."
            )
        elif describe_engine.variant_partial(model_id, variant):
            self.ad_state_note.SetLabel(
                f"{entry['name']} is installed, but this variant's download "
                "stopped part-way: its files are incomplete, so it is fetched "
                "again - continuing where it stopped - the first time it is "
                "described, or now from Settings > Manage the video describer."
            )
        else:
            self.ad_state_note.SetLabel(
                f"{entry['name']} is installed, but this variant is not "
                "downloaded yet: its files are fetched the first time it is "
                "described, or now from Settings > Manage the video describer."
            )
        self.ad_state_note.Wrap(680)
        self.ad_model_note.SetLabel(
            f"{entry['name']} ({entry['license']}): download about "
            f"{describe_catalog.variant_size_gb(model_id, variant):.1f} GB, "
            f"needs about "
            f"{describe_catalog.variant_vram_gb(model_id, variant):.1f} GB of "
            f"video memory.  {entry['notes']}  Hallucination: "
            f"{describe_catalog.hallucination_note(model_id)}"
        )
        self.ad_model_note.Wrap(680)

        duration = self._ad_total_duration
        if duration <= 0:
            self.ad_note.SetLabel(
                "Add a video to see how long describing it is estimated to "
                "take."
            )
        else:
            info = describe_pipeline.estimate(
                duration,
                model_id=model_id,
                variant=variant,
                fps=self.selected_ad_fps(),
                device=self.selected_ad_compute(),
                chunk_seconds=self.selected_ad_chunk(),
            )
            lines = [
                f"Estimated time: about "
                f"{describe_pipeline.format_duration(info['seconds'])} for "
                f"{describe_pipeline.format_duration(duration)} of video, in "
                f"{info['chunks']} chunk(s) and "
                f"{info['descriptions']} description(s) - one for every "
                f"{AD_WINDOW_SECONDS:g} seconds of video, which is what keeps a "
                "change in the picture from being summarised away."
            ]
            if describe_pipeline.needs_warning(duration, AD_WARN_SECONDS):
                lines.append(
                    "This is a long video: describing it takes far longer than "
                    "transcribing it. Shorter chunks are recommended - they "
                    "use less memory and the run can be stopped and resumed "
                    "between them."
                )
            self.ad_note.SetLabel("\n".join(lines))
        self.ad_note.Wrap(680)
        self.Layout()

    def _refresh_ad_sources(self, message: str = "") -> None:
        """Refresh the video list, the count and the (probed) total length."""
        sources = self.wizard.describe_sources
        self.ad_sources_list.Set([os.path.basename(s) or s for s in sources])
        if sources:
            announce.status_text(
                self.ad_sources_count,
                message or f"{len(sources)} video(s) added.",
                key="ad-sources",
            )
        else:
            announce.status_text(self.ad_sources_count,
                                 "No videos added yet.")
        # The length is probed here rather than in _refresh_ad_notes: a probe is
        # a subprocess, and the estimate is refreshed on every combo change.
        self._ad_total_duration = describe_media.total_duration(sources)
        self._refresh_ad_notes()

    def _on_ad_add(self, _):
        """Pick video files to describe; a non-video choice is refused."""
        wildcard = (
            "Video files|"
            + ";".join("*" + ext for ext in
                       sorted(describe_media.VIDEO_EXTENSIONS))
            + "|All files|*.*"
        )
        style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE
        with wx.FileDialog(self, "Choose video file(s) to describe",
                           wildcard=wildcard, style=style) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            chosen = list(dlg.GetPaths())
        accepted = [path for path in chosen
                    if describe_media.is_video_file(path)]
        rejected = [os.path.basename(path) for path in chosen
                    if not describe_media.is_video_file(path)]
        if accepted:
            self.wizard.describe_sources.extend(accepted)
        if rejected:
            wx.MessageBox(
                "These files are not video files and were not added:\n"
                + "\n".join(rejected),
                "Add video", style=wx.OK | wx.ICON_WARNING,
            )
        self._refresh_ad_sources()

    # -- speech to text accessors -----------------------------------------
    def selected_output(self) -> str:
        sel = self.stt_output_combo.GetSelection()
        return (self.stt_output_combo.GetClientData(sel) if sel >= 0
                else STT_DEFAULT_OUTPUT)

    def include_timestamp(self) -> bool:
        return bool(self.timestamp_check.GetValue())

    def selected_source_type(self) -> str:
        sel = self.stt_source_combo.GetSelection()
        return (self.stt_source_combo.GetClientData(sel) if sel >= 0
                else STT_DEFAULT_SOURCE)

    def _refresh_sources(self, message: str = "") -> None:
        sources = self.wizard.stt_sources
        self.sources_list.Set([os.path.basename(s) or s for s in sources])
        if sources:
            announce.status_text(
                self.sources_count,
                message or f"{len(sources)} source(s) added.",
                key="stt-sources",
            )
        else:
            announce.status_text(self.sources_count,
                                 "No sources added yet.")
        self.Layout()

    def _on_add_source(self, _):
        """Open the dialog that matches the selected source type."""
        kind = self.selected_source_type()
        wildcard = (
            "Audio and video files|"
            "*.wav;*.mp3;*.flac;*.m4a;*.aac;*.ogg;*.oga;*.opus;*.wma;"
            "*.aiff;*.aif;*.amr;*.mp4;*.mkv;*.mov;*.avi;*.webm;*.flv;"
            "*.wmv;*.m4v;*.mpg;*.mpeg;*.3gp|"
            "All files|*.*"
        )
        if kind == STT_SOURCE_FOLDER:
            with wx.DirDialog(
                self, "Choose a folder of audio or video files",
                style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
            ) as dlg:
                if dlg.ShowModal() != wx.ID_OK:
                    return
                self.wizard.stt_sources.append(dlg.GetPath())
                self._refresh_sources()
            return
        if kind == STT_SOURCE_MULTIPLE:
            style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE
        else:
            style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        with wx.FileDialog(self, "Choose audio or video file(s)",
                           wildcard=wildcard, style=style) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            chosen = list(dlg.GetPaths())
        if kind == STT_SOURCE_SINGLE:
            self.wizard.stt_sources = chosen[:1]
        else:
            self.wizard.stt_sources.extend(chosen)
        self._refresh_sources()

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


class _SttModePage(WizardPage):
    """Second wizard page for Speech-to-Text projects: file creation.

    "A document file for each source file" writes one transcript per source
    (named after it: ``lecture 1``); "Time by time separation" cuts the
    transcript at the chosen interval into numbered parts
    (``lecture 1 part 1``, ``lecture 1 part 2``, ...).
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
        sizer.Add(wx.StaticText(self, label="File creation"), 0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="Choose how the transcribed text is written to "
                      "documents, then press Finish to open the transcribe "
                      "window.",
            ),
            0, wx.ALL, 6,
        )

        self.radios = []
        current = settings.get("stt.split_mode", STT_DEFAULT_SPLIT)
        first = True
        for value, label in STT_SPLIT_CHOICES:
            radio = wx.RadioButton(
                self, label=label, style=wx.RB_GROUP if first else 0,
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

        time_row = wx.BoxSizer(wx.HORIZONTAL)
        time_lbl = wx.StaticText(self, label="Minutes per document:")
        time_lbl.SetName("Minutes per document")
        time_row.Add(time_lbl, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 4)
        self.minutes_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Minutes per document")
        for minutes in STT_TIME_CHOICES:
            self.minutes_combo.Append(f"{minutes} minutes", minutes)
        current_minutes = int(settings.get("stt.minutes", STT_DEFAULT_MINUTES) or 30)
        minute_idx = next(
            (i for i, m in enumerate(STT_TIME_CHOICES) if m == current_minutes), 0
        )
        self.minutes_combo.SetSelection(minute_idx)
        time_row.Add(self.minutes_combo, 0, wx.RIGHT, 4)
        sizer.Add(time_row, 0, wx.ALL, 4)

        self.hint = wx.StaticText(self, label="")
        self.hint.Wrap(680)
        sizer.Add(self.hint, 0, wx.ALL, 6)

        self._on_select(self.selected())
        self.SetSizer(sizer)

    def _on_select(self, value: str):
        self.minutes_combo.Enable(value == STT_SPLIT_BY_TIME)
        if value == STT_SPLIT_BY_TIME:
            self.hint.SetLabel(
                "Each source file is cut at the chosen interval and written "
                "as 'name part 1', 'name part 2', ...  For example 'lecture 1 "
                "part 1' and 'lecture 1 part 2'."
            )
        else:
            self.hint.SetLabel(
                "One document is written for each source file, named after "
                "it - for example 'lecture 1', then 'lecture 2'."
            )
        self.hint.Wrap(680)

    def selected(self) -> str:
        for radio, value in self.radios:
            if radio.GetValue():
                return value
        return STT_DEFAULT_SPLIT

    def minutes(self) -> int:
        sel = self.minutes_combo.GetSelection()
        return self.minutes_combo.GetClientData(sel) if sel >= 0 else STT_DEFAULT_MINUTES


class _DaisyPage(WizardPage):
    """Second wizard page for DAISY project types.

    Replaces the generic audio file creation page when the user chooses a
    DAISY book type: here the chapter splitting is chosen.  The DAISY book
    information (title, creator, dates and so on) is asked on the next page
    (:class:`_DaisyInfoPage`).  The choices are stored in the project and in
    Settings > DAISY settings.
    """

    def __init__(self, wizard, settings: Settings):
        super().__init__(wizard)
        self.wizard = wizard
        self._build_ui(settings)

    def GetNext(self):
        return self.wizard.page_info

    def GetPrev(self):
        return self.wizard.page_details

    def _build_ui(self, settings: Settings):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="DAISY book setup"),
                  0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(self, label="Choose how the document is split into "
                                      "DAISY chapters, then press Next to "
                                      "enter the DAISY book information."),
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

    def _update_description(self):
        self.description.SetLabel(
            DAISY_SPLIT_DESCRIPTIONS.get(self.selected_splitting(), "")
        )
        self.description.Wrap(680)


class _DaisyInfoPage(WizardPage):
    """Third wizard page (DAISY types): the DAISY book information.

    Asks Title, Creator, Date, Language, Publisher, Subject and Narrator in
    edit boxes and stores them in the project; the builders write them into
    the generated book metadata.  Running time is measured automatically from
    the recorded audio and Producer/Software are filled automatically.
    """

    # (key, label, settings key, default).  Order defines tab order.
    _FIELDS = (
        ("title", "Title", None, ""),
        ("creator", "Creator", "daisy.creator", ""),
        ("date", "Date", "daisy.date", ""),
        ("language", "Language", "daisy.language", "en"),
        ("publisher", "Publisher", "daisy.publisher", ""),
        ("subject", "Subject", "daisy.subject", ""),
        ("narrator", "Narrator", "daisy.narrator", ""),
        ("producer", "Producer", "daisy.producer", ""),
    )

    def __init__(self, wizard, settings: Settings):
        super().__init__(wizard)
        self.wizard = wizard
        self.settings = settings
        self._last_name_value = ""
        self._build_ui(settings)

    def GetNext(self):
        return None

    def GetPrev(self):
        return self.wizard.page_daisy

    def _build_ui(self, settings: Settings):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="DAISY book information"),
                  0, wx.ALL, 6)
        sizer.Add(
            wx.StaticText(self, label="Enter the book information stored in "
                                      "the DAISY book. Running time is "
                                      "filled automatically from the "
                                      "recording."),
            0, wx.ALL, 6,
        )

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)

        self._controls: dict = {}
        for key, label, skey, default in self._FIELDS:
            ctrl = wx.TextCtrl(self)
            ctrl.SetName(f"DAISY {label}")
            if skey:
                ctrl.SetValue(str(self.settings.get(skey, default) or default))
            else:
                ctrl.SetValue(default)
            add_labeled(self, grid, label, ctrl,
                        flag=wx.LEFT | wx.RIGHT, border=2)
            self._controls[key] = ctrl

        # Title defaults to the project name (kept in sync until edited).
        self._controls["title"].SetValue(
            self.wizard.page_details.name_ctrl.GetValue().strip()
        )
        self.wizard.page_details.name_ctrl.Bind(
            wx.EVT_TEXT, self._on_name_changed
        )

        # Optional software information: when unchecked the book carries no
        # producer/software metadata at all.
        self.software_check = add_check(
            self, sizer, "Show software information in book",
            checked=bool(settings.get("daisy.show_software", True)),
        )

        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)
        sizer.Add(
            wx.StaticText(
                self,
                label="Press Finish to prepare the document and open the "
                      "Recording window.",
            ),
            0, wx.ALL, 6,
        )
        self.SetSizer(sizer)

    def _on_name_changed(self, evt):
        """Keep the Title box in sync with the project name while the title
        has not been customized."""
        title_ctrl = self._controls["title"]
        if title_ctrl.GetValue().strip() in ("", self._last_name_value):
            title_ctrl.SetValue(evt.GetString().strip())
        self._last_name_value = evt.GetString().strip()

    def focus_first_field(self):
        """Move focus to the Title box (the page's first field)."""
        self._controls["title"].SetFocus()

    def metadata(self) -> dict:
        """Collected DAISY book information for ``project.json``."""
        out: dict = {}
        for key, _label, _skey, _default in self._FIELDS:
            out[key] = self._controls[key].GetValue().strip()
        out["language"] = (out.get("language") or "en").lower()
        out["show_software"] = self.software_check.GetValue()
        # Date: fill with today when left empty (yyyy-mm-dd).
        if not out.get("date"):
            out["date"] = datetime.date.today().isoformat()
        return out
