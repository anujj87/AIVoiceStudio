"""Recording window (SPEC 3.6).

Opened from the New Project wizard, the main menu (Ctrl+Shift+R) or File ->
Open Project. All TTS settings chosen here apply only to this project and are
stored in ``project.json``. Segments are synthesized in a background thread and
**saved to disk immediately**; if the app stops or crashes, reopening the
project resumes from the first unsaved segment.
"""

from __future__ import annotations

import logging
import os
import threading
import wx

from .. import compute, project
from ..audio import ffmpeg as ffmpeg_mod
from ..constants import (
    AUDIO_MODE_DESCRIPTIONS,
    DAISY3_PACKAGE_FILE,
    DAISY3_OUTPUT_DIR_NAME,
    DAISY_NCC_FILE,
    DAISY_OUTPUT_DIR_NAME,
    FORMAT_WAV,
    MODE_ONE_FILE,
    OUTPUT_FORMAT_CHOICES,
    PITCH_MAX,
    PITCH_MIN,
    RATE_MAX,
    RATE_MIN,
    VOLUME_MAX,
    VOLUME_MIN,
)
from ..documents.splitter import Segment
from ..jobs.synthesizer import SynthesisWorker
from .. import venv_packages
from ..settings import Settings
from ..tts import catalog, windows_tts
from ..util import sanitize_filename
from ..tts.models import ModelStore
from . import dialogs, language_choice
from .a11y import (
    add_labeled,
    finalize_accessibility,
    set_accessible_name,
    update_accessible_name,
)
from .events import (
    EVT_SYNTH_ERROR,
    EVT_SYNTH_FINISHED,
    EVT_SYNTH_SEGMENT_DONE,
    EVT_SYNTH_STATUS,
    SynthErrorEvent,
    SynthFinishedEvent,
    SynthSegmentDoneEvent,
    SynthStatusEvent,
)
from .progress import TaskProgressDialog

log = logging.getLogger(__name__)

# Project types that produce a DAISY book (2.02 audio-only or the legacy
# 2.02 audio+text, plus DAISY 3 audio+text with images).
_DAISY_PROJECT_TYPES = (
    "daisy_audio", "daisy_audio_text", "daisy3_audio_text",
)


def _daisy_entry_point(project_dir: str, ptype: str) -> tuple[str, str]:
    """Return ``(kind, entry_path)`` for a DAISY project type.

    ``kind`` is ``"2.02 audio"``, ``"2.02 audio+text"`` or ``"DAISY 3"``;
    ``entry_path`` is the file a player opens (the NCC or the OPF).
    """
    if ptype == "daisy_audio":
        return "2.02 audio", os.path.join(
            project_dir, DAISY_OUTPUT_DIR_NAME, DAISY_NCC_FILE)
    if ptype == "daisy_audio_text":
        return "2.02 audio+text", os.path.join(
            project_dir, DAISY_OUTPUT_DIR_NAME, DAISY_NCC_FILE)
    return "DAISY 3", os.path.join(
        project_dir, DAISY3_OUTPUT_DIR_NAME, DAISY3_PACKAGE_FILE)


def _daisy_book_exists(project_dir: str, ptype: str) -> bool:
    """True when the generated book for ``ptype`` exists on disk."""
    _kind, entry = _daisy_entry_point(project_dir, ptype)
    return os.path.isfile(entry)

_COMPUTE_LABELS = {
    compute.COMPUTE_AUTO: "Auto (best available)",
    compute.COMPUTE_CPU: "CPU (ONNX)",
    compute.COMPUTE_CUDA: "GPU (ONNX)",
    compute.COMPUTE_DML: "NPU (DirectML)",
    "cuda_gpu": "CUDA GPU (OmniVoice)",
}

# TTS engines available for each compute mode.
# Local engines (Piper, Kokoro, Kitten, the built-in Windows system voices,
# etc.) work with CPU/GPU compute.  OmniVoice engines (Server, Triton,
# Hybrid) require CUDA GPU.  The Voice Lab engines (Pocket TTS, Bark, F5-TTS)
# run on the CPU and on the GPU, so their compute combo offers the CPU and -
# when a GPU is detected - the GPU as well.
_ONNX_TTS_ENGINES = {
    "piper", "kokoro", "kitten", "matcha", "pocket",
    "sapi5", "windows_core",
}
_OMNIVOICE_ENGINES = {"omnivoice", "omnivoice_server"}
_VOICE_LAB_ENGINES = {"pocket_tts", "bark", "f5tts"}


class RecordingDialog(wx.Dialog):
    def __init__(
        self,
        parent,
        project_dir: str,
        settings: Settings,
        store: ModelStore,
        start_index: int | None = None,
        single_segment: bool = False,
    ):
        """``start_index`` (0-based) starts the next recording at that segment
        instead of the first segment that is still pending; ``single_segment``
        stops the run after that one segment (Edit > Start Selected
        Recording)."""
        data = project.load_project(project_dir)
        super().__init__(parent, title=f"Recording - {data.get('name', project_dir)}",
                         size=(720, 640))
        self.project_dir = project_dir
        self.data = data
        self.settings = settings
        self.store = store
        self._segments: list[Segment] = []
        self._voices: list = []
        self._worker: SynthesisWorker | None = None
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._progress_dlg: TaskProgressDialog | None = None
        self._warned_one_file = False
        self._start_index = None if start_index is None else max(0, int(start_index))
        self._single_segment = bool(single_segment)

        self._load_segments()
        self._build_ui()
        self._populate_voices()
        self._apply_project_tts()
        self._update_progress()
        note = self._pending_note()
        if note:
            self.status.SetLabel(note)
        # Announce the project name and focus the text preview.
        self.SetName(f"Recording: {self.data.get('name', '')}")
        # Real MSAA accNames for every labelled control.
        finalize_accessibility(self)
        wx.CallAfter(self.text_preview.SetFocus)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        title_text = self.data.get('name', '')
        title = wx.StaticText(
            self,
            label=f"Project: {title_text}",
        )
        title.SetName(f"Recording project: {title_text}")
        sizer.Add(title, 0, wx.ALL, 6)

        # -- text to record --------------------------------------------------
        # Read-only view of what will be spoken. It is the first control in
        # tab order so screen readers announce the content first, then the
        # Compute back-end combo box that follows it.
        text_label = wx.StaticText(self, label="Text to record:")
        text_label.SetName("Text to record label")
        sizer.Add(text_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.text_preview = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY,
            size=(-1, 130),
        )
        self.text_preview.SetName("Text to record")
        self.text_preview.SetToolTip("Read-only preview of the text that will be synthesized into audio")
        self.text_preview.SetValue(
            "\n\n".join(seg.text for seg in self._segments)
        )
        sizer.Add(self.text_preview, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        self.text_preview.SetFocus()

        # -- model selectors ------------------------------------------------
        # Order: the TTS engine combo lists every downloaded/ready engine;
        # the Compute back-end combo right below it offers only the options
        # that engine supports (ONNX engines: Auto/CPU/GPU ONNX; OmniVoice
        # engines: CUDA GPU). Language / variant / voice follow the cascade.
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.compute_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Compute back-end")
        self.tts_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                     name="Select TTS engine")
        self.lang_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Select language")
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Select variant")
        self.voice_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Select voice")
        add_labeled(self, grid, "TTS engine", self.tts_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Compute back-end", self.compute_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Language", self.lang_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Variant", self.variant_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "Voice", self.voice_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 6)

        # -- voice parameters ----------------------------------------------
        params = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        params.AddGrowableCol(1)
        self.rate = self._slider(params, "Rate", RATE_MIN, RATE_MAX)
        self.pitch = self._slider(params, "Pitch", PITCH_MIN, PITCH_MAX)
        self.volume = self._slider(params, "Volume", VOLUME_MIN, VOLUME_MAX)
        self.format_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Output audio format")
        for value, label in OUTPUT_FORMAT_CHOICES:
            self.format_combo.Append(label, value)
        self.format_combo.SetSelection(0)
        self.format_combo.SetToolTip(
            "Choose the audio file format: WAV is always available, MP3 and "
            "FLAC need FFmpeg. WAV is recommended for DAISY projects.")
        self.format_combo.Bind(wx.EVT_COMBOBOX, self._on_format_changed)
        fmt_label = add_labeled(self, params, "Output format", self.format_combo,
                                flag=wx.LEFT | wx.RIGHT, border=2)
        # The label cell is itself announced by screen readers; give it the
        # same accessible name as the combo so the row always reads as
        # "Output format" and never as a bare value or default class name.
        fmt_label.SetName("Output format")
        sizer.Add(params, 0, wx.EXPAND | wx.ALL, 6)

        # -- OmniVoice voice options (per project) ---------------------------
        omni_row = wx.BoxSizer(wx.HORIZONTAL)
        self.omni_btn = wx.Button(self, label="OmniVoice voice options...")
        self.omni_btn.SetName("OmniVoice voice options")
        self.omni_btn.SetToolTip(
            "Choose auto / voice design / voice clone and the advanced "
            "OmniVoice generation settings for this project."
        )
        omni_row.Add(self.omni_btn, 0, wx.ALL, 4)
        self.omni_summary = wx.StaticText(self, label="")
        self.omni_summary.SetName("OmniVoice voice options summary")
        omni_row.Add(self.omni_summary, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        self.omni_btn.Hide()
        self.omni_summary.Hide()
        sizer.Add(omni_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # -- Voice Lab tuning (per engine, per project) ----------------------
        tune_row = wx.BoxSizer(wx.HORIZONTAL)
        self.engine_options_btn = wx.Button(self, label="Engine tuning...")
        self.engine_options_btn.SetName("Voice engine tuning")
        self.engine_options_btn.SetToolTip(
            "Diffusion steps, sampling temperatures, quantization and the "
            "other generation settings of the selected Voice Lab engine."
        )
        tune_row.Add(self.engine_options_btn, 0, wx.ALL, 4)
        self.engine_options_summary = wx.StaticText(self, label="")
        self.engine_options_summary.SetName("Voice engine tuning summary")
        tune_row.Add(self.engine_options_summary, 1,
                     wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        self.engine_options_btn.Hide()
        self.engine_options_summary.Hide()
        sizer.Add(tune_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # -- controls --------------------------------------------------------
        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self, label="Start recording")
        self.pause_btn = wx.Button(self, label="Pause")
        self.resume_btn = wx.Button(self, label="Resume")
        self.stop_btn = wx.Button(self, label="Stop")
        for btn, name, tip in ((
            self.start_btn, "Start recording", "Begin or resume synthesizing audio for all pending segments"),
            (self.pause_btn, "Pause recording", "Pause after the current segment finishes"),
            (self.resume_btn, "Resume recording", "Continue recording from where you paused"),
            (self.stop_btn, "Stop recording", "Stop after the current segment finishes"),
        ):
            btn.SetName(name)
            btn.SetToolTip(tip)
        self.pause_btn.Disable()
        self.resume_btn.Disable()
        self.stop_btn.Disable()
        btns.Add(self.start_btn, 0, wx.ALL, 4)
        btns.Add(self.pause_btn, 0, wx.ALL, 4)
        btns.Add(self.resume_btn, 0, wx.ALL, 4)
        btns.Add(self.stop_btn, 0, wx.ALL, 4)
        sizer.Add(btns, 0, wx.LEFT, 2)

        self.gauge = wx.Gauge(self, range=100, size=(-1, 22))
        self.gauge.SetName("Recording progress bar")
        self.gauge.SetToolTip("Shows how many segments have been recorded")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.ALL, 6)
        self.status = wx.StaticText(self, label="Ready.")
        self.status.SetName("Recording status")
        self.status.SetToolTip("Current status of the recording process")
        sizer.Add(self.status, 0, wx.ALL, 6)

        sizer.Add(
            wx.StaticText(self, label="Segments are saved immediately. Close and "
                                      "reopen to resume later."),
            0, wx.ALL, 6,
        )

        close_btn = wx.Button(self, label="Close")
        close_btn.SetName("Close recording window")
        close_btn.SetToolTip("Close this window (recording will resume when you reopen the project)")
        close_sizer = wx.BoxSizer(wx.HORIZONTAL)
        close_sizer.AddStretchSpacer(1)
        # DAISY export button (hidden until recording completes)
        ptype = self.data.get("project_type", "audio_playlist")
        self.export_daisy_btn = wx.Button(self, label="Export DAISY as ZIP")
        self.export_daisy_btn.SetName("Export DAISY book as ZIP archive")
        self.export_daisy_btn.SetToolTip("Package the DAISY book into a ZIP file for distribution")
        self.export_daisy_btn.Show(ptype in _DAISY_PROJECT_TYPES)
        self.export_daisy_btn.Disable()
        close_sizer.Add(self.export_daisy_btn, 0, wx.ALL, 4)
        close_sizer.Add(close_btn, 0, wx.ALL, 4)
        sizer.Add(close_sizer, 0, wx.EXPAND)

        self.SetSizer(sizer)

        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start)
        self.pause_btn.Bind(wx.EVT_BUTTON, self._on_pause)
        self.resume_btn.Bind(wx.EVT_BUTTON, self._on_resume)
        self.stop_btn.Bind(wx.EVT_BUTTON, self._on_stop)
        close_btn.Bind(wx.EVT_BUTTON, lambda _: self.Close())
        self.export_daisy_btn.Bind(wx.EVT_BUTTON, self._on_export_daisy)
        self.tts_combo.Bind(wx.EVT_COMBOBOX, self._on_tts)
        self.lang_combo.Bind(wx.EVT_COMBOBOX, self._on_lang)
        self.variant_combo.Bind(wx.EVT_COMBOBOX, self._on_variant)
        self.voice_combo.Bind(wx.EVT_COMBOBOX, self._on_voice_change)
        self.omni_btn.Bind(wx.EVT_BUTTON, self._on_omni_options)
        self.engine_options_btn.Bind(wx.EVT_BUTTON, self._on_engine_options)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(EVT_SYNTH_STATUS, self._on_synth_status)
        self.Bind(EVT_SYNTH_SEGMENT_DONE, self._on_segment_done)
        self.Bind(EVT_SYNTH_FINISHED, self._on_finished)
        self.Bind(EVT_SYNTH_ERROR, self._on_error)

    def _selected_compute(self) -> str:
        """Return the selected compute mode key (with 'auto' resolved).

        Voice Lab engines resolve 'auto' to the GPU when one is present, the
        ONNX engines resolve it through the application's compute detection.
        """
        sel = self.compute_combo.GetSelection()
        raw = self.compute_combo.GetClientData(sel) if sel >= 0 else "auto"
        if self._selected_tts_id() in _VOICE_LAB_ENGINES:
            from ..voicelab import resolve_device  # noqa: PLC0415

            return resolve_device(raw)
        if raw == compute.COMPUTE_AUTO:
            return compute.resolve_compute(raw)
        return raw

    def _is_omnivoice_compute(self) -> bool:
        """True when CUDA GPU (OmniVoice) compute mode is selected."""
        return self._selected_compute() == "cuda_gpu"

    def _slider(self, grid, name: str, lo: float, hi: float) -> wx.Slider:
        """A labelled slider row (see settings dialog ``_slider_row``).

        The current value is part of the row label's own text ("Rate: 1.00")
        instead of a separate bare-number static text, so a screen reader
        always announces the word together with the number.
        """
        label = wx.StaticText(self, label=f"{name}: 1.00")
        label.SetName(name + " value")
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        slider = wx.Slider(self, minValue=int(lo * 100), maxValue=int(hi * 100),
                           value=int(1.0 * 100))
        set_accessible_name(slider, f"{name}: 1.00")
        slider.value_label = label  # type: ignore[attr-defined]
        grid.Add(slider, 1, wx.EXPAND)
        slider.Bind(
            wx.EVT_SLIDER,
            lambda evt, s=slider, lb=label, n=name: (
                lb.SetLabel(f"{n}: {s.GetValue() / 100.0:.2f}"),
                update_accessible_name(s, f"{n}: {s.GetValue() / 100.0:.2f}"),
            ),
        )
        return slider

    # ------------------------------------------------------------- project
    def _load_segments(self):
        self._segments = []
        for seg in self.data.get("segments", []):
            self._segments.append(
                Segment(
                    index=seg.get("index", 1),
                    title=seg.get("title", ""),
                    text=seg.get("text", ""),
                )
            )

    def _populate_voices(self):
        """Load voices and group by TTS engine, filtering by compute mode."""
        self._all_voices = self.store.installed_voices()
        # Built-in Windows voices (SAPI5 / Windows Core): always available.
        windows_tts.add_installed_voices(self._all_voices, self._on_builtin_voices)
        # Universal OmniVoice voice library: created voices are engine
        # agnostic, so register them under every installed OmniVoice engine
        # and they appear no matter which TTS version is selected.
        try:
            from ..omnivoice import voice_store  # noqa: PLC0415
            if voice_store.omni_custom_voices(self.store):
                installed = voice_store.engine_ids_installed()
                self._all_voices.extend(
                    voice_store.consumer_entries(self.store, installed)
                )
        except Exception:  # noqa: BLE001
            pass
        # Inject pip-installed OmniVoice voices (not in artifact system)
        self._inject_omnivoice_voices()
        # Bind compute combo change to refresh the voice cascade (only once:
        # this method runs again when background voice discovery finishes).
        if not getattr(self, "_compute_change_bound", False):
            self._compute_change_bound = True
            self.compute_combo.Bind(wx.EVT_COMBOBOX, self._on_compute_change)
        self._populate_tts_engines()

    def _inject_omnivoice_voices(self):
        """Add pip-installed engines' voices (OmniVoice, Voice Lab).

        OmniVoice voices come from the catalog; the Voice Lab engines (Pocket
        TTS, Bark, F5-TTS) bring their own pre-made voices, which are listed
        from their installed package.
        """
        try:
            from ..voicelab import builtin_voice_entries  # noqa: PLC0415

            self._all_voices.extend(builtin_voice_entries())
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
                                wx.CallAfter(self._on_package_probe, p, e),
                        )
                    continue
                for lang in tts_entry.get("languages", []):
                    for variant in lang.get("variants", []):
                        for voice in variant.get("voices", []):
                            self._all_voices.append(
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

    def _on_compute_change(self, _):
        """Refresh the voice cascade when the compute back-end changes."""
        self._refresh_voices_for_selection()

    # -- background voice discovery -----------------------------------------
    def _on_builtin_voices(self, _voices=None):
        """The Windows voice enumeration finished (worker thread)."""
        try:
            wx.CallAfter(self._rebuild_voice_choices)
        except Exception:  # noqa: BLE001
            pass

    def _on_package_probe(self, package: str, engine: str | None = None):
        """A managed-venv probe finished; re-list the engines when present."""
        try:
            if venv_packages.version(package, engine=engine):
                wx.CallAfter(self._rebuild_voice_choices)
        except Exception:  # noqa: BLE001
            pass

    def _remember_tts_selection(self) -> dict:
        voice = self._selected_voice()
        return {
            "tts": self._selected_tts_id(),
            "language": voice.get("language") if voice else None,
            "variant": voice.get("variant") if voice else None,
            "voice": voice.get("voice") if voice else None,
            "compute": self._selected_compute(),
        }

    def _rebuild_voice_choices(self):
        """Re-run voice discovery without losing the current selection."""
        try:
            saved = self._remember_tts_selection()
            self._populate_voices()
            tts_id = saved.get("tts")
            if tts_id:
                for index in range(self.tts_combo.GetCount()):
                    if self.tts_combo.GetClientData(index) == tts_id:
                        self.tts_combo.SetSelection(index)
                        self._on_tts(None)
                        break
            self._select_project_cascade(saved)
            compute_choice = saved.get("compute")
            if compute_choice:
                for index in range(self.compute_combo.GetCount()):
                    if self.compute_combo.GetClientData(index) == compute_choice:
                        self.compute_combo.SetSelection(index)
                        break
            self._update_omni_ui()
        except Exception:  # noqa: BLE001
            log.debug("Could not rebuild the voice list", exc_info=True)

    def _populate_tts_engines(self):
        """List every downloaded/ready TTS engine (compute-agnostic)."""
        tts_ids = sorted({v["tts"] for v in self._all_voices})
        self.tts_combo.Clear()
        for tts_id in tts_ids:
            tts = catalog.find_tts(tts_id)
            self.tts_combo.Append(tts["name"] if tts else tts_id, tts_id)
        if self.tts_combo.GetCount():
            self.tts_combo.SetSelection(0)
            self._on_tts(None)
        else:
            for combo in (self.compute_combo, self.lang_combo,
                          self.variant_combo, self.voice_combo):
                combo.Clear()
            self.start_btn.Disable()
            self.status.SetLabel(
                "No voices are installed. Download some from "
                "Settings > Download and remove."
            )

    def _compute_options_for_tts(self, tts_id: str | None):
        """Fill the compute combo with the options the selected TTS supports.

        ONNX engines (Piper, Kokoro, Kitten, ...) run on Auto / CPU / GPU
        (ONNX); OmniVoice engines run on CUDA GPU; the Voice Lab engines run
        on the CPU and on the GPU (CPU always, GPU plus Auto when detected).
        """
        self.compute_combo.Clear()
        if tts_id in _VOICE_LAB_ENGINES:
            from ..voicelab import device_options  # noqa: PLC0415

            for value, label in device_options():
                self.compute_combo.Append(label, value)
            # CPU first (the option that works everywhere) unless the user
            # already picked something else for this engine.
            preferred = self.settings.get("clone_engines.device", "cpu")
            for index in range(self.compute_combo.GetCount()):
                if self.compute_combo.GetClientData(index) == preferred:
                    self.compute_combo.SetSelection(index)
                    break
            else:
                self.compute_combo.SetSelection(0)
            return
        if tts_id in _OMNIVOICE_ENGINES:
            self.compute_combo.Append(_COMPUTE_LABELS["cuda_gpu"], "cuda_gpu")
        else:
            for value in compute.available_compute():
                self.compute_combo.Append(_COMPUTE_LABELS.get(value, value), value)
        if self.compute_combo.GetCount():
            self.compute_combo.SetSelection(0)

    def _refresh_voices_for_selection(self):
        """Keep language/variant/voice lists in sync with the selected TTS
        engine and compute back-end."""
        tts_sel = self.tts_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        is_omni = self._is_omnivoice_compute() or (tts_id in _OMNIVOICE_ENGINES)
        self._voices = []
        for v in self._all_voices:
            if v.get("tts") != tts_id:
                continue
            engine = v.get("engine", "vits")
            if is_omni:
                if engine in _OMNIVOICE_ENGINES or v.get("custom_omni"):
                    self._voices.append(v)
            else:
                if engine in _ONNX_TTS_ENGINES or engine in _VOICE_LAB_ENGINES:
                    self._voices.append(v)
        if not self._voices:
            for combo in (self.lang_combo, self.variant_combo, self.voice_combo):
                combo.Clear()
            self.start_btn.Disable()
            self.status.SetLabel(
                "No voices available for this TTS engine with the selected "
                "compute back-end."
            )
            self._update_omni_ui()
            return
        self.start_btn.Enable()
        self._populate_langs(tts_id)
        self._update_omni_ui()

    def _populate_langs(self, tts_id):
        tts = catalog.find_tts(tts_id) if tts_id else None
        if language_choice.is_omnivoice(tts_id):
            # OmniVoice: the Language box is the engine's language hint (Auto
            # plus all 646 languages of the model), not a key into the voice
            # list - every OmniVoice voice can speak every one of them.
            language_choice.fill(self.lang_combo, self._omni_language_hint())
            self._on_lang(None)
            return
        langs = sorted({v["language"] for v in self._voices if v["tts"] == tts_id})
        self.lang_combo.Clear()
        for code in langs:
            label = catalog.language_display_name(tts, code) if tts else code
            self.lang_combo.Append(label, code)
        if self.lang_combo.GetCount():
            self.lang_combo.SetSelection(0)
        self._on_lang(None)

    def _voices_for(self):
        tts_sel = self.tts_combo.GetSelection()
        lang_sel = self.lang_combo.GetSelection()
        variant_sel = self.variant_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        lang = language_choice.code_at(self.lang_combo, lang_sel) if lang_sel >= 0 else None
        variant = language_choice.code_at(self.variant_combo, variant_sel) if variant_sel >= 0 else None
        if language_choice.is_omnivoice(tts_id):
            # The pinned language is a hint to the engine, so it never removes
            # voices from the list.
            lang = None
        return [
            v for v in self._voices
            if v["tts"] == tts_id
            and (v["language"] == lang or not lang)
            and (v["variant"] == variant or not variant)
        ]

    def _on_tts(self, _):
        """TTS engine changed: rebuild the compute options for that engine,
        then refresh the language/variant/voice cascade."""
        sel = self.tts_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(sel) if sel >= 0 else None
        self._compute_options_for_tts(tts_id)
        self._refresh_voices_for_selection()

    def _on_lang(self, _):
        tts_sel = self.tts_combo.GetSelection()
        lang_sel = self.lang_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        tts = catalog.find_tts(tts_id) if tts_id else None
        if language_choice.is_omnivoice(tts_id):
            # The chosen language is stored as this project's OmniVoice hint
            # (Auto clears it); the variants stay complete either way.
            self._store_omni_language(language_choice.hint_of(self.lang_combo))
            variants = sorted({
                v["variant"] for v in self._voices if v["tts"] == tts_id
            })
            self.variant_combo.Clear()
            for vid in variants:
                variant = (catalog.find_variant(tts, language_choice.AUTO_KEY, vid)
                           if tts else None)
                self.variant_combo.Append(variant["name"] if variant else vid, vid)
            if self.variant_combo.GetCount():
                self.variant_combo.SetSelection(0)
            self._on_variant(None)
            return
        lang = language_choice.code_at(self.lang_combo, lang_sel) if lang_sel >= 0 else None
        variants = sorted({
            v["variant"] for v in self._voices
            if v["tts"] == tts_id and v["language"] == lang
        })
        self.variant_combo.Clear()
        for vid in variants:
            label = vid
            if tts:
                variant = catalog.find_variant(tts, lang, vid)
                if variant:
                    label = variant["name"]
            self.variant_combo.Append(label, vid)
        if self.variant_combo.GetCount():
            self.variant_combo.SetSelection(0)
        self._on_variant(None)

    def _on_variant(self, _):
        self.voice_combo.Clear()
        for v in self._voices_for():
            self.voice_combo.Append(v["voice_name"], v)
        if self.voice_combo.GetCount():
            self.voice_combo.SetSelection(0)
        self._update_omni_ui()

    def _on_voice_change(self, _):
        """Another voice was picked: refresh the OmniVoice options/summary
        (a voice-library voice carries its own language pin)."""
        self._update_omni_ui()

    # -- OmniVoice options (per project) ------------------------------------
    def _selected_tts_id(self):
        sel = self.tts_combo.GetSelection()
        return self.tts_combo.GetClientData(sel) if sel >= 0 else None

    def _is_omnivoice_selected(self) -> bool:
        """True when the selected TTS engine is an OmniVoice engine."""
        return self._selected_tts_id() in _OMNIVOICE_ENGINES

    def _omni_settings(self) -> dict:
        return self.data.get("tts", {}).get("omni") or {}

    def _omni_language_hint(self):
        """The OmniVoice language this project will pin (``None`` = auto).

        The project's own choice (Settings-free: it is kept in
        ``project.json``) wins; a voice-library voice carries its own pin,
        which is what the Language box shows while the project has none.
        """
        voice = self._selected_voice()
        stored = None
        if voice and voice.get("custom_omni"):
            stored = (voice.get("omni") or {}).get("language")
        return self._omni_settings().get("language") or stored or None

    def _store_omni_language(self, hint):
        """Keep ``hint`` (a language id, or ``None`` for auto) with the rest
        of this project's OmniVoice options."""
        tts = self.data.setdefault("tts", {})
        omni = tts.get("omni")
        if not isinstance(omni, dict):
            omni = {}
        if hint:
            omni["language"] = hint
        else:
            omni.pop("language", None)
        if omni:
            tts["omni"] = omni

    @staticmethod
    def _omni_summary_text(omni: dict, engine: str | None = None) -> str:
        """Short human summary of the stored per-project OmniVoice options."""
        if not omni:
            return ""
        from ..omnivoice import languages, spec  # noqa: PLC0415
        mode = omni.get("mode", "auto")
        if mode == "clone":
            ref = omni.get("ref_audio") or ""
            who = os.path.basename(ref) if ref else "(no reference set)"
            detail = f"clone of '{who}'"
        elif mode == "design":
            detail = f"design: {omni.get('instruct') or '(auto attributes)'}"
        else:
            detail = "auto voice"
        bits = []
        if omni.get("num_step"):
            bits.append(f"{omni['num_step']} steps")
        if omni.get("guidance_scale") is not None:
            bits.append(f"guidance {omni['guidance_scale']:g}")
        if omni.get("language"):
            # A pinned language is worth showing: it is the fix for short
            # lines that OmniVoice would otherwise detect as a neighbour.
            bits.append(f"language {languages.display_name(omni['language'])}")
        if omni.get("seed") is not None:
            bits.append(f"seed {omni['seed']}")
        suffix = f" - {', '.join(bits)}" if bits else ""
        return f"{detail}{suffix}."

    def _update_omni_ui(self):
        """Show the OmniVoice options button only for OmniVoice engines and
        refresh its summary from the project's stored options.

        Voice-library voices (created in Settings > OmniVoice engines) carry
        their own clone/design identity, so the per-project options button is
        hidden for them and the summary describes the library voice instead.
        """
        voice = self._selected_voice()
        is_library = bool(voice and voice.get("custom_omni"))
        visible = self._is_omnivoice_selected()
        self.omni_btn.Show(visible and not is_library)
        self.omni_summary.Show(visible)
        if visible:
            # Keep the Language box on the language this project will send:
            # the project's choice wins, else the library voice's own pin.
            language_choice.select(
                self.lang_combo,
                self._omni_language_hint() or language_choice.AUTO_KEY,
            )
            if is_library:
                omni = voice.get("omni") or {}
                if omni.get("mode") == "clone":
                    ref = voice.get("ref_audio") or omni.get("ref_audio") or ""
                    text = ("Voice library voice - clone of "
                            f"'{os.path.basename(ref) if ref else '?'}'.")
                else:
                    text = ("Voice library voice - design: "
                            f"{omni.get('instruct') or '(auto attributes)'}")
                text += (" Language: "
                         + language_choice.label(self._omni_language_hint())
                         + ".")
            else:
                omni = self._omni_settings()
                text = self._omni_summary_text(omni)
                text = (text if text
                        else "No OmniVoice options set - auto voice will be used.")
            self.omni_summary.SetLabel(text)
            self.omni_summary.SetName("OmniVoice voice options summary: " + text)
        self._update_engine_options_ui()
        self.Layout()

    def _on_omni_options(self, _):
        """Open the full-feature OmniVoice options dialog for this project."""
        from .omnivoice_options_dialog import OmniVoiceOptionsDialog  # noqa: PLC0415
        tts_id = self._selected_tts_id()
        engine_label = "OmniVoice Server" if tts_id == "omnivoice_server" else "OmniVoice"
        dlg = OmniVoiceOptionsDialog(
            self,
            engine_label=engine_label,
            omni=self._omni_settings(),
            project_name=self.data.get("name", ""),
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            self.data.setdefault("tts", {})["omni"] = dlg.get_omni()
        finally:
            dlg.Destroy()
        self._update_omni_ui()
        self.status.SetLabel(
            "OmniVoice options saved for this project. Press Start recording."
        )

    # -- Voice Lab tuning options (per project, per engine) ------------------
    def _is_voicelab_selected(self) -> bool:
        return self._selected_tts_id() in _VOICE_LAB_ENGINES

    def _engine_options_for_engine(self) -> dict:
        """The tuning overrides this project stores for the selected engine."""
        stored = self.data.get("tts", {}).get("engine_options") or {}
        tts_id = self._selected_tts_id()
        if stored.get("engine") == tts_id:
            values = stored.get("values") or {}
            return values if isinstance(values, dict) else {}
        return {}

    def _engine_options_defaults(self) -> dict:
        """The tuning defaults saved in Settings > Voice Clone for the engine."""
        from ..voicelab import options as tuning  # noqa: PLC0415

        tts_id = self._selected_tts_id()
        if tts_id not in _VOICE_LAB_ENGINES:
            return {}
        saved = self.settings.get(tuning.settings_key(tts_id), {})
        return saved if isinstance(saved, dict) else {}

    def _effective_engine_options(self) -> dict:
        """Project overrides, else the saved defaults, else nothing."""
        from ..voicelab import options as tuning  # noqa: PLC0415

        tts_id = self._selected_tts_id()
        if tts_id not in _VOICE_LAB_ENGINES:
            return {}
        return tuning.resolve(
            tts_id, self._engine_options_for_engine(),
            self._engine_options_defaults(),
        )

    def _update_engine_options_ui(self):
        """Show the tuning button for Voice Lab engines and describe the
        values this project will actually use."""
        visible = self._is_voicelab_selected()
        self.engine_options_btn.Show(visible)
        self.engine_options_summary.Show(visible)
        if not visible:
            return
        from ..voicelab import options as tuning  # noqa: PLC0415

        tts_id = self._selected_tts_id()
        values = self._effective_engine_options()
        text = tuning.describe_overrides(tts_id, values)
        if text:
            source = "" if self._engine_options_for_engine() else " (Settings default)"
            summary = f"Tuning: {text}{source}."
        else:
            summary = "Tuning: engine defaults."
        notes = tuning.problems(tts_id, values, self._selected_compute())
        if notes:
            summary += " " + " ".join(notes)
        self.engine_options_summary.SetLabel(summary)
        self.engine_options_summary.SetName(
            "Voice engine tuning summary: " + summary
        )

    def _on_engine_options(self, _):
        """Open the per-engine tuning dialog for this project."""
        from .voicelab_options_dialog import VoiceLabOptionsDialog  # noqa: PLC0415

        tts_id = self._selected_tts_id()
        if tts_id not in _VOICE_LAB_ENGINES:
            return
        dlg = VoiceLabOptionsDialog(
            self,
            engine_id=tts_id,
            values=self._engine_options_for_engine(),
            project_name=self.data.get("name", ""),
            scope="project",
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            values = dlg.get_options()
        finally:
            dlg.Destroy()
        tts = self.data.setdefault("tts", {})
        if values:
            tts["engine_options"] = {"engine": tts_id, "values": values}
        else:
            tts.pop("engine_options", None)
        self._update_engine_options_ui()
        self.status.SetLabel(
            "Engine tuning saved for this project. Press Start recording."
        )

    def _selected_voice(self):
        sel = self.voice_combo.GetSelection()
        if sel < 0:
            return None
        return self.voice_combo.GetClientData(sel)

    def _apply_project_tts(self):
        tts = self.data.get("tts", {})
        # Select the stored TTS engine first (this rebuilds the compute
        # options for that engine).
        stored_tts = tts.get("tts")
        if stored_tts:
            idx = next((i for i in range(self.tts_combo.GetCount())
                        if self.tts_combo.GetClientData(i) == stored_tts), -1)
            if idx >= 0:
                self.tts_combo.SetSelection(idx)
                self._on_tts(None)
        if tts.get("compute") and self.compute_combo.GetCount():
            # Map stored "cuda" compute to "cuda_gpu" combo value if present
            target = tts["compute"]
            if target == "cuda":
                # Check if cuda_gpu option exists in the combo
                for i in range(self.compute_combo.GetCount()):
                    if self.compute_combo.GetClientData(i) == "cuda_gpu":
                        target = "cuda_gpu"
                        break
            idx = next((i for i in range(self.compute_combo.GetCount())
                        if self.compute_combo.GetClientData(i) == target), 0)
            self.compute_combo.SetSelection(idx)
            self._refresh_voices_for_selection()
        # best-effort preselect the voice the project used
        self._select_project_cascade(tts)
        for name, slider in (("rate", self.rate), ("pitch", self.pitch), ("volume", self.volume)):
            value = float(tts.get(name, 1.0))
            slider.SetValue(int(value * 100))
            slider.value_label.SetLabel(f"{name}: {value:.2f}")  # type: ignore[attr-defined]
            update_accessible_name(slider, f"{name}: {value:.2f}")
        self._update_omni_ui()
        fmt = tts.get("output_format", FORMAT_WAV)
        idx = next((i for i in range(self.format_combo.GetCount())
                    if self.format_combo.GetClientData(i) == fmt), 0)
        self.format_combo.SetSelection(idx)

    def _on_format_changed(self, evt):
        """Warn on every non-WAV format selection in a DAISY project.

        DAISY book players are much more reliable with WAV narration; the
        engines' MP3s cut out in strict players unless the book builder can
        re-encode them.  The warning intentionally has no "don't show
        again" option: it appears every time MP3 or FLAC is picked for a
        DAISY project (any DAISY project type).
        """
        try:
            sel = self.format_combo.GetSelection()
            fmt = self.format_combo.GetClientData(sel) if sel >= 0 else None
        except Exception:  # noqa: BLE001
            fmt = None
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype in _DAISY_PROJECT_TYPES and fmt and fmt != FORMAT_WAV:
            wx.MessageBox(
                "MP3 and FLAC recordings can cut out in DAISY book players. "
                "WAV is the recommended output format for DAISY projects.",
                "DAISY audio format warning",
                style=wx.OK | wx.ICON_WARNING,
            )
        evt.Skip()

    def _select_project_cascade(self, tts):
        """Preselect language / variant / voice stored in the project."""
        if language_choice.is_omnivoice(tts.get("tts")):
            # The pin lives in the project's OmniVoice options; the catalog
            # language of every OmniVoice voice stays "auto".
            language_choice.select(
                self.lang_combo,
                self._omni_language_hint() or language_choice.AUTO_KEY,
            )
            self._on_lang(None)
        elif tts.get("language"):
            for i in range(self.lang_combo.GetCount()):
                if language_choice.code_at(self.lang_combo, i) == tts["language"]:
                    self.lang_combo.SetSelection(i)
                    self._on_lang(None)
                    break
        if tts.get("variant"):
            for i in range(self.variant_combo.GetCount()):
                if language_choice.code_at(self.variant_combo, i) == tts["variant"]:
                    self.variant_combo.SetSelection(i)
                    self._on_variant(None)
                    break
        voice_id = tts.get("voice")
        if voice_id:
            for i in range(self.voice_combo.GetCount()):
                voice = self.voice_combo.GetClientData(i)
                if voice and voice.get("voice") == voice_id:
                    self.voice_combo.SetSelection(i)
                    break

    # ------------------------------------------------------------- actions
    def _current_params(self):
        compute_choice = self._selected_compute()
        # Map "cuda_gpu" back to "cuda" for the engine (OmniVoice uses CUDA)
        engine_compute = "cuda" if compute_choice == "cuda_gpu" else compute_choice
        # Punctuation is chosen when the project is created (New Project
        # wizard) and stored in the project file; the Recording window no
        # longer shows or changes it.
        punct = self.data.get("tts", {}).get("punctuation", "default")
        fmt_sel = self.format_combo.GetSelection()
        fmt = self.format_combo.GetClientData(fmt_sel) if fmt_sel >= 0 else "wav"
        return {
            "compute": engine_compute,
            "rate": self.rate.GetValue() / 100.0,
            "pitch": self.pitch.GetValue() / 100.0,
            "volume": self.volume.GetValue() / 100.0,
            "punctuation": punct,
            "output_format": fmt,
        }

    def _save_tts_to_project(self):
        voice = self._selected_voice()
        params = self._current_params()
        omni = self.data.get("tts", {}).get("omni") or {}
        tts_data = {
            "tts": voice.get("tts") if voice else None,
            "language": voice.get("language") if voice else None,
            "variant": voice.get("variant") if voice else None,
            "voice": voice.get("voice") if voice else None,
            "rate": params["rate"],
            "pitch": params["pitch"],
            "volume": params["volume"],
            "punctuation": params["punctuation"],
            "output_format": params["output_format"],
            "compute": params["compute"],
        }
        if omni:
            tts_data["omni"] = omni
        if self._is_voicelab_selected():
            tuning = self._engine_options_for_engine()
            if tuning:
                tts_data["engine_options"] = {
                    "engine": self._selected_tts_id(),
                    "values": tuning,
                }
        # Merge into the freshest on-disk state instead of saving the stale
        # in-memory copy: the recording worker marks segments done in
        # project.json while the dialog runs, and a wholesale save here would
        # revert them to "pending" (which breaks DAISY builds and resume).
        try:
            fresh = project.load_project(self.project_dir)
        except Exception:  # noqa: BLE001
            fresh = None
        if fresh is not None:
            fresh["tts"] = tts_data
            project.save_project(self.project_dir, fresh)
        else:
            self.data["tts"] = tts_data
            project.save_project(self.project_dir, self.data)
        # Remember the selection so the next new project can seed its
        # per-TTS rate/pitch/volume defaults (Settings > Recording settings).
        if voice:
            try:
                self.settings.set("last_model", {
                    "tts": voice.get("tts"),
                    "language": voice.get("language"),
                    "variant": voice.get("variant"),
                    "voice": voice.get("voice"),
                })
            except Exception:  # noqa: BLE001
                pass

    def _on_start(self, _):
        voice = self._selected_voice()
        if not voice:
            wx.MessageBox("Select a voice first.", "Recording",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        # "One audio file" projects cannot be resumed from the middle of the
        # single file; warn about it once per window so the user does not
        # close it and lose a long recording.
        if self.data.get("audio_mode") == MODE_ONE_FILE and not self._warned_one_file:
            self._warned_one_file = True
            wx.MessageBox(
                "This project records the whole document into one audio "
                "file. Resume is not supported: if you stop or close this "
                "window before the file is finished, the recording starts "
                "again from the beginning. Keep this window open until "
                "recording is complete.",
                "One audio file", style=wx.OK | wx.ICON_INFORMATION,
            )

        # Enrich the voice entry with this project's OmniVoice options
        # (cloning reference, design instructions, language hint, knobs).
        # Voice-library voices (Settings > OmniVoice engines) carry their own
        # identity, so the project's options never override them.
        if voice.get("engine") in _OMNIVOICE_ENGINES:
            from ..omnivoice import spec  # noqa: PLC0415

            if voice.get("custom_omni"):
                # A voice-library voice keeps its own clone/design identity;
                # an explicit project language still overrides its own pin.
                omni = dict(voice.get("omni") or {})
                pin = self._omni_settings().get("language")
                if pin:
                    omni["language"] = pin
                    voice = dict(voice)
                    voice["omni"] = omni
            else:
                voice = spec.apply_omni_to_voice(voice, self._omni_settings())
                omni = voice.get("omni") or {}
            if omni.get("mode") == "clone":
                ref = voice.get("ref_audio") or omni.get("ref_audio") or ""
                if not ref or not os.path.isfile(ref):
                    if voice.get("custom_omni"):
                        wx.MessageBox(
                            "This voice library clone's sample audio is missing. "
                            "Re-create the voice from Settings > OmniVoice engines.",
                            "OmniVoice clone",
                            style=wx.OK | wx.ICON_INFORMATION,
                        )
                    else:
                        wx.MessageBox(
                            "Voice clone mode needs a reference audio sample. "
                            "Choose one in OmniVoice voice options before recording.",
                            "OmniVoice clone",
                            style=wx.OK | wx.ICON_INFORMATION,
                        )
                    return
                self.omni_summary.SetLabel(
                    f"Cloning the voice from: {os.path.basename(ref)}."
                )
        # Voice Lab engines: hand this project's tuning overrides (or the
        # Settings defaults for the engine) to the synthesis worker.
        if voice.get("engine") in _VOICE_LAB_ENGINES:
            from ..voicelab import options as tuning  # noqa: PLC0415

            values = self._effective_engine_options()
            voice = tuning.apply_to_voice(voice, values)
            for note in tuning.problems(voice.get("engine"), values,
                                        self._selected_compute()):
                self.status.SetLabel(note)

        params = self._current_params()
        fmt = params["output_format"]
        ffmpeg_exe = None
        if fmt != FORMAT_WAV:
            ffmpeg_exe = ffmpeg_mod.ensure_ffmpeg()
            if not ffmpeg_exe:
                answer = wx.MessageBox(
                    "FFmpeg is required to create " + fmt.upper() + " files. "
                    "It is not installed yet. Download FFmpeg now? It will be "
                    "saved to your user folder (AppData\\Roaming\\AIVoiceStudio\\ffmpeg).",
                    "FFmpeg required",
                    style=wx.YES_NO | wx.ICON_QUESTION,
                )
                if answer == wx.YES:
                    self.status.SetLabel("Downloading FFmpeg...")
                    self.start_btn.Disable()
                    threading.Thread(target=self._download_ffmpeg_job, daemon=True).start()
                    return
                # No -> fall back to WAV
                fmt = FORMAT_WAV
                params["output_format"] = fmt
                idx = next((i for i in range(self.format_combo.GetCount())
                            if self.format_combo.GetClientData(i) == fmt), 0)
                self.format_combo.SetSelection(idx)

        self._save_tts_to_project()
        if self._start_index is None:
            start_index = project.first_pending_index(self.project_dir)
        else:
            start_index = min(self._start_index, len(self._segments))
        # "Only record selected file" stops after that one segment; otherwise
        # the run goes on to the end of the project as before.
        end_index = start_index + 1 if self._single_segment else None
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._worker = SynthesisWorker(
            segments=self._segments,
            voice_entry=voice,
            params=params,
            output_dir=self.project_dir,
            ffmpeg_exe=ffmpeg_exe,
            start_index=start_index,
            end_index=end_index,
            on_segment_done=lambda idx, title, path: self._segment_done(idx, title, path),
            on_all_done=lambda: wx.PostEvent(self, SynthFinishedEvent()),
            on_error=lambda msg: wx.PostEvent(self, SynthErrorEvent(msg)),
            cancel_event=self._cancel_event,
            pause_event=self._pause_event,
            on_status=lambda msg: wx.PostEvent(self, SynthStatusEvent(msg)),
        )
        self._set_running(True)
        self._show_progress_dialog()
        self._worker.start()
        if self._single_segment:
            self.status.SetLabel(
                f"Recording started (only segment {start_index + 1})."
            )
        else:
            self.status.SetLabel(
                f"Recording started (resuming from segment {start_index + 1})."
            )

    def _download_ffmpeg_job(self):
        try:
            ffmpeg_mod.download_ffmpeg()
            wx.CallAfter(self._ffmpeg_done, None)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._ffmpeg_done, f"FFmpeg download failed: {exc}")

    def _ffmpeg_done(self, error):
        self.start_btn.Enable()
        if error:
            wx.MessageBox(error + " Switching to WAV output.", "FFmpeg",
                          style=wx.OK | wx.ICON_ERROR)
            self.format_combo.SetSelection(0)
            return
        self.status.SetLabel("FFmpeg installed. Press Start recording again.")
        wx.MessageBox("FFmpeg installed. Press Start recording again.",
                      "FFmpeg", style=wx.OK | wx.ICON_INFORMATION)

    # worker callbacks (worker thread) --------------------------------------
    def _segment_done(self, index, title, path):
        project.mark_segment_done(self.project_dir, index + 1, path)
        wx.PostEvent(self, SynthSegmentDoneEvent(index, title, path))

    # recording progress dialog ---------------------------------------------
    def _show_progress_dialog(self):
        """Show the non-modal progress bar dialog for the current run."""
        if self._progress_dlg is not None:
            return
        self._progress_dlg = TaskProgressDialog(
            self,
            title="Recording in progress",
            message="Starting recording...",
            cancel_label="Stop recording",
            on_cancel=lambda: self._on_stop(None),
        )
        self._progress_dlg.Show()
        self._progress_dlg.start_pulse()

    def _finish_progress_dialog(self):
        dlg = self._progress_dlg
        self._progress_dlg = None
        if dlg is not None:
            dlg.finish()

    def _update_progress_dialog(self, value: int | None = None,
                                message: str | None = None):
        dlg = self._progress_dlg
        if dlg is None:
            return
        if value is None:
            dlg.set_message(message)
        elif message is None:
            dlg.show_progress(value)
        else:
            dlg.update(value, message)

    # UI handlers -----------------------------------------------------------
    def _on_synth_status(self, evt: SynthStatusEvent):
        self.status.SetLabel(evt.message)
        if evt.message == "Stopped.":
            # Stop (not pause): the worker has finished; re-enable the
            # controls and close the progress dialog.
            self._finish_progress_dialog()
            self._set_running(False)
        elif self._progress_dlg is not None:
            self._update_progress_dialog(message=evt.message)

    def _on_segment_done(self, evt: SynthSegmentDoneEvent):
        done = evt.index + 1
        total = len(self._segments)
        value = int(done * 100 / total) if total else 0
        self.gauge.SetValue(value)
        self.status.SetLabel(f"Saved segment {done} of {total}: {evt.title}")
        self._update_progress_dialog(
            value=value,
            message=f"Saved segment {done} of {total}.",
        )

    def _on_finished(self, _):
        self._set_running(False)
        self.status.SetLabel("Recording complete.")
        self.gauge.SetValue(100)
        self._finish_progress_dialog()
        # Generate the DAISY/EPUB book structure if this is a DAISY project
        self._build_daisy_if_needed()
        # Generate the M3U8/PLS/WPL playlists for audio-playlist projects.
        self._build_playlists_if_needed()
        # Enable DAISY export if applicable
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype in _DAISY_PROJECT_TYPES:
            self.export_daisy_btn.Enable()
        dialogs.show_recording_complete(
            self,
            "Recording complete. All audio files were saved to the "
            "project folder." + self._daisy_summary()
            + self._playlist_summary(),
            self.project_dir,
        )

    def _playlist_summary(self) -> str:
        """Return a note about generated playlist files (audio projects)."""
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype != "audio_playlist" or not self._playlist_files:
            return ""
        names = ", ".join(os.path.basename(p) for p in self._playlist_files)
        return ("\n\nPlaylists were generated in the project folder "
                f"({names}) - open them in VLC, Windows Media Player, "
                "PotPlayer or any modern player to play the whole "
                "recording in order.")

    def _build_playlists_if_needed(self):
        """Write M3U8/PLS/WPL playlists after recording completes.

        Only for ``audio_playlist`` projects: the playlists reference the
        recorded audio by file name so modern players (VLC, PotPlayer, GOM,
        Windows Media Player, ...) can play the whole narration in order.
        """
        self._playlist_files = []
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype != "audio_playlist":
            return
        try:
            fresh = project.load_project(self.project_dir)
            from ..documents.playlist_builder import build_playlists  # noqa: PLC0415
            self._playlist_files = build_playlists(
                self.project_dir,
                fresh.get("name", "playlist"),
                fresh.get("segments", []),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Playlist generation failed: %s", exc)

    def _daisy_summary(self) -> str:
        """Return a note about DAISY files if they were generated."""
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype not in _DAISY_PROJECT_TYPES:
            return ""
        kind, entry = _daisy_entry_point(self.project_dir, ptype)
        if os.path.isfile(entry):
            return (f"\n\nThe {kind} book was generated in the project folder. "
                    "Use 'Export DAISY as ZIP' to share it.")
        return ("\n\nDAISY book not generated yet - no segments had been "
                "recorded when recording finished.")

    def _build_daisy_if_needed(self):
        """Build the DAISY book structure after recording completes.

        Reads the fresh segment status from disk (the in-memory copy loaded at
        dialog open never sees the 'done' status written by the worker), then
        generates the book:

        * ``daisy_audio`` / ``daisy_audio_text`` -> DAISY 2.02 in ``DAISY/``
        * ``daisy3_audio_text``                  -> DAISY 3 in ``DAISY3/``
        """
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype not in _DAISY_PROJECT_TYPES:
            return
        try:
            fresh = project.load_project(self.project_dir)
            # Per-project DAISY settings (chosen in the wizard) win; fall back
            # to the global Settings > DAISY defaults.
            pd = fresh.get("daisy") or {}
            meta = {
                "title": pd.get("title") or fresh.get("name", "Untitled"),
                "creator": pd.get("creator")
                    or self.settings.get("daisy.creator", ""),
                "date": pd.get("date"),
                "subject": pd.get("subject")
                    or self.settings.get("daisy.subject", ""),
                "narrator": pd.get("narrator")
                    or self.settings.get("daisy.narrator", ""),
                "producer": pd.get("producer")
                    or self.settings.get("daisy.producer", ""),
                "show_software": pd.get(
                    "show_software",
                    self.settings.get("daisy.show_software", True),
                ),
            }
            lang = pd.get("language") or self.settings.get("daisy.language", "en")
            publisher = pd.get("publisher") or self.settings.get("daisy.publisher", "")
            include_text = ptype == "daisy_audio_text" and bool(
                pd.get("include_text", self.settings.get("daisy.include_text", True))
            )
            fmt = fresh.get("tts", {}).get("output_format", "wav")
            common = dict(
                output_dir=self.project_dir,
                project_name=fresh.get("name", "Untitled"),
                segments=fresh.get("segments", []),
                audio_format=fmt,
                language=lang,
                publisher=publisher,
                source_file=fresh.get("source_file", ""),
                meta=meta,
            )
            if ptype == "daisy3_audio_text":
                from ..documents.daisy3_builder import build_daisy3_book  # noqa: PLC0415
                result = build_daisy3_book(**common)
            else:
                from ..documents.daisy_builder import build_daisy_book  # noqa: PLC0415
                result = build_daisy_book(include_text=include_text, **common)
            if result:
                self.export_daisy_btn.Enable()
            else:
                log.warning("DAISY book not built: no completed segments on disk")
        except Exception as exc:  # noqa: BLE001
            log.warning("DAISY book generation failed: %s", exc)

    def _on_error(self, evt: SynthErrorEvent):
        self._set_running(False)
        self._finish_progress_dialog()
        self.status.SetLabel(evt.message)
        dialogs.notify_engine_error(self, "Recording error", evt.message)

    def _on_pause(self, _):
        if self._worker:
            self._worker.pause()
            self.pause_btn.Disable()
            self.resume_btn.Enable()
            self.status.SetLabel("Paused. The current segment will finish, then "
                                 "recording stops until you resume.")
            self._update_progress_dialog(message="Paused. The current segment will "
                                                 "finish, then recording stops.")

    def _on_resume(self, _):
        if self._worker:
            self._worker.resume()
            self.resume_btn.Disable()
            self.pause_btn.Enable()
            self.status.SetLabel("Resumed.")
            self._update_progress_dialog(message="Resumed.")

    def _on_stop(self, _):
        if self._worker:
            self._worker.cancel()
            self.status.SetLabel("Stopping after the current segment...")
            self._update_progress_dialog(
                message="Stopping after the current segment..."
            )
            self.stop_btn.Disable()

    def _set_running(self, running: bool):
        if running:
            self.start_btn.Disable()
        else:
            self.start_btn.Enable()
        self.pause_btn.Enable(running and not self._pause_event.is_set())
        self.resume_btn.Enable(running and self._pause_event.is_set())
        self.stop_btn.Enable(running)

    def _pending_note(self) -> str:
        """'Ready' status line for a run started at a chosen segment."""
        if self._start_index is None or self._start_index >= len(self._segments):
            return ""
        total = len(self._segments)
        title = self._segments[self._start_index].title
        if self._single_segment:
            return (f"Ready: only segment {self._start_index + 1} of {total} "
                    f"({title}) will be recorded.")
        return (f"Ready: segment {self._start_index + 1} of {total} ({title}) "
                "and every file after it will be recorded.")

    def _update_progress(self):
        total = len(self._segments)
        pending = project.first_pending_index(self.project_dir)
        if total:
            self.gauge.SetValue(int(pending * 100 / total))
        self.status.SetLabel(
            f"Project has {total} segments; {pending} already saved. "
            "Press Start recording to continue."
        )

    def _on_export_daisy(self, _):
        """Package the DAISY book into a ZIP file for distribution."""
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype not in _DAISY_PROJECT_TYPES:
            return
        if not _daisy_book_exists(self.project_dir, ptype):
            wx.MessageBox(
                "No DAISY book has been generated yet. Finish recording every "
                "segment first (press Start recording and let it complete), "
                "then export again.",
                "DAISY book missing",
                style=wx.OK | wx.ICON_INFORMATION,
            )
            return

        default_name = sanitize_filename(self.data.get("name", "daisy_book"), 50) + ".zip"
        with wx.FileDialog(
            self,
            "Save DAISY book as ZIP",
            defaultFile=default_name,
            wildcard="ZIP files (*.zip)|*.zip",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            zip_path = dlg.GetPath()

        self.status.SetLabel("Creating DAISY ZIP archive...")
        self.export_daisy_btn.Disable()
        threading.Thread(
            target=self._export_daisy_job,
            args=(zip_path, self.data.get("project_type", "audio_playlist")),
            daemon=True,
        ).start()

    def _export_daisy_job(self, zip_path: str, ptype: str):
        """Background thread to build the ZIP."""
        try:
            if ptype == "daisy3_audio_text":
                from ..documents.daisy3_builder import export_daisy3_zip  # noqa: PLC0415
                result = export_daisy3_zip(self.project_dir, zip_path)
            else:
                from ..documents.daisy_builder import export_daisy_zip  # noqa: PLC0415
                result = export_daisy_zip(self.project_dir, zip_path)
            wx.CallAfter(self._export_daisy_done, None, result)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._export_daisy_done, str(exc), None)

    def _export_daisy_done(self, error: str | None, path: str | None):
        self.export_daisy_btn.Enable()
        if error:
            self.status.SetLabel("Export failed.")
            wx.MessageBox(f"Export failed: {error}", "Export error",
                          style=wx.OK | wx.ICON_ERROR)
        else:
            self.status.SetLabel("DAISY book exported.")
            wx.MessageBox(
                f"DAISY book exported to:\n{path}",
                "Export complete",
                style=wx.OK | wx.ICON_INFORMATION,
            )

    def _on_close(self, _):
        if self._worker and self._worker.is_alive():
            if wx.MessageBox(
                "Recording is still running. Stop it and close this window? "
                "Already-finished segments are saved and the project can be "
                "resumed later.",
                "Close recording",
                style=wx.YES_NO | wx.ICON_QUESTION,
            ) != wx.YES:
                return
            self._worker.cancel()
            self._worker.join(timeout=3)
        self._finish_progress_dialog()
        self._save_tts_to_project()
        self.EndModal(wx.ID_CLOSE)
