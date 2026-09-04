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
    FORMAT_WAV,
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
from ..settings import Settings
from ..tts import catalog
from ..util import sanitize_filename
from ..tts.models import ModelStore
from .a11y import add_labeled
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

log = logging.getLogger(__name__)

_COMPUTE_LABELS = {
    compute.COMPUTE_AUTO: "Auto (best available)",
    compute.COMPUTE_CPU: "CPU (ONNX)",
    compute.COMPUTE_CUDA: "GPU (ONNX)",
    compute.COMPUTE_DML: "NPU (DirectML)",
    "cuda_gpu": "CUDA GPU (OmniVoice)",
}

# TTS engines available for each compute mode.
# ONNX engines (Piper, Kokoro, Kitten, etc.) work with CPU/GPU compute.
# OmniVoice engines (Server, Triton, Hybrid) require CUDA GPU.
_ONNX_TTS_ENGINES = {"piper", "kokoro", "kitten", "matcha", "pocket"}
_OMNIVOICE_ENGINES = {"omnivoice", "omnivoice_server"}


class RecordingDialog(wx.Dialog):
    def __init__(self, parent, project_dir: str, settings: Settings, store: ModelStore):
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

        self._load_segments()
        self._build_ui()
        self._populate_voices()
        self._apply_project_tts()
        self._update_progress()
        # Announce the project name and focus the text preview.
        self.SetName(f"Recording: {self.data.get('name', '')}")
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
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.compute_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Compute back-end")
        for value in compute.available_compute():
            self.compute_combo.Append(_COMPUTE_LABELS.get(value, value), value)
        # Add CUDA GPU (OmniVoice) option if NVIDIA GPU is detected
        self._add_cuda_gpu_option()
        if self.compute_combo.GetCount():
            self.compute_combo.SetSelection(0)
        self.tts_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                     name="Select TTS engine")
        self.lang_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                      name="Select language")
        self.variant_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                         name="Select variant")
        self.voice_combo = wx.ComboBox(self, style=wx.CB_READONLY,
                                       name="Select voice")
        add_labeled(self, grid, "Compute back-end", self.compute_combo, flag=wx.LEFT | wx.RIGHT, border=2)
        add_labeled(self, grid, "TTS engine", self.tts_combo, flag=wx.LEFT | wx.RIGHT, border=2)
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
        self.format_combo.SetToolTip("Choose the audio file format: WAV is always available, MP3 and FLAC need FFmpeg")
        add_labeled(self, params, "Output format", self.format_combo, flag=wx.LEFT | wx.RIGHT, border=2)
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
        self.export_daisy_btn.Show(ptype in ("daisy_audio", "daisy_audio_text"))
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
        self.omni_btn.Bind(wx.EVT_BUTTON, self._on_omni_options)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(EVT_SYNTH_STATUS, self._on_synth_status)
        self.Bind(EVT_SYNTH_SEGMENT_DONE, self._on_segment_done)
        self.Bind(EVT_SYNTH_FINISHED, self._on_finished)
        self.Bind(EVT_SYNTH_ERROR, self._on_error)

    def _add_cuda_gpu_option(self):
        """Add CUDA GPU (OmniVoice) option if NVIDIA GPU is available."""
        try:
            import subprocess as _sp  # noqa: PLC0415
            result = _sp.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.compute_combo.Append("CUDA GPU (OmniVoice)", "cuda_gpu")
        except Exception:  # noqa: BLE001
            pass

    def _selected_compute(self) -> str:
        """Return the selected compute mode key (resolves auto)."""
        sel = self.compute_combo.GetSelection()
        raw = self.compute_combo.GetClientData(sel) if sel >= 0 else "auto"
        if raw == compute.COMPUTE_AUTO:
            return compute.resolve_compute(raw)
        return raw

    def _is_omnivoice_compute(self) -> bool:
        """True when CUDA GPU (OmniVoice) compute mode is selected."""
        return self._selected_compute() == "cuda_gpu"

    def _slider(self, grid, name: str, lo: float, hi: float) -> wx.Slider:
        label = wx.StaticText(self, label=name + ":")
        label.SetName(name + " label")
        slider = wx.Slider(self, minValue=int(lo * 100), maxValue=int(hi * 100),
                           value=int(1.0 * 100))
        slider.SetName(f"{name}: 1.00")
        value_label = wx.StaticText(self, label="1.00")
        value_label.SetName(name + " value")
        slider.value_label = value_label  # type: ignore[attr-defined]
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(slider, 1, wx.EXPAND)
        box.Add(value_label, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6)
        grid.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2)
        grid.Add(box, 1, wx.EXPAND)
        slider.Bind(
            wx.EVT_SLIDER,
            lambda evt, s=slider, vl=value_label, n=name: (
                vl.SetLabel(f"{s.GetValue() / 100.0:.2f}"),
                s.SetName(f"{n}: {s.GetValue() / 100.0:.2f}"),
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
        for voice in self.store.custom_voices():
            self._all_voices.append(
                {
                    "tts": voice["tts"],
                    "tts_name": voice["tts"],
                    "language": "custom",
                    "variant": "custom",
                    "voice": voice["name"],
                    "voice_name": f"Cloned voice: {voice['name']}",
                    "sid": 0,
                    "engine": voice.get("engine", "vits"),
                    "dir": voice["dir"],
                    "custom": True,
                    "sample": voice.get("sample", ""),
                    "reference": voice.get("reference", ""),
                    "xtts_lang": voice.get("language", "en"),
                    "ref_text": voice.get("ref_text", ""),
                    "model_dir": voice.get("model_dir", ""),
                    "instruct": voice.get("instruct", ""),
                }
            )
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
        # Bind compute combo change to refresh TTS list
        self.compute_combo.Bind(wx.EVT_COMBOBOX, self._on_compute_change)
        self._refresh_tts_for_compute()

    def _inject_omnivoice_voices(self):
        """Add OmniVoice voices from catalog if pip package is installed."""
        try:
            from ..python_runtime import get_runtime  # noqa: PLC0415
            rt = get_runtime()
            if not rt.is_created:
                return
            for tts_entry in catalog.get_tts_list():
                pkg = tts_entry.get("requires_package")
                if not pkg:
                    continue
                result = rt.run_in_env(
                    f"import importlib.metadata; "
                    f"print(importlib.metadata.version('{pkg}'))"
                )
                if result.returncode != 0 or not result.stdout.strip():
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
        """Refresh TTS engine list when compute mode changes."""
        self._refresh_tts_for_compute()

    def _refresh_tts_for_compute(self):
        """Show only TTS engines compatible with the selected compute mode."""
        is_omni = self._is_omnivoice_compute()
        self._voices = []
        for v in self._all_voices:
            engine = v.get("engine", "vits")
            if is_omni:
                # CUDA GPU mode: show OmniVoice engines only
                if engine in _OMNIVOICE_ENGINES:
                    self._voices.append(v)
            else:
                # CPU/GPU ONNX mode: show ONNX engines only
                if engine in _ONNX_TTS_ENGINES:
                    self._voices.append(v)
        tts_ids = sorted({v["tts"] for v in self._voices})
        self.tts_combo.Clear()
        for tts_id in tts_ids:
            tts = catalog.find_tts(tts_id)
            self.tts_combo.Append(tts["name"] if tts else tts_id, tts_id)
        if self.tts_combo.GetCount():
            self.tts_combo.SetSelection(0)
            self._on_tts(None)
        else:
            self.start_btn.Disable()
            self.status.SetLabel(
                "No voices available for this compute mode. "
                + ("Install OmniVoice from the Compute tab." if is_omni
                   else "Download voices from Settings > Download and remove.")
            )
            for combo in (self.lang_combo, self.variant_combo, self.voice_combo):
                combo.Clear()
        self._update_omni_ui()

    def _voices_for(self):
        tts_sel = self.tts_combo.GetSelection()
        lang_sel = self.lang_combo.GetSelection()
        variant_sel = self.variant_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        lang = self.lang_combo.GetClientData(lang_sel) if lang_sel >= 0 else None
        variant = self.variant_combo.GetClientData(variant_sel) if variant_sel >= 0 else None
        return [
            v for v in self._voices
            if v["tts"] == tts_id
            and (v["language"] == lang or not lang)
            and (v["variant"] == variant or not variant)
        ]

    def _on_tts(self, _):
        sel = self.tts_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(sel) if sel >= 0 else None
        tts = catalog.find_tts(tts_id) if tts_id else None
        langs = sorted({v["language"] for v in self._voices if v["tts"] == tts_id})
        self.lang_combo.Clear()
        for code in langs:
            label = catalog.language_display_name(tts, code) if tts else code
            self.lang_combo.Append(label, code)
        if self.lang_combo.GetCount():
            self.lang_combo.SetSelection(0)
        self._on_lang(None)

    def _on_lang(self, _):
        tts_sel = self.tts_combo.GetSelection()
        lang_sel = self.lang_combo.GetSelection()
        tts_id = self.tts_combo.GetClientData(tts_sel) if tts_sel >= 0 else None
        lang = self.lang_combo.GetClientData(lang_sel) if lang_sel >= 0 else None
        tts = catalog.find_tts(tts_id) if tts_id else None
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

    # -- OmniVoice options (per project) ------------------------------------
    def _selected_tts_id(self):
        sel = self.tts_combo.GetSelection()
        return self.tts_combo.GetClientData(sel) if sel >= 0 else None

    def _is_omnivoice_selected(self) -> bool:
        """True when the selected TTS engine is an OmniVoice engine."""
        return self._selected_tts_id() in _OMNIVOICE_ENGINES

    def _omni_settings(self) -> dict:
        return self.data.get("tts", {}).get("omni") or {}

    @staticmethod
    def _omni_summary_text(omni: dict, engine: str | None = None) -> str:
        """Short human summary of the stored per-project OmniVoice options."""
        if not omni:
            return ""
        from ..omnivoice import spec  # noqa: PLC0415
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
            if is_library:
                omni = voice.get("omni") or {}
                if omni.get("mode") == "clone":
                    ref = voice.get("ref_audio") or omni.get("ref_audio") or ""
                    text = ("Voice library voice - clone of "
                            f"'{os.path.basename(ref) if ref else '?'}'.")
                else:
                    text = ("Voice library voice - design: "
                            f"{omni.get('instruct') or '(auto attributes)'}")
            else:
                omni = self._omni_settings()
                text = self._omni_summary_text(omni)
                text = (text if text
                        else "No OmniVoice options set - auto voice will be used.")
            self.omni_summary.SetLabel(text)
            self.omni_summary.SetName("OmniVoice voice options summary: " + text)
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

    def _selected_voice(self):
        sel = self.voice_combo.GetSelection()
        if sel < 0:
            return None
        return self.voice_combo.GetClientData(sel)

    def _apply_project_tts(self):
        tts = self.data.get("tts", {})
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
        # best-effort preselect the voice the project used
        self._select_project_voice(tts)
        for name, slider in (("rate", self.rate), ("pitch", self.pitch), ("volume", self.volume)):
            value = float(tts.get(name, 1.0))
            slider.SetValue(int(value * 100))
            slider.value_label.SetLabel(f"{value:.2f}")  # type: ignore[attr-defined]
        self._update_omni_ui()
        fmt = tts.get("output_format", FORMAT_WAV)
        idx = next((i for i in range(self.format_combo.GetCount())
                    if self.format_combo.GetClientData(i) == fmt), 0)
        self.format_combo.SetSelection(idx)

    def _select_project_voice(self, tts):
        voice_id = tts.get("voice")
        if not voice_id:
            return
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
        self.data["tts"] = tts_data
        project.save_project(self.project_dir, self.data)

    def _on_start(self, _):
        voice = self._selected_voice()
        if not voice:
            wx.MessageBox("Select a voice first.", "Recording",
                          style=wx.OK | wx.ICON_INFORMATION)
            return
        # Enrich the voice entry with this project's OmniVoice options
        # (cloning reference, design instructions, language hint, knobs).
        # Voice-library voices (Settings > OmniVoice engines) carry their own
        # identity, so the project's options never override them.
        if voice.get("engine") in _OMNIVOICE_ENGINES:
            from ..omnivoice import spec  # noqa: PLC0415

            if voice.get("custom_omni"):
                omni = voice.get("omni") or {}
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
        start_index = project.first_pending_index(self.project_dir)
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._worker = SynthesisWorker(
            segments=self._segments,
            voice_entry=voice,
            params=params,
            output_dir=self.project_dir,
            ffmpeg_exe=ffmpeg_exe,
            start_index=start_index,
            on_segment_done=lambda idx, title, path: self._segment_done(idx, title, path),
            on_all_done=lambda: wx.PostEvent(self, SynthFinishedEvent()),
            on_error=lambda msg: wx.PostEvent(self, SynthErrorEvent(msg)),
            cancel_event=self._cancel_event,
            pause_event=self._pause_event,
            on_status=lambda msg: wx.PostEvent(self, SynthStatusEvent(msg)),
        )
        self._set_running(True)
        self._worker.start()
        self.status.SetLabel(f"Recording started (resuming from segment {start_index + 1}).")

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

    # UI handlers -----------------------------------------------------------
    def _on_synth_status(self, evt: SynthStatusEvent):
        self.status.SetLabel(evt.message)

    def _on_segment_done(self, evt: SynthSegmentDoneEvent):
        done = evt.index + 1
        total = len(self._segments)
        self.gauge.SetValue(int(done * 100 / total) if total else 0)
        self.status.SetLabel(f"Saved segment {done} of {total}: {evt.title}")

    def _on_finished(self, _):
        self._set_running(False)
        self.status.SetLabel("Recording complete.")
        self.gauge.SetValue(100)
        # Generate DAISY book structure if this is a DAISY project
        self._build_daisy_if_needed()
        # Enable DAISY export if applicable
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype in ("daisy_audio", "daisy_audio_text"):
            self.export_daisy_btn.Enable()
        wx.MessageBox("Recording complete. All audio files were saved to the "
                      "project folder." + self._daisy_summary(),
                      "Recording complete",
                      style=wx.OK | wx.ICON_INFORMATION)

    def _daisy_summary(self) -> str:
        """Return a note about DAISY files if they were generated."""
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype in ("daisy_audio", "daisy_audio_text"):
            return ("\n\nDAISY 2.02 files (ncc.html, SMIL, package.opf) were "
                    "generated in the project folder.")
        return ""

    def _build_daisy_if_needed(self):
        """Build DAISY 2.02 book structure after recording completes."""
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype not in ("daisy_audio", "daisy_audio_text"):
            return
        from ..documents.daisy_builder import build_daisy_book  # noqa: PLC0415
        try:
            lang = self.settings.get("daisy.language", "en")
            publisher = self.settings.get("daisy.publisher", "")
            include_text = ptype == "daisy_audio_text" and self.settings.get("daisy.include_text", True)
            fmt = self.data.get("tts", {}).get("output_format", "wav")
            build_daisy_book(
                output_dir=self.project_dir,
                project_name=self.data.get("name", "Untitled"),
                segments=self.data.get("segments", []),
                audio_format=fmt,
                include_text=include_text,
                language=lang,
                publisher=publisher,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("DAISY book generation failed: %s", exc)

    def _on_error(self, evt: SynthErrorEvent):
        self._set_running(False)
        self.status.SetLabel(evt.message)
        wx.MessageBox(evt.message, "Recording error", style=wx.OK | wx.ICON_ERROR)

    def _on_pause(self, _):
        if self._worker:
            self._worker.pause()
            self.pause_btn.Disable()
            self.resume_btn.Enable()
            self.status.SetLabel("Paused. The current segment will finish, then "
                                 "recording stops until you resume.")

    def _on_resume(self, _):
        if self._worker:
            self._worker.resume()
            self.resume_btn.Disable()
            self.pause_btn.Enable()
            self.status.SetLabel("Resumed.")

    def _on_stop(self, _):
        if self._worker:
            self._worker.cancel()
            self.status.SetLabel("Stopping after the current segment...")
            self.stop_btn.Disable()

    def _set_running(self, running: bool):
        if running:
            self.start_btn.Disable()
        else:
            self.start_btn.Enable()
        self.pause_btn.Enable(running and not self._pause_event.is_set())
        self.resume_btn.Enable(running and self._pause_event.is_set())
        self.stop_btn.Enable(running)

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
        from ..documents.daisy_builder import export_daisy_zip  # noqa: PLC0415
        ptype = self.data.get("project_type", "audio_playlist")
        if ptype not in ("daisy_audio", "daisy_audio_text"):
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
            target=self._export_daisy_job, args=(zip_path,), daemon=True
        ).start()

    def _export_daisy_job(self, zip_path: str):
        """Background thread to build the ZIP."""
        from ..documents.daisy_builder import export_daisy_zip  # noqa: PLC0415
        try:
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
        self._save_tts_to_project()
        self.EndModal(wx.ID_CLOSE)
