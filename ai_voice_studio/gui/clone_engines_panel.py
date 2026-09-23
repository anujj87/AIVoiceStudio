"""Settings -> Voice Clone category.

The **Voice Lab** engines (Pocket TTS by Kyutai, Suno Bark and F5-TTS) live
here.  They are all *voice cloning* engines that run on a normal CPU; when an
NVIDIA GPU is detected the GPU is offered as an extra choice next to the CPU -
never instead of it.

The panel does four things, in reading order:

1. pick the engine and install / remove its package,
2. pick the device (CPU, plus GPU when one is detected),
3. choose a voice: one of the engine's built-in voices, or clone a voice from
   a short recording of your own,
4. preview any voice that is ready to use.

Cloned voices are stored in the model store (``kind=clone_reference``) and are
listed by every other part of the app, so a voice created here can be used in
the Recording window straight away.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import wx

from .. import venv_packages, voicelab
from ..settings import Settings
from ..tts.models import ModelStore
from ..voicelab import engines as voice_lab
from ..voicelab import options as voice_options
from .a11y import add_labeled, finalize_accessibility
from . import access_keys, dialogs
from .voicelab_options_dialog import tune_options as voice_options_tune

log = logging.getLogger(__name__)

#: Engines whose built-in voice inventory can be probed cheaply (the worker
#: answers without loading a model).
_VOICE_PROBE_ENGINES = {"f5tts"}


class VoiceClonePanel(access_keys.AccessKeyHints, wx.Panel):
    """Settings category: clone a voice with a CPU-first engine."""

    title = "Voice Clone"
    description = (
        "Voice cloning engines that run on your CPU and use the GPU when one "
        "is detected."
    )

    def __init__(self, parent, settings: Settings, store: ModelStore):
        super().__init__(parent)
        self.settings = settings
        self.store = store
        self._preview_sound = None
        self._voices: List[Dict[str, Any]] = []

        sizer = wx.BoxSizer(wx.VERTICAL)
        intro = wx.StaticText(
            self,
            label=(
                "These engines clone a voice from a short recording of it. "
                "They run on your CPU; when an NVIDIA GPU is detected the GPU "
                "can be used as well."
            ),
        )
        intro.Wrap(660)
        sizer.Add(intro, 0, wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.SetSizer(sizer)

        self._build_engine_box(sizer)
        self._build_device_box(sizer)
        self._build_voice_box(sizer)
        self._build_library_box(sizer)

        self._select_saved_engine()
        self._on_engine(None)
        self._on_source(None)
        finalize_accessibility(self)

    # ------------------------------------------------------------------ UI
    def _build_engine_box(self, sizer):
        box = wx.StaticBox(self, label="1. Engine")
        inner = wx.StaticBoxSizer(box, wx.VERTICAL)

        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.engine_combo = wx.ComboBox(box, style=wx.CB_READONLY,
                                        name="TTS engine")
        add_labeled(box, grid, "TTS engine", self.engine_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        inner.Add(grid, 0, wx.EXPAND | wx.ALL, 4)

        self.engine_note = wx.StaticText(box, label="")
        self.engine_note.Wrap(640)
        inner.Add(self.engine_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.install_btn = wx.Button(box, label="Install engine...")
        self.install_btn.SetName("Install selected engine")
        self.install_btn.SetToolTip(
            "Install the Python packages this engine needs into the "
            "application's managed environment."
        )
        self.remove_btn = wx.Button(box, label="Remove engine")
        self.remove_btn.SetName("Remove selected engine")
        self.tuning_btn = wx.Button(box, label="Engine tuning...")
        self.tuning_btn.SetName("Engine tuning defaults")
        self.tuning_btn.SetToolTip(
            "Diffusion steps, sampling temperatures, quantization and the "
            "other generation settings of the selected engine. Saved as the "
            "default for new projects; each project can override them in the "
            "Recording window."
        )
        row.Add(self.install_btn, 0, wx.ALL, 2)
        row.Add(self.remove_btn, 0, wx.ALL, 2)
        row.Add(self.tuning_btn, 0, wx.ALL, 2)
        inner.Add(row, 0, wx.LEFT, 2)

        self.engine_status = wx.StaticText(box, label="")
        self.engine_status.SetName("Engine install status")
        inner.Add(self.engine_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        self.tuning_note = wx.StaticText(box, label="")
        self.tuning_note.Wrap(640)
        self.tuning_note.SetName("Engine tuning summary")
        inner.Add(self.tuning_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        sizer.Add(inner, 0, wx.EXPAND | wx.ALL, 6)

        self.engine_combo.Bind(wx.EVT_COMBOBOX, self._on_engine)
        self.install_btn.Bind(wx.EVT_BUTTON, self._on_install)
        self.remove_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.tuning_btn.Bind(wx.EVT_BUTTON, self._on_tuning)

    def _build_device_box(self, sizer):
        box = wx.StaticBox(self, label="2. Device")
        inner = wx.StaticBoxSizer(box, wx.VERTICAL)
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        grid.AddGrowableCol(1)
        self.device_combo = wx.ComboBox(box, style=wx.CB_READONLY,
                                        name="Device")
        for value, label in voicelab.device_options():
            self.device_combo.Append(label, value)
        saved = self.settings.get(voicelab.DEVICE_SETTING, voicelab.DEVICE_CPU)
        options = [value for value, _label in voicelab.device_options()]
        index = options.index(saved) if saved in options else 0
        self.device_combo.SetSelection(index)
        add_labeled(box, grid, "Device", self.device_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        inner.Add(grid, 0, wx.EXPAND | wx.ALL, 4)

        self.device_note = wx.StaticText(box, label="")
        self.device_note.Wrap(640)
        inner.Add(self.device_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        sizer.Add(inner, 0, wx.EXPAND | wx.ALL, 6)

        self.device_combo.Bind(wx.EVT_COMBOBOX, self._on_device)

    def _build_voice_box(self, sizer):
        box = wx.StaticBox(self, label="3. Voice")
        inner = wx.StaticBoxSizer(box, wx.VERTICAL)

        source_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        source_grid.AddGrowableCol(1)
        self.source_combo = wx.ComboBox(box, style=wx.CB_READONLY,
                                        name="Voice source")
        self.source_combo.Append("Use a built-in voice", "builtin")
        self.source_combo.Append("Clone a voice from a sample", "clone")
        self.source_combo.SetSelection(0)
        add_labeled(box, source_grid, "Voice source", self.source_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        inner.Add(source_grid, 0, wx.EXPAND | wx.ALL, 4)

        # -- built-in voice -------------------------------------------------
        self.builtin_panel = wx.Panel(box)
        b_sizer = wx.BoxSizer(wx.VERTICAL)
        b_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        b_grid.AddGrowableCol(1)
        self.builtin_combo = wx.ComboBox(self.builtin_panel, style=wx.CB_READONLY,
                                         name="Built-in voice")
        add_labeled(self.builtin_panel, b_grid, "Built-in voice", self.builtin_combo,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        b_sizer.Add(b_grid, 0, wx.EXPAND | wx.ALL, 2)
        self.builtin_count = wx.StaticText(self.builtin_panel, label="")
        self.builtin_count.SetName("Built-in voice count")
        b_sizer.Add(self.builtin_count, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)
        self.builtin_panel.SetSizer(b_sizer)
        inner.Add(self.builtin_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)

        # -- clone from a sample -------------------------------------------
        self.clone_panel = wx.Panel(box)
        c_sizer = wx.BoxSizer(wx.VERTICAL)

        sample_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        sample_grid.AddGrowableCol(1)
        self.sample_ctrl = wx.TextCtrl(self.clone_panel)
        self.sample_ctrl.SetName("Reference recording")
        browse_btn = wx.Button(self.clone_panel, label="Browse...")
        browse_btn.SetName("Browse for a reference recording")
        sample_row = wx.BoxSizer(wx.HORIZONTAL)
        sample_row.Add(self.sample_ctrl, 1, wx.EXPAND)
        sample_row.Add(browse_btn, 0, wx.LEFT, 4)
        sample_grid.Add(
            wx.StaticText(self.clone_panel, label="Reference recording:"),
            0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 2,
        )
        sample_grid.Add(sample_row, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        c_sizer.Add(sample_grid, 0, wx.EXPAND | wx.ALL, 2)
        sample_hint = wx.StaticText(
            self.clone_panel,
            label=(
                "5-15 seconds of clean speech (WAV/MP3/FLAC/OGG). Bark clones "
                "from a speaker-embedding file (.npz) instead."
            ),
        )
        sample_hint.Wrap(600)
        c_sizer.Add(sample_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)

        name_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        name_grid.AddGrowableCol(1)
        self.name_ctrl = wx.TextCtrl(self.clone_panel)
        self.name_ctrl.SetName("Voice name")
        add_labeled(self.clone_panel, name_grid, "Voice name", self.name_ctrl,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        c_sizer.Add(name_grid, 0, wx.EXPAND | wx.ALL, 2)

        text_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        text_grid.AddGrowableCol(1)
        self.ref_text_ctrl = wx.TextCtrl(self.clone_panel)
        self.ref_text_ctrl.SetName("Transcript of the recording (optional)")
        add_labeled(
            self.clone_panel, text_grid,
            "Transcript of the recording (optional)", self.ref_text_ctrl,
            flag=wx.LEFT | wx.RIGHT, border=2,
        )
        c_sizer.Add(text_grid, 0, wx.EXPAND | wx.ALL, 2)
        text_hint = wx.StaticText(
            self.clone_panel,
            label=(
                "Typing what the recording says improves the clone; leave it "
                "empty and the engine works it out itself."
            ),
        )
        text_hint.Wrap(600)
        c_sizer.Add(text_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)

        self.create_btn = wx.Button(self.clone_panel, label="Create cloned voice")
        self.create_btn.SetName("Create cloned voice")
        c_sizer.Add(self.create_btn, 0, wx.ALL, 2)
        self.clone_panel.SetSizer(c_sizer)
        inner.Add(self.clone_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)

        sizer.Add(inner, 0, wx.EXPAND | wx.ALL, 6)

        self.source_combo.Bind(wx.EVT_COMBOBOX, self._on_source)
        self.builtin_combo.Bind(wx.EVT_COMBOBOX, self._on_builtin_select)
        browse_btn.Bind(wx.EVT_BUTTON, self._on_browse)
        self.create_btn.Bind(wx.EVT_BUTTON, self._on_create)

    def _build_library_box(self, sizer):
        box = wx.StaticBox(self, label="4. Voices ready to use")
        inner = wx.StaticBoxSizer(box, wx.VERTICAL)

        self.voices_list = wx.ListBox(box, size=(-1, 130),
                                      name="Voices ready to use")
        inner.Add(self.voices_list, 1, wx.EXPAND | wx.ALL, 4)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.delete_btn = wx.Button(box, label="Delete cloned voice")
        self.delete_btn.SetName("Delete cloned voice")
        btns.Add(self.delete_btn, 0, wx.ALL, 2)
        inner.Add(btns, 0, wx.LEFT, 2)

        preview_grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        preview_grid.AddGrowableCol(1)
        self.preview_text = wx.TextCtrl(
            box,
            value="Welcome to AI Voice Studio. This is a preview of the "
                  "selected voice.",
        )
        self.preview_text.SetName("Preview text")
        add_labeled(box, preview_grid, "Preview text", self.preview_text,
                    flag=wx.LEFT | wx.RIGHT, border=2)
        inner.Add(preview_grid, 0, wx.EXPAND | wx.ALL, 2)

        self.preview_btn = wx.Button(box, label="&Preview selected voice")
        self.preview_btn.SetName("Preview selected voice")
        inner.Add(self.preview_btn, 0, wx.ALL, 2)
        self.preview_status = wx.StaticText(box, label="")
        self.preview_status.SetName("Preview status")
        inner.Add(self.preview_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)

        sizer.Add(inner, 0, wx.EXPAND | wx.ALL, 6)

        self.voices_list.Bind(wx.EVT_LISTBOX, self._on_voice_select)
        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
        self.delete_btn.Bind(wx.EVT_BUTTON, self._on_delete)

    # ------------------------------------------------------- lifecycle
    def on_activated(self):
        """Refresh availability when the category is opened."""
        self.Show()
        self.Layout()
        self._refresh_engine_status()
        self._refresh_voices()

    def on_deactivated(self):
        self.Hide()

    def apply_to_settings(self):
        self.settings.set(voicelab.DEVICE_SETTING, self._selected_device())
        self.settings.set("clone_engines.engine", self._selected_engine_id())

    def isValid(self) -> bool:
        return True

    # ------------------------------------------------------- selection
    def _select_saved_engine(self):
        saved = self.settings.get("clone_engines.engine", "")
        for index, info in enumerate(voice_lab.ENGINES):
            self.engine_combo.Append(info["name"], info["id"])
            if info["id"] == saved:
                self.engine_combo.SetSelection(index)
        if self.engine_combo.GetCount() and self.engine_combo.GetSelection() < 0:
            self.engine_combo.SetSelection(0)

    def _selected_engine_id(self) -> str:
        index = self.engine_combo.GetSelection()
        if index < 0:
            return voice_lab.engine_ids()[0]
        return self.engine_combo.GetClientData(index) or voice_lab.engine_ids()[0]

    def select_engine(self, engine_id: str) -> bool:
        """Select an engine by id; returns False when it is not in the list."""
        for index in range(self.engine_combo.GetCount()):
            if self.engine_combo.GetClientData(index) == engine_id:
                self.engine_combo.SetSelection(index)
                self._on_engine(None)
                return True
        return False

    def _selected_device(self) -> str:
        index = self.device_combo.GetSelection()
        if index < 0:
            return voicelab.DEVICE_CPU
        return self.device_combo.GetClientData(index) or voicelab.DEVICE_CPU

    def _selected_builtin_voice(self) -> Optional[Dict[str, Any]]:
        index = self.builtin_combo.GetSelection()
        if index < 0:
            return None
        return self.builtin_combo.GetClientData(index)

    def _on_builtin_select(self, _evt):
        """Picking a built-in voice selects its row in the list below."""
        voice = self._selected_builtin_voice()
        if not voice:
            return
        for index, entry in enumerate(self._voices):
            if (entry.get("voice") == voice.get("voice")
                    and entry.get("variant") == voice.get("variant")):
                self.voices_list.SetSelection(index)
                self._update_voice_buttons()
                self.preview_status.SetLabel(
                    f"Selected: {voice.get('voice_name', voice.get('voice', ''))}."
                )
                return

    def _on_engine(self, _evt):
        engine_id = self._selected_engine_id()
        info = voice_lab.engine(engine_id) or {}
        self.engine_note.SetLabel(
            f"{info.get('note', '')}  (about {info.get('size_hint', '?')} "
            f"on disk; {info.get('license', '')})"
        )
        self.engine_note.Wrap(640)
        self._refresh_engine_status()
        self._refresh_tuning_note()
        self._refresh_builtin_voices()
        self._refresh_voices()
        self.Layout()

    # -- engine tuning defaults --------------------------------------------
    def _saved_tuning(self, engine_id: Optional[str] = None) -> dict:
        """The engine's saved default tuning overrides (Settings scope)."""
        engine_id = engine_id or self._selected_engine_id()
        saved = self.settings.get(voice_options.settings_key(engine_id), {})
        if not isinstance(saved, dict):
            return {}
        try:
            return voice_options.clean(engine_id, saved)
        except ValueError:
            return {}

    def _refresh_tuning_note(self):
        engine_id = self._selected_engine_id()
        values = self._saved_tuning(engine_id)
        text = voice_options.describe_overrides(engine_id, values)
        if text:
            note = (f"Tuning defaults: {text}. Each project can change them; "
                    "a project without its own tuning uses these.")
        else:
            note = ("Tuning defaults: engine defaults. Choose 'Engine "
                    "tuning...' to change them for every new project.")
        notes = voice_options.problems(engine_id, values, self._selected_device())
        if notes:
            note += " " + " ".join(notes)
        self.tuning_note.SetLabel(note)
        self.tuning_note.Wrap(640)
        self.tuning_note.SetName("Engine tuning summary: " + note)

    def _on_tuning(self, _evt):
        """Edit the tuning defaults of the selected engine."""
        engine_id = self._selected_engine_id()
        saved = voice_options_tune(
            self, engine_id, self._saved_tuning(engine_id), scope="default"
        )
        if saved is None:
            return
        self.settings.set(voice_options.settings_key(engine_id), saved)
        self._refresh_tuning_note()
        text = voice_options.describe_overrides(engine_id, saved)
        self.engine_status.SetLabel(
            f"{voice_lab.engine_name(engine_id)} tuning saved"
            + (f": {text}." if text else " (engine defaults).")
        )
        self.Layout()

    def _on_device(self, _evt):
        self._refresh_device_note()
        self._refresh_voices()

    def _refresh_device_note(self):
        device = voicelab.resolve_device(self._selected_device())
        if device == voicelab.DEVICE_CPU:
            text = "Voices are generated on the CPU. This works on every computer."
        else:
            text = ("Voices are generated on the NVIDIA GPU. "
                    "The CPU stays available: choose it per project in the "
                    "Recording window if you need it.")
        if not self._is_installed():
            text += " Install the engine first."
        self.device_note.SetLabel(text)
        self.device_note.Wrap(640)

    def _on_source(self, _evt):
        source = self.source_combo.GetClientData(self.source_combo.GetSelection())
        self.builtin_panel.Show(source != "clone")
        self.clone_panel.Show(source == "clone")
        self.Layout()

    def _on_browse(self, _evt):
        engine_id = self._selected_engine_id()
        if engine_id == "bark":
            wildcard = ("Bark speaker embedding (*.npz)|*.npz|All files (*.*)|*.*")
        else:
            wildcard = ("Audio files (*.wav;*.mp3;*.flac;*.ogg)|*.wav;*.mp3;*.flac;*.ogg"
                        "|All files (*.*)|*.*")
        with wx.FileDialog(
            self,
            "Choose a reference recording",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.sample_ctrl.SetValue(dlg.GetPath())

    # ------------------------------------------------------- install
    def _is_installed(self) -> bool:
        return voicelab.is_installed(self._selected_engine_id())

    def _refresh_engine_status(self):
        engine_id = self._selected_engine_id()
        packages = voicelab.packages_for(engine_id)
        if voicelab.is_installed(engine_id):
            version = voicelab.installed_version(engine_id)
            self.engine_status.SetLabel(
                "Installed"
                + (f" (version {version})" if version else "")
                + ". Its voices can be used now."
            )
            self.install_btn.Disable()
            self.remove_btn.Enable()
        else:
            self.engine_status.SetLabel(
                "Not installed yet. Press 'Install engine...' to add "
                + ", ".join(packages) + "."
            )
            self.install_btn.Enable()
            self.remove_btn.Disable()
            package = voice_lab.probe_package(engine_id)
            if package and not venv_packages.is_known(package, engine=engine_id):
                venv_packages.request(
                    package, engine=engine_id,
                    on_ready=lambda _v: wx.CallAfter(self._refresh_engine_status),
                )
        self._refresh_device_note()

    def _on_install(self, _evt):
        engine_id = self._selected_engine_id()
        packages = voicelab.packages_for(engine_id)
        if not packages:
            return
        gpu_note = ""
        if voice_lab.gpu_supported(engine_id) and voicelab.has_cuda():
            gpu_note = (
                "\n\nAn NVIDIA GPU was detected: the CUDA build of PyTorch is "
                "installed into this engine's environment, so you can run it "
                "on the GPU (or on the CPU - both stay available)."
            )
        if wx.MessageBox(
            f"Install {voice_lab.engine_name(engine_id)}?\n\n"
            "This downloads the Python packages " + ", ".join(packages) +
            " into the engine's own Python environment (each engine gets its "
            "own, so their dependencies never conflict). The engine's model "
            "is downloaded the first time you synthesize with it."
            + gpu_note,
            "Install engine",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        self._install_dialog = self._make_progress_dialog(
            f"Installing {voice_lab.engine_name(engine_id)}", "Starting..."
        )
        self.install_btn.Disable()
        self.remove_btn.Disable()
        threading.Thread(
            target=self._install_job, args=(engine_id,), daemon=True
        ).start()

    def _make_progress_dialog(self, title: str, message: str):
        dlg = wx.Dialog(self, title=title, style=wx.DEFAULT_DIALOG_STYLE,
                        size=(460, 130))
        sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(dlg, label=message)
        sizer.Add(label, 0, wx.ALL | wx.EXPAND, 10)
        gauge = wx.Gauge(dlg, range=0, size=(-1, 22),
                         style=wx.GA_HORIZONTAL | wx.GA_SMOOTH)
        sizer.Add(gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        dlg.SetSizer(sizer)
        dlg.Centre()
        dlg.Show()
        dlg._aivs_label = label  # type: ignore[attr-defined]
        return dlg

    def _install_job(self, engine_id):
        from ..python_runtime import get_runtime  # noqa: PLC0415

        try:
            # Each engine gets its own environment: its packages (and their
            # version pins) cannot break another engine's.
            runtime = get_runtime(engine_id)
            runtime.ensure_pip()

            def progress(message, _done, _total):
                wx.CallAfter(self._update_progress, message)

            notes: list[str] = []
            for packages, index_url in voicelab.install_plan(engine_id):
                result = runtime.pip_install(
                    list(packages), progress=progress, index_url=index_url
                )
                if result.get("ok"):
                    continue
                if index_url:
                    # The CUDA wheel index does not carry a build for this
                    # Python interpreter (it stops at an older version), or
                    # the mirror is unreachable.  Install the same packages
                    # from PyPI instead, so the engine still works on the CPU
                    # rather than failing the whole install.
                    notes.append(
                        "The CUDA build of PyTorch is not available for this "
                        "Python version, so the CPU build was installed "
                        "instead. The engine works, but cannot use the GPU "
                        "until a CUDA build is available."
                    )
                    wx.CallAfter(self._update_progress,
                                 "CUDA PyTorch unavailable - installing the "
                                 "CPU build instead...")
                    result = runtime.pip_install(
                        list(packages), progress=progress, index_url=None
                    )
                    if result.get("ok"):
                        continue
                wx.CallAfter(
                    self._install_done, engine_id,
                    result.get("error") or "Installation failed.", notes,
                )
                return
            # pip succeeding is not the same as a working engine: it can
            # resolve a combination the engine's own code cannot import (a
            # dependency that dropped a module the engine still imports, for
            # example).  Import the engine once, in its own environment, and
            # say so when it fails.
            wx.CallAfter(self._update_progress,
                         "Checking that the engine loads...")
            problem = voicelab.verify_engine_import(engine_id)
            if problem:
                notes.append(problem)
            wx.CallAfter(self._install_done, engine_id, None, notes)
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._install_done, engine_id, str(exc))

    def _update_progress(self, message: str):
        dlg = getattr(self, "_install_dialog", None)
        if dlg is None:
            return
        try:
            dlg._aivs_label.SetLabel((message or "")[:200] or "Working...")
        except Exception:  # noqa: BLE001
            pass

    def _install_done(self, engine_id, error, notes=()):
        dlg = getattr(self, "_install_dialog", None)
        if dlg is not None:
            try:
                dlg.Hide()
                dlg.Destroy()
            except Exception:  # noqa: BLE001
                pass
            self._install_dialog = None
        venv_packages.invalidate(
            *voicelab.packages_for(engine_id), engine=engine_id
        )
        self._refresh_engine_status()
        self._refresh_builtin_voices()
        self._refresh_voices()
        if error:
            dialogs.notify_engine_error(self, "Installation failed", error)
        else:
            message = (
                f"{voice_lab.engine_name(engine_id)} is installed. Its voices "
                "now appear in Available TTS, in the Recording window and in "
                "the Built-in voice list above."
            )
            if notes:
                message += "\n\n" + "\n\n".join(notes)
            wx.MessageBox(
                message,
                "Install engine",
                style=wx.OK | wx.ICON_INFORMATION,
            )

    def _on_remove(self, _evt):
        engine_id = self._selected_engine_id()
        packages = voicelab.packages_for(engine_id)
        if not packages:
            return
        if not self._is_installed():
            return
        if wx.MessageBox(
            f"Remove {voice_lab.engine_name(engine_id)}?\n\n"
            "This uninstalls " + ", ".join(packages) +
            " from the engine's own Python environment. Voices you cloned "
            "with it are kept, but they cannot be spoken until you install "
            "the engine again.",
            "Remove engine",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        self._install_dialog = self._make_progress_dialog(
            f"Removing {voice_lab.engine_name(engine_id)}", "Uninstalling..."
        )
        self.install_btn.Disable()
        self.remove_btn.Disable()
        threading.Thread(
            target=self._remove_job, args=(engine_id, packages), daemon=True
        ).start()

    def _remove_job(self, engine_id, packages):
        from ..python_runtime import get_runtime  # noqa: PLC0415

        try:
            result = get_runtime(engine_id).pip_uninstall(list(packages))
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "Removal failed.")
            # An engine installed into the older shared environment is removed
            # from there as well, so nothing is left behind.
            shared = get_runtime().pip_uninstall(list(packages))
            ok = bool(shared.get("ok")) or not get_runtime().is_created
            wx.CallAfter(
                self._install_done, engine_id,
                None if ok else (shared.get("error") or "Removal failed."),
            )
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._install_done, engine_id, str(exc))

    # ------------------------------------------------------- voices
    def _refresh_builtin_voices(self):
        engine_id = self._selected_engine_id()
        voices = (
            voice_lab.builtin_voices(engine_id)
            if voicelab.is_installed(engine_id)
            else []
        )
        self.builtin_combo.Clear()
        for voice in voices:
            self.builtin_combo.Append(voice["voice_name"], voice)
        if self.builtin_combo.GetCount():
            self.builtin_combo.SetSelection(0)
        if not voicelab.is_installed(engine_id):
            self.builtin_count.SetLabel(
                "Install the engine to use its built-in voices."
            )
        else:
            self.builtin_count.SetLabel(
                f"{len(voices)} built-in "
                f"voice{'s' if len(voices) != 1 else ''} available for this engine."
            )
            self._probe_builtin_availability(engine_id)

    def _probe_builtin_availability(self, engine_id: str):
        """Ask the worker, in the background, which voices are really there.

        F5-TTS speaks with reference files that ship inside its package: the
        probe reports the ones this installation actually has, without
        blocking the panel (the worker may have to start first).
        """
        if engine_id not in _VOICE_PROBE_ENGINES:
            return
        device = voicelab.resolve_device(self._selected_device())

        def job():
            try:
                worker = voicelab.worker_for(engine_id, device)
                available = worker.available_voices(engine_id)
            except Exception:  # noqa: BLE001
                log.debug("Built-in voice probe failed for %s", engine_id,
                          exc_info=True)
                return
            wx.CallAfter(self._show_availability, engine_id, available)

        threading.Thread(target=job, daemon=True,
                         name=f"aivs-voices-{engine_id}").start()

    def _show_availability(self, engine_id: str, available: Dict[str, bool]):
        """Refine the built-in voice count once the probe answers.

        Runs from ``wx.CallAfter``: the panel may already be gone by then (the
        user closed Settings while the probe was in flight), so everything is
        guarded.
        """
        try:
            self._apply_availability(engine_id, available)
        except Exception:  # noqa: BLE001
            log.debug("Could not show built-in voice availability", exc_info=True)

    def _apply_availability(self, engine_id: str, available: Dict[str, bool]):
        if engine_id != self._selected_engine_id():
            return
        voices = voice_lab.builtin_voices(engine_id)
        missing = [v for v in voices if not available.get(v["voice"], True)]
        total = len(voices)
        if not total:
            return
        if missing:
            self.builtin_count.SetLabel(
                f"{total - len(missing)} of {total} built-in voices are ready; "
                f"{len(missing)} need a reference file the installed package "
                "does not include - use 'Clone a voice from a sample' instead."
            )
        else:
            self.builtin_count.SetLabel(
                f"All {total} built-in voices are ready to use."
            )

    def _refresh_voices(self):
        """Voices ready to use: built-in ones plus the user's clones."""
        engine_id = self._selected_engine_id()
        self._voices = []
        if voicelab.is_installed(engine_id):
            self._voices.extend(voice_lab.builtin_voices(engine_id))
        try:
            for entry in self.store.installed_voices():
                if entry.get("engine") == engine_id and entry.get("cloned"):
                    self._voices.append(entry)
        except Exception:  # noqa: BLE001
            log.debug("Could not list cloned voices", exc_info=True)
        self.voices_list.Clear()
        for voice in self._voices:
            kind = "cloned" if voice.get("cloned") else "built-in"
            device = voicelab.resolve_device(self._selected_device())
            self.voices_list.Append(
                f"{voice.get('voice_name', voice.get('voice', ''))} "
                f"({kind}, {self._variant_label(engine_id, voice)}, "
                f"{device.upper()})"
            )
        if self._voices:
            self.voices_list.SetSelection(0)
        self._update_voice_buttons()
        self.preview_status.SetLabel(
            f"{len(self._voices)} voice{'s' if len(self._voices) != 1 else ''} "
            "ready for this engine."
            if self._voices
            else "No voices ready yet. Install the engine or create a clone."
        )

    @staticmethod
    def _variant_label(engine_id: str, voice: Dict[str, Any]) -> str:
        info = voice_lab.engine(engine_id) or {}
        for lang in info.get("languages", []):
            for variant in lang.get("variants", []):
                if variant["id"] == voice.get("variant"):
                    return variant.get("name", variant["id"])
        return voice.get("variant", "")

    def _selected_voice(self) -> Optional[Dict[str, Any]]:
        index = self.voices_list.GetSelection()
        if index < 0 or index >= len(self._voices):
            return None
        return self._voices[index]

    def _on_voice_select(self, _evt):
        self._update_voice_buttons()

    def _update_voice_buttons(self):
        voice = self._selected_voice()
        usable = bool(voice) and voicelab.is_installed(voice.get("engine", ""))
        if usable:
            self.preview_btn.Enable()
        else:
            self.preview_btn.Disable()
        if voice and voice.get("cloned"):
            self.delete_btn.Enable()
        else:
            self.delete_btn.Disable()

    def _on_create(self, _evt):
        engine_id = self._selected_engine_id()
        try:
            entry = voicelab.create_clone(
                self.store,
                engine_id,
                self.name_ctrl.GetValue(),
                self.sample_ctrl.GetValue(),
                self.ref_text_ctrl.GetValue(),
                language=self._default_language(engine_id),
                variant=self._default_variant(engine_id),
            )
        except ValueError as exc:
            dialogs.notify_engine_error(self, "Create voice", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not create the cloned voice")
            dialogs.notify_engine_error(self, "Create voice", str(exc))
            return
        self.name_ctrl.SetValue("")
        self.sample_ctrl.SetValue("")
        self.ref_text_ctrl.SetValue("")
        self._refresh_voices()
        for index, voice in enumerate(self._voices):
            if voice.get("voice_name") == entry.get("name"):
                self.voices_list.SetSelection(index)
                break
        self._update_voice_buttons()
        self.preview_status.SetLabel(
            f"Voice '{entry.get('name')}' created. It is available in the "
            "Recording window and in Available TTS."
        )

    @staticmethod
    def _default_language(engine_id: str) -> str:
        info = voice_lab.engine(engine_id) or {}
        languages = info.get("languages") or []
        return languages[0]["code"] if languages else "en"

    @staticmethod
    def _default_variant(engine_id: str) -> str:
        info = voice_lab.engine(engine_id) or {}
        for lang in info.get("languages", []):
            for variant in lang.get("variants", []):
                return variant["id"]
        return "default"

    def _on_delete(self, _evt):
        voice = self._selected_voice()
        if not voice or not voice.get("cloned"):
            return
        name = voice.get("voice_name", "")
        if wx.MessageBox(
            f"Delete the cloned voice '{name}'? The copy of the reference "
            "recording stored with it is deleted too.",
            "Delete cloned voice",
            style=wx.YES_NO | wx.ICON_QUESTION,
        ) != wx.YES:
            return
        if voicelab.delete_clone(self.store, name):
            self._refresh_voices()
            self.preview_status.SetLabel(f"Deleted '{name}'.")

    # ------------------------------------------------------- preview
    def _on_preview(self, evt=None):
        # One Alt+P press reaches this handler more than once (see
        # access_keys.once): the guard keeps it to one preview.
        if not access_keys.once(evt if evt is not None else self.preview_btn):
            return
        voice = self._selected_voice()
        if not voice:
            return
        if not voicelab.is_installed(voice.get("engine", "")):
            dialogs.notify_engine_error(
                self, "Preview",
                "Install this engine before previewing its voices.",
            )
            return
        text = self.preview_text.GetValue().strip() or "Hello."
        device = voicelab.resolve_device(self._selected_device())
        entry = dict(voice)
        entry["device"] = device
        # Preview with the tuning defaults saved for this engine, so what you
        # hear here is what a new project will sound like.
        applied = voice_options.apply_to_voice(
            entry, self._saved_tuning(entry.get("engine"))
        )
        tuned = voice_options.describe_overrides(
            entry.get("engine", ""), applied.get("options")
        )
        self.preview_btn.Disable()
        self.preview_status.SetLabel(
            "Synthesizing a preview on the "
            + ("GPU" if device == "cuda" else "CPU")
            + (f" with {tuned}" if tuned else " with the engine defaults")
            + "... (the first use downloads the model and can take a while.)"
        )
        threading.Thread(
            target=self._preview_job, args=(applied, text, device), daemon=True
        ).start()

    def _preview_job(self, entry, text, device):
        from ..audio.output import write_wav  # noqa: PLC0415
        from ..tts.engine import (  # noqa: PLC0415
            EngineUnavailableError,
            get_engine,
            process_punctuation,
        )

        try:
            engine = get_engine(entry, provider=device)
            punct = self.settings.get("recording.punctuation", "default")
            samples = engine.synthesize(
                process_punctuation(text, punct), sid=0, speed=1.0
            )
            import tempfile  # noqa: PLC0415

            fd, tmp = tempfile.mkstemp(prefix="aivs_clone_preview_", suffix=".wav")
            os.close(fd)
            write_wav(samples, engine.sample_rate, tmp)
            wx.CallAfter(self._preview_done, tmp, None)
        except EngineUnavailableError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except voicelab.VoicelabError as exc:
            wx.CallAfter(self._preview_done, None, str(exc))
        except Exception as exc:  # noqa: BLE001
            wx.CallAfter(self._preview_done, None, f"Preview failed: {exc}")

    def _preview_done(self, tmp, error):
        import wx.adv  # noqa: PLC0415

        self._update_voice_buttons()
        if error:
            self.preview_status.SetLabel(error)
            dialogs.notify_engine_error(self, "Preview failed", error)
            return
        self._stop_preview_sound()
        sound = wx.adv.Sound(tmp)
        if sound.IsOk():
            # Keep a reference: wxSound must outlive Play(SOUND_ASYNC).
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
